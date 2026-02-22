###############################################################################
# CELL 1: PREPROCESSING & DATASET PIPELINE
# This cell defines the Dataset, all augmentations (aligned with the report),
# and the class mapping logic.
###############################################################################

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.notebook import tqdm
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================
TRAIN_DIR = '/content/dataset/train/Offroad_Segmentation_Training_Dataset/train'
VAL_DIR   = '/content/dataset/train/Offroad_Segmentation_Training_Dataset/val'
TEST_DIR  = '/content/dataset/test/Offroad_Segmentation_testImages'

TRAIN_IMG_DIR  = os.path.join(TRAIN_DIR, 'Color_Images')
TRAIN_MASK_DIR = os.path.join(TRAIN_DIR, 'Segmentation')
VAL_IMG_DIR    = os.path.join(VAL_DIR, 'Color_Images')
VAL_MASK_DIR   = os.path.join(VAL_DIR, 'Segmentation')
TEST_IMG_DIR   = os.path.join(TEST_DIR, 'Color_Images')
TEST_MASK_DIR  = os.path.join(TEST_DIR, 'Segmentation')

# Desert segmentation classes (from the problem statement)
CLASS_NAMES = [
    'Landscape',    # 0 - general ground
    'Sky',          # 1
    'Trees',        # 2
    'Lush Bushes',  # 3
    'Dry Grass',    # 4
    'Dry Bushes',   # 5
    'Rocks',        # 6
    'Ground Clutter',# 7
    'Flowers',      # 8
    'Logs',         # 9
]

IMG_SIZE = 512  # Crop size for training (fits T4 memory with AMP)
BATCH_SIZE = 4

# ============================================================================
# STEP 1: AUTO-DISCOVER MASK VALUES & BUILD CLASS MAPPING
# ============================================================================
def discover_class_mapping(mask_dir, max_scan=500):
    """Scans training masks to find all unique pixel values and builds a mapping."""
    valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(valid_ext)]
    
    all_vals = set()
    pixel_counts = Counter()
    
    for m in tqdm(mask_files[:max_scan], desc="Scanning masks for class mapping"):
        mask = cv2.imread(os.path.join(mask_dir, m), 0)
        unique, counts = np.unique(mask, return_counts=True)
        all_vals.update(unique.tolist())
        for v, c in zip(unique, counts):
            pixel_counts[int(v)] += int(c)
    
    sorted_vals = sorted(list(all_vals))
    # Build mapping: raw pixel value -> sequential class index (0, 1, 2, ...)
    value_to_class = {v: idx for idx, v in enumerate(sorted_vals)}
    
    num_classes = len(sorted_vals)
    print(f"\n✅ Found {num_classes} unique mask values: {sorted_vals}")
    print(f"   Mapping: {value_to_class}")
    
    # Compute class weights (inverse frequency) for Weighted Cross-Entropy
    total_pixels = sum(pixel_counts.values())
    class_weights = []
    for v in sorted_vals:
        freq = pixel_counts[v] / total_pixels
        # Inverse frequency, capped to avoid extreme weights
        weight = min(1.0 / (freq + 1e-6), 50.0)
        class_weights.append(weight)
    
    # Normalize weights so they average to 1.0
    mean_w = np.mean(class_weights)
    class_weights = [w / mean_w for w in class_weights]

    print(f"   Class weights: {[f'{w:.2f}' for w in class_weights]}")
    
    return value_to_class, num_classes, class_weights

# Run the scan
VALUE_TO_CLASS, NUM_CLASSES, CLASS_WEIGHTS = discover_class_mapping(TRAIN_MASK_DIR)

# Adjust CLASS_NAMES to match actual number of classes found
if len(CLASS_NAMES) != NUM_CLASSES:
    print(f"\n⚠️  Found {NUM_CLASSES} classes but CLASS_NAMES has {len(CLASS_NAMES)} entries.")
    print(f"   Auto-generating generic names for {NUM_CLASSES} classes.")
    CLASS_NAMES = [f'Class_{i}' for i in range(NUM_CLASSES)]

# ============================================================================
# STEP 2: DATASET CLASS
# ============================================================================
class DesertSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, value_to_class, augmentation=None, preprocessing=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.value_to_class = value_to_class
        self.num_classes = len(value_to_class)
        
        valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        img_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(valid_ext)])
        
        # Robust pairing by filename
        mask_dict = {os.path.splitext(m)[0]: m for m in mask_files}
        self.images, self.masks = [], []
        for img in img_files:
            base = os.path.splitext(img)[0]
            if base in mask_dict:
                self.images.append(img)
                self.masks.append(mask_dict[base])
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.images)

    def remap_mask(self, mask):
        """Maps raw grayscale pixel values to sequential class indices."""
        remapped = np.zeros_like(mask, dtype=np.uint8)
        for raw_val, class_idx in self.value_to_class.items():
            remapped[mask == raw_val] = class_idx
        return remapped

    def extract_boundaries(self, mask):
        """Sobel boundary map for boundary-aware loss."""
        mask_float = mask.astype(np.float32)
        sx = cv2.Sobel(mask_float, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(mask_float, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.clip(cv2.magnitude(sx, sy), 0, 1)
        return edges.astype(np.float32)

    def __getitem__(self, i):
        image = cv2.imread(os.path.join(self.image_dir, self.images[i]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, self.masks[i]), 0)
        
        # Remap mask values to 0..N-1
        mask = self.remap_mask(mask)
        
        # Fix shape mismatch
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Boundary map (computed BEFORE tensor conversion)
        boundary = self.extract_boundaries(mask)

        # Preprocessing (Normalize + ToTensor)
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Manual tensor conversion for boundary (avoids Albumentations shape crash)
        boundary = torch.from_numpy(boundary).float()
        
        # Safety clamp
        mask = torch.clamp(mask.long(), 0, self.num_classes - 1)

        return image, mask, boundary

# ============================================================================
# STEP 3: AUGMENTATION PIPELINES (Aligned with Report)
# ============================================================================
def get_training_augmentation():
    """
    Production-grade domain randomization pipeline:
    - Geometric: Scale, Crop, Flip, Rotate
    - Photometric: CLAHE, Gamma, ColorJitter, Grayscale
    - Sensor Noise: GaussNoise, GaussianBlur, MotionBlur, JPEG
    - Physics-Based: RandomShadow, CoarseDropout
    """
    return A.Compose([
        # === Geometric ===
        A.SmallestMaxSize(max_size=768, always_apply=True),
        A.RandomScale(scale_limit=(-0.25, 0.5), p=0.5),
        A.RandomCrop(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),

        # === Photometric (Domain Randomization) ===
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.5),
        A.ToGray(p=0.2),             # Force color invariance (report: RandomGrayscale)

        # === Sensor Noise Simulation ===
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.MotionBlur(blur_limit=5, p=0.15),  # UGV vibration simulation
        A.ImageCompression(quality_lower=75, quality_upper=100, p=0.1),

        # === Physics-Based Shadow & Occlusion ===
        A.RandomShadow(
            shadow_roi=(0, 0.3, 1, 1),     # Shadows on bottom 70% of image
            num_shadows_limit=(1, 3),
            shadow_dimension=5,
            p=0.3
        ),
        A.CoarseDropout(                   # Random erasing for occlusion robustness
            max_holes=6, max_height=40, max_width=40,
            min_holes=1, min_height=10, min_width=10,
            fill_value=0, p=0.2
        ),
    ])

def get_validation_augmentation():
    """Deterministic resize + center crop for validation."""
    return A.Compose([
        A.SmallestMaxSize(max_size=768, always_apply=True),
        A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
    ])

def get_test_augmentation():
    """Simple resize for test inference."""
    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
    ])

def get_preprocessing():
    """ImageNet normalization + tensor conversion."""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# ============================================================================
# STEP 4: CREATE DATASETS & DATALOADERS
# ============================================================================
train_dataset = DesertSegmentationDataset(
    TRAIN_IMG_DIR, TRAIN_MASK_DIR, VALUE_TO_CLASS,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing()
)

val_dataset = DesertSegmentationDataset(
    VAL_IMG_DIR, VAL_MASK_DIR, VALUE_TO_CLASS,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing()
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)

print(f"\n📊 Dataset Summary:")
print(f"   Training:   {len(train_dataset)} images")
print(f"   Validation: {len(val_dataset)} images")
print(f"   Classes:    {NUM_CLASSES}")
print(f"   Crop size:  {IMG_SIZE}x{IMG_SIZE}")
print(f"   Batch size: {BATCH_SIZE}")

# Quick sanity check
img, mask, boundary = train_dataset[0]
print(f"\n✅ Sanity Check Passed:")
print(f"   Image:    {img.shape}  (C, H, W)")
print(f"   Mask:     {mask.shape}  (H, W), range [{mask.min()}, {mask.max()}]")
print(f"   Boundary: {boundary.shape}  (H, W)")
