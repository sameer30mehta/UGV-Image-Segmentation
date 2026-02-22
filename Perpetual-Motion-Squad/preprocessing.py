import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 1. Define the Heavy Augmentation Pipeline for Training
# This forces the model to ignore synthetic textures and focus on shapes
train_transform = A.Compose([
    # Geometric: Scale invariance and flipping
    A.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    
    # Photometric: Lighting, Contrast, and Desert Sun simulation
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    
    # Artifacts: Camera noise and motion blur from bumpy UGV rides
    A.MotionBlur(blur_limit=5, p=0.3),
    A.GaussNoise(std_range=(0.012, 0.027), p=0.3),
    A.ISONoise(p=0.3),
    
    # Standard Normalization for ImageNet weights
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Validation Pipeline (No crazy augmentations, just resize and normalize)
val_transform = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# 2. Define the Dataset Class
class DesertSegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.images = sorted(os.listdir(self.image_dir))
        
        # Mapping raw pixel values to class IDs (0 to 9)
        self.value_map = {
            0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 
            550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        # Load Image (Convert BGR to RGB for Albumentations/PyTorch)
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Mask (Keep unchanged to read raw values like 100, 7100, etc.)
        mask_path = os.path.join(self.masks_dir, img_name)
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if raw_mask is None:
            raise FileNotFoundError(f"Could not load mask: {mask_path}")
        
        # Convert raw mask values to 0-9 class IDs
        mask = np.zeros_like(raw_mask, dtype=np.int64)
        for raw_val, class_id in self.value_map.items():
            mask[raw_mask == raw_val] = class_id
            
        # Apply Albumentations transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Mask needs to be long tensor for CrossEntropyLoss
        return image, mask.long()

def get_dataloaders(train_dir, val_dir, batch_size=8, num_workers=2):
    """Creates and returns the training and validation dataloaders."""
    train_dataset = DesertSegmentationDataset(train_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_dataset = DesertSegmentationDataset(val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")
    return train_loader, val_loader

if __name__ == "__main__":
    # Quick test to verify the dataset loads correctly
    TRAIN_DIR = 'Offroad_Segmentation_Training_Dataset/train'
    VAL_DIR = 'Offroad_Segmentation_Training_Dataset/val'
    
    if os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR):
        train_loader, val_loader = get_dataloaders(TRAIN_DIR, VAL_DIR)
        images, masks = next(iter(train_loader))
        print(f"Batch image shape: {images.shape}")
        print(f"Batch mask shape: {masks.shape}")
    else:
        print(f"Dataset not found at {TRAIN_DIR} or {VAL_DIR}. Please check your paths.")