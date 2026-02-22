###############################################################################
# CELL 3: TESTING / INFERENCE ON UNSEEN TEST DATA
# Run this cell after training completes.
###############################################################################

import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# ============================================================================
# 1. LOAD BEST MODEL
# ============================================================================
model.load_state_dict(torch.load('/content/best_desert_segmentation.pth', map_location=DEVICE))
model.eval()
print("✅ Best model loaded for inference.")

# ============================================================================
# 2. COLORMAP FOR VISUALIZATION
# ============================================================================
# Distinct colors for each class (visually separable)
COLORMAP = np.array([
    [139, 119,  101],  # 0: Landscape  - tan
    [135, 206,  235],  # 1: Sky        - sky blue
    [  0, 128,    0],  # 2: Trees      - dark green
    [ 50, 205,   50],  # 3: Lush Bushes - lime green
    [218, 165,   32],  # 4: Dry Grass  - goldenrod
    [160,  82,   45],  # 5: Dry Bushes - sienna
    [128, 128,  128],  # 6: Rocks      - gray
    [210, 180,  140],  # 7: Ground Clutter - wheat
    [255,   0,  255],  # 8: Flowers    - magenta
    [139,  69,   19],  # 9: Logs       - saddle brown
], dtype=np.uint8)

# If NUM_CLASSES > 10, extend colormap
while len(COLORMAP) < NUM_CLASSES:
    COLORMAP = np.vstack([COLORMAP, np.random.randint(50, 200, (1, 3), dtype=np.uint8)])

def mask_to_rgb(mask_tensor):
    """Convert class index mask to RGB visualization."""
    if isinstance(mask_tensor, torch.Tensor):
        mask = mask_tensor.cpu().numpy()
    else:
        mask = mask_tensor
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        rgb[mask == c] = COLORMAP[c]
    return rgb


def denormalize(tensor):
    """Undo ImageNet normalization for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.permute(1, 2, 0).numpy()
    return np.clip(img * 255, 0, 255).astype(np.uint8)

# ============================================================================
# 3. INFERENCE ON TEST SET
# ============================================================================
TEST_IMG_DIR_PATH = TEST_IMG_DIR  # from Cell 1
valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
test_images = sorted([f for f in os.listdir(TEST_IMG_DIR_PATH) if f.lower().endswith(valid_ext)])

print(f"\n📸 Running inference on {len(test_images)} test images...")

os.makedirs('/content/predictions', exist_ok=True)
os.makedirs('/content/predictions/masks', exist_ok=True)
os.makedirs('/content/predictions/overlays', exist_ok=True)

test_preprocessing = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

all_predictions = []

for img_name in tqdm(test_images, desc="Test Inference"):
    # Load original image
    img_path = os.path.join(TEST_IMG_DIR_PATH, img_name)
    original = cv2.imread(img_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = original.shape[:2]
    
    # Preprocess
    sample = test_preprocessing(image=original)
    img_tensor = sample['image'].unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            logits = model(img_tensor)
    
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    
    # Resize prediction back to original size
    pred_fullres = cv2.resize(pred.astype(np.uint8), (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)
    
    all_predictions.append({'name': img_name, 'pred': pred_fullres})
    
    # Save prediction mask
    pred_rgb = mask_to_rgb(pred_fullres)
    cv2.imwrite(f'/content/predictions/masks/{img_name}', cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
    
    # Save overlay (original + semi-transparent prediction)
    overlay = cv2.addWeighted(original, 0.5, 
                               cv2.resize(pred_rgb, (orig_w, orig_h)), 0.5, 0)
    cv2.imwrite(f'/content/predictions/overlays/{img_name}', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print(f"✅ Saved {len(test_images)} predictions to /content/predictions/masks/ and /content/predictions/overlays/")

# ============================================================================
# 4. VISUALIZE SAMPLE PREDICTIONS
# ============================================================================
def show_predictions(predictions, num_samples=6):
    """Display a grid of test predictions."""
    n = min(num_samples, len(predictions))
    fig, axes = plt.subplots(n, 3, figsize=(18, 6 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    
    for i in range(n):
        entry = predictions[i]
        img_path = os.path.join(TEST_IMG_DIR_PATH, entry['name'])
        original = cv2.imread(img_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        pred_rgb = mask_to_rgb(entry['pred'])
        overlay = cv2.addWeighted(original, 0.5,
                                   cv2.resize(pred_rgb, (original.shape[1], original.shape[0])), 0.5, 0)
        
        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f"Input: {entry['name']}", fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(pred_rgb)
        axes[i, 1].set_title("Predicted Mask", fontsize=10)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay", fontsize=10)
        axes[i, 2].axis('off')
    
    # Legend
    legend_patches = []
    import matplotlib.patches as mpatches
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'Class_{c}'
        color = COLORMAP[c] / 255.0
        legend_patches.append(mpatches.Patch(color=color, label=name))
    fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.savefig('/content/predictions/sample_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📊 Saved to /content/predictions/sample_predictions.png")

show_predictions(all_predictions, num_samples=6)

# ============================================================================
# 5. VISUALIZE VALIDATION SET (with Ground Truth comparison)
# ============================================================================
def show_val_predictions(model, val_dataset, num_samples=4):
    """Show validation predictions alongside ground truth."""
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    
    indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    
    for row, idx in enumerate(indices):
        img_tensor, mask_tensor, _ = val_dataset[idx]
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                logits = model(img_tensor.unsqueeze(0).to(DEVICE))
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        
        original = denormalize(img_tensor)
        gt_rgb = mask_to_rgb(mask_tensor.numpy())
        pred_rgb = mask_to_rgb(pred)
        
        # Difference map (red = wrong, green = correct)
        correct = (pred == mask_tensor.numpy())
        diff_map = np.zeros((*correct.shape, 3), dtype=np.uint8)
        diff_map[correct] = [0, 200, 0]    # Green = correct
        diff_map[~correct] = [255, 0, 0]   # Red = incorrect
        
        axes[row, 0].imshow(original)
        axes[row, 0].set_title("Input Image", fontsize=10)
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(gt_rgb)
        axes[row, 1].set_title("Ground Truth", fontsize=10)
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(pred_rgb)
        axes[row, 2].set_title("Prediction", fontsize=10)
        axes[row, 2].axis('off')
        
        axes[row, 3].imshow(diff_map)
        axes[row, 3].set_title("Error Map (Red=Wrong)", fontsize=10)
        axes[row, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('/content/predictions/val_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📊 Saved to /content/predictions/val_comparison.png")

show_val_predictions(model, val_dataset, num_samples=4)
