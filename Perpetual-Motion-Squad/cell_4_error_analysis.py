###############################################################################
# CELL 4: ERROR ANALYSIS & PER-CLASS METRICS
# Run this cell after Cell 3 (Testing) completes.
###############################################################################

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
import seaborn as sns

# ============================================================================
# 1. COMPUTE FULL VALIDATION METRICS
# ============================================================================
print("📊 Computing comprehensive validation metrics...")

model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for images, masks, boundaries in tqdm(val_loader, desc="Computing Metrics"):
        images = images.to(DEVICE)
        with torch.amp.autocast('cuda'):
            logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy().flatten()
        targets = masks.cpu().numpy().flatten()
        
        all_preds.append(preds)
        all_targets.append(targets)

all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

# ============================================================================
# 2. PER-CLASS IoU, PRECISION, RECALL, F1
# ============================================================================
def compute_per_class_metrics(preds, targets, num_classes):
    """Computes IoU, Precision, Recall, F1 for each class."""
    metrics = []
    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)
        
        tp = np.sum(pred_c & target_c)
        fp = np.sum(pred_c & ~target_c)
        fn = np.sum(~pred_c & target_c)
        tn = np.sum(~pred_c & ~target_c)
        
        iou = tp / (tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        pixel_acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
        support = np.sum(target_c)
        
        metrics.append({
            'class': c,
            'name': CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'Class_{c}',
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pixel_acc': pixel_acc,
            'support': support
        })
    return metrics

metrics = compute_per_class_metrics(all_preds, all_targets, NUM_CLASSES)

# Print detailed metrics table
print(f"\n{'='*90}")
print(f"{'Class':<20} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Pixel Acc':>10} {'Support':>10}")
print(f"{'='*90}")
for m in metrics:
    print(f"{m['name']:<20} {m['iou']:>8.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} "
          f"{m['f1']:>8.4f} {m['pixel_acc']:>10.4f} {m['support']:>10.0f}")
print(f"{'='*90}")
mean_iou = np.mean([m['iou'] for m in metrics])
mean_f1 = np.mean([m['f1'] for m in metrics])
global_acc = np.sum(all_preds == all_targets) / len(all_targets)
print(f"{'MEAN':<20} {mean_iou:>8.4f} {'':>10} {'':>8} {mean_f1:>8.4f} {global_acc:>10.4f}")
print(f"{'='*90}")

# ============================================================================
# 3. PER-CLASS IoU BAR CHART
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
names = [m['name'] for m in metrics]
ious = [m['iou'] for m in metrics]
colors = [COLORMAP[i] / 255.0 for i in range(NUM_CLASSES)]

bars = ax.bar(names, ious, color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(y=mean_iou, color='red', linestyle='--', linewidth=1.5, label=f'mIoU = {mean_iou:.4f}')
ax.set_ylabel('IoU Score', fontsize=12)
ax.set_title('Per-Class Intersection over Union (IoU)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.legend(fontsize=11)

# Add value labels on bars
for bar, iou in zip(bars, ious):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{iou:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.xticks(rotation=30, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig('/content/predictions/per_class_iou.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: /content/predictions/per_class_iou.png")

# ============================================================================
# 4. CONFUSION MATRIX
# ============================================================================
print("\n📊 Computing Confusion Matrix...")

# Subsample for efficiency if dataset is very large
max_samples = 2_000_000
if len(all_preds) > max_samples:
    indices = np.random.choice(len(all_preds), max_samples, replace=False)
    cm_preds = all_preds[indices]
    cm_targets = all_targets[indices]
else:
    cm_preds = all_preds
    cm_targets = all_targets

cm = confusion_matrix(cm_targets, cm_preds, labels=list(range(NUM_CLASSES)))

# Normalize by row (true class) to get percentages
cm_normalized = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-6)

fig, axes = plt.subplots(1, 2, figsize=(22, 9))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=names, yticklabels=names, linewidths=0.5)
axes[0].set_title('Confusion Matrix (Raw Counts)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Predicted Class', fontsize=11)
axes[0].set_ylabel('True Class', fontsize=11)

# Normalized
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Oranges', ax=axes[1],
            xticklabels=names, yticklabels=names, linewidths=0.5, vmin=0, vmax=1)
axes[1].set_title('Confusion Matrix (Normalized %)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Predicted Class', fontsize=11)
axes[1].set_ylabel('True Class', fontsize=11)

plt.tight_layout()
plt.savefig('/content/predictions/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: /content/predictions/confusion_matrix.png")

# ============================================================================
# 5. TOP CONFUSION PAIRS (Most confused classes)
# ============================================================================
print(f"\n🔍 Top Class Confusions (Off-Diagonal):")
print(f"{'True Class':<20} → {'Predicted As':<20} {'Confusion Rate':>15}")
print("-" * 60)

confusion_pairs = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and cm_normalized[i, j] > 0.01:  # More than 1% confusion
            confusion_pairs.append((names[i], names[j], cm_normalized[i, j]))

confusion_pairs.sort(key=lambda x: x[2], reverse=True)
for true_cls, pred_cls, rate in confusion_pairs[:10]:
    print(f"  {true_cls:<20} → {pred_cls:<20} {rate:>14.2%}")

# ============================================================================
# 6. TRAINING CURVES
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Loss curves
axes[0].plot(history['train_loss'], label='Train Loss', color='#2196F3', linewidth=2)
axes[0].plot(history['val_loss'], label='Val Loss', color='#F44336', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# mIoU curves
axes[1].plot(history['train_miou'], label='Train mIoU', color='#4CAF50', linewidth=2)
axes[1].plot(history['val_miou'], label='Val mIoU', color='#FF9800', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('mIoU')
axes[1].set_title('Training & Validation mIoU', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Learning rate
axes[2].plot(history['lr'], color='#9C27B0', linewidth=2)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Learning Rate')
axes[2].set_title('OneCycleLR Schedule', fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/predictions/training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: /content/predictions/training_curves.png")

# ============================================================================
# 7. FAILURE CASE ANALYSIS
# ============================================================================
print("\n🔍 Failure Case Analysis:")
print("=" * 60)

# Identify weakest and strongest classes
sorted_by_iou = sorted(metrics, key=lambda x: x['iou'])
weakest = sorted_by_iou[:3]
strongest = sorted_by_iou[-3:]

print("\n❌ WEAKEST Classes (Lowest IoU):")
for m in weakest:
    print(f"   {m['name']:<20} IoU: {m['iou']:.4f}  Recall: {m['recall']:.4f}  Support: {m['support']:.0f}")
    # Find what this class is most confused with
    c = m['class']
    top_confusion_idx = np.argsort(cm_normalized[c])[::-1]
    confused_with = [(names[j], cm_normalized[c, j]) for j in top_confusion_idx if j != c and cm_normalized[c, j] > 0.01]
    if confused_with:
        print(f"     → Most confused with: {confused_with[0][0]} ({confused_with[0][1]:.1%})")

print(f"\n✅ STRONGEST Classes (Highest IoU):")
for m in strongest:
    print(f"   {m['name']:<20} IoU: {m['iou']:.4f}  Recall: {m['recall']:.4f}  Support: {m['support']:.0f}")

# ============================================================================
# 8. FINAL SUMMARY
# ============================================================================
print(f"\n{'='*60}")
print(f"📋 FINAL MODEL PERFORMANCE SUMMARY")
print(f"{'='*60}")
print(f"  Model:           FPN + MiT-B3 (SegFormer Transformer)")
print(f"  Input Size:      {IMG_SIZE}x{IMG_SIZE}")
print(f"  Num Classes:     {NUM_CLASSES}")
print(f"  Training Epochs: {NUM_EPOCHS}")
print(f"  Loss Function:   WCE + Focal + Dice + Boundary")
print(f"  Optimizer:       AdamW + OneCycleLR")
print(f"  Mixed Precision: Enabled (FP16 AMP)")
print(f"  ")
print(f"  Global Pixel Accuracy: {global_acc:.4f}")
print(f"  Mean IoU (mIoU):       {mean_iou:.4f}")
print(f"  Mean F1-Score:         {mean_f1:.4f}")
print(f"{'='*60}")

# ============================================================================
# 9. VISUALIZE HARDEST VALIDATION SAMPLES (highest error rate)
# ============================================================================
print("\n🔍 Finding hardest validation samples...")

sample_errors = []
for i in range(min(100, len(val_dataset))):
    img_tensor, mask_tensor, _ = val_dataset[i]
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            logits = model(img_tensor.unsqueeze(0).to(DEVICE))
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    gt = mask_tensor.numpy()
    error_rate = 1.0 - np.mean(pred == gt)
    sample_errors.append((i, error_rate))

sample_errors.sort(key=lambda x: x[1], reverse=True)
hardest = sample_errors[:4]

fig, axes = plt.subplots(len(hardest), 4, figsize=(20, 5 * len(hardest)))
if len(hardest) == 1:
    axes = axes[np.newaxis, :]

for row, (idx, err_rate) in enumerate(hardest):
    img_tensor, mask_tensor, _ = val_dataset[idx]
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            logits = model(img_tensor.unsqueeze(0).to(DEVICE))
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    
    original = denormalize(img_tensor)
    gt_rgb = mask_to_rgb(mask_tensor.numpy())
    pred_rgb = mask_to_rgb(pred)
    
    correct = (pred == mask_tensor.numpy())
    diff_map = np.zeros((*correct.shape, 3), dtype=np.uint8)
    diff_map[correct] = [0, 200, 0]
    diff_map[~correct] = [255, 0, 0]
    
    axes[row, 0].imshow(original)
    axes[row, 0].set_title(f"Input (Error: {err_rate:.1%})", fontsize=10)
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

plt.suptitle("🔴 Hardest Validation Samples (Highest Error Rate)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/content/predictions/hardest_samples.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: /content/predictions/hardest_samples.png")
