###############################################################################
# CELL 2: MODEL, LOSS, & TRAINING PIPELINE
# Run this cell after Cell 1 (Preprocessing) completes successfully.
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import torch.optim as optim
from tqdm.notebook import tqdm
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# 1. MODEL: FPN + MiT-B3 (CNN/Transformer Hybrid from SegFormer family)
# ============================================================================
print(f"\n🏗️  Building FPN-Transformer Network for {NUM_CLASSES} classes...")

model = smp.FPN(
    encoder_name="mit_b3",          # Mix Vision Transformer backbone (SegFormer)
    encoder_weights="imagenet",     # ImageNet pre-training for fast convergence
    in_channels=3,
    classes=NUM_CLASSES,
    decoder_dropout=0.2             # Regularization against synthetic memorization
)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total params:     {total_params / 1e6:.1f}M")
print(f"   Trainable params: {trainable_params / 1e6:.1f}M")

# ============================================================================
# 2. COMPOSITE LOSS: Weighted CE + Focal + Dice + Boundary (from report)
# ============================================================================
class FocalLoss(nn.Module):
    """Focal Loss: down-weights easy pixels, focuses on hard/ambiguous ones."""
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight,
                                  ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CompositeLoss(nn.Module):
    """
    Multi-objective loss topology (aligned with report):
      L_total = λ1 * WCE + λ2 * Focal + λ3 * Dice + λ4 * Boundary
    """
    def __init__(self, class_weights, num_classes, lambda_wce=0.3, lambda_focal=0.3,
                 lambda_dice=0.3, lambda_boundary=0.1):
        super().__init__()
        
        weight_tensor = torch.FloatTensor(class_weights).to(DEVICE)
        
        # Weighted Cross-Entropy (inverse frequency weights for class imbalance)
        self.wce_loss = nn.CrossEntropyLoss(weight=weight_tensor)
        
        # Focal Loss (dynamic hard-pixel mining)
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0, weight=weight_tensor)
        
        # Dice Loss (spatial overlap, IoU proxy)
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass')
        
        # Boundary Loss (BCEWithLogits on edge confidence)
        self.boundary_loss = nn.BCEWithLogitsLoss()
        
        self.lambda_wce = lambda_wce
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.lambda_boundary = lambda_boundary

    def forward(self, logits, masks, boundary_gt):
        # Segmentation losses
        loss_wce = self.wce_loss(logits, masks)
        loss_focal = self.focal_loss(logits, masks)
        loss_dice = self.dice_loss(logits, masks)
        
        # Boundary loss: low-confidence regions should align with actual edges
        probs = torch.softmax(logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        edge_proxy = 1.0 - max_probs
        loss_boundary = self.boundary_loss(edge_proxy, boundary_gt)
        
        total = (self.lambda_wce * loss_wce +
                 self.lambda_focal * loss_focal +
                 self.lambda_dice * loss_dice +
                 self.lambda_boundary * loss_boundary)
        
        return total, {
            'wce': loss_wce.item(),
            'focal': loss_focal.item(),
            'dice': loss_dice.item(),
            'boundary': loss_boundary.item(),
            'total': total.item()
        }

criterion = CompositeLoss(CLASS_WEIGHTS, NUM_CLASSES)

# ============================================================================
# 3. OPTIMIZER & SCHEDULER (OneCycleLR from report)
# ============================================================================
NUM_EPOCHS = 20
MAX_LR = 3e-4

optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=1e-4)

# OneCycleLR: warm-up then cosine decay (report-aligned)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=MAX_LR,
    steps_per_epoch=len(train_loader),
    epochs=NUM_EPOCHS,
    pct_start=0.15,     # Warm-up for first 15% of training
    anneal_strategy='cos'
)

# Mixed Precision (AMP) for T4 Tensor Core acceleration
scaler = torch.amp.GradScaler('cuda')

# ============================================================================
# 4. METRIC: Mean Intersection over Union (mIoU)
# ============================================================================
class IoUMetric:
    """Computes per-class IoU and mIoU."""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)
    
    def update(self, preds, targets):
        preds = preds.argmax(dim=1).cpu()
        targets = targets.cpu()
        for c in range(self.num_classes):
            pred_c = (preds == c)
            target_c = (targets == c)
            self.intersection[c] += (pred_c & target_c).sum().float()
            self.union[c] += (pred_c | target_c).sum().float()
    
    def compute(self):
        iou = self.intersection / (self.union + 1e-6)
        return iou, iou.mean()

# ============================================================================
# 5. TRAINING LOOP
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    metric = IoUMetric(NUM_CLASSES)
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for images, masks, boundaries in pbar:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        boundaries = boundaries.to(DEVICE)
        
        optimizer.zero_grad(set_to_none=True)
        
        # AMP forward pass
        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss, loss_dict = criterion(logits, masks, boundaries)
        
        # AMP backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        running_loss += loss.item()
        metric.update(logits.detach(), masks)
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.6f}"
        })
    
    _, miou = metric.compute()
    avg_loss = running_loss / len(loader)
    return avg_loss, miou.item()


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    metric = IoUMetric(NUM_CLASSES)
    
    for images, masks, boundaries in tqdm(loader, desc="Validating", leave=False):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        boundaries = boundaries.to(DEVICE)
        
        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss, _ = criterion(logits, masks, boundaries)
        
        running_loss += loss.item()
        metric.update(logits, masks)
    
    per_class_iou, miou = metric.compute()
    avg_loss = running_loss / len(loader)
    return avg_loss, miou.item(), per_class_iou

# ============================================================================
# 6. MAIN TRAINING LOOP
# ============================================================================
print(f"\n🚀 Starting training: {NUM_EPOCHS} epochs on {DEVICE}")
print(f"   AMP: Enabled (FP16 Tensor Core acceleration)")
print(f"   Scheduler: OneCycleLR (max_lr={MAX_LR})")
print("=" * 60)

history = {'train_loss': [], 'val_loss': [], 'train_miou': [], 'val_miou': [], 'lr': []}
best_miou = 0.0
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    
    # Train
    train_loss, train_miou = train_one_epoch(
        model, train_loader, criterion, optimizer, scheduler, scaler, epoch, NUM_EPOCHS
    )
    
    # Validate
    val_loss, val_miou, per_class_iou = validate(model, val_loader, criterion)
    
    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_miou'].append(train_miou)
    history['val_miou'].append(val_miou)
    history['lr'].append(scheduler.get_last_lr()[0])
    
    epoch_time = time.time() - epoch_start
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}  |  Time: {epoch_time:.0f}s  |  Total: {total_time/60:.1f}min")
    print(f"  Train Loss: {train_loss:.4f}  |  Train mIoU: {train_miou:.4f}")
    print(f"  Val   Loss: {val_loss:.4f}  |  Val   mIoU: {val_miou:.4f}")
    
    # Print per-class IoU
    print(f"\n  Per-Class IoU:")
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'Class_{c}'
        print(f"    {name:20s}: {per_class_iou[c]:.4f}")
    
    # Save best model
    if val_miou > best_miou:
        best_miou = val_miou
        torch.save(model.state_dict(), '/content/best_desert_segmentation.pth')
        print(f"\n  ⭐ New best model saved! mIoU: {best_miou:.4f}")
    
    # Save latest checkpoint
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_miou': best_miou,
        'history': history,
        'value_to_class': VALUE_TO_CLASS,
        'class_names': CLASS_NAMES,
        'num_classes': NUM_CLASSES,
    }, '/content/latest_checkpoint.pth')
    
    torch.cuda.empty_cache()

total_training_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"✅ Training Complete!")
print(f"   Total time: {total_training_time/60:.1f} minutes")
print(f"   Best Val mIoU: {best_miou:.4f}")
print(f"   Model saved to: /content/best_desert_segmentation.pth")
