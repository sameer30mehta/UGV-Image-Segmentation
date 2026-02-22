"""
Rapid Few-Shot Fine-Tuning Module
Clean, modular implementation for personalized model training
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from PIL import Image
import os
import base64
from io import BytesIO
import numpy as np
import torchvision.transforms.functional as TF
from backend.gradcam import SegmentationGradCAM, apply_heatmap


NUM_CLASSES = 10
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 1e-5


class SimpleSegDataset(Dataset):
    """Minimal dataset for images and masks"""
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = img_name.rsplit('.', 1)[0] + '.png'
        
        img_path = os.path.join(self.images_path, img_name)
        mask_path = os.path.join(self.masks_path, mask_name)
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        target_size = (512, 512)
        image = image.resize(target_size, Image.BILINEAR)
        mask = mask.resize(target_size, Image.NEAREST)
        
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        mask = torch.tensor(list(mask.getdata())).reshape(target_size[0], target_size[1]).long()
        mask = torch.clamp(mask, 0, NUM_CLASSES - 1)
        
        return image, mask


class CombinedLoss(nn.Module):
    """CE + Dice loss"""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = smp.losses.DiceLoss(mode='multiclass')
    
    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)


def load_base_model(model_path):
    """Load pretrained segmentation model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = smp.FPN(
        encoder_name='mit_b3',
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        decoder_dropout=0.2
    )
    
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        model = smp.FPN(
            encoder_name='mit_b3',
            encoder_weights='imagenet',
            in_channels=3,
            classes=NUM_CLASSES,
            decoder_dropout=0.2
        )
    
    model = model.to(device)
    
    return model, device


def freeze_backbone(model):
    """Freeze all layers except final segmentation head for fast training"""
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Freeze decoder
    for param in model.decoder.parameters():
        param.requires_grad = False
    
    # Only train segmentation head (last layer)
    for param in model.segmentation_head.parameters():
        param.requires_grad = True
    
    return model


def prepare_dataloader(dataset_path):
    """Create dataloader from dataset directory"""
    images_path = os.path.join(dataset_path, 'images')
    masks_path = os.path.join(dataset_path, 'masks')
    
    dataset = SimpleSegDataset(images_path, masks_path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    return dataloader


def run_finetune(model, dataloader, device):
    """Fine-tune model for specified epochs with metrics tracking"""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    criterion = CombinedLoss()
    
    model.train()
    
    metrics = {
        'epoch_losses': [],
        'final_loss': 0.0,
        'initial_loss': 0.0,
        'improvement': 0.0
    }
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        metrics['epoch_losses'].append(round(avg_loss, 4))
        
        if epoch == 0:
            metrics['initial_loss'] = avg_loss
        if epoch == EPOCHS - 1:
            metrics['final_loss'] = avg_loss
    
    if metrics['initial_loss'] > 0:
        metrics['improvement'] = round(
            ((metrics['initial_loss'] - metrics['final_loss']) / metrics['initial_loss']) * 100, 2
        )
    
    return model, metrics


def save_user_model(model, user_id):
    """Save personalized model to models directory"""
    os.makedirs('models', exist_ok=True)
    save_path = f'models/user_{user_id}.pth'
    torch.save(model.state_dict(), save_path)
    return save_path


def generate_comparison(base_model, finetuned_model, dataset_path, device):
    """Generate before/after comparison with GradCAM visualization"""
    try:
        # Load first image from dataset
        images_path = os.path.join(dataset_path, 'images')
        image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        if not image_files:
            return None
        
        # Load and preprocess image
        img_path = os.path.join(images_path, image_files[0])
        original_img = Image.open(img_path).convert('RGB')
        original_img = original_img.resize((512, 512), Image.BILINEAR)
        original_np = np.array(original_img)
        
        # Prepare tensor
        img_tensor = TF.to_tensor(original_img)
        img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Before: Base model inference + GradCAM
        base_model.eval()
        with torch.no_grad():
            base_pred = base_model(img_tensor).argmax(dim=1).squeeze().cpu().numpy()
        
        base_gradcam = SegmentationGradCAM(base_model)
        base_cam, _ = base_gradcam.generate(img_tensor)
        base_overlay = apply_heatmap(original_np, base_cam, alpha=0.4)
        base_gradcam.cleanup()
        
        # After: Fine-tuned model inference + GradCAM
        finetuned_model.eval()
        with torch.no_grad():
            finetuned_pred = finetuned_model(img_tensor).argmax(dim=1).squeeze().cpu().numpy()
        
        finetuned_gradcam = SegmentationGradCAM(finetuned_model)
        finetuned_cam, _ = finetuned_gradcam.generate(img_tensor)
        finetuned_overlay = apply_heatmap(original_np, finetuned_cam, alpha=0.4)
        finetuned_gradcam.cleanup()
        
        # Convert to base64
        def img_to_base64(img_array):
            img = Image.fromarray(img_array.astype(np.uint8))
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode()
        
        return {
            'original': img_to_base64(original_np),
            'base_gradcam': img_to_base64(base_overlay),
            'finetuned_gradcam': img_to_base64(finetuned_overlay)
        }
    except Exception as e:
        print(f"Comparison generation failed: {e}")
        return None


def run_finetune_pipeline(dataset_path, user_id):
    """Main entry point for fine-tuning pipeline with progress tracking"""
    base_model_path = 'backend/weights/best_desert_segmentation.pth'
    
    # Load base model (keep copy for comparison)
    base_model, device = load_base_model(base_model_path)
    
    # Load model for training
    model, _ = load_base_model(base_model_path)
    
    # Freeze all except last layer
    model = freeze_backbone(model)
    
    # Prepare data
    dataloader = prepare_dataloader(dataset_path)
    
    # Train
    model, metrics = run_finetune(model, dataloader, device)
    
    # Generate comparison
    comparison = generate_comparison(base_model, model, dataset_path, device)
    
    # Save
    save_path = save_user_model(model, user_id)
    
    return save_path, metrics, comparison
