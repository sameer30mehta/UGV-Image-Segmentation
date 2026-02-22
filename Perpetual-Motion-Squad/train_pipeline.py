import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from tqdm import tqdm
import os

# Import the dataloaders from our preprocessing script
from preprocessing import get_dataloaders

def train_model(train_dir, val_dir, epochs=20, batch_size=8, learning_rate=1e-4):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cpu':
        print("WARNING: Training on CPU will be very slow. Please ensure CUDA is available.")

    # 2. Get DataLoaders
    train_loader, val_loader = get_dataloaders(train_dir, val_dir, batch_size=batch_size)

    # 3. Initialize Model: Unet with SegFormer (MiT-B2) backbone
    print("Initializing SegFormer model...")
    model = smp.Unet(
        encoder_name="mit_b2",        # SegFormer backbone (excellent for domain generalization)
        encoder_weights="imagenet",   # Pre-trained on ImageNet
        in_channels=3,                # RGB images
        classes=10                    # 10 classes in your dataset
    ).to(device)

    # 4. Define Hybrid Loss (CrossEntropy + Dice)
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = smp.losses.DiceLoss(mode='multiclass')

    def criterion(y_pred, y_true):
        return 0.5 * ce_loss(y_pred, y_true) + 0.5 * dice_loss(y_pred, y_true)

    # 5. Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 6. AMP Scaler for Free GPU Memory Optimization
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # 7. Training Loop
    best_val_loss = float('inf')
    save_path = "best_segformer_model.pth"

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc="Training")
        for images, masks in train_bar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Mixed Precision Forward Pass (if using CUDA)
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                # Mixed Precision Backward Pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard Forward/Backward Pass (for CPU)
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validation")
            for images, masks in val_bar:
                images, masks = images.to(device), masks.to(device)
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())
                
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step() # Update learning rate
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model to {save_path}!")

    print(f"\nTraining Complete! Best model saved as '{save_path}'")

if __name__ == "__main__":
    # Define paths to your local dataset
    TRAIN_DIR = 'Offroad_Segmentation_Training_Dataset/train'
    VAL_DIR = 'Offroad_Segmentation_Training_Dataset/val'
    
    # Check if dataset exists before starting
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print(f"Error: Dataset not found at {TRAIN_DIR} or {VAL_DIR}.")
        print("Please ensure the 'Offroad_Segmentation_Training_Dataset' folder is in the same directory as this script.")
    else:
        # Start training
        # Note: Batch size 8 is safe for 16GB GPUs. Reduce to 4 if you get OutOfMemory errors.
        train_model(
            train_dir=TRAIN_DIR, 
            val_dir=VAL_DIR, 
            epochs=20, 
            batch_size=8
        )