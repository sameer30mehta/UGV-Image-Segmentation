"""
TerrainAI Backend — Model Registry
Handles loading/managing multiple segmentation model variants.
"""
import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
import cv2

# Default class configuration (matches Colab training)
DEFAULT_CLASS_NAMES = [
    'Landscape', 'Sky', 'Trees', 'Lush Bushes', 'Dry Grass',
    'Dry Bushes', 'Rocks', 'Ground Clutter', 'Flowers', 'Logs'
]

# Safety classification for terrain traversability
SAFETY_MAP = {
    'Landscape': 'safe',
    'Sky': 'neutral',
    'Trees': 'obstacle',
    'Lush Bushes': 'caution',
    'Dry Grass': 'safe',
    'Dry Bushes': 'caution',
    'Rocks': 'obstacle',
    'Ground Clutter': 'caution',
    'Flowers': 'safe',
    'Logs': 'obstacle',
}

# Distinct colors per class
COLORMAP = np.array([
    [139, 119, 101],   # Landscape - tan
    [135, 206, 235],   # Sky - sky blue
    [0,   128,   0],   # Trees - dark green
    [50,  205,  50],   # Lush Bushes - lime
    [218, 165,  32],   # Dry Grass - goldenrod
    [160,  82,  45],   # Dry Bushes - sienna
    [128, 128, 128],   # Rocks - gray
    [210, 180, 140],   # Ground Clutter - wheat
    [255,   0, 255],   # Flowers - magenta
    [139,  69,  19],   # Logs - saddle brown
], dtype=np.uint8)


MODEL_CONFIGS = {
    'mit_b3': {
        'name': 'MiT-B3 (High Accuracy)',
        'encoder': 'mit_b3',
        'description': 'Best accuracy. Mix Vision Transformer with deep feature extraction. Best for complex scenes with many similar classes.',
        'params': '~47M',
        'speed': 'Medium',
        'accuracy': 'Highest',
        'use_case': 'Complex monochromatic scenes, many small objects',
    },
    'mit_b1': {
        'name': 'MiT-B1 (Balanced)',
        'encoder': 'mit_b1',
        'description': 'Good balance of speed and accuracy. Suitable for most scenes.',
        'params': '~17M',
        'speed': 'Fast',
        'accuracy': 'High',
        'use_case': 'General purpose, good color contrast scenes',
    },
    'mit_b0': {
        'name': 'MiT-B0 (Real-Time)',
        'encoder': 'mit_b0',
        'description': 'Ultra-lightweight for real-time UGV navigation. Trades accuracy for speed.',
        'params': '~7M',
        'speed': 'Very Fast',
        'accuracy': 'Good',
        'use_case': 'Real-time navigation, video processing',
    },
}


class ModelRegistry:
    """Manages multiple segmentation models for intelligent selection."""
    
    def __init__(self, num_classes=6, device='cuda'):
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.active_model_key = None
        self.class_names = DEFAULT_CLASS_NAMES[:num_classes]
    
    def load_model(self, key, weights_path=None):
        """Load a model variant by key."""
        if key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {key}. Available: {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[key]
        
        # Check if weights file is valid (not a Git LFS pointer)
        use_pretrained_weights = True
        if weights_path and os.path.exists(weights_path) and os.path.getsize(weights_path) > 1000:
            use_pretrained_weights = False
        
        model = smp.FPN(
            encoder_name=config['encoder'],
            encoder_weights='imagenet' if use_pretrained_weights else None,
            in_channels=3,
            classes=self.num_classes,
            decoder_dropout=0.2
        )
        
        if weights_path and not use_pretrained_weights:
            state_dict = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded weights from {weights_path}")
        elif use_pretrained_weights:
            print(f"  Using ImageNet pretrained weights (custom weights not available)")
        
        model = model.to(self.device)
        model.eval()
        self.models[key] = model
        
        if self.active_model_key is None:
            self.active_model_key = key
        
        print(f"[OK] Loaded: {config['name']} ({config['params']} params)")
        return model
    
    def get_model(self, key=None):
        """Get a model by key, or the active model."""
        key = key or self.active_model_key
        if key not in self.models:
            self.load_model(key)
        return self.models[key]
    
    def set_active(self, key):
        """Set the active model."""
        if key not in self.models:
            self.load_model(key)
        self.active_model_key = key
    
    def load_user_model(self, user_id, weights_path):
        """Load a custom user fine-tuned model."""
        if not os.path.exists(weights_path):
            raise ValueError(f"User model not found: {weights_path}")
        
        model_key = f"user_{user_id}"
        
        # Use mit_b3 architecture (same as base model)
        model = smp.FPN(
            encoder_name='mit_b3',
            encoder_weights=None,
            in_channels=3,
            classes=self.num_classes,
            decoder_dropout=0.2
        )
        
        # Load user weights
        state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict, strict=False)
        
        model = model.to(self.device)
        model.eval()
        self.models[model_key] = model
        self.active_model_key = model_key
        
        print(f"[OK] Loaded custom model: {model_key} from {weights_path}")
        return model_key
    
    def predict(self, image_tensor, key=None):
        """Run inference with the specified or active model."""
        model = self.get_model(key)
        with torch.no_grad():
            device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
            with torch.amp.autocast(device_type, enabled=(device_type == 'cuda')):
                logits = model(image_tensor.to(self.device))
        return logits
    
    def get_info(self):
        """Return info about all registered models."""
        info = {}
        for key, config in MODEL_CONFIGS.items():
            info[key] = {
                **config,
                'loaded': key in self.models,
                'active': key == self.active_model_key,
            }
        return info


def mask_to_rgb(mask, num_classes=6):
    """Convert class index mask to RGB visualization."""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    h, w = mask.shape
    colormap = COLORMAP[:num_classes]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(num_classes):
        rgb[mask == c] = colormap[c]
    return rgb


def mask_to_safety(mask, class_names=None, num_classes=6):
    """Convert segmentation mask to traversability heatmap (green/yellow/red)."""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    class_names = class_names or DEFAULT_CLASS_NAMES[:num_classes]
    
    safety_colors = {
        'safe': [0, 200, 0],       # Green
        'caution': [255, 200, 0],   # Yellow
        'obstacle': [255, 0, 0],    # Red
        'neutral': [100, 100, 100], # Gray
    }
    
    h, w = mask.shape
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(num_classes):
        name = class_names[c] if c < len(class_names) else 'neutral'
        safety = SAFETY_MAP.get(name, 'neutral')
        heatmap[mask == c] = safety_colors[safety]
    return heatmap
