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
from pathlib import Path

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
        'name': 'FPN + MiT-B3',
        'official_name': 'FPN + MiT-B3',
        'arch': 'fpn',
        'encoder': 'mit_b3',
        'description': 'Best accuracy. Mix Vision Transformer with deep feature extraction. Best for complex scenes with many similar classes.',
        'params': '~47M',
        'speed': 'Medium',
        'accuracy': 'Highest',
        'use_case': 'Complex monochromatic scenes, many small objects',
    },
    'mit_b1': {
        'name': 'DeepLabV3+ + EfficientNet-B4',
        'official_name': 'DeepLabV3+ + EfficientNet-B4',
        'arch': 'deeplabv3plus',
        'encoder': 'efficientnet-b4',
        'native_classes': 10,
        'description': 'EfficientNet-based DeepLabV3+ model (Lovasz-trained checkpoint compatible). High quality with good throughput.',
        'params': '~25M',
        'speed': 'Medium-Fast',
        'accuracy': 'High',
        'use_case': 'General purpose scenes where efficientnet checkpoint is preferred',
    },
    'mit_b0': {
        'name': 'Linknet + MobileNetV2',
        'official_name': 'Linknet + MobileNetV2',
        'arch': 'linknet',
        'encoder': 'mobilenet_v2',
        'description': 'Ultra-fast non-FPN runtime model for high-motion segments. Prioritizes latency over fine detail accuracy.',
        'params': '~4M',
        'speed': 'Very Fast',
        'accuracy': 'Moderate',
        'use_case': 'Real-time navigation, video processing',
    },
}


def discover_weights(weights_dir=None):
    """Discover and assign available .pth files from weights directory to model tiers."""
    if weights_dir is None:
        weights_dir = Path(__file__).parent / "weights"
    else:
        weights_dir = Path(weights_dir)

    mapping = {
        'mit_b3': None,
        'mit_b1': None,
        'mit_b0': None,
    }
    if not weights_dir.exists() or not weights_dir.is_dir():
        return mapping

    pths = sorted(weights_dir.glob("*.pth"))
    if not pths:
        return mapping

    # Highest-priority (best): explicitly prefer best_desert_segmentation.pth
    best_b3 = next((p for p in pths if p.name.lower() == "best_desert_segmentation.pth"), None)
    if best_b3 is None:
        best_b3 = pths[0]
    mapping['mit_b3'] = str(best_b3)

    # Second-tier (almost as good): explicitly prefer best_model.pth, else any other available
    best_b1 = next((p for p in pths if p.name.lower() == "best_model.pth" and p != best_b3), None)
    if best_b1 is None:
        alternatives = [p for p in pths if p != best_b3]
        if alternatives:
            best_b1 = alternatives[0]
    if best_b1 is not None:
        mapping['mit_b1'] = str(best_b1)

    # Third tier (ultra-fast) intentionally remains lightweight fallback if no compatible weight exists.
    # If a third file exists, assign it to mit_b0.
    used = {best_b3, best_b1}
    third_candidates = [p for p in pths if p not in used]
    if third_candidates:
        mapping['mit_b0'] = str(third_candidates[0])

    return mapping


class ModelRegistry:
    """Manages multiple segmentation models for intelligent selection."""
    
    def __init__(self, num_classes=6, device='cuda', weights_dir=None):
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.active_model_key = None
        self.class_names = DEFAULT_CLASS_NAMES[:num_classes]
        self.weights_dir = weights_dir or str(Path(__file__).parent / "weights")
        self.weight_paths = discover_weights(self.weights_dir)

    def _build_model(self, config, use_pretrained_weights):
        """Build model instance from config."""
        encoder_weights = 'imagenet' if use_pretrained_weights else None
        arch = config.get('arch', 'fpn')

        model_classes = int(config.get('native_classes', self.num_classes))

        if arch == 'fpn':
            return smp.FPN(
                encoder_name=config['encoder'],
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=model_classes,
                decoder_dropout=0.2
            )

        if arch == 'deeplabv3plus':
            return smp.DeepLabV3Plus(
                encoder_name=config['encoder'],
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=model_classes,
            )

        if arch == 'linknet':
            return smp.Linknet(
                encoder_name=config['encoder'],
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=model_classes,
            )

        raise ValueError(f"Unsupported model arch: {arch}")

    @staticmethod
    def _extract_state_dict(payload):
        """Handle common checkpoint wrappers."""
        if not isinstance(payload, dict):
            return payload
        if 'model_state_dict' in payload and isinstance(payload['model_state_dict'], dict):
            return payload['model_state_dict']
        if 'state_dict' in payload and isinstance(payload['state_dict'], dict):
            return payload['state_dict']
        return payload
    
    def load_model(self, key, weights_path=None):
        """Load a model variant by key."""
        if key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {key}. Available: {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[key]
        if weights_path is None:
            weights_path = self.weight_paths.get(key)
        
        # Check if weights file is valid (not a Git LFS pointer)
        use_pretrained_weights = True
        if weights_path and os.path.exists(weights_path) and os.path.getsize(weights_path) > 1000:
            use_pretrained_weights = False
        elif config.get('pretrained_fallback', True) is False:
            use_pretrained_weights = False
        
        model = self._build_model(config, use_pretrained_weights)
        
        if weights_path and not use_pretrained_weights:
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                state_dict = self._extract_state_dict(state_dict)
                incompatible = model.load_state_dict(state_dict, strict=False)
                total_params = len(model.state_dict())
                missing = len(getattr(incompatible, 'missing_keys', []))
                unexpected = len(getattr(incompatible, 'unexpected_keys', []))
                matched = max(total_params - missing, 0)
                match_ratio = matched / max(total_params, 1)
                print(f"  Loaded weights from {weights_path}")
                print(f"  Checkpoint compatibility: matched={matched}/{total_params} ({match_ratio:.1%}), missing={missing}, unexpected={unexpected}")
                if match_ratio < 0.5:
                    print("  [WARN] Low checkpoint compatibility for this architecture. The checkpoint may belong to a different backbone/model family.")
            except Exception as exc:
                print(f"  [WARN] Failed loading {weights_path} for {key}: {exc}")
                print("  [WARN] Falling back to ImageNet-pretrained encoder weights")
        elif use_pretrained_weights:
            print(f"  Using ImageNet pretrained weights (custom weights not available)")
        
        model = model.to(self.device)
        model.eval()
        self.models[key] = model
        
        if self.active_model_key is None:
            self.active_model_key = key
        
        print(f"[OK] Loaded: {config['name']} [{config.get('official_name', config.get('arch', 'model'))}] ({config['params']} params)")
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
        key = key or self.active_model_key
        model = self.get_model(key)
        config = MODEL_CONFIGS.get(key, {})
        native_classes = int(config.get('native_classes', self.num_classes))
        with torch.no_grad():
            device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
            with torch.amp.autocast(device_type, enabled=(device_type == 'cuda')):
                logits = model(image_tensor.to(self.device))
        # Keep runtime interfaces consistent even when a model uses native class count.
        if native_classes > self.num_classes:
            logits = logits[:, :self.num_classes, ...]
        elif native_classes < self.num_classes:
            pad = torch.zeros(
                (logits.shape[0], self.num_classes - native_classes, logits.shape[2], logits.shape[3]),
                device=logits.device,
                dtype=logits.dtype,
            )
            logits = torch.cat([logits, pad], dim=1)
        return logits
    
    def get_info(self):
        """Return info about all registered models."""
        info = {}
        for key, config in MODEL_CONFIGS.items():
            info[key] = {
                **config,
                'loaded': key in self.models,
                'active': key == self.active_model_key,
                'weights_path': self.weight_paths.get(key),
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
