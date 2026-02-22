"""
TerrainAI Backend — GradCAM Explainability
Generates class activation heatmaps showing why the model made its predictions.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2


class SegmentationGradCAM:
    """
    GradCAM for semantic segmentation models.
    Hooks into the last encoder layer and computes activation maps
    showing which regions influenced each class prediction.
    """
    
    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Auto-detect the last encoder layer
        if target_layer is None:
            # For SMP models, encoder stages are accessible
            target_layer = self._find_last_layer()
        
        self.target_layer = target_layer
        self._register_hooks()
    
    def _find_last_layer(self):
        """Find the last convolutional/attention layer in the encoder."""
        last_layer = None
        for name, module in self.model.encoder.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.LayerNorm)):
                last_layer = module
        return last_layer
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output.detach()
        
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
    
    def generate(self, image_tensor, target_class=None):
        """
        Generate GradCAM heatmap.
        
        Args:
            image_tensor: (1, 3, H, W) input tensor
            target_class: If None, uses the most predicted class
        
        Returns:
            heatmap: (H, W) numpy array normalized to [0, 1]
            overlay: (H, W, 3) colored heatmap overlay
        """
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(True)
        
        # Forward pass
        logits = self.model(image_tensor)  # (1, C, H, W)
        
        # If no target class specified, use the dominant predicted class
        if target_class is None:
            pred = logits.argmax(dim=1)  # (1, H, W)
            # Find the most common predicted class (excluding background)
            unique, counts = torch.unique(pred, return_counts=True)
            # Sort by count, skip class 0 if possible
            sorted_idx = counts.argsort(descending=True)
            target_class = unique[sorted_idx[0]].item()
            if target_class == 0 and len(sorted_idx) > 1:
                target_class = unique[sorted_idx[1]].item()
        
        # Create a score for the target class
        score = logits[0, target_class].sum()
        
        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            # Fallback: return uniform heatmap
            h, w = image_tensor.shape[2], image_tensor.shape[3]
            return np.ones((h, w), dtype=np.float32) * 0.5, target_class
        
        # Compute weights (global average pooling of gradients)
        gradients = self.gradients
        activations = self.activations
        
        # Handle different tensor shapes
        if len(gradients.shape) == 3:
            # (B, L, C) from transformer — need to reshape to spatial
            B, L, C = gradients.shape
            H = W = int(L ** 0.5)
            if H * W == L:
                gradients = gradients.view(B, H, W, C).permute(0, 3, 1, 2)
                activations = activations.view(B, H, W, C).permute(0, 3, 1, 2)
            else:
                # Can't reshape perfectly, use mean
                weights = gradients.mean(dim=1, keepdim=True)  # (B, 1, C)
                cam = (weights * activations).sum(dim=-1).squeeze()  # (B, L)
                cam = F.relu(cam)
                h, w = image_tensor.shape[2], image_tensor.shape[3]
                cam = cam.view(1, 1, H, W) if H * W == L else cam.unsqueeze(0).unsqueeze(0)
                cam = F.interpolate(cam.float(), size=(h, w), mode='bilinear', align_corners=False)
                cam = cam.squeeze().cpu().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                return cam, target_class
        
        # Standard 4D case: (B, C, H, W)
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)
        
        # Upsample to input size
        h, w = image_tensor.shape[2], image_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, target_class
    
    def cleanup(self):
        """Remove hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []


def apply_heatmap(image, cam, alpha=0.5):
    """
    Apply colored heatmap overlay on the original image.
    
    Args:
        image: (H, W, 3) RGB numpy array (0-255)
        cam: (H, W) float array normalized to [0, 1]
        alpha: blending factor
    
    Returns:
        overlay: (H, W, 3) RGB numpy array
    """
    # Convert cam to colormap
    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match image
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay
