"""
TerrainAI Backend — Preprocessing Visualizer
Generates step-by-step DIP transformation images for the UI carousel.
"""
import cv2
import numpy as np
import base64
from io import BytesIO


def apply_clahe(image_rgb, clip_limit=2.0, grid_size=(8, 8)):
    """Apply CLAHE contrast enhancement."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def apply_gamma(image_rgb, gamma=1.0):
    """Apply gamma correction."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(256)]).astype('uint8')
    return cv2.LUT(image_rgb, table)


def apply_edge_detection(image_rgb):
    """Visualize edges using Canny detector."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Make it 3-channel for display
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # Overlay edges on dimmed original
    dimmed = (image_rgb * 0.3).astype(np.uint8)
    edges_highlight = dimmed.copy()
    edges_highlight[edges > 0] = [0, 255, 128]  # Green edges
    return edges_highlight


def apply_normalize_preview(image_rgb):
    """Show the effect of ImageNet normalization (visualization only)."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (image_rgb.astype(np.float32) / 255.0 - mean) / std
    # Rescale to 0-255 for visualization
    vis = normalized - normalized.min()
    vis = (vis / (vis.max() + 1e-6) * 255).astype(np.uint8)
    return vis


def image_to_base64(image_rgb):
    """Convert numpy RGB image to base64 JPEG string."""
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), 
                              [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def generate_preprocessing_steps(image_rgb, target_size=512):
    """
    Generate all DIP preprocessing steps as base64 images.
    Returns a list of {name, description, image_base64} dicts.
    """
    # Step 0: Original (resized for display)
    h, w = image_rgb.shape[:2]
    scale = target_size / min(h, w)
    display = cv2.resize(image_rgb, (int(w * scale), int(h * scale)))
    
    # Crop to square for consistent display
    dh, dw = display.shape[:2]
    cy, cx = dh // 2, dw // 2
    half = target_size // 2
    display = display[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    display = cv2.resize(display, (target_size, target_size))
    
    steps = []
    
    # 1. Original Input
    steps.append({
        'name': 'Original Input',
        'description': 'Raw uploaded image before any processing',
        'icon': '[IMG]',
        'image': image_to_base64(display),
    })
    
    # 2. Resize
    resized = cv2.resize(image_rgb, (target_size, target_size))
    steps.append({
        'name': 'Resize (512×512)',
        'description': 'Normalized to model input dimensions',
        'icon': '[RSZ]',
        'image': image_to_base64(resized),
    })
    
    # 3. CLAHE
    clahe_img = apply_clahe(resized)
    steps.append({
        'name': 'CLAHE Enhancement',
        'description': 'Adaptive histogram equalization boosts local contrast between terrain classes',
        'icon': '[CLH]',
        'image': image_to_base64(clahe_img),
    })
    
    # 4. Gamma Correction
    gamma_img = apply_gamma(clahe_img, gamma=0.9)
    steps.append({
        'name': 'Gamma Correction',
        'description': 'Adjusts exposure to normalize lighting variations',
        'icon': '[GAM]',
        'image': image_to_base64(gamma_img),
    })
    
    # 5. Edge Detection (visualization)
    edge_img = apply_edge_detection(resized)
    steps.append({
        'name': 'Edge Analysis',
        'description': 'Boundary detection reveals class transition zones',
        'icon': '[EDG]',
        'image': image_to_base64(edge_img),
    })
    
    # 6. Normalized
    norm_img = apply_normalize_preview(clahe_img)
    steps.append({
        'name': 'ImageNet Normalize',
        'description': 'Standardized to ImageNet statistics for the MiT-B3 transformer backbone',
        'icon': '[NRM]',
        'image': image_to_base64(norm_img),
    })
    
    return steps


def preprocess_for_inference(image_rgb, target_size=512):
    """
    Apply the actual DIP pipeline for model inference.
    Returns a preprocessed numpy array ready for tensor conversion.
    """
    # Resize
    resized = cv2.resize(image_rgb, (target_size, target_size))
    
    # CLAHE
    enhanced = apply_clahe(resized, clip_limit=2.0)
    
    # Gamma
    corrected = apply_gamma(enhanced, gamma=1.0)
    
    # ImageNet normalize
    img_float = corrected.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (img_float - mean) / std
    
    # HWC -> CHW for PyTorch
    tensor_ready = normalized.transpose(2, 0, 1).astype(np.float32)
    
    return tensor_ready
