"""
TerrainAI Backend — Smart Model Selector
Analyzes input image characteristics and recommends the optimal model.
"""
import numpy as np
import cv2


def analyze_image(image_rgb):
    """
    Analyze image properties to recommend the best model variant.
    
    Returns a dict with analysis results and recommendation.
    """
    h, w = image_rgb.shape[:2]
    
    # 1. Color variance analysis
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue_std = np.std(hsv[:, :, 0])
    sat_mean = np.mean(hsv[:, :, 1])
    val_std = np.std(hsv[:, :, 2])
    
    # 2. Edge density (texture complexity)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)
    
    # 3. Contrast (dynamic range)
    contrast = np.std(gray.astype(np.float32))
    
    # 4. Resolution
    megapixels = (h * w) / 1e6
    
    # Build analysis
    analysis = {
        'resolution': f'{w}x{h}',
        'megapixels': round(megapixels, 2),
        'color_variance': round(float(hue_std), 2),
        'saturation': round(float(sat_mean), 2),
        'brightness_variance': round(float(val_std), 2),
        'edge_density': round(float(edge_density), 4),
        'contrast': round(float(contrast), 2),
    }
    
    # Decision logic
    score_b3 = 0  # High accuracy
    score_b1 = 0  # Balanced
    score_b0 = 0  # Fast
    reasons = []
    
    # Low color variance → monochromatic scene → needs deeper features
    if hue_std < 20:
        score_b3 += 3
        reasons.append("Low color variance detected — deep features needed to distinguish similar classes")
    elif hue_std < 40:
        score_b1 += 2
        reasons.append("Moderate color variance — balanced model sufficient")
    else:
        score_b0 += 1
        reasons.append("High color variance — classes are visually distinct")
    
    # High edge density → complex scene → needs more capacity
    if edge_density > 0.15:
        score_b3 += 2
        reasons.append("High edge density — complex scene with many boundaries")
    elif edge_density > 0.08:
        score_b1 += 2
        reasons.append("Moderate texture complexity")
    else:
        score_b0 += 1
        reasons.append("Simple scene structure — lightweight model works well")
    
    # Low contrast → hard to segment → needs powerful model
    if contrast < 30:
        score_b3 += 2
        reasons.append("Low contrast — model needs strong feature extraction")
    elif contrast < 50:
        score_b1 += 1
    
    # High resolution → can afford heavier model
    if megapixels > 2:
        score_b3 += 1
        reasons.append("High resolution input — can leverage detailed features")
    elif megapixels < 0.5:
        score_b0 += 2
        reasons.append("Low resolution — lightweight model optimal")
    
    # Determine recommendation
    scores = {'mit_b3': score_b3, 'mit_b1': score_b1, 'mit_b0': score_b0}
    recommended = max(scores, key=scores.get)
    
    model_names = {
        'mit_b3': 'FPN + MiT-B3',
        'mit_b1': 'DeepLabV3+ + EfficientNet-B4',
        'mit_b0': 'Linknet + MobileNetV2',
    }
    
    return {
        'analysis': analysis,
        'scores': scores,
        'recommended': recommended,
        'recommended_name': model_names[recommended],
        'reasons': reasons,
        'confidence': round(max(scores.values()) / (sum(scores.values()) + 1e-6), 2),
    }
