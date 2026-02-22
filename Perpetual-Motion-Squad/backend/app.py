import os
import sys
import time
import uuid
import shutil
import base64
import tempfile
from io import BytesIO
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Local imports
from models import ModelRegistry, mask_to_rgb, mask_to_safety, COLORMAP, DEFAULT_CLASS_NAMES, SAFETY_MAP
from gradcam import SegmentationGradCAM, apply_heatmap
from smart_selector import analyze_image
# video_processor is imported locally inside the segment-video endpoint
from preprocessing_viz import generate_preprocessing_steps, preprocess_for_inference, image_to_base64
from finetune import run_finetune_pipeline

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_WEIGHTS_PATH = os.environ.get("MODEL_WEIGHTS", None)
if MODEL_WEIGHTS_PATH is None:
    # Check common locations
    _candidates = [
        "../best_desert_segmentation.pth",
        "weights/best_desert_segmentation.pth",
        "weights/best_segformer_model.pth",
        "../best_segformer_model.pth",
    ]
    for _c in _candidates:
        if os.path.exists(_c):
            MODEL_WEIGHTS_PATH = _c
            break
    if MODEL_WEIGHTS_PATH is None:
        MODEL_WEIGHTS_PATH = "weights/best_desert_segmentation.pth"

NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "6"))
IMG_SIZE = 512
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# MODEL LOADING
# ============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
registry = ModelRegistry(num_classes=NUM_CLASSES, device=DEVICE)

@asynccontextmanager
async def lifespan(app):
    # Startup
    print(f"\n[START] TerrainAI starting on {DEVICE}")
    if DEVICE == "cuda":
        print(f"  [GPU] {torch.cuda.get_device_name(0)}")
    else:
        print(f"  [CPU] Running on CPU — CUDA not available")
    weights = Path(MODEL_WEIGHTS_PATH)
    if weights.exists():
        registry.load_model("mit_b3", str(weights))
        print(f"  [OK] Loaded weights: {weights}")
    else:
        registry.load_model("mit_b3")
        print(f"  [WARN] No weights found at {weights}, using ImageNet pre-trained")
    print(f"  Classes: {NUM_CLASSES}")
    print(f"  Device: {DEVICE}")
    print(f"  Ready at http://localhost:8000\n")
    yield
    # Shutdown
    print("\n[STOP] TerrainAI shutting down")

# ============================================================================
# APP INIT
# ============================================================================
app = FastAPI(title="TerrainAI", version="1.0.0",
              description="Intelligent Terrain Analysis for Autonomous UGV Navigation",
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Serve output files
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the frontend index.html."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return JSONResponse({"error": "Frontend not found"}, status_code=404)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def read_upload_image(file_bytes):
    """Read uploaded image bytes into RGB numpy array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image file")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_inference(image_rgb, model_key=None):
    """Run full inference pipeline on an RGB image."""
    orig_h, orig_w = image_rgb.shape[:2]
    
    # Preprocess with DIP pipeline
    tensor_data = preprocess_for_inference(image_rgb, target_size=IMG_SIZE)
    tensor = torch.from_numpy(tensor_data).unsqueeze(0).to(DEVICE)
    
    # Inference
    start = time.time()
    logits = registry.predict(tensor, key=model_key)
    inference_time = time.time() - start
    
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    
    # Resize back to original
    pred_full = cv2.resize(pred.astype(np.uint8), (orig_w, orig_h),
                           interpolation=cv2.INTER_NEAREST)
    
    return pred_full, logits, inference_time


def compute_safety_percentages(distribution, num_classes=None):
    """Compute safe/caution/obstacle percentages from class distribution."""
    num_classes = num_classes or NUM_CLASSES
    names = DEFAULT_CLASS_NAMES[:num_classes]
    totals = {'safe': 0, 'caution': 0, 'obstacle': 0, 'neutral': 0}
    for d in distribution:
        name = d['name']
        safety = SAFETY_MAP.get(name, 'neutral')
        totals[safety] += d['percentage']
    return {
        'safe': round(totals['safe'], 1),
        'caution': round(totals['caution'], 1),
        'obstacle': round(totals['obstacle'], 1),
        'neutral': round(totals['neutral'], 1),
    }


def compute_class_distribution(mask, num_classes=None):
    """Compute pixel count per class."""
    num_classes = num_classes or NUM_CLASSES
    names = DEFAULT_CLASS_NAMES[:num_classes]
    total = mask.size
    dist = []
    for c in range(num_classes):
        count = int(np.sum(mask == c))
        dist.append({
            'class_id': c,
            'name': names[c] if c < len(names) else f'Class_{c}',
            'pixels': count,
            'percentage': round(count / total * 100, 2),
            'color': COLORMAP[c].tolist() if c < len(COLORMAP) else [128, 128, 128],
        })
    return dist


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return 204 to avoid favicon 404 noise when no favicon file is provided."""
    return JSONResponse(status_code=204, content=None)


@app.get("/api/health")
async def health():
    return {
        "status": "running",
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU",
        "models_loaded": list(registry.models.keys()),
        "active_model": registry.active_model_key,
    }


@app.get("/api/models")
async def list_models():
    """List all available models with metadata."""
    return registry.get_info()


@app.post("/api/set-model")
async def set_model(model_key: str = Form(...)):
    """Switch the active model."""
    try:
        registry.set_active(model_key)
        return {"status": "ok", "active_model": model_key}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/load-user-model")
async def load_user_model(user_id: str = Form(...)):
    """Load a custom fine-tuned model by user ID."""
    model_path = Path(f"models/user_{user_id}.pth")
    
    if not model_path.exists():
        raise HTTPException(404, f"User model not found for ID: {user_id}")
    
    try:
        model_key = registry.load_user_model(user_id, str(model_path))
        return {
            "status": "ok",
            "model_key": model_key,
            "active_model": registry.active_model_key,
            "message": f"Custom model loaded: {model_key}"
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to load user model: {str(e)}")


@app.post("/api/reset-to-base-model")
async def reset_to_base_model():
    """Reset to the default base model."""
    try:
        registry.set_active("mit_b3")
        return {
            "status": "ok",
            "active_model": "mit_b3",
            "message": "Reset to base model"
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to reset model: {str(e)}")


@app.post("/api/segment")
async def segment_image(file: UploadFile = File(...), model_key: str = Form(None)):
    """
    Segment a single image. Returns segmented mask, overlay, safety heatmap,
    class distribution, preprocessing steps, and per-pixel confidence grid.
    """
    file_bytes = await file.read()
    image_rgb = read_upload_image(file_bytes)
    orig_h, orig_w = image_rgb.shape[:2]
    
    # 1. Generate preprocessing steps (for UI carousel)
    steps = generate_preprocessing_steps(image_rgb, target_size=IMG_SIZE)
    
    # 2. Run inference
    pred_mask, logits, inference_time = run_inference(image_rgb, model_key)
    
    # 3. Generate outputs
    mask_rgb = mask_to_rgb(pred_mask, NUM_CLASSES)
    safety_map = mask_to_safety(pred_mask, num_classes=NUM_CLASSES)
    overlay = cv2.addWeighted(image_rgb, 0.5, 
                               cv2.resize(mask_rgb, (orig_w, orig_h)), 0.5, 0)
    
    # 4. Class distribution
    distribution = compute_class_distribution(pred_mask)
    
    # 5. Confidence grid for hover tooltip
    #    Compute softmax probabilities, downsample to grid for efficient transfer
    GRID_SIZE = 64
    probs = torch.softmax(logits, dim=1).squeeze(0)  # (C, H, W) at model resolution
    max_conf, pred_classes = torch.max(probs, dim=0)  # (H, W) each
    
    # Resize to match original image dimensions, then downsample to grid
    max_conf_np = max_conf.cpu().numpy()
    pred_cls_np = pred_classes.cpu().numpy().astype(np.uint8)
    
    max_conf_full = cv2.resize(max_conf_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    pred_cls_full = cv2.resize(pred_cls_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # Downsample to GRID_SIZE x GRID_SIZE for JSON transfer
    conf_grid = cv2.resize(max_conf_full, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_LINEAR)
    cls_grid = cv2.resize(pred_cls_full, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_NEAREST)
    
    names = DEFAULT_CLASS_NAMES[:NUM_CLASSES]
    confidence_grid = {
        "grid_w": GRID_SIZE,
        "grid_h": GRID_SIZE,
        "img_w": orig_w,
        "img_h": orig_h,
        "cells": []  # flat array: row-major [y][x] -> {class_id, class_name, confidence}
    }
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            c = int(cls_grid[y, x])
            confidence_grid["cells"].append({
                "c": c,
                "n": names[c] if c < len(names) else f"Class_{c}",
                "p": round(float(conf_grid[y, x]) * 100, 1),
            })
    
    # 6. Confidence heatmap image (visual)
    conf_vis = (max_conf_full * 255).astype(np.uint8)
    conf_colored = cv2.applyColorMap(conf_vis, cv2.COLORMAP_VIRIDIS)
    conf_colored = cv2.cvtColor(conf_colored, cv2.COLOR_BGR2RGB)
    
    # 7. Safety percentages
    safety_percentages = compute_safety_percentages(distribution)
    
    # 8. Encode outputs as base64
    result = {
        "original": image_to_base64(image_rgb),
        "mask": image_to_base64(mask_rgb),
        "overlay": image_to_base64(overlay),
        "safety": image_to_base64(safety_map),
        "confidence": image_to_base64(conf_colored),
        "confidence_grid": confidence_grid,
        "preprocessing_steps": steps,
        "class_distribution": distribution,
        "safety_percentages": safety_percentages,
        "inference_time_ms": round(inference_time * 1000, 1),
        "input_size": f"{orig_w}x{orig_h}",
        "model_used": model_key or registry.active_model_key,
    }
    
    return result


@app.post("/api/explain")
async def explain_prediction(file: UploadFile = File(...)):
    """Generate GradCAM explainability heatmap for the uploaded image."""
    file_bytes = await file.read()
    image_rgb = read_upload_image(file_bytes)
    
    # Preprocess
    tensor_data = preprocess_for_inference(image_rgb, target_size=IMG_SIZE)
    tensor = torch.from_numpy(tensor_data).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)
    
    # Get active model
    model = registry.get_model()
    
    # Generate GradCAM
    gradcam = SegmentationGradCAM(model)
    try:
        cam, target_class = gradcam.generate(tensor)
        
        # Apply heatmap overlay
        resized_img = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
        heatmap_overlay = apply_heatmap(resized_img, cam, alpha=0.5)
        
        # Resize back
        heatmap_overlay = cv2.resize(heatmap_overlay, 
                                      (image_rgb.shape[1], image_rgb.shape[0]))
        
        class_name = DEFAULT_CLASS_NAMES[target_class] if target_class < len(DEFAULT_CLASS_NAMES) else f"Class_{target_class}"
        
        return {
            "gradcam": image_to_base64(heatmap_overlay),
            "target_class": target_class,
            "target_class_name": class_name,
            "description": f"Regions most influential for predicting '{class_name}'. Warm colors (red/yellow) indicate high activation.",
        }
    finally:
        gradcam.cleanup()


@app.post("/api/recommend-model")
async def recommend_model(file: UploadFile = File(...)):
    """Analyze image and recommend the best model."""
    file_bytes = await file.read()
    image_rgb = read_upload_image(file_bytes)
    result = analyze_image(image_rgb)
    return result


@app.post("/api/segment-video")
async def segment_video(
    file: UploadFile = File(...),
    interval_ms: int = Form(200),
):
    """
    Segment a video file frame-by-frame.
    
    1. Extract frames at fixed time intervals (100ms or 200ms)
    2. Run each frame through the full DIP + segmentation pipeline
    3. Generate separate videos for: overlay, mask, safety map, GradCAM
    4. Return URLs for all videos + stats
    """
    from video_processor import extract_frames_by_interval, stitch_video, stitch_sidebyside
    from velocity_estimator import estimate_velocities_for_frames
    
    # Save uploaded video
    video_id = str(uuid.uuid4())[:8]
    input_path = str(UPLOAD_DIR / f"{video_id}_input.mp4")
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Extract frames at fixed intervals
    video_data = extract_frames_by_interval(input_path, interval_ms=interval_ms)
    frames = video_data['frames']
    output_fps = video_data['output_fps']
    
    if not frames:
        os.remove(input_path)
        raise HTTPException(400, "Could not extract frames from video")
    
    # ── Velocity-based adaptive model selection ──────────────────────────
    # Estimate per-frame velocity via Lucas-Kanade sparse optical flow.
    # dt = interval between frames in seconds.
    dt = interval_ms / 1000.0
    velocity_info = estimate_velocities_for_frames(frames, dt=dt)
    
    # Ensure all required models are loaded
    for vdata in velocity_info:
        mkey = vdata["model_key"]
        if mkey not in registry.models:
            registry.load_model(mkey)
    
    # Process each frame through the full pipeline
    overlay_frames = []
    mask_frames = []
    safety_frames = []
    gradcam_frames = []
    total_inference = 0
    per_frame_details = []   # velocity + model info returned to frontend
    
    for i, frame in enumerate(frames):
        orig_h, orig_w = frame.shape[:2]
        vdata = velocity_info[i]
        selected_model_key = vdata["model_key"]
        
        # Run inference with the velocity-selected model
        pred_mask, logits, inf_time = run_inference(frame, model_key=selected_model_key)
        total_inference += inf_time
        
        # Record per-frame details
        per_frame_details.append({
            **vdata,
            "inference_ms": round(inf_time * 1000, 1),
        })
        
        # Generate all view types at original frame resolution
        mask_rgb = mask_to_rgb(pred_mask, NUM_CLASSES)
        mask_rgb = cv2.resize(mask_rgb, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        safety_rgb = mask_to_safety(pred_mask, num_classes=NUM_CLASSES)
        safety_rgb = cv2.resize(safety_rgb, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        overlay = cv2.addWeighted(frame, 0.5, mask_rgb, 0.5, 0)
        
        # GradCAM for each frame (use the selected model)
        try:
            model = registry.get_model(selected_model_key)
            tensor_data = preprocess_for_inference(frame, target_size=IMG_SIZE)
            tensor = torch.from_numpy(tensor_data).unsqueeze(0).to(DEVICE)
            tensor.requires_grad_(True)
            gradcam_gen = SegmentationGradCAM(model)
            cam, _ = gradcam_gen.generate(tensor)
            resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            heatmap = apply_heatmap(resized_frame, cam, alpha=0.5)
            heatmap = cv2.resize(heatmap, (orig_w, orig_h))
            gradcam_gen.cleanup()
        except Exception:
            heatmap = frame.copy()
        
        overlay_frames.append(overlay)
        mask_frames.append(mask_rgb)
        safety_frames.append(safety_rgb)
        gradcam_frames.append(heatmap)
    
    # Stitch all video types
    videos = {}
    
    # 1. Side-by-side (original | overlay)
    sbs_path = str(OUTPUT_DIR / f"{video_id}_sidebyside.mp4")
    stitch_sidebyside(frames, overlay_frames, sbs_path, output_fps)
    videos["sidebyside"] = f"/outputs/{video_id}_sidebyside.mp4"
    
    # 2. Overlay only
    overlay_path = str(OUTPUT_DIR / f"{video_id}_overlay.mp4")
    stitch_video(overlay_frames, overlay_path, output_fps)
    videos["overlay"] = f"/outputs/{video_id}_overlay.mp4"
    
    # 3. Mask only
    mask_path = str(OUTPUT_DIR / f"{video_id}_mask.mp4")
    stitch_video(mask_frames, mask_path, output_fps)
    videos["mask"] = f"/outputs/{video_id}_mask.mp4"
    
    # 4. Safety heatmap
    safety_path = str(OUTPUT_DIR / f"{video_id}_safety.mp4")
    stitch_video(safety_frames, safety_path, output_fps)
    videos["safety"] = f"/outputs/{video_id}_safety.mp4"
    
    # 5. GradCAM
    gradcam_path = str(OUTPUT_DIR / f"{video_id}_gradcam.mp4")
    stitch_video(gradcam_frames, gradcam_path, output_fps)
    videos["gradcam"] = f"/outputs/{video_id}_gradcam.mp4"
    
    # Preview images (first frame of each type)
    previews = {
        "overlay": image_to_base64(overlay_frames[0]),
        "mask": image_to_base64(mask_frames[0]),
        "safety": image_to_base64(safety_frames[0]),
        "gradcam": image_to_base64(gradcam_frames[0]),
    }
    
    # Cleanup input video
    os.remove(input_path)
    
    # ── Velocity summary stats ──────────────────────────────────────────
    velocities = [d["velocity"] for d in per_frame_details]
    model_usage = {}
    for d in per_frame_details:
        mk = d["model_key"]
        model_usage[mk] = model_usage.get(mk, 0) + 1
    
    return {
        "videos": videos,
        "previews": previews,
        "frames_processed": len(frames),
        "interval_ms": interval_ms,
        "output_fps": output_fps,
        "total_inference_ms": round(total_inference * 1000, 1),
        "avg_inference_per_frame_ms": round(total_inference / len(frames) * 1000, 1) if frames else 0,
        "avg_fps": round(len(frames) / total_inference, 1) if total_inference > 0 else 0,
        "video_info": {
            "input_fps": video_data['input_fps'],
            "width": video_data['width'],
            "height": video_data['height'],
            "duration_ms": round(video_data['duration_ms'], 1),
            "duration_s": round(video_data['duration_ms'] / 1000, 1),
        },
        # ── NEW: velocity-adaptive model selection data ──
        "velocity": {
            "per_frame": per_frame_details,
            "avg_velocity": round(float(np.mean(velocities)), 2) if velocities else 0,
            "max_velocity": round(float(np.max(velocities)), 2) if velocities else 0,
            "min_velocity": round(float(np.min(velocities)), 2) if velocities else 0,
            "model_usage": model_usage,
            "thresholds": {"low": 15.0, "high": 40.0},
        },
    }


@app.post("/api/preprocessing-steps")
async def get_preprocessing_steps(file: UploadFile = File(...)):
    """Get DIP preprocessing step visualizations for an image."""
    file_bytes = await file.read()
    image_rgb = read_upload_image(file_bytes)
    steps = generate_preprocessing_steps(image_rgb, target_size=IMG_SIZE)
    return {"steps": steps}


@app.post("/api/compare-models")
async def compare_models(file: UploadFile = File(...)):
    """
    Run the same image through all 3 model variants (MiT-B0, B1, B3)
    and return side-by-side segmentation results with timing info.
    """
    file_bytes = await file.read()
    image_rgb = read_upload_image(file_bytes)
    orig_h, orig_w = image_rgb.shape[:2]
    
    results = {}
    for model_key in ['mit_b0', 'mit_b1', 'mit_b3']:
        try:
            pred_mask, logits, inference_time = run_inference(image_rgb, model_key)
            mask_rgb = mask_to_rgb(pred_mask, NUM_CLASSES)
            overlay = cv2.addWeighted(image_rgb, 0.5,
                                       cv2.resize(mask_rgb, (orig_w, orig_h)), 0.5, 0)
            distribution = compute_class_distribution(pred_mask)
            safety_pcts = compute_safety_percentages(distribution)
            
            results[model_key] = {
                'overlay': image_to_base64(overlay),
                'mask': image_to_base64(mask_rgb),
                'inference_time_ms': round(inference_time * 1000, 1),
                'class_distribution': distribution,
                'safety_percentages': safety_pcts,
            }
        except Exception as e:
            results[model_key] = {'error': str(e)}
    
    return {'results': results}


@app.post("/api/finetune")
async def finetune_model(
    images: list[UploadFile] = File(...),
    masks: list[UploadFile] = File(...),
):
    """
    Fine-tune the model on user-provided images and masks.
    Max 10 images. Returns path to personalized model.
    """
    if len(images) != len(masks):
        raise HTTPException(400, "Number of images and masks must match")
    
    if len(images) > 10:
        raise HTTPException(400, "Maximum 10 images allowed")
    
    if len(images) < 1:
        raise HTTPException(400, "At least 1 image required")
    
    # Create unique user ID for this session
    user_id = str(uuid.uuid4())[:8]
    
    # Create temporary dataset directory
    dataset_dir = UPLOAD_DIR / f"finetune_{user_id}"
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save uploaded images
        for i, img_file in enumerate(images):
            img_bytes = await img_file.read()
            img_path = images_dir / f"img_{i:03d}.png"
            with open(img_path, "wb") as f:
                f.write(img_bytes)
        
        # Save uploaded masks
        for i, mask_file in enumerate(masks):
            mask_bytes = await mask_file.read()
            mask_path = masks_dir / f"img_{i:03d}.png"
            with open(mask_path, "wb") as f:
                f.write(mask_bytes)
        
        # Run fine-tuning pipeline
        model_path, metrics, comparison = run_finetune_pipeline(str(dataset_dir), user_id)
        
        # Cleanup dataset directory
        shutil.rmtree(dataset_dir)
        
        return {
            "success": True,
            "model_path": model_path,
            "user_id": user_id,
            "num_images": len(images),
            "message": f"Model fine-tuned successfully on {len(images)} images",
            "metrics": metrics,
            "comparison": comparison
        }
    
    except Exception as e:
        # Cleanup on error
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        raise HTTPException(500, f"Fine-tuning failed: {str(e)}")


# ============================================================================
# ROBUSTNESS / SHAKE TEST
# ============================================================================
def apply_perturbation(image_rgb, perturbation_type, severity):
    """
    Apply image perturbations for robustness testing.
    Types: noise, blur, brightness
    Severity: low, medium, high
    """
    img = image_rgb.copy().astype(np.float32)
    h, w = img.shape[:2]
    
    # Severity levels
    severity_map = {
        'noise': {'low': 15, 'medium': 30, 'high': 50},
        'blur': {'low': 3, 'medium': 7, 'high': 15},
        'brightness': {'low': 0.15, 'medium': 0.30, 'high': 0.50},
    }
    
    level = severity_map[perturbation_type][severity]
    
    if perturbation_type == 'noise':
        # Gaussian noise
        noise = np.random.normal(0, level, img.shape).astype(np.float32)
        img = img + noise
        
    elif perturbation_type == 'blur':
        # Gaussian blur
        ksize = level if level % 2 == 1 else level + 1
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
    elif perturbation_type == 'brightness':
        # Random brightness shift (darken or brighten)
        factor = 1.0 + (np.random.choice([-1, 1]) * level)
        img = img * factor
    
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


@app.post("/api/robustness-test")
async def robustness_test(file: UploadFile = File(...)):
    """
    Run robustness/shake test: apply noise, blur, and brightness perturbations
    at low/medium/high severity and measure segmentation changes.
    """
    file_bytes = await file.read()
    image_rgb = read_upload_image(file_bytes)
    orig_h, orig_w = image_rgb.shape[:2]
    
    # First, get baseline segmentation
    baseline_mask, _, _ = run_inference(image_rgb)
    baseline_dist = compute_class_distribution(baseline_mask)
    baseline_safety = compute_safety_percentages(baseline_dist)
    
    perturbation_types = ['noise', 'blur', 'brightness']
    severities = ['low', 'medium', 'high']
    
    results = {}
    
    for p_type in perturbation_types:
        results[p_type] = {}
        for sev in severities:
            # Apply perturbation
            perturbed_img = apply_perturbation(image_rgb, p_type, sev)
            
            # Run inference
            start = time.time()
            pred_mask, _, _ = run_inference(perturbed_img)
            inference_time = time.time() - start
            
            # Generate overlay
            mask_rgb = mask_to_rgb(pred_mask, NUM_CLASSES)
            overlay = cv2.addWeighted(
                perturbed_img, 0.5,
                cv2.resize(mask_rgb, (orig_w, orig_h)), 0.5, 0
            )
            
            # Compute safety delta compared to baseline
            dist = compute_class_distribution(pred_mask)
            safety_pcts = compute_safety_percentages(dist)
            safety_delta = {
                'safe': round(safety_pcts['safe'] - baseline_safety['safe'], 1),
                'caution': round(safety_pcts['caution'] - baseline_safety['caution'], 1),
                'obstacle': round(safety_pcts['obstacle'] - baseline_safety['obstacle'], 1),
            }
            
            results[p_type][sev] = {
                'overlay': image_to_base64(overlay),
                'safety_percentages': safety_pcts,
                'safety_delta': safety_delta,
                'inference_time_ms': round(inference_time * 1000, 1),
            }
    
    return {
        'baseline_safety': baseline_safety,
        'results': results,
    }


# ============================================================================
# MODEL DISAGREEMENT ANALYSIS
# ============================================================================
@app.post("/api/model-disagreement")
async def model_disagreement(file: UploadFile = File(...)):
    """
    Run image through all model variants (B0, B1, B3) and compute
    pixel-wise disagreement statistics and heatmap.
    """
    file_bytes = await file.read()
    image_rgb = read_upload_image(file_bytes)
    orig_h, orig_w = image_rgb.shape[:2]
    
    model_keys = ['mit_b0', 'mit_b1', 'mit_b3']
    predictions = {}
    overlays = {}
    
    # Run inference for all models
    for model_key in model_keys:
        try:
            pred_mask, _, _ = run_inference(image_rgb, model_key)
            predictions[model_key] = pred_mask
            
            # Generate overlay for this model
            mask_rgb = mask_to_rgb(pred_mask, NUM_CLASSES)
            overlay = cv2.addWeighted(
                image_rgb, 0.5,
                cv2.resize(mask_rgb, (orig_w, orig_h)), 0.5, 0
            )
            overlays[model_key] = image_to_base64(overlay)
        except Exception as e:
            print(f"[WARN] Model {model_key} failed: {e}")
            # Use zeros as fallback
            predictions[model_key] = np.zeros((orig_h, orig_w), dtype=np.uint8)
            overlays[model_key] = image_to_base64(image_rgb)
    
    # Compute disagreement statistics
    masks = [predictions[k] for k in model_keys]
    
    # Resize all masks to same size for comparison
    target_h, target_w = masks[0].shape[:2]
    masks_resized = []
    for m in masks:
        if m.shape != (target_h, target_w):
            m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        masks_resized.append(m)
    
    # Stack and compute agreement
    stacked = np.stack(masks_resized, axis=0)  # (3, H, W)
    
    # For each pixel, count unique predictions
    # Full agreement: all 3 same, Partial: 2 same, Full disagreement: all different
    total_pixels = target_h * target_w
    
    full_agree = 0
    partial_disagree = 0
    full_disagree = 0
    
    # Compute per-pixel disagreement for heatmap
    disagreement_map = np.zeros((target_h, target_w), dtype=np.float32)
    
    for y in range(target_h):
        for x in range(target_w):
            pixel_preds = stacked[:, y, x]
            unique_count = len(np.unique(pixel_preds))
            
            if unique_count == 1:
                full_agree += 1
                disagreement_map[y, x] = 0.0
            elif unique_count == 2:
                partial_disagree += 1
                disagreement_map[y, x] = 0.5
            else:  # all 3 different
                full_disagree += 1
                disagreement_map[y, x] = 1.0
    
    # Convert to percentages
    statistics = {
        'full_agreement': round(full_agree / total_pixels * 100, 1),
        'partial_disagreement': round(partial_disagree / total_pixels * 100, 1),
        'full_disagreement': round(full_disagree / total_pixels * 100, 1),
    }
    
    # Generate disagreement heatmap overlay
    # Resize to original image size
    disagreement_full = cv2.resize(disagreement_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    disagreement_vis = (disagreement_full * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(disagreement_vis, cv2.COLORMAP_HOT)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    heatmap_overlay = cv2.addWeighted(image_rgb, 0.4, heatmap_colored, 0.6, 0)
    
    return {
        'statistics': statistics,
        'overlay': image_to_base64(heatmap_overlay),
        'per_model_predictions': overlays,
    }


# ============================================================================
# ROBUSTNESS / SHAKE TEST
# ============================================================================
def apply_perturbation(image_rgb, perturbation_type, severity):
    """
    Apply image perturbations for robustness testing.
    Types: noise, blur, brightness
    Severity: low, medium, high
    """
    img = image_rgb.copy().astype(np.float32)
    h, w = img.shape[:2]
    
    # Severity levels
    severity_map = {
        'noise': {'low': 15, 'medium': 30, 'high': 50},
        'blur': {'low': 3, 'medium': 7, 'high': 15},
        'brightness': {'low': 0.15, 'medium': 0.30, 'high': 0.50},
    }
    
    level = severity_map[perturbation_type][severity]
    
    if perturbation_type == 'noise':
        # Gaussian noise
        noise = np.random.normal(0, level, img.shape).astype(np.float32)
        img = img + noise
        
    elif perturbation_type == 'blur':
        # Gaussian blur
        ksize = level if level % 2 == 1 else level + 1
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
    elif perturbation_type == 'brightness':
        # Random brightness shift (darken or brighten)
        factor = 1.0 + (np.random.choice([-1, 1]) * level)
        img = img * factor
    
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


@app.post("/api/robustness-test")
async def robustness_test(file: UploadFile = File(...)):
    """
    Run robustness/shake test: apply noise, blur, and brightness perturbations
    at low/medium/high severity and measure segmentation changes.
    """
    file_bytes = await file.read()
    image_rgb = read_upload_image(file_bytes)
    orig_h, orig_w = image_rgb.shape[:2]
    
    # First, get baseline segmentation
    baseline_mask, _, _ = run_inference(image_rgb)
    baseline_dist = compute_class_distribution(baseline_mask)
    baseline_safety = compute_safety_percentages(baseline_dist)
    
    perturbation_types = ['noise', 'blur', 'brightness']
    severities = ['low', 'medium', 'high']
    
    results = {}
    
    for p_type in perturbation_types:
        results[p_type] = {}
        for sev in severities:
            # Apply perturbation
            perturbed_img = apply_perturbation(image_rgb, p_type, sev)
            
            # Run inference
            start = time.time()
            pred_mask, _, _ = run_inference(perturbed_img)
            inference_time = time.time() - start
            
            # Generate overlay
            mask_rgb = mask_to_rgb(pred_mask, NUM_CLASSES)
            overlay = cv2.addWeighted(
                perturbed_img, 0.5,
                cv2.resize(mask_rgb, (orig_w, orig_h)), 0.5, 0
            )
            
            # Compute safety delta compared to baseline
            dist = compute_class_distribution(pred_mask)
            safety_pcts = compute_safety_percentages(dist)
            safety_delta = {
                'safe': round(safety_pcts['safe'] - baseline_safety['safe'], 1),
                'caution': round(safety_pcts['caution'] - baseline_safety['caution'], 1),
                'obstacle': round(safety_pcts['obstacle'] - baseline_safety['obstacle'], 1),
            }
            
            results[p_type][sev] = {
                'overlay': image_to_base64(overlay),
                'safety_percentages': safety_pcts,
                'safety_delta': safety_delta,
                'inference_time_ms': round(inference_time * 1000, 1),
            }
    
    return {
        'baseline_safety': baseline_safety,
        'results': results,
    }


# ============================================================================
# MODEL DISAGREEMENT ANALYSIS
# ============================================================================
@app.post("/api/model-disagreement")
async def model_disagreement(file: UploadFile = File(...)):
    """
    Run image through all model variants (B0, B1, B3) and compute
    pixel-wise disagreement statistics and heatmap.
    """
    file_bytes = await file.read()
    image_rgb = read_upload_image(file_bytes)
    orig_h, orig_w = image_rgb.shape[:2]
    
    model_keys = ['mit_b0', 'mit_b1', 'mit_b3']
    predictions = {}
    overlays = {}
    
    # Run inference for all models
    for model_key in model_keys:
        try:
            pred_mask, _, _ = run_inference(image_rgb, model_key)
            predictions[model_key] = pred_mask
            
            # Generate overlay for this model
            mask_rgb = mask_to_rgb(pred_mask, NUM_CLASSES)
            overlay = cv2.addWeighted(
                image_rgb, 0.5,
                cv2.resize(mask_rgb, (orig_w, orig_h)), 0.5, 0
            )
            overlays[model_key] = image_to_base64(overlay)
        except Exception as e:
            print(f"[WARN] Model {model_key} failed: {e}")
            # Use zeros as fallback
            predictions[model_key] = np.zeros((orig_h, orig_w), dtype=np.uint8)
            overlays[model_key] = image_to_base64(image_rgb)
    
    # Compute disagreement statistics
    masks = [predictions[k] for k in model_keys]
    
    # Resize all masks to same size for comparison
    target_h, target_w = masks[0].shape[:2]
    masks_resized = []
    for m in masks:
        if m.shape != (target_h, target_w):
            m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        masks_resized.append(m)
    
    # Stack and compute agreement
    stacked = np.stack(masks_resized, axis=0)  # (3, H, W)
    
    # For each pixel, count unique predictions
    # Full agreement: all 3 same, Partial: 2 same, Full disagreement: all different
    total_pixels = target_h * target_w
    
    full_agree = 0
    partial_disagree = 0
    full_disagree = 0
    
    # Compute per-pixel disagreement for heatmap
    disagreement_map = np.zeros((target_h, target_w), dtype=np.float32)
    
    for y in range(target_h):
        for x in range(target_w):
            pixel_preds = stacked[:, y, x]
            unique_count = len(np.unique(pixel_preds))
            
            if unique_count == 1:
                full_agree += 1
                disagreement_map[y, x] = 0.0
            elif unique_count == 2:
                partial_disagree += 1
                disagreement_map[y, x] = 0.5
            else:  # all 3 different
                full_disagree += 1
                disagreement_map[y, x] = 1.0
    
    # Convert to percentages
    statistics = {
        'full_agreement': round(full_agree / total_pixels * 100, 1),
        'partial_disagreement': round(partial_disagree / total_pixels * 100, 1),
        'full_disagreement': round(full_disagree / total_pixels * 100, 1),
    }
    
    # Generate disagreement heatmap overlay
    # Resize to original image size
    disagreement_full = cv2.resize(disagreement_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    disagreement_vis = (disagreement_full * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(disagreement_vis, cv2.COLORMAP_HOT)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    heatmap_overlay = cv2.addWeighted(image_rgb, 0.4, heatmap_colored, 0.6, 0)
    
    return {
        'statistics': statistics,
        'overlay': image_to_base64(heatmap_overlay),
        'per_model_predictions': overlays,
    }


# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  TerrainAI - Intelligent Terrain Analysis Platform")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
