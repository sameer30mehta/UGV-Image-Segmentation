# Desert Semantic Segmentation — Pipeline Explanation

## Overview

We built a **pixel-level semantic segmentation** system that classifies every pixel in a synthetic desert image into one of ~10 classes (landscape, sky, trees, bushes, rocks, etc.). The key challenge: the model trains on **colorful synthetic data** but must generalize to **monochromatic, unseen real desert images**.

---

## Architecture: FPN + MiT-B3 (CNN–Transformer Hybrid)

### What is it?
We use **Feature Pyramid Network (FPN)** as the decoder with a **Mix Vision Transformer (MiT-B3)** as the encoder backbone, from the **SegFormer** family.

### Why this combination?

| Component | Role | Why it helps |
|-----------|------|-------------|
| **MiT-B3 (Encoder)** | Extracts features from the image | Uses **self-attention** to look at the *entire* image at once, not just local patches. This resolves ambiguities like "is this brown patch a rock or dirt?" by using global context. |
| **FPN (Decoder)** | Fuses multi-scale features into a segmentation mask | Combines fine-grained detail (edges of small rocks) with high-level context (sky vs ground), producing sharp boundaries. |
| **ImageNet Pre-training** | Initializes encoder weights | Instead of learning from scratch, we start with weights that already understand edges, textures, and shapes — drastically reducing training time. |

### Key specs
- **Parameters:** ~47M (MiT-B3 + FPN decoder)
- **Input:** 512×512 RGB crops
- **Output:** 512×512 mask with per-pixel class predictions

### How it works (simplified):

```
Input Image (512x512x3)
    │
    ▼
┌─────────────────────────┐
│  MiT-B3 Transformer     │  ← Self-attention at 4 different scales
│  Encoder (Backbone)      │     Produces feature maps at 1/4, 1/8, 1/16, 1/32 resolution
└─────────────────────────┘
    │  Multi-scale features
    ▼
┌─────────────────────────┐
│  FPN Decoder             │  ← Fuses all 4 scales together
│  (Feature Pyramid)       │     Upsamples back to 1/4 resolution
└─────────────────────────┘
    │
    ▼
  Final 1x1 Conv → NUM_CLASSES channels
    │
    ▼
  Predicted Mask (512x512)
```

---

## Preprocessing Pipeline

### Why heavy preprocessing?
The training data is **synthetic** (rendered by Unreal Engine) and **colorful**. The test data is **real-world** and nearly **monochromatic brown**. Without preprocessing, the model memorizes synthetic colors and fails on real data.

### What we do:

| Step | Technique | Purpose |
|------|-----------|---------|
| 1 | **Resize** (shortest side → 768px) | Preserve aspect ratio, control memory |
| 2 | **Random Scale** (0.75x–1.5x) | Scale invariance |
| 3 | **Random Crop** (512×512) | Fits GPU memory, varied viewpoints |
| 4 | **Horizontal Flip** | Layout invariance |
| 5 | **CLAHE** (clip=2.0, 8×8 grid) | Enhances local contrast between similar-looking classes (dry grass vs dirt) |
| 6 | **Random Gamma** (0.8–1.2) | Simulates different sun exposure |
| 7 | **Color Jitter** (brightness/contrast ±30%) | Destroys reliance on specific synthetic colors |
| 8 | **ToGray** (p=0.2) | Forces model to learn from **texture/shape**, not color |
| 9 | **Gaussian Noise** | Simulates real camera sensor noise |
| 10 | **Motion Blur** | Simulates UGV vibration |
| 11 | **Random Shadow** | Prevents confusing shadows with rocks |
| 12 | **Coarse Dropout** | Random rectangular erasure → forces contextual reasoning |
| 13 | **JPEG Compression** | Robustness to compression artifacts |
| 14 | **ImageNet Normalize** | Required for pre-trained backbone |

### Auto Class Mapping
The dataset masks use arbitrary grayscale values (0, 1, 2, 3, 27, 39) instead of sequential (0–5). We auto-scan the training masks and build a `{raw_value → class_index}` dictionary to remap them to 0, 1, 2, 3, 4, 5 before training.

---

## Loss Function: Composite Multi-Objective

We don't use a single loss. We combine **four** losses, each solving a different problem:

### 1. Weighted Cross-Entropy (WCE) — λ=0.3
- Standard pixel classification loss
- **Weighted** by inverse class frequency: rare classes like rocks/logs get 10–20x higher penalty than landscape/sky
- Prevents the model from ignoring small objects

### 2. Focal Loss — λ=0.3
- Dynamically **down-weights easy pixels** (sky, big landscape patches)
- **Amplifies loss on hard pixels** (bush-vs-ground boundaries)
- Formula: `FL = -α(1-pt)^γ * log(pt)` where γ=2.0
- Focuses the optimizer on the confusing boundaries

### 3. Dice Loss — λ=0.3
- Directly optimizes **spatial overlap** (IoU proxy)
- Resistant to class imbalance by design
- Ensures small objects (flowers, logs) maintain their spatial structure

### 4. Boundary Loss — λ=0.1
- Computes Sobel edges on the ground truth mask
- Compares model's low-confidence regions with actual class boundaries
- Forces crisp, accurate edges between classes

### Total loss:
```
L_total = 0.3 × WCE + 0.3 × Focal + 0.3 × Dice + 0.1 × Boundary
```

---

## Training Strategy

| Setting | Value | Why |
|---------|-------|-----|
| **Optimizer** | AdamW | Proper weight decay for transformers (decoupled from gradient) |
| **Learning Rate** | 3e-4 (max) | OneCycleLR handles the schedule |
| **Scheduler** | OneCycleLR | Warm-up → peak → cosine decay. Avoids wasting early epochs with tiny LR |
| **Weight Decay** | 1e-4 | Prevents overfitting to synthetic data |
| **Mixed Precision** | AMP (FP16) | Halves memory, 1.5–3x faster on T4 Tensor Cores |
| **Batch Size** | 4 | Max that fits T4 with AMP enabled |
| **Epochs** | 20 | Sufficient with OneCycleLR + pre-trained weights |

### Mixed Precision (AMP) Explained
- Forward pass runs in **FP16** (16-bit) on T4 Tensor Cores → 2x faster math
- Gradients scaled by `GradScaler` to prevent underflow
- Weights updated in **FP32** for precision
- Net effect: **~2x faster training, 50% less VRAM**

---

## Testing & Inference

1. Load best model checkpoint (lowest val loss / highest mIoU)
2. Resize test images to 512×512
3. Forward pass with AMP
4. `argmax` on logits → predicted class per pixel
5. Resize prediction back to original resolution (nearest-neighbor interpolation)
6. Save colored mask overlays

---

## Error Analysis & Metrics

| Metric | What it measures |
|--------|-----------------|
| **Per-class IoU** | Overlap between predicted and true mask *for each class* |
| **mIoU** | Average of all per-class IoUs (the competition metric) |
| **Precision** | Of all pixels predicted as class X, how many were correct? |
| **Recall** | Of all true class X pixels, how many did we find? |
| **F1-Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Shows exactly which classes get confused with which |

### Visualizations produced:
- Per-class IoU bar chart
- Normalized confusion matrix heatmap
- Training loss & mIoU curves
- Learning rate schedule curve
- Hardest validation samples (highest error rate) with error maps
- Side-by-side: Input → Ground Truth → Prediction → Error Map

---

## Summary Diagram

```
┌──────────────┐     ┌──────────────────┐     ┌────────────────┐
│  Synthetic   │     │  Augmentation    │     │  FPN + MiT-B3  │
│  Desert Data │────▶│  Pipeline        │────▶│  (Transformer  │
│  (RGB+Mask)  │     │  (14 transforms) │     │   Hybrid)      │
└──────────────┘     └──────────────────┘     └───────┬────────┘
                                                      │
                                               ┌──────▼──────┐
                                               │ Composite   │
                                               │ Loss (4-way)│
                                               │ WCE+Focal+  │
                                               │ Dice+Boundary│
                                               └──────┬──────┘
                                                      │
                                               ┌──────▼──────┐
                                               │  AdamW +    │
                                               │  OneCycleLR │
                                               │  + AMP      │
                                               └──────┬──────┘
                                                      │
                                               ┌──────▼──────┐
                                               │  Trained    │
                                               │  Model      │──▶ Inference on
                                               │  (.pth)     │    Unseen Test Data
                                               └─────────────┘
```
