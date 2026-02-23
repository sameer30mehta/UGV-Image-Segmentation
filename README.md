# TerrainAI — UGV Image Segmentation

AI-powered semantic segmentation platform for autonomous off-road UGV (Unmanned Ground Vehicle) navigation. TerrainAI classifies every pixel in terrain images into semantic classes (landscape, sky, trees, rocks, bushes, etc.) to enable safe autonomous traversal of desert and off-road environments.

## Key Features

- **Semantic Segmentation** — 10-class pixel-level terrain labeling
- **Domain Generalization** — Trained on synthetic data, generalizes to real-world desert images via a 14-step augmentation pipeline
- **Multiple Model Variants** — Accuracy/speed trade-off across three architectures (MiT-B3 / EfficientNet-B4 / MobileNetV2)
- **Video Processing** — Frame-by-frame segmentation with H.264-encoded output
- **Few-Shot Fine-Tuning** — Personalize models with as few as 1–10 labeled image/mask pairs
- **Explainability** — GradCAM heatmaps showing which regions influenced predictions
- **Multi-Model Comparison** — A/B test different architectures on the same input
- **Traversability Scoring** — Safety classification (safe / caution / obstacle) from segmentation output
- **Premium Web UI** — Cinematic Martian-themed interface with drag-and-drop upload, before/after sliders, and interactive charts

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | MiT-B3 (Mix Vision Transformer, ~47M params) |
| **Decoder** | FPN (Feature Pyramid Network) |
| **Library** | segmentation-models-pytorch + timm |
| **Loss** | Composite: 0.3×WCE + 0.3×Focal + 0.3×Dice + 0.1×Boundary |
| **Optimizer** | AdamW with OneCycleLR scheduling |
| **Precision** | AMP mixed precision (FP16) |

## Project Structure

```
UGV-Image-Segmentation/
└── Perpetual-Motion-Squad/
    ├── backend/                     # FastAPI server & ML pipelines
    │   ├── app.py                   # REST API endpoints
    │   ├── models.py                # Model registry (3 variants)
    │   ├── finetune.py              # Few-shot fine-tuning pipeline
    │   ├── video_processor.py       # Video frame extraction & encoding
    │   ├── gradcam.py               # GradCAM explainability
    │   ├── smart_selector.py        # Automatic model routing
    │   ├── velocity_estimator.py    # Traversability estimation
    │   └── preprocessing_viz.py     # Augmentation visualization
    ├── frontend/                    # Web interface
    │   ├── index.html               # Single-page application
    │   ├── app.js                   # Interactive UI logic
    │   └── style.css                # Cinematic Martian theme
    ├── cell_0_setup.py              # Pipeline stage: environment setup
    ├── cell_1_preprocessing.py      # Pipeline stage: data preprocessing
    ├── cell_2_training.py           # Pipeline stage: model training
    ├── cell_3_testing.py            # Pipeline stage: inference & evaluation
    ├── cell_4_error_analysis.py     # Pipeline stage: metrics & analysis
    ├── preprocessing.py             # Augmentation configuration
    ├── train_pipeline.py            # End-to-end training loop
    ├── train_segmentation.py        # Alternative training script
    ├── test_segmentation.py         # Validation & evaluation script
    ├── run.py                       # Application launcher
    ├── requirements.txt             # Python dependencies
    ├── pipeline_explanation.md       # Architecture & methods documentation
    ├── INTEGRATION_GUIDE.md         # Fine-tuning integration guide
    ├── MODEL_PERSISTENCE.md         # Model persistence system docs
    └── train_segformer.ipynb        # Jupyter notebook for training
```

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU recommended (NVIDIA T4 or better)

### Installation

```bash
# Clone the repository
git clone https://github.com/sameer30mehta/UGV-Image-Segmentation.git
cd UGV-Image-Segmentation/Perpetual-Motion-Squad

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
python run.py
```

This starts the FastAPI server on `http://localhost:8000`, loads all model variants, and serves the web UI.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/segment` | Predict segmentation mask for an image |
| `POST` | `/api/segment-video` | Process a video file frame-by-frame |
| `POST` | `/api/finetune` | Fine-tune a model on user-provided data |
| `POST` | `/api/load-user-model` | Activate a fine-tuned model |
| `GET`  | `/api/gradcam` | Generate GradCAM explanation heatmap |
| `POST` | `/api/compare-models` | Compare inference across model variants |

## Model Variants

| Variant | Encoder | Decoder | Params | Use Case |
|---------|---------|---------|--------|----------|
| `mit_b3` | MiT-B3 | FPN | ~47M | Best accuracy |
| `mit_b1` | EfficientNet-B4 | DeepLabV3+ | ~25M | Balanced accuracy/speed |
| `mit_b0` | MobileNetV2 | Linknet | ~4M | Ultra-fast, edge deployment |

## Training Pipeline

The training pipeline is split into modular stages (`cell_0` through `cell_4`), originally developed in Jupyter notebooks:

1. **Setup** — Environment configuration and dataset extraction
2. **Preprocessing** — 14-step augmentation pipeline addressing the synthetic-to-real domain gap (CLAHE, color jitter, grayscale conversion, motion blur, random shadows, etc.)
3. **Training** — FPN + MiT-B3 with composite loss, OneCycleLR scheduler, and AMP mixed precision
4. **Testing** — Inference, RGB mask visualization, and per-class metrics
5. **Error Analysis** — Confusion matrices, per-class IoU/F1/precision/recall, and hardest sample identification

## Fine-Tuning

TerrainAI supports few-shot fine-tuning to personalize models for specific terrain types:

- Upload 1–10 image/mask pairs through the web UI
- Only the decoder and segmentation head are trained; the encoder stays frozen
- Fine-tuned models are saved as `models/user_<uuid>.pth` and persist across sessions

See [INTEGRATION_GUIDE.md](Perpetual-Motion-Squad/INTEGRATION_GUIDE.md) and [MODEL_PERSISTENCE.md](Perpetual-Motion-Squad/MODEL_PERSISTENCE.md) for details.

## Documentation

- [Pipeline Explanation](Perpetual-Motion-Squad/pipeline_explanation.md) — Detailed architecture and methodology
- [Integration Guide](Perpetual-Motion-Squad/INTEGRATION_GUIDE.md) — Fine-tuning integration
- [Model Persistence](Perpetual-Motion-Squad/MODEL_PERSISTENCE.md) — Model saving and loading system
