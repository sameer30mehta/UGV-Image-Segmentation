"""
TerrainAI — Launch Script
Starts the FastAPI backend server.

Usage:
    cd "SPIT Hack"
    python run.py

Then open http://localhost:8000 in your browser.

To use your trained model, place the .pth file at:
    backend/weights/best_desert_segmentation.pth

Or set the environment variable:
    set MODEL_WEIGHTS=path/to/your/model.pth
"""
import os
import sys
from pathlib import Path


def discover_weights_local(weights_dir):
    """Lightweight weight discovery for launcher logging."""
    mapping = {'mit_b3': None, 'mit_b1': None, 'mit_b0': None}
    pths = sorted(weights_dir.glob("*.pth")) if weights_dir.exists() else []
    if not pths:
        return mapping

    best_b3 = next((p for p in pths if p.name.lower() == "best_desert_segmentation.pth"), None) or pths[0]
    mapping['mit_b3'] = str(best_b3)

    best_b1 = next((p for p in pths if p.name.lower() == "best_model.pth" and p != best_b3), None)
    if best_b1 is None:
        alternatives = [p for p in pths if p != best_b3]
        if alternatives:
            best_b1 = alternatives[0]
    if best_b1 is not None:
        mapping['mit_b1'] = str(best_b1)

    used = {best_b3, best_b1}
    third = [p for p in pths if p not in used]
    if third:
        mapping['mit_b0'] = str(third[0])
    return mapping

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Create required directories
os.makedirs('backend/weights', exist_ok=True)
os.makedirs('backend/uploads', exist_ok=True)
os.makedirs('backend/outputs', exist_ok=True)

if __name__ == '__main__':
    import uvicorn
    
    print()
    print("=" * 60)
    print("  TerrainAI - Intelligent Terrain Analysis Platform")
    print("=" * 60)
    print()
    print("  Frontend:  http://localhost:8000")
    print("  API Docs:  http://localhost:8000/docs")
    print()
    
    # Check and report model weights
    root_dir = Path(__file__).parent
    weights_dir = Path(os.environ.get("MODEL_WEIGHTS_DIR", str(root_dir / "backend" / "weights")))
    weight_map = discover_weights_local(weights_dir)
    pth_files = sorted(weights_dir.glob("*.pth")) if weights_dir.exists() else []

    if pth_files:
        print(f"  [OK] Weight files detected in {weights_dir}:")
        for p in pth_files:
            print(f"     - {p}")
        print("  Assigned model tiers:")
        print(f"     - mit_b3 (best): {weight_map.get('mit_b3') or 'ImageNet fallback'}")
        print(f"     - mit_b1 (second): {weight_map.get('mit_b1') or 'ImageNet fallback'}")
        print(f"     - mit_b0 (ultra-fast): {weight_map.get('mit_b0') or 'ImageNet fallback'}")
    else:
        print(f"  [WARN] No .pth files found in: {weights_dir}")
        print("     Using ImageNet pre-trained initialization")
        print("     Copy your .pth files into backend/weights")
    
    print()
    print("=" * 60)
    print()
    
    # Set working directory to backend for relative imports
    os.chdir(os.path.join(os.path.dirname(__file__), 'backend'))
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
