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
    
    # Check for model weights
    weights_path = os.environ.get("MODEL_WEIGHTS", "backend/weights/best_desert_segmentation.pth")
    if os.path.exists(weights_path):
        print(f"  [OK] Model weights found: {weights_path}")
    else:
        print(f"  [WARN] No weights at: {weights_path}")
        print(f"     Using ImageNet pre-trained (results will be untrained)")
        print(f"     Copy your .pth file to: backend/weights/best_desert_segmentation.pth")
    
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
