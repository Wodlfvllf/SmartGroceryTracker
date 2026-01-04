"""
Modal Labs Cloud Configuration for Smart Grocery Tracker
=========================================================

This module provides shared Modal configuration for running DETR training
and inference on Modal Labs cloud infrastructure with GPU support.

Usage:
    # Run training on Modal cloud
    modal run train.py --train-images /data/train --train-annotations /data/train/_annotations.coco.json

    # Run zero-shot inference on Modal cloud
    modal run zero_shot.py --image-dir /data/test

    # Deploy as a persistent service
    modal deploy modal_run.py
"""

import modal

# =============================================================================
# MODAL APP CONFIGURATION
# =============================================================================

APP_NAME = "smart-grocery-tracker"

# Create the Modal app
app = modal.App(APP_NAME)

# =============================================================================
# DOCKER IMAGE CONFIGURATION
# =============================================================================

# Build container image with all required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx",  # Required for OpenCV
        "libglib2.0-0",     # Required for OpenCV
        "git",
    )
    .pip_install(
        # Core PyTorch
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        # Hugging Face (for pretrained DETR)
        "transformers>=4.30.0",
        # Scientific computing
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        # Image processing
        "opencv-python>=4.8.0",
        "Pillow>=9.0.0",
        "albumentations>=1.3.0",
        # Data handling
        "pandas>=2.0.0",
        "pycocotools>=2.0.6",
        # Configuration
        "pyyaml>=6.0",
        # Progress bar
        "tqdm>=4.65.0",
        # Visualization
        "matplotlib>=3.7.0",
        # Machine learning utilities
        "scikit-learn>=1.2.0",
    )
)

# =============================================================================
# VOLUME CONFIGURATION
# =============================================================================

# Volume for storing dataset
data_volume = modal.Volume.from_name("grocery-data", create_if_missing=True)

# Volume for storing checkpoints and outputs
checkpoints_volume = modal.Volume.from_name("grocery-checkpoints", create_if_missing=True)

# Volume mount paths (used inside Modal containers)
DATA_PATH = "/data"
CHECKPOINTS_PATH = "/checkpoints"
OUTPUT_PATH = "/output"

# =============================================================================
# GPU CONFIGURATIONS
# =============================================================================

# GPU options for different workloads
GPU_TRAINING = "A10G"      # Good balance of performance and cost for training
GPU_INFERENCE = "T4"        # Cost-effective for inference
GPU_HEAVY = "A100"          # For large batch training or complex models

# Timeout configurations (in seconds)
TIMEOUT_TRAINING = 86400    # 24 hours for training jobs
TIMEOUT_INFERENCE = 3600    # 1 hour for inference jobs

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@app.function(image=image)
def health_check():
    """Simple health check to verify Modal setup is working."""
    import torch
    
    # Return only simple Python types (str, int, bool) to avoid
    # deserialization issues when torch isn't installed locally
    cuda_available = torch.cuda.is_available()
    
    info = {
        "status": "healthy",
        "torch_version": str(torch.__version__),
        "cuda_available": str(cuda_available),
        "cuda_device_count": str(torch.cuda.device_count() if cuda_available else 0),
    }
    
    if cuda_available:
        info["cuda_device_name"] = str(torch.cuda.get_device_name(0))
    
    return info


@app.function(image=image, volumes={DATA_PATH: data_volume})
def list_data():
    """List contents of the data volume."""
    import os
    
    contents = []
    for root, dirs, files in os.walk(DATA_PATH):
        level = root.replace(DATA_PATH, '').count(os.sep)
        indent = ' ' * 2 * level
        contents.append(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:10]:  # Limit files shown
            contents.append(f"{subindent}{file}")
        if len(files) > 10:
            contents.append(f"{subindent}... and {len(files) - 10} more files")
    
    return "\n".join(contents)


@app.local_entrypoint()
def main():
    """Local entrypoint for testing Modal setup."""
    print("=" * 60)
    print("Smart Grocery Tracker - Modal Labs Cloud Setup")
    print("=" * 60)
    
    print("\nRunning health check...")
    result = health_check.remote()
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\nData volume contents:")
    try:
        contents = list_data.remote()
        print(contents if contents else "  (empty)")
    except Exception as e:
        print(f"  Error listing data: {e}")
    
    print("\n" + "=" * 60)
    print("Setup verified successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Upload your dataset to the 'grocery-data' volume")
    print("  2. Run training: modal run train.py --help")
    print("  3. Run inference: modal run zero_shot.py --help")
