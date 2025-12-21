# Smart Grocery Tracker - DETR Implementation

Vision-Based Automatic Grocery Tracking System using **DETR (DEtection TRansformer)** architecture for detecting grocery items in cluttered refrigerator environments.

Inspired by the research paper "Vision-Based Automatic Groceries Tracking System - Smart Homes", but reimplemented using DETR instead of YOLO for improved handling of small, overlapping objects.

## 🚀 Quick Start

### Installation

```bash
cd SmartGroceryTracker
pip install -r requirements.txt
```

### Zero-Shot Inference (No Training Required)

Test with pretrained COCO model to establish baseline:

```bash
python zero_shot.py --image_path your_fridge_image.jpg --output_dir output/
```

### Training

Fine-tune on your grocery dataset:

```bash
python train.py \
    --train_images data/train/images \
    --train_annotations data/train/annotations.json \
    --val_images data/val/images \
    --val_annotations data/val/annotations.json \
    --epochs 100
```

### Inference

Run predictions with trained model:

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image_dir test_images/ \
    --output_dir results/
```

## 📁 Project Structure

```
SmartGroceryTracker/
├── configs/
│   └── default.yaml          # Configuration (classes, hyperparameters)
├── data/
│   ├── dataset.py            # COCO format dataset loader
│   └── transforms.py         # Data augmentation pipeline
├── models/
│   ├── backbone.py           # ResNet backbone
│   ├── position_encoding.py  # Sinusoidal positional encodings
│   ├── transformer.py        # Transformer encoder/decoder
│   ├── detr.py               # Main DETR (pretrained & scratch)
│   └── matcher.py            # Hungarian matcher
├── training/
│   ├── criterion.py          # DETR loss (CE + L1 + GIoU)
│   └── trainer.py            # Training loop
├── utils/
│   ├── box_ops.py            # Bounding box utilities
│   └── misc.py               # General utilities
├── zero_shot.py              # Script A: Zero-shot inference
├── train.py                  # Script B: Fine-tuning
└── inference.py              # Script C: Flexible inference
```

## 🏗️ Model Options

### Option A: Pretrained Fine-tuning (Recommended)

```python
from models.detr import build_detr

model = build_detr(
    num_classes=25,        # Your grocery classes
    use_pretrained=True,   # Load COCO pretrained weights
)
```

### Option B: From Scratch (Educational)

```python
model = build_detr(
    num_classes=25,
    use_pretrained=False,  # Full manual implementation
)
```

## 📊 Dataset Format

Data should be in COCO format:

```
data/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── ...
│   └── annotations.json
└── val/
    ├── images/
    └── annotations.json
```

**annotations.json** structure:
```json
{
  "images": [{"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h]}],
  "categories": [{"id": 1, "name": "banana"}]
}
```

## 🔧 Key Features

- **Hungarian Matching**: Optimal 1-to-1 assignment between predictions and ground truth
- **No NMS Required**: DETR's set-based predictions eliminate need for post-processing
- **Object Queries**: 100 learnable embeddings for detecting up to 100 objects per image
- **Multi-scale Training**: Random resize augmentation for scale invariance

## 📈 Training Tips

1. **Batch Size**: Use 2-4 due to memory constraints
2. **Learning Rate**: 1e-4 for transformer, 1e-5 for backbone
3. **Gradient Clipping**: Essential (max_norm=0.1)
4. **Epochs**: 300 for full training, 50-100 for fine-tuning

## 📚 References

- [DETR Paper](https://arxiv.org/abs/2005.12872) - End-to-End Object Detection with Transformers
- [Hugging Face DETR](https://huggingface.co/facebook/detr-resnet-50)
