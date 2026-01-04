#!/usr/bin/env python
"""
Zero-Shot Inference Script (Modal Labs Cloud)
==============================================

Script A: Load a pre-trained DETR model (COCO weights) and run inference
on Custom Refrigerator Dataset images with ground truth comparison.

Features:
- Zero-shot inference using pretrained DETR
- Ground truth comparison using COCO annotations
- Generates accuracy report and visualization

Usage (Modal Cloud):
    # Run on validation set (has annotations for comparison)
    modal run zero_shot.py
    
    # Custom options
    modal run zero_shot.py --image-dir /data/valid --confidence 0.5
"""

import modal

# =============================================================================
# MODAL CONFIGURATION (self-contained)
# =============================================================================

APP_NAME = "smart-grocery-zero-shot"
app = modal.App(APP_NAME)

# Docker image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "opencv-python>=4.8.0",
        "Pillow>=9.0.0",
        "numpy>=1.24.0",
        "timm",
    )
)

# Volumes
data_volume = modal.Volume.from_name("grocery-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("grocery-checkpoints", create_if_missing=True)

DATA_PATH = "/data"
CHECKPOINTS_PATH = "/checkpoints"
GPU_INFERENCE = "T4"

# =============================================================================
# COCO CLASSES (pretrained DETR knows these)
# =============================================================================

COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Food-related COCO classes
FOOD_CLASSES = {
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'refrigerator'
}

# =============================================================================
# MODAL FUNCTION
# =============================================================================

@app.function(
    gpu=GPU_INFERENCE,
    image=image,
    volumes={
        DATA_PATH: data_volume,
        CHECKPOINTS_PATH: checkpoints_volume,
    },
    timeout=3600,
)
def run_zero_shot_inference(
    image_dir: str = None,
    annotations_file: str = None,
    output_dir: str = f"{CHECKPOINTS_PATH}/zero_shot",
    confidence: float = 0.5,
    model_name: str = "facebook/detr-resnet-50",
    max_samples: int = None,
) -> dict:
    """Run zero-shot inference with ground truth comparison."""
    import os
    import json
    from pathlib import Path
    from typing import List, Dict
    from collections import defaultdict
    
    import torch
    import cv2
    import numpy as np
    from PIL import Image
    from transformers import DetrForObjectDetection, DetrImageProcessor
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print("Zero-Shot Inference with Ground Truth Comparison")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model_name}")
    print(f"Confidence: {confidence}")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading pretrained DETR...")
    processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)
    model.to(device)
    model.eval()
    print("Model loaded!\n")
    
    # Load ground truth annotations if available
    gt_data = None
    gt_by_image = {}
    gt_categories = {}
    
    if annotations_file and os.path.exists(annotations_file):
        print(f"Loading ground truth: {annotations_file}")
        with open(annotations_file, 'r') as f:
            gt_data = json.load(f)
        
        # Build category mapping
        for cat in gt_data.get('categories', []):
            gt_categories[cat['id']] = cat['name']
        
        # Build image filename to annotations mapping
        img_id_to_file = {img['id']: img['file_name'] for img in gt_data.get('images', [])}
        
        for ann in gt_data.get('annotations', []):
            img_id = ann['image_id']
            filename = img_id_to_file.get(img_id, '')
            if filename not in gt_by_image:
                gt_by_image[filename] = []
            gt_by_image[filename].append({
                'label': gt_categories.get(ann['category_id'], 'unknown'),
                'bbox': ann['bbox'],  # [x, y, w, h] format
            })
        
        print(f"  Categories: {list(gt_categories.values())[:5]}...")
        print(f"  Images with annotations: {len(gt_by_image)}")
    
    # Prediction function
    @torch.no_grad()
    def predict(img: Image.Image) -> List[dict]:
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        
        target_sizes = torch.tensor([img.size[::-1]], device=device)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=confidence
        )[0]
        
        detections = []
        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            label_idx = label.item()
            class_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else 'unknown'
            detections.append({
                'label': class_name,
                'confidence': score.item(),
                'bbox': box.cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                'is_food': class_name in FOOD_CLASSES,
            })
        return detections
    
    # Visualization function with LABELS
    def visualize(img: Image.Image, predictions: List[dict], ground_truth: List[dict], save_path: str):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
        
        # Draw ground truth boxes (BLUE, dashed effect)
        for gt in ground_truth:
            x, y, bw, bh = gt['bbox']
            x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue
            
            # Label for GT
            label = f"GT: {gt['label']}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_cv, (x1, y2), (x1 + tw + 4, y2 + th + 6), (255, 0, 0), -1)
            cv2.putText(img_cv, label, (x1 + 2, y2 + th + 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw predictions (GREEN for food, ORANGE for other)
        for pred in predictions:
            x1, y1, x2, y2 = [int(c) for c in pred['bbox']]
            color = (0, 255, 0) if pred['is_food'] else (0, 165, 255)  # Green / Orange
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
            
            # Label for prediction
            label = f"{pred['label']}: {pred['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_cv, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img_cv, label, (x1 + 2, y1 - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Legend
        cv2.putText(img_cv, f"Pred: {len(predictions)} | GT: {len(ground_truth)}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img_cv, "GREEN=Pred | BLUE=GT", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        cv2.imwrite(save_path, img_cv)
    
    # Process images
    os.makedirs(output_dir, exist_ok=True)
    
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = sorted([f for f in Path(image_dir).iterdir() if f.suffix.lower() in extensions])
    
    if max_samples:
        image_files = image_files[:max_samples]
    
    print(f"Processing {len(image_files)} images from {image_dir}\n")
    
    # Results storage
    all_results = []
    pred_class_counts = defaultdict(int)
    gt_class_counts = defaultdict(int)
    
    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        img = Image.open(img_path).convert('RGB')
        
        # Get predictions
        predictions = predict(img)
        
        # Get ground truth
        ground_truth = gt_by_image.get(img_path.name, [])
        
        # Count classes
        for p in predictions:
            pred_class_counts[p['label']] += 1
        for g in ground_truth:
            gt_class_counts[g['label']] += 1
        
        # Store result
        result = {
            'image': img_path.name,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'num_predictions': len(predictions),
            'num_ground_truth': len(ground_truth),
        }
        all_results.append(result)
        
        print(f"  Predictions: {len(predictions)} | Ground Truth: {len(ground_truth)}")
        for p in predictions:
            print(f"    PRED: {p['label']} ({p['confidence']:.2f})")
        for g in ground_truth:
            print(f"    GT:   {g['label']}")
        
        # Save visualization
        save_path = os.path.join(output_dir, img_path.stem + '_comparison.jpg')
        visualize(img, predictions, ground_truth, save_path)
    
    # Generate summary report
    report = {
        'total_images': len(image_files),
        'total_predictions': sum(r['num_predictions'] for r in all_results),
        'total_ground_truth': sum(r['num_ground_truth'] for r in all_results),
        'prediction_classes': dict(pred_class_counts),
        'ground_truth_classes': dict(gt_class_counts),
        'per_image_results': all_results,
    }
    
    # Save detailed JSON report
    report_path = os.path.join(output_dir, 'results_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Images processed: {report['total_images']}")
    print(f"Total predictions: {report['total_predictions']}")
    print(f"Total ground truth: {report['total_ground_truth']}")
    
    print("\nPredicted Class Distribution:")
    for cls, count in sorted(pred_class_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cls}: {count}")
    
    print("\nGround Truth Class Distribution:")
    for cls, count in sorted(gt_class_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cls}: {count}")
    
    # Note about COCO vs custom classes
    print("\n" + "="*60)
    print("NOTE: Zero-shot uses COCO classes (banana, apple, orange, etc.)")
    print("Your dataset may have different class names (Apples, Banana, etc.)")
    print("Class name mismatch is expected - this shows baseline detection ability.")
    print("="*60)
    
    checkpoints_volume.commit()
    
    return {
        "total_images": report['total_images'],
        "total_predictions": report['total_predictions'],
        "total_ground_truth": report['total_ground_truth'],
        "output_dir": output_dir,
        "report_file": report_path,
    }


@app.local_entrypoint()
def main(
    image_dir: str = None,
    annotations_file: str = None,
    output_dir: str = f"{CHECKPOINTS_PATH}/zero_shot",
    confidence: float = 0.5,
    model: str = "facebook/detr-resnet-50",
    max_samples: int = None,
):
    """
    Run zero-shot inference with ground truth comparison.
    
    Example:
        modal run zero_shot.py
        modal run zero_shot.py --max-samples 10 --confidence 0.3
    """
    # Default to validation set (has annotations)
    if image_dir is None:
        image_dir = f"{DATA_PATH}/SmartFridgeV2_Final.v8i.coco/valid"
    if annotations_file is None:
        annotations_file = f"{DATA_PATH}/SmartFridgeV2_Final.v8i.coco/valid/_annotations.coco.json"
    
    print("Starting Zero-Shot Inference with GT Comparison...")
    print(f"  Image dir: {image_dir}")
    print(f"  Annotations: {annotations_file}")
    print(f"  Max samples: {max_samples or 'all'}")
    print(f"  GPU: {GPU_INFERENCE}")
    
    result = run_zero_shot_inference.remote(
        image_dir=image_dir,
        annotations_file=annotations_file,
        output_dir=output_dir,
        confidence=confidence,
        model_name=model,
        max_samples=max_samples,
    )
    
    print(f"\nCompleted!")
    print(f"  Images: {result['total_images']}")
    print(f"  Predictions: {result['total_predictions']}")
    print(f"  Ground Truth: {result['total_ground_truth']}")
    print(f"  Report: {result['report_file']}")
    print(f"\nDownload results: modal volume get grocery-checkpoints /zero_shot ./local_results/")
    
    return result
