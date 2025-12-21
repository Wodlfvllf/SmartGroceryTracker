#!/usr/bin/env python
"""
Zero-Shot Inference Script
==========================

Script A: Load a pre-trained DETR model (COCO weights) and run inference
on Custom Refrigerator Dataset images to establish baseline performance
without any training.

This script demonstrates:
1. Loading pretrained DETR from Hugging Face
2. Running inference on new images
3. Visualizing detections with bounding boxes

Usage:
    python zero_shot.py --image_path path/to/image.jpg
    python zero_shot.py --image_dir path/to/images/ --output_dir results/
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
import cv2
import numpy as np
from PIL import Image

# COCO class names (91 classes that pretrained DETR knows)
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

# Food-related COCO classes that might appear in refrigerator images
FOOD_CLASSES = {
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'refrigerator'
}


class ZeroShotInference:
    """
    Zero-shot inference using pretrained DETR on COCO.
    
    This class loads a pretrained DETR and runs inference on new images.
    Since COCO has some food classes, it can detect items like apple,
    banana, orange, bottle, etc. without any training.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        confidence_threshold: float = 0.7,
        device: str = None,
    ):
        """
        Args:
            model_name: Hugging Face model name
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on
        """
        self.confidence_threshold = confidence_threshold
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)
        
        print(f"Loading pretrained DETR: {model_name}")
        print(f"Device: {self.device}")
        
        # Load model and processor
        try:
            from transformers import DetrForObjectDetection, DetrImageProcessor
            
            self.processor = DetrImageProcessor.from_pretrained(model_name)
            self.model = DetrForObjectDetection.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            print("Model loaded successfully!")
            print(f"Model can detect {len(COCO_CLASSES)} COCO classes")
            print(f"Food-related classes: {len(FOOD_CLASSES)}")
            
        except ImportError:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )
    
    @torch.no_grad()
    def predict(self, image: Image.Image) -> List[dict]:
        """
        Run inference on a single image.
        
        Args:
            image: PIL Image
        
        Returns:
            List of detections, each with:
                - label: class name
                - confidence: detection confidence
                - bbox: [x_min, y_min, x_max, y_max] in pixels
        """
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.confidence_threshold,
        )[0]
        
        # Convert to list of dicts
        detections = []
        for score, label, box in zip(
            results['scores'],
            results['labels'],
            results['boxes'],
        ):
            label_idx = label.item()
            class_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else 'unknown'
            
            detections.append({
                'label': class_name,
                'confidence': score.item(),
                'bbox': box.cpu().numpy().tolist(),
                'is_food': class_name in FOOD_CLASSES,
            })
        
        return detections
    
    def visualize(
        self,
        image: Image.Image,
        detections: List[dict],
        output_path: str = None,
    ) -> np.ndarray:
        """
        Visualize detections on image.
        
        Args:
            image: Original PIL Image
            detections: List of detection dicts
            output_path: Path to save visualization
        
        Returns:
            Visualization as numpy array (BGR)
        """
        # Convert to OpenCV format
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Colors for visualization
        food_color = (0, 255, 0)     # Green for food items
        other_color = (255, 165, 0)  # Orange for other items
        
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det['bbox']]
            label = det['label']
            conf = det['confidence']
            is_food = det['is_food']
            
            # Choose color
            color = food_color if is_food else other_color
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_text = f"{label}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Background for label
            cv2.rectangle(
                img,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0] + 5, y1),
                color,
                -1,
            )
            
            # Label text
            cv2.putText(
                img,
                label_text,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        
        # Add summary text
        food_count = sum(1 for d in detections if d['is_food'])
        summary = f"Detections: {len(detections)} (Food: {food_count})"
        cv2.putText(
            img, summary, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )
        
        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            cv2.imwrite(output_path, img)
            print(f"Saved: {output_path}")
        
        return img


def process_image(
    inferencer: ZeroShotInference,
    image_path: str,
    output_path: str = None,
) -> Tuple[List[dict], np.ndarray]:
    """Process a single image."""
    print(f"\nProcessing: {image_path}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    print(f"  Image size: {image.size}")
    
    # Run inference
    detections = inferencer.predict(image)
    
    # Print results
    print(f"  Detections: {len(detections)}")
    for det in detections:
        marker = "🍎" if det['is_food'] else "📦"
        print(f"    {marker} {det['label']}: {det['confidence']:.2f}")
    
    # Visualize
    vis = inferencer.visualize(image, detections, output_path)
    
    return detections, vis


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot inference with pretrained DETR"
    )
    parser.add_argument(
        '--image_path', type=str,
        help='Path to single image'
    )
    parser.add_argument(
        '--image_dir', type=str,
        help='Path to directory of images'
    )
    parser.add_argument(
        '--output_dir', type=str, default='output/zero_shot',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--confidence', type=float, default=0.7,
        help='Confidence threshold for detections'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device (cuda, mps, cpu)'
    )
    parser.add_argument(
        '--model', type=str, default='facebook/detr-resnet-50',
        help='Hugging Face model name'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.image_path and not args.image_dir:
        parser.error("Must provide either --image_path or --image_dir")
    
    # Initialize inferencer
    inferencer = ZeroShotInference(
        model_name=args.model,
        confidence_threshold=args.confidence,
        device=args.device,
    )
    
    # Process images
    if args.image_path:
        # Single image
        output_path = os.path.join(
            args.output_dir,
            Path(args.image_path).stem + '_detection.jpg'
        )
        process_image(inferencer, args.image_path, output_path)
    
    elif args.image_dir:
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [
            f for f in Path(args.image_dir).iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        print(f"\nFound {len(image_files)} images in {args.image_dir}")
        
        all_detections = []
        for img_path in image_files:
            output_path = os.path.join(
                args.output_dir,
                img_path.stem + '_detection.jpg'
            )
            detections, _ = process_image(inferencer, str(img_path), output_path)
            all_detections.extend(detections)
        
        # Summary
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Total images processed: {len(image_files)}")
        print(f"Total detections: {len(all_detections)}")
        
        # Class distribution
        class_counts = {}
        for det in all_detections:
            cls = det['label']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        print("\nClass distribution:")
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            marker = "🍎" if cls in FOOD_CLASSES else "📦"
            print(f"  {marker} {cls}: {count}")


if __name__ == '__main__':
    main()
