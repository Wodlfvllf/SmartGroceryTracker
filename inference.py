#!/usr/bin/env python
"""
Flexible Inference Script
=========================

Script C: Run inference using either:
- Fine-tuned weights (from train.py)
- From-scratch weights
- Pretrained COCO weights (for comparison)

This script can load any trained model and run inference on new images,
outputting visualized predictions with bounding boxes.

Usage:
    # Inference with fine-tuned model
    python inference.py --checkpoint checkpoints/best_model.pth \
                        --image_path path/to/image.jpg
    
    # Inference on directory
    python inference.py --checkpoint checkpoints/best_model.pth \
                        --image_dir path/to/images/ \
                        --output_dir results/
    
    # Using from-scratch model
    python inference.py --checkpoint checkpoints/scratch_model.pth \
                        --from_scratch \
                        --image_path path/to/image.jpg
    
    # Using different class configuration
    python inference.py --checkpoint checkpoints/best_model.pth \
                        --config configs/default.yaml \
                        --image_path path/to/image.jpg
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T

from models.detr import build_detr
from utils.misc import load_config, get_device, load_checkpoint
from utils.box_ops import box_cxcywh_to_xyxy, rescale_boxes


class GroceryInference:
    """
    Inference class for trained DETR models.
    
    Handles loading checkpoints, preprocessing images, running inference,
    and visualizing results.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = 'configs/default.yaml',
        use_pretrained: bool = True,
        confidence_threshold: float = 0.7,
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to configuration file
            use_pretrained: Whether model was pretrained (vs from scratch)
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on
            class_names: Optional list of class names
        """
        self.confidence_threshold = confidence_threshold
        
        # Set device
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)
        
        print(f"Device: {self.device}")
        
        # Load config
        config = {}
        if os.path.exists(config_path):
            config = load_config(config_path)
        
        # Get class names
        if class_names is None:
            self.class_names = config.get('classes', [
                'banana', 'apple', 'orange', 'avocado', 'lemon',
                'strawberry', 'grape', 'tomato', 'lettuce', 'carrot',
                'cucumber', 'broccoli', 'onion', 'pepper', 'milk',
                'juice', 'yogurt', 'cheese', 'butter', 'egg_carton',
                'bottle', 'can', 'container', 'jar',
            ])
        else:
            self.class_names = class_names
        
        self.num_classes = len(self.class_names)
        print(f"Classes: {self.num_classes}")
        
        # Build model
        print(f"Building model (pretrained={use_pretrained})...")
        model_config = config.get('model', {})
        
        self.model = build_detr(
            num_classes=self.num_classes,
            use_pretrained=use_pretrained,
            hidden_dim=model_config.get('hidden_dim', 256),
            num_queries=model_config.get('num_queries', 100),
        )
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
        
        # Image transforms
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @torch.no_grad()
    def predict(self, image: Image.Image) -> List[Dict]:
        """
        Run inference on a single image.
        
        Args:
            image: PIL Image
        
        Returns:
            List of detections with:
                - label: class name
                - confidence: detection confidence
                - bbox: [x_min, y_min, x_max, y_max] in pixels
        """
        # Original size for rescaling boxes
        orig_w, orig_h = image.size
        
        # Transform image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Create mask (no padding for single images)
        from utils.misc import NestedTensor
        mask = torch.zeros((1, *img_tensor.shape[-2:]), dtype=torch.bool, device=self.device)
        samples = NestedTensor(img_tensor, mask)
        
        # Run model
        outputs = self.model(samples)
        
        # Post-process outputs
        pred_logits = outputs['pred_logits'][0]  # [N, num_classes + 1]
        pred_boxes = outputs['pred_boxes'][0]    # [N, 4]
        
        # Get probabilities
        probs = pred_logits.softmax(-1)
        
        # Get scores and labels (excluding "no object" class)
        scores, labels = probs[:, :-1].max(-1)
        
        # Filter by confidence
        keep = scores > self.confidence_threshold
        
        scores = scores[keep]
        labels = labels[keep]
        boxes = pred_boxes[keep]
        
        # Rescale boxes to image coordinates
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        scale = torch.tensor([orig_w, orig_h, orig_w, orig_h], device=self.device)
        boxes_scaled = boxes_xyxy * scale
        
        # Build detection list
        detections = []
        for score, label, box in zip(scores, labels, boxes_scaled):
            label_idx = label.item()
            class_name = self.class_names[label_idx] if label_idx < len(self.class_names) else f'class_{label_idx}'
            
            detections.append({
                'label': class_name,
                'confidence': score.item(),
                'bbox': box.cpu().numpy().tolist(),
            })
        
        return detections
    
    def visualize(
        self,
        image: Image.Image,
        detections: List[Dict],
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> np.ndarray:
        """
        Visualize detections on image.
        
        Args:
            image: Original PIL Image
            detections: List of detection dicts
            output_path: Path to save visualization
            show: Whether to display the image
        
        Returns:
            Visualization as numpy array (BGR)
        """
        # Convert to OpenCV format
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Generate colors for each class
        np.random.seed(42)
        colors = {name: tuple(map(int, np.random.randint(0, 255, 3)))
                  for name in self.class_names}
        
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det['bbox']]
            label = det['label']
            conf = det['confidence']
            
            # Get color for this class
            color = colors.get(label, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label_text = f"{label}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(
                img,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0] + 10, y1),
                color,
                -1,
            )
            
            # Draw label text
            cv2.putText(
                img,
                label_text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        
        # Add detection count
        cv2.putText(
            img,
            f"Detected: {len(detections)} items",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        
        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            cv2.imwrite(output_path, img)
            print(f"Saved: {output_path}")
        
        # Show if requested
        if show:
            cv2.imshow('Detection', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return img
    
    def process_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Process a single image file.
        
        Args:
            image_path: Path to image
            output_path: Path to save visualization
            show: Whether to display result
        
        Returns:
            Tuple of (detections, visualization)
        """
        print(f"\nProcessing: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        print(f"  Image size: {image.size}")
        
        # Run inference
        import time
        start = time.time()
        detections = self.predict(image)
        inference_time = time.time() - start
        
        print(f"  Inference time: {inference_time*1000:.1f}ms")
        print(f"  Detections: {len(detections)}")
        
        for det in detections:
            print(f"    - {det['label']}: {det['confidence']:.2f}")
        
        # Visualize
        vis = self.visualize(image, detections, output_path, show)
        
        return detections, vis


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with trained DETR model"
    )
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--from_scratch', action='store_true',
                        help='Model was trained from scratch (not pretrained)')
    
    # Input arguments
    parser.add_argument('--image_path', type=str,
                        help='Path to single image')
    parser.add_argument('--image_dir', type=str,
                        help='Path to directory of images')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output/inference',
                        help='Output directory for visualizations')
    parser.add_argument('--show', action='store_true',
                        help='Display results (press any key to continue)')
    
    # Inference arguments
    parser.add_argument('--confidence', type=float, default=0.7,
                        help='Confidence threshold')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda, mps, cpu)')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.image_path and not args.image_dir:
        parser.error("Must provide either --image_path or --image_dir")
    
    # Initialize inferencer
    inferencer = GroceryInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        use_pretrained=not args.from_scratch,
        confidence_threshold=args.confidence,
        device=args.device,
    )
    
    # Process images
    if args.image_path:
        # Single image
        output_path = os.path.join(
            args.output_dir,
            Path(args.image_path).stem + '_prediction.jpg'
        )
        inferencer.process_image(args.image_path, output_path, args.show)
    
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
                img_path.stem + '_prediction.jpg'
            )
            detections, _ = inferencer.process_image(
                str(img_path), output_path, args.show
            )
            all_detections.extend(detections)
        
        # Summary
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Total images: {len(image_files)}")
        print(f"Total detections: {len(all_detections)}")
        
        # Class distribution
        class_counts = {}
        for det in all_detections:
            cls = det['label']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        print("\nClass distribution:")
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"  {cls}: {count}")


if __name__ == '__main__':
    main()
