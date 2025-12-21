"""
Data Transforms for Smart Grocery Tracker
==========================================

Data augmentation pipeline for DETR training on grocery images.
These transforms are designed for object detection and handle both
images and their corresponding bounding box annotations.

Key considerations for refrigerator/grocery images:
- Preserve aspect ratio (items can be at any angle)
- Handle reflective surfaces (brightness/contrast augmentation)
- Maintain small object visibility (careful with aggressive crops)
"""

import random
from typing import Dict, List, Optional, Tuple, Any

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image


class Compose:
    """Compose multiple transforms that work on both image and target."""
    
    def __init__(self, transforms: List[Any]):
        self.transforms = transforms
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert PIL Image to tensor."""
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        return F.to_tensor(image), target


class Normalize:
    """Normalize image with ImageNet stats."""
    
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std
    
    def __call__(self, image: torch.Tensor, target: Dict) -> Tuple[torch.Tensor, Dict]:
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomHorizontalFlip:
    """Random horizontal flip with probability p."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[Image.Image, Dict]:
        if random.random() < self.p:
            image = F.hflip(image)
            
            # Flip bounding boxes
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes'].clone()
                # boxes are in [cx, cy, w, h] format, normalized
                # Flip cx: new_cx = 1 - cx
                boxes[:, 0] = 1 - boxes[:, 0]
                target['boxes'] = boxes
        
        return image, target


class RandomResize:
    """Resize image to a random size from the given list."""
    
    def __init__(self, sizes: List[int], max_size: int = 1333):
        """
        Args:
            sizes: List of possible shorter edge sizes
            max_size: Maximum size for the longer edge
        """
        self.sizes = sizes
        self.max_size = max_size
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[Image.Image, Dict]:
        size = random.choice(self.sizes)
        return resize(image, target, size, self.max_size)


class Resize:
    """Resize image to a fixed size."""
    
    def __init__(self, size: int, max_size: int = 1333):
        self.size = size
        self.max_size = max_size
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[Image.Image, Dict]:
        return resize(image, target, self.size, self.max_size)


class ColorJitter:
    """Apply color jittering for data augmentation."""
    
    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
    ):
        self.transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[Image.Image, Dict]:
        return self.transform(image), target


class RandomSelect:
    """Randomly select between two transforms."""
    
    def __init__(self, transforms1: Any, transforms2: Any, p: float = 0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[Any, Dict]:
        if random.random() < self.p:
            return self.transforms1(image, target)
        return self.transforms2(image, target)


class RandomCrop:
    """Random crop with constraints to keep at least one object visible."""
    
    def __init__(self, size: Tuple[int, int]):
        self.size = size
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[Image.Image, Dict]:
        # Get image dimensions
        w, h = image.size
        th, tw = self.size
        
        if h < th or w < tw:
            # Image is smaller than crop size, return as is
            return image, target
        
        # Random crop position
        top = random.randint(0, h - th)
        left = random.randint(0, w - tw)
        
        # Crop image
        image = F.crop(image, top, left, th, tw)
        
        # Adjust bounding boxes
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            
            # Denormalize boxes to absolute coordinates
            boxes[:, 0] *= w  # cx
            boxes[:, 1] *= h  # cy
            boxes[:, 2] *= w  # w
            boxes[:, 3] *= h  # h
            
            # Convert to xyxy format for easier clipping
            x1 = boxes[:, 0] - boxes[:, 2] / 2
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            x2 = boxes[:, 0] + boxes[:, 2] / 2
            y2 = boxes[:, 1] + boxes[:, 3] / 2
            
            # Adjust for crop
            x1 = x1 - left
            y1 = y1 - top
            x2 = x2 - left
            y2 = y2 - top
            
            # Clip to crop boundaries
            x1 = x1.clamp(min=0, max=tw)
            y1 = y1.clamp(min=0, max=th)
            x2 = x2.clamp(min=0, max=tw)
            y2 = y2.clamp(min=0, max=th)
            
            # Filter out boxes that are too small or completely outside
            valid = (x2 > x1 + 1) & (y2 > y1 + 1)
            
            # Convert back to cxcywh and normalize
            new_boxes = torch.zeros_like(boxes[valid])
            new_boxes[:, 0] = (x1[valid] + x2[valid]) / 2 / tw
            new_boxes[:, 1] = (y1[valid] + y2[valid]) / 2 / th
            new_boxes[:, 2] = (x2[valid] - x1[valid]) / tw
            new_boxes[:, 3] = (y2[valid] - y1[valid]) / th
            
            target['boxes'] = new_boxes
            target['labels'] = target['labels'][valid]
            target['area'] = target['area'][valid] if 'area' in target else None
            target['iscrowd'] = target['iscrowd'][valid] if 'iscrowd' in target else None
        
        # Update size
        target['size'] = torch.tensor([th, tw])
        
        return image, target


def resize(
    image: Image.Image,
    target: Dict,
    size: int,
    max_size: int = 1333,
) -> Tuple[Image.Image, Dict]:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: PIL Image
        target: Target dictionary
        size: Target size for shorter edge
        max_size: Maximum size for longer edge
    
    Returns:
        Resized image and updated target
    """
    w, h = image.size
    
    # Calculate new size
    if max(h, w) * size / min(h, w) > max_size:
        # Longer edge would exceed max_size
        if h > w:
            new_h = max_size
            new_w = int(w * max_size / h)
        else:
            new_w = max_size
            new_h = int(h * max_size / w)
    else:
        # Scale by shorter edge
        if h < w:
            new_h = size
            new_w = int(w * size / h)
        else:
            new_w = size
            new_h = int(h * size / w)
    
    # Resize image
    image = F.resize(image, (new_h, new_w))
    
    # Update target size (boxes are already normalized, so no adjustment needed)
    target['size'] = torch.tensor([new_h, new_w])
    
    return image, target


def get_train_transforms(image_size: int = 800) -> Compose:
    """
    Get training transforms with data augmentation.
    
    Training augmentations for grocery detection:
    1. Random horizontal flip (fridges can be mirrored)
    2. Color jitter (handle varying lighting conditions)
    3. Random resize (scale invariance)
    
    Args:
        image_size: Target image size
    
    Returns:
        Composed transforms
    """
    # Multiple scales for multi-scale training
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    
    return Compose([
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        RandomResize(scales, max_size=1333),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],    # ImageNet std
        ),
    ])


def get_val_transforms(image_size: int = 800) -> Compose:
    """
    Get validation transforms (no augmentation).
    
    Args:
        image_size: Target image size
    
    Returns:
        Composed transforms
    """
    return Compose([
        Resize(image_size, max_size=1333),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],    # ImageNet std
        ),
    ])
