"""
COCO Format Dataset Loader for Smart Grocery Tracker
=====================================================

This module provides a dataset class for loading grocery images and annotations
in COCO format. DETR expects this format because:
1. It uses normalized bounding boxes (center_x, center_y, width, height)
2. Each image can have variable number of objects
3. Class IDs are mapped to custom grocery categories

The dataset handles padding for batching variable-length annotations.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from .transforms import get_train_transforms, get_val_transforms


class GroceryCocoDataset(Dataset):
    """
    Dataset class for loading grocery images in COCO format.
    
    COCO format structure:
    {
        "images": [{"id": 1, "file_name": "img.jpg", "width": 640, "height": 480}, ...],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h]}, ...],
        "categories": [{"id": 1, "name": "banana"}, ...]
    }
    
    Note: COCO bbox format is [x_min, y_min, width, height] (top-left corner)
    DETR expects [center_x, center_y, width, height] normalized by image size
    """
    
    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        transforms: Optional[Any] = None,
        return_masks: bool = False,
    ):
        """
        Initialize the dataset.
        
        Args:
            img_folder: Path to the folder containing images
            ann_file: Path to the COCO format annotations JSON file
            transforms: Optional transforms to apply to images and targets
            return_masks: Whether to return segmentation masks (not used for bbox detection)
        """
        self.img_folder = Path(img_folder)
        self.transforms = transforms
        self.return_masks = return_masks
        
        # Load COCO annotations
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
        
        # Build image id to annotations mapping
        self.img_id_to_anns: Dict[int, List[Dict]] = {}
        for ann in self.coco.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)
        
        # Build image id to image info mapping
        self.img_id_to_info: Dict[int, Dict] = {
            img['id']: img for img in self.coco.get('images', [])
        }
        
        # List of image ids (valid images that exist)
        self.img_ids = [
            img_id for img_id in self.img_id_to_info.keys()
            if (self.img_folder / self.img_id_to_info[img_id]['file_name']).exists()
        ]
        
        # Build category mapping
        self.cat_id_to_name: Dict[int, str] = {
            cat['id']: cat['name'] for cat in self.coco.get('categories', [])
        }
        
        # Create contiguous class mapping (DETR needs 0-indexed classes)
        self.cat_id_to_class: Dict[int, int] = {
            cat_id: idx for idx, cat_id in enumerate(sorted(self.cat_id_to_name.keys()))
        }
        
        self.num_classes = len(self.cat_id_to_name)
        
        print(f"Loaded {len(self.img_ids)} images with {self.num_classes} classes")
    
    def __len__(self) -> int:
        return len(self.img_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a single sample.
        
        Returns:
            image: Tensor of shape [C, H, W] normalized
            target: Dictionary containing:
                - boxes: Tensor of shape [N, 4] with normalized [cx, cy, w, h]
                - labels: Tensor of shape [N] with class indices
                - image_id: Tensor with image ID
                - area: Tensor of shape [N] with box areas
                - iscrowd: Tensor of shape [N] with crowd flags
                - orig_size: Tensor with original [H, W]
                - size: Tensor with current [H, W]
        """
        img_id = self.img_ids[idx]
        img_info = self.img_id_to_info[img_id]
        
        # Load image
        img_path = self.img_folder / img_info['file_name']
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        
        # Get annotations for this image
        anns = self.img_id_to_anns.get(img_id, [])
        
        # Convert COCO format [x, y, w, h] to normalized [cx, cy, w, h]
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
            
            # Convert to center format and normalize
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            
            boxes.append([cx, cy, nw, nh])
            labels.append(self.cat_id_to_class[ann['category_id']])
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            # Handle images with no annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd,
            'orig_size': torch.tensor([orig_h, orig_w]),
            'size': torch.tensor([orig_h, orig_w]),
        }
        
        # Apply transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def get_class_names(self) -> List[str]:
        """Get list of class names in order."""
        return [
            self.cat_id_to_name[cat_id] 
            for cat_id in sorted(self.cat_id_to_name.keys())
        ]


def build_dataloader(
    img_folder: str,
    ann_file: str,
    batch_size: int = 2,
    num_workers: int = 4,
    is_train: bool = True,
    image_size: int = 800,
) -> Tuple[DataLoader, GroceryCocoDataset]:
    """
    Build a DataLoader for the grocery dataset.
    
    Args:
        img_folder: Path to image folder
        ann_file: Path to annotations file
        batch_size: Batch size
        num_workers: Number of data loading workers
        is_train: Whether this is for training (affects augmentations)
        image_size: Target image size
    
    Returns:
        DataLoader and dataset objects
    """
    from utils.misc import collate_fn
    
    # Get appropriate transforms
    if is_train:
        transforms = get_train_transforms(image_size)
    else:
        transforms = get_val_transforms(image_size)
    
    # Create dataset
    dataset = GroceryCocoDataset(
        img_folder=img_folder,
        ann_file=ann_file,
        transforms=transforms,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=is_train,
        pin_memory=True,
    )
    
    return dataloader, dataset
