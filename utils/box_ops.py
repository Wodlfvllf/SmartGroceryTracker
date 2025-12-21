"""
Bounding Box Operations for DETR
================================

Utility functions for bounding box manipulation and IoU calculation.
These are essential for:
1. Converting between box formats (COCO vs DETR)
2. Computing losses (IoU, GIoU)
3. Visualization

Box Formats:
- COCO: [x_min, y_min, width, height] (top-left corner + size)
- DETR: [center_x, center_y, width, height] (center + size), normalized to [0, 1]
- xyxy: [x_min, y_min, x_max, y_max] (corners)
"""

import torch
from torch import Tensor
from typing import Tuple


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """
    Convert boxes from center format to corner format.
    
    Args:
        boxes: Tensor of shape [..., 4] with [center_x, center_y, width, height]
    
    Returns:
        Tensor of shape [..., 4] with [x_min, y_min, x_max, y_max]
    """
    cx, cy, w, h = boxes.unbind(-1)
    x_min = cx - 0.5 * w
    y_min = cy - 0.5 * h
    x_max = cx + 0.5 * w
    y_max = cy + 0.5 * h
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """
    Convert boxes from corner format to center format.
    
    Args:
        boxes: Tensor of shape [..., 4] with [x_min, y_min, x_max, y_max]
    
    Returns:
        Tensor of shape [..., 4] with [center_x, center_y, width, height]
    """
    x_min, y_min, x_max, y_max = boxes.unbind(-1)
    w = x_max - x_min
    h = y_max - y_min
    cx = x_min + 0.5 * w
    cy = y_min + 0.5 * h
    return torch.stack([cx, cy, w, h], dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    """
    Compute area of boxes.
    
    Args:
        boxes: Tensor of shape [..., 4] with [x_min, y_min, x_max, y_max]
    
    Returns:
        Tensor of shape [...] with areas
    """
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape [N, 4] in xyxy format
        boxes2: Tensor of shape [M, 4] in xyxy format
    
    Returns:
        iou: Tensor of shape [N, M] with IoU values
        union: Tensor of shape [N, M] with union areas
    """
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]
    
    # Intersection coordinates
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    
    # Intersection area
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # Union area
    union = area1[:, None] + area2[None, :] - inter  # [N, M]
    
    # IoU
    iou = inter / (union + 1e-6)  # [N, M]
    
    return iou, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Compute Generalized IoU between two sets of boxes.
    
    GIoU = IoU - (area of enclosing box - union) / area of enclosing box
    
    GIoU is better than IoU for gradient-based optimization because:
    - IoU is 0 for non-overlapping boxes regardless of distance
    - GIoU provides gradients even for non-overlapping boxes
    - GIoU ranges from -1 to 1 (1 = perfect overlap, -1 = far apart)
    
    Args:
        boxes1: Tensor of shape [N, 4] in xyxy format
        boxes2: Tensor of shape [M, 4] in xyxy format
    
    Returns:
        Tensor of shape [N, M] with GIoU values
    
    Reference:
        "Generalized Intersection over Union: A Metric and A Loss for Bounding
        Box Regression" - https://arxiv.org/abs/1902.09630
    """
    # Ensure boxes are valid (x2 > x1, y2 > y1)
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "boxes1 invalid"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "boxes2 invalid"
    
    # Compute IoU and union
    iou, union = box_iou(boxes1, boxes2)
    
    # Compute enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    
    # Area of enclosing box
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    area_enclosing = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # GIoU
    giou = iou - (area_enclosing - union) / (area_enclosing + 1e-6)
    
    return giou


def rescale_boxes(boxes: Tensor, size: Tuple[int, int]) -> Tensor:
    """
    Rescale normalized boxes to image coordinates.
    
    Args:
        boxes: Tensor of shape [..., 4] in cxcywh format, normalized to [0, 1]
        size: Tuple of (height, width) of the image
    
    Returns:
        Tensor of shape [..., 4] in xyxy format, absolute coordinates
    """
    h, w = size
    
    # Convert to xyxy
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)
    
    # Scale to image coordinates
    scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
    boxes_scaled = boxes_xyxy * scale
    
    return boxes_scaled
