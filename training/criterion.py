"""
DETR Loss Criterion
===================

The DETR loss combines multiple components after Hungarian matching:

1. Classification Loss (Cross-Entropy):
   - For matched predictions: CE(predicted_class, gt_class)
   - For unmatched predictions: CE(predicted_class, "no_object")
   - "no_object" class is downweighted to handle class imbalance

2. Bounding Box L1 Loss:
   - L1 distance between predicted and ground truth boxes
   - Only for matched predictions
   - In normalized [cx, cy, w, h] format

3. GIoU Loss:
   - Generalized IoU loss for better convergence
   - Only for matched predictions
   - Provides gradients even for non-overlapping boxes

Auxiliary Losses:
   - DETR applies the same losses to intermediate decoder outputs
   - Helps with gradient flow through deep transformer
   - Weights are the same for all decoder layers

Total Loss = λ_ce * CE + λ_bbox * L1 + λ_giou * (1 - GIoU)
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.matcher import HungarianMatcher, build_matcher
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class SetCriterion(nn.Module):
    """
    DETR Loss Criterion.
    
    Performs the loss computation after Hungarian matching.
    The criterion is a "Set" criterion because DETR predicts a set of objects.
    """
    
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float = 0.1,
        losses: List[str] = ['labels', 'boxes'],
    ):
        """
        Args:
            num_classes: Number of object classes (excluding "no object")
            matcher: Hungarian matcher for bipartite matching
            weight_dict: Dict mapping loss names to their weights
            eos_coef: Weight for "no object" class in classification
            losses: List of losses to compute ['labels', 'boxes']
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        
        # Class weights: downweight "no object" to handle imbalance
        # Most predictions will be "no object" (N queries, but few GT objects)
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef  # Last class is "no object"
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_labels(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict],
        indices: List[tuple],
        num_boxes: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute classification loss.
        
        Args:
            outputs: Model predictions
            targets: Ground truth targets
            indices: Matched indices from Hungarian matching
            num_boxes: Total number of boxes for normalization
        
        Returns:
            Dict with 'loss_ce' key
        """
        assert 'pred_logits' in outputs
        
        src_logits = outputs['pred_logits']  # [B, N, num_classes + 1]
        bs, num_queries, _ = src_logits.shape
        
        # Get source and target indices
        idx = self._get_src_permutation_idx(indices)
        
        # Build target classes tensor
        # Start with all "no object" (last class)
        target_classes = torch.full(
            (bs, num_queries),
            self.num_classes,  # "no object" class
            dtype=torch.int64,
            device=src_logits.device,
        )
        
        # Fill in matched targets
        target_classes_o = torch.cat([
            t['labels'][J] for t, (_, J) in zip(targets, indices)
        ])
        target_classes[idx] = target_classes_o
        
        # Compute cross-entropy loss
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),  # [B, C, N]
            target_classes,               # [B, N]
            self.empty_weight,
        )
        
        return {'loss_ce': loss_ce}
    
    def loss_boxes(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict],
        indices: List[tuple],
        num_boxes: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute bounding box losses (L1 + GIoU).
        
        Args:
            outputs: Model predictions
            targets: Ground truth targets
            indices: Matched indices from Hungarian matching
            num_boxes: Total number of boxes for normalization
        
        Returns:
            Dict with 'loss_bbox' and 'loss_giou' keys
        """
        assert 'pred_boxes' in outputs
        
        idx = self._get_src_permutation_idx(indices)
        
        # Get matched predictions
        src_boxes = outputs['pred_boxes'][idx]  # [num_matched, 4]
        
        # Get matched targets
        target_boxes = torch.cat([
            t['boxes'][J] for t, (_, J) in zip(targets, indices)
        ], dim=0)
        
        # Handle case with no matches
        if len(src_boxes) == 0:
            return {
                'loss_bbox': torch.tensor(0.0, device=outputs['pred_boxes'].device),
                'loss_giou': torch.tensor(0.0, device=outputs['pred_boxes'].device),
            }
        
        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes
        
        # GIoU loss
        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        
        # Clamp to valid boxes (prevent negative sizes)
        src_boxes_xyxy = src_boxes_xyxy.clamp(min=0, max=1)
        target_boxes_xyxy = target_boxes_xyxy.clamp(min=0, max=1)
        
        # GIoU is computed pairwise, but we only need diagonal (matched pairs)
        giou = torch.diag(generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy))
        loss_giou = (1 - giou).sum() / num_boxes
        
        return {
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
        }
    
    def _get_src_permutation_idx(
        self,
        indices: List[tuple],
    ) -> tuple:
        """
        Get batch and source indices for permuting predictions.
        
        Converts matched indices to batch-level indices for indexing
        the flattened predictions tensor.
        """
        batch_idx = torch.cat([
            torch.full_like(src, i)
            for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def get_loss(
        self,
        loss: str,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict],
        indices: List[tuple],
        num_boxes: int,
    ) -> Dict[str, torch.Tensor]:
        """Get a specific loss by name."""
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f"Unknown loss: {loss}"
        return loss_map[loss](outputs, targets, indices, num_boxes)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            outputs: Model predictions containing:
                - pred_logits: [B, N, num_classes + 1]
                - pred_boxes: [B, N, 4]
                - aux_outputs: (optional) list of intermediate outputs
            targets: List of B target dicts
        
        Returns:
            Dict of all losses (weighted)
        """
        # Only use final layer output for matching
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != 'aux_outputs'
        }
        
        # Hungarian matching
        indices = self.matcher(outputs_without_aux, targets)
        
        # Count total boxes for normalization
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = max(num_boxes, 1)  # Avoid division by zero
        
        # Compute losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs_without_aux, targets, indices, num_boxes
            ))
        
        # Auxiliary losses (from intermediate decoder layers)
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices_aux = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_aux, num_boxes
                    )
                    # Rename with layer index
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        return losses


def build_criterion(
    num_classes: int,
    cost_class: float = 1.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0,
    loss_ce: float = 1.0,
    loss_bbox: float = 5.0,
    loss_giou: float = 2.0,
    eos_coef: float = 0.1,
) -> SetCriterion:
    """
    Build the DETR criterion.
    
    Args:
        num_classes: Number of object classes
        cost_class: Matching cost weight for classification
        cost_bbox: Matching cost weight for L1 bbox
        cost_giou: Matching cost weight for GIoU
        loss_ce: Loss weight for classification
        loss_bbox: Loss weight for L1 bbox
        loss_giou: Loss weight for GIoU
        eos_coef: Weight for "no object" class
    
    Returns:
        SetCriterion instance
    """
    # Build matcher
    matcher = build_matcher(
        cost_class=cost_class,
        cost_bbox=cost_bbox,
        cost_giou=cost_giou,
    )
    
    # Loss weights
    weight_dict = {
        'loss_ce': loss_ce,
        'loss_bbox': loss_bbox,
        'loss_giou': loss_giou,
    }
    
    # Add weights for auxiliary losses (same weights)
    for i in range(5):  # Assuming max 6 decoder layers - 1
        weight_dict[f'loss_ce_{i}'] = loss_ce
        weight_dict[f'loss_bbox_{i}'] = loss_bbox
        weight_dict[f'loss_giou_{i}'] = loss_giou
    
    return SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=eos_coef,
        losses=['labels', 'boxes'],
    )
