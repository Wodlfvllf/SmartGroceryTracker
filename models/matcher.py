"""
Hungarian Matcher for DETR
==========================

The Hungarian Matcher implements bipartite matching between predictions and
ground truth objects. This is a KEY INNOVATION of DETR that replaces the
complex anchor-based matching in traditional detectors.

How Bipartite Matching Works:
============================

Traditional Detectors (YOLO, Faster R-CNN):
    - Use anchors at multiple scales/positions
    - Match GT to nearby anchors based on IoU
    - Multiple anchors can match same GT → need NMS

DETR's Bipartite Matching:
    - No anchors, just N object queries
    - Find optimal 1-to-1 matching using Hungarian algorithm
    - Each GT matched to exactly one prediction
    - No NMS needed!

The Matching Process:
====================

1. DETR outputs N predictions (N=100 typically)
2. Image has M ground truth objects (M << N usually)
3. We need to find optimal assignment of M GTs to M predictions

Cost Matrix [N × M]:
    For each (prediction_i, gt_j) pair:
        cost = λ_cls * class_cost + λ_L1 * L1_bbox_cost + λ_giou * giou_cost

Hungarian Algorithm:
    - Finds minimum cost bipartite matching
    - O(N³) complexity, implemented in scipy
    - Returns indices: which prediction matches which GT

After Matching:
    - Matched predictions: compute loss against their GT
    - Unmatched predictions: predict "no object" class
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for optimal bipartite matching.
    
    Computes an assignment between predictions and ground truth objects
    that minimizes a cost function combining classification and localization.
    
    This module doesn't require gradients - it's just for computing optimal
    assignment. The actual loss computation happens in the criterion.
    """
    
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        """
        Args:
            cost_class: Weight for classification cost
            cost_bbox: Weight for L1 bounding box cost
            cost_giou: Weight for GIoU cost
        
        These weights are different from loss weights!
        Matching weights determine which prediction-GT pairs are matched.
        Loss weights determine how much each matched pair contributes to loss.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        
        assert cost_class > 0 or cost_bbox > 0 or cost_giou > 0, \
            "At least one cost weight must be positive"
    
    @torch.no_grad()
    def forward(
        self,
        outputs: dict,
        targets: List[dict],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute optimal matching between predictions and ground truth.
        
        Args:
            outputs: Dictionary containing:
                - "pred_logits": [B, N, num_classes] classification logits
                - "pred_boxes": [B, N, 4] predicted boxes (cxcywh, normalized)
            targets: List of B dictionaries, each containing:
                - "labels": [M_i] ground truth class labels
                - "boxes": [M_i, 4] ground truth boxes (cxcywh, normalized)
        
        Returns:
            List of B tuples (pred_indices, gt_indices):
                - pred_indices: [M_i] indices of matched predictions
                - gt_indices: [M_i] indices of matched ground truths
        
        Example:
            If returns [(tensor([3, 7]), tensor([0, 1]))], it means:
            - prediction[3] matches ground_truth[0]
            - prediction[7] matches ground_truth[1]
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Flatten batch dimension for efficient computation
        # [B, N, C] -> [B*N, C]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*N, 4]
        
        # Concatenate all targets
        tgt_ids = torch.cat([t["labels"] for t in targets])  # [sum(M_i)]
        tgt_bbox = torch.cat([t["boxes"] for t in targets])  # [sum(M_i), 4]
        
        # Handle case with no targets
        if len(tgt_ids) == 0:
            return [(
                torch.tensor([], dtype=torch.int64, device=outputs["pred_logits"].device),
                torch.tensor([], dtype=torch.int64, device=outputs["pred_logits"].device)
            ) for _ in range(bs)]
        
        # =================================================================
        # STEP 1: Compute Classification Cost
        # =================================================================
        # Higher probability for correct class = lower cost
        # We use negative probability as cost
        cost_class = -out_prob[:, tgt_ids]  # [B*N, sum(M_i)]
        
        # =================================================================
        # STEP 2: Compute L1 Bounding Box Cost
        # =================================================================
        # L1 distance between predicted and target boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # [B*N, sum(M_i)]
        
        # =================================================================
        # STEP 3: Compute GIoU Cost
        # =================================================================
        # GIoU provides better gradients than IoU
        # Convert to xyxy format for GIoU computation
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )  # [B*N, sum(M_i)]
        
        # =================================================================
        # STEP 4: Compute Total Cost Matrix
        # =================================================================
        C = (
            self.cost_class * cost_class +
            self.cost_bbox * cost_bbox +
            self.cost_giou * cost_giou
        )
        
        # Reshape back to [B, N, sum(M_i)]
        C = C.view(bs, num_queries, -1).cpu()
        
        # =================================================================
        # STEP 5: Run Hungarian Algorithm for Each Image
        # =================================================================
        # Split by the number of targets per image
        sizes = [len(t["labels"]) for t in targets]
        indices = []
        
        offset = 0
        for i, size in enumerate(sizes):
            if size == 0:
                # No targets in this image
                indices.append((
                    torch.tensor([], dtype=torch.int64),
                    torch.tensor([], dtype=torch.int64)
                ))
            else:
                # Extract cost matrix for this image
                cost_i = C[i, :, offset:offset + size]
                
                # Run Hungarian algorithm
                # Returns (row_indices, col_indices) for optimal matching
                pred_idx, gt_idx = linear_sum_assignment(cost_i.numpy())
                
                indices.append((
                    torch.tensor(pred_idx, dtype=torch.int64),
                    torch.tensor(gt_idx, dtype=torch.int64)
                ))
            
            offset += size
        
        return indices


def build_matcher(
    cost_class: float = 1.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0,
) -> HungarianMatcher:
    """
    Build the Hungarian Matcher.
    
    Args:
        cost_class: Classification cost weight
        cost_bbox: L1 bbox cost weight
        cost_giou: GIoU cost weight
    
    Returns:
        HungarianMatcher instance
    """
    return HungarianMatcher(
        cost_class=cost_class,
        cost_bbox=cost_bbox,
        cost_giou=cost_giou,
    )
