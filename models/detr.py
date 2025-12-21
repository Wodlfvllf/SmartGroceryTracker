"""
DETR Model - Detection Transformer
===================================

Main DETR model implementation with TWO options:

Option A: Pretrained/Fine-tuning (use_pretrained=True)
    - Loads DETR from Hugging Face, pretrained on COCO
    - Replaces classification head for custom grocery classes
    - Best for: Quick fine-tuning, production use

Option B: From Scratch (use_pretrained=False)
    - Full manual implementation of DETR architecture
    - Backbone → Position Encoding → Transformer → FFN Heads
    - Best for: Educational purposes, full control, research

DETR Architecture Overview:
===========================

    Input Image [B, 3, H, W]
            │
            ▼
    ┌───────────────────┐
    │  ResNet Backbone   │  →  Feature Map [B, 2048, H/32, W/32]
    └───────────────────┘
            │
            ▼
    ┌───────────────────┐
    │  1×1 Convolution   │  →  Reduce to hidden_dim [B, 256, H/32, W/32]
    └───────────────────┘
            │
            ▼
    ┌───────────────────┐
    │  + Position Enc.   │  →  Add 2D sinusoidal position info
    └───────────────────┘
            │
            ▼
    ┌───────────────────┐
    │   Transformer      │
    │   Encoder         │  →  Global context [B, H*W/1024, 256]
    │   (6 layers)      │
    └───────────────────┘
            │
            ▼
    ┌───────────────────┐
    │   Transformer      │
    │   Decoder         │  →  Object features [B, N, 256]
    │   (6 layers)      │     N = num_queries (100)
    │   + Object Queries│
    └───────────────────┘
            │
            ▼
    ┌───────────────────┐
    │  FFN Heads        │
    │  - Class head     │  →  Class logits [B, N, num_classes + 1]
    │  - BBox head      │  →  Box coords [B, N, 4]
    └───────────────────┘
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone, Backbone, Joiner
from .position_encoding import build_position_encoding
from .transformer import build_transformer, Transformer
from utils.misc import NestedTensor, nested_tensor_from_tensor_list


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for the prediction heads.
    
    Used for both classification and bounding box regression.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    """
    DETR (DEtection TRansformer) - From Scratch Implementation
    
    This is Option B: full manual implementation for educational purposes
    and complete control over the architecture.
    
    Key Components:
        1. Backbone: ResNet-50 for feature extraction
        2. Position Encoding: Sinusoidal 2D encodings
        3. Transformer: Encoder + Decoder
        4. Prediction Heads: Class + BBox FFN heads
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        transformer: Transformer,
        num_classes: int,
        num_queries: int = 100,
        hidden_dim: int = 256,
        aux_loss: bool = True,
    ):
        """
        Args:
            backbone: Backbone network with position encoding (Joiner)
            transformer: Transformer encoder-decoder
            num_classes: Number of object classes (excluding "no object")
            num_queries: Number of object queries (max objects per image)
            hidden_dim: Hidden dimension of the transformer
            aux_loss: Return intermediate decoder outputs for auxiliary losses
        """
        super().__init__()
        
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.aux_loss = aux_loss
        
        self.transformer = transformer
        
        # Backbone with position encoding
        self.backbone = backbone
        
        # Project backbone features to hidden_dim
        self.input_proj = nn.Conv2d(
            backbone.num_channels,
            hidden_dim,
            kernel_size=1,
        )
        
        # Learnable object queries
        # These are the "slots" that will be filled with detected objects
        # Each query learns to specialize for objects at certain positions/scales
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Prediction heads
        # Class head: predicts class logits for each query
        # Note: +1 for "no object" class
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        
        # BBox head: predicts normalized [cx, cy, w, h] for each query
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
    
    def forward(
        self,
        samples: NestedTensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DETR.
        
        Args:
            samples: NestedTensor with:
                - tensors: Batched images [B, 3, H, W]
                - mask: Padding mask [B, H, W]
        
        Returns:
            Dictionary containing:
                - pred_logits: Class logits [B, N, num_classes + 1]
                - pred_boxes: Box coordinates [B, N, 4] (cxcywh, normalized)
                - aux_outputs: (optional) List of intermediate predictions
        """
        # Handle list input
        if isinstance(samples, (list, tuple)):
            samples = nested_tensor_from_tensor_list(samples)
        
        # Extract features and position encodings from backbone
        features, pos = self.backbone(samples)
        
        # Take last feature level
        src, mask = features[-1].decompose()
        pos_embed = pos[-1]
        
        # Project to hidden dimension
        src = self.input_proj(src)
        
        # Pass through transformer
        hs, memory = self.transformer(
            src,
            mask,
            self.query_embed.weight,
            pos_embed,
        )
        
        # Prediction heads
        # hs shape: [num_decoder_layers, B, N, hidden_dim]
        outputs_class = self.class_embed(hs)  # [num_layers, B, N, num_classes + 1]
        outputs_coord = self.bbox_embed(hs).sigmoid()  # [num_layers, B, N, 4]
        
        # Final output (from last decoder layer)
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
        }
        
        # Auxiliary outputs for intermediate supervision
        if self.aux_loss:
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        
        return out


class DETRPretrained(nn.Module):
    """
    DETR - Pretrained/Fine-tuning Implementation
    
    This is Option A: load pretrained DETR from Hugging Face and
    replace the classification head for custom grocery classes.
    
    Advantages:
        - Faster training (pretrained on COCO)
        - Better performance with limited data
        - Production-ready
    """
    
    def __init__(
        self,
        num_classes: int,
        pretrained_model: str = "facebook/detr-resnet-50",
        freeze_backbone: bool = False,
        num_queries: int = 100,
    ):
        """
        Args:
            num_classes: Number of grocery classes (excluding "no object")
            pretrained_model: Hugging Face model name
            freeze_backbone: Whether to freeze backbone weights
            num_queries: Number of object queries (must match pretrained)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # Load pretrained DETR from Hugging Face
        try:
            from transformers import DetrForObjectDetection, DetrConfig
            
            # Load pretrained model
            self.model = DetrForObjectDetection.from_pretrained(pretrained_model)
            
            # Replace classification head for custom classes
            # Original: 91 COCO classes + 1 "no object"
            # New: num_classes + 1 "no object"
            hidden_dim = self.model.config.d_model
            self.model.class_labels_classifier = nn.Linear(
                hidden_dim, num_classes + 1
            )
            
            # Reinitialize the new classification head
            nn.init.xavier_uniform_(self.model.class_labels_classifier.weight)
            nn.init.constant_(self.model.class_labels_classifier.bias, 0)
            
            # Optionally freeze backbone
            if freeze_backbone:
                for param in self.model.model.backbone.parameters():
                    param.requires_grad = False
            
            self.hidden_dim = hidden_dim
            self.aux_loss = True  # HF DETR supports auxiliary loss
            
            print(f"Loaded pretrained DETR with {num_classes} classes")
            print(f"  - Backbone: {'frozen' if freeze_backbone else 'trainable'}")
            print(f"  - Object queries: {num_queries}")
            
        except ImportError:
            raise ImportError(
                "transformers library required for pretrained DETR. "
                "Install with: pip install transformers"
            )
    
    def forward(
        self,
        samples: NestedTensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through pretrained DETR.
        
        Args:
            samples: NestedTensor or tensor with images
        
        Returns:
            Dictionary with pred_logits and pred_boxes
        """
        # Handle NestedTensor input
        if isinstance(samples, NestedTensor):
            pixel_values = samples.tensors
            pixel_mask = ~samples.mask  # HF uses inverted mask (True = valid)
        elif isinstance(samples, (list, tuple)):
            samples = nested_tensor_from_tensor_list(samples)
            pixel_values = samples.tensors
            pixel_mask = ~samples.mask
        else:
            pixel_values = samples
            pixel_mask = None
        
        # Forward through HF model
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_hidden_states=False,
            output_attentions=False,
        )
        
        # Convert to our format
        out = {
            'pred_logits': outputs.logits,
            'pred_boxes': outputs.pred_boxes,
        }
        
        # Add auxiliary outputs if available
        if hasattr(outputs, 'auxiliary_outputs') and outputs.auxiliary_outputs is not None:
            out['aux_outputs'] = outputs.auxiliary_outputs
        
        return out


def build_detr(
    num_classes: int,
    use_pretrained: bool = True,
    hidden_dim: int = 256,
    num_heads: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    num_queries: int = 100,
    backbone_name: str = 'resnet50',
    pretrained_backbone: bool = True,
    freeze_backbone: bool = False,
    aux_loss: bool = True,
    pretrained_model: str = "facebook/detr-resnet-50",
) -> nn.Module:
    """
    Build DETR model.
    
    Args:
        num_classes: Number of grocery classes (excluding "no object")
        use_pretrained: If True, load pretrained DETR (Option A)
                       If False, build from scratch (Option B)
        hidden_dim: Hidden dimension for transformer
        num_heads: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        dim_feedforward: FFN intermediate dimension
        dropout: Dropout rate
        num_queries: Number of object queries
        backbone_name: Backbone network name
        pretrained_backbone: Load pretrained backbone weights
        freeze_backbone: Freeze backbone during training
        aux_loss: Use auxiliary losses from intermediate decoder layers
        pretrained_model: Hugging Face model name for pretrained DETR
    
    Returns:
        DETR model (either pretrained or from scratch)
    """
    if use_pretrained:
        # Option A: Pretrained/Fine-tuning
        return DETRPretrained(
            num_classes=num_classes,
            pretrained_model=pretrained_model,
            freeze_backbone=freeze_backbone,
            num_queries=num_queries,
        )
    else:
        # Option B: From Scratch
        # Build backbone
        backbone = build_backbone(
            name=backbone_name,
            hidden_dim=hidden_dim,
            train_backbone=not freeze_backbone,
            pretrained=pretrained_backbone,
            freeze_bn=True,
        )
        
        # Build position encoding
        position_encoding = build_position_encoding(hidden_dim=hidden_dim)
        
        # Combine backbone and position encoding
        backbone_with_pos = Joiner(backbone, position_encoding)
        backbone_with_pos.num_channels = backbone.num_channels
        
        # Build transformer
        transformer = build_transformer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        
        # Build DETR
        return DETR(
            backbone=backbone_with_pos,
            transformer=transformer,
            num_classes=num_classes,
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            aux_loss=aux_loss,
        )
