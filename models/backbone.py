"""
ResNet Backbone for DETR
========================

The backbone extracts visual features from input images. DETR uses a ResNet
backbone (typically ResNet-50) that outputs feature maps at 1/32 resolution.

Key Design Choices:
1. Remove the final classification layer (we only need features)
2. Optionally freeze BatchNorm layers for fine-tuning stability
3. Output feature maps + position encodings for the transformer

Architecture:
    Input Image [3, H, W]
        ↓
    ResNet (conv layers)
        ↓
    Feature Map [C, H/32, W/32] where C=2048 for ResNet-50
        ↓
    1x1 Conv (reduce to hidden_dim)
        ↓
    Output [hidden_dim, H/32, W/32]
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from utils.misc import NestedTensor


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d with frozen running stats and affine parameters.
    
    Used when fine-tuning to prevent the backbone's BatchNorm statistics
    from changing due to the small batch sizes typically used with DETR.
    
    This is important because:
    - DETR training uses small batches (2-4 images) due to memory constraints
    - Small batches lead to noisy BatchNorm statistics
    - Freezing prevents this from destabilizing the pretrained backbone
    """
    
    def __init__(self, n: int):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape for broadcasting
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        
        return x * scale + bias


class BackboneBase(nn.Module):
    """
    Base class for backbone networks.
    
    Wraps a torchvision backbone to:
    1. Extract intermediate features
    2. Track output channels
    3. Handle NestedTensor input
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_layers: Dict[str, str],
    ):
        super().__init__()
        
        # Only train backbone if specified
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        
        # Get intermediate layer outputs
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels
    
    def forward(self, tensor_list: NestedTensor) -> Dict[str, NestedTensor]:
        """
        Forward pass through the backbone.
        
        Args:
            tensor_list: NestedTensor with images and mask
        
        Returns:
            Dict mapping layer names to NestedTensors of features
        """
        xs = self.body(tensor_list.tensors)
        out = {}
        
        for name, x in xs.items():
            # Downsample the mask to match feature map size
            mask = tensor_list.mask
            assert mask is not None
            
            # Use interpolate to resize mask
            mask = F.interpolate(
                mask[None].float(),
                size=x.shape[-2:],
                mode='nearest'
            )[0].to(torch.bool)
            
            out[name] = NestedTensor(x, mask)
        
        return out


class Backbone(BackboneBase):
    """
    ResNet backbone for DETR.
    
    Uses torchvision's ResNet with optional:
    - Pretrained ImageNet weights
    - Frozen BatchNorm layers
    - Training only later layers (fine-tuning)
    """
    
    def __init__(
        self,
        name: str = 'resnet50',
        train_backbone: bool = True,
        dilation: bool = False,
        pretrained: bool = True,
        freeze_bn: bool = True,
    ):
        """
        Args:
            name: Backbone name ('resnet50' or 'resnet101')
            train_backbone: Whether to train the backbone
            dilation: Use dilation instead of stride in last block (increases resolution)
            pretrained: Load ImageNet pretrained weights
            freeze_bn: Freeze BatchNorm layers
        """
        # Select norm layer
        norm_layer = FrozenBatchNorm2d if freeze_bn else nn.BatchNorm2d
        
        # Load backbone
        if name == 'resnet50':
            backbone = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
                replace_stride_with_dilation=[False, False, dilation],
                norm_layer=norm_layer,
            )
            num_channels = 2048
        elif name == 'resnet101':
            backbone = torchvision.models.resnet101(
                weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None,
                replace_stride_with_dilation=[False, False, dilation],
                norm_layer=norm_layer,
            )
            num_channels = 2048
        else:
            raise ValueError(f"Unknown backbone: {name}")
        
        # Only return layer4 output (last conv layer before pooling)
        return_layers = {'layer4': '0'}
        
        super().__init__(
            backbone=backbone,
            train_backbone=train_backbone,
            num_channels=num_channels,
            return_layers=return_layers,
        )


class Joiner(nn.Module):
    """
    Combines backbone with position encoding.
    
    DETR needs both:
    1. Visual features from backbone
    2. Positional encodings for spatial information
    
    This module joins them together for the transformer.
    """
    
    def __init__(self, backbone: Backbone, position_encoding: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.position_encoding = position_encoding
        self.num_channels = backbone.num_channels
    
    def forward(self, tensor_list: NestedTensor):
        """
        Forward pass through backbone + position encoding.
        
        Args:
            tensor_list: NestedTensor with images and mask
        
        Returns:
            features: List of NestedTensors with backbone features
            pos: List of position encodings corresponding to each feature level
        """
        # Get backbone features
        xs = self.backbone(tensor_list)
        
        # Add position encodings
        out = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self.position_encoding(x).to(x.tensors.dtype))
        
        return out, pos


def build_backbone(
    name: str = 'resnet50',
    hidden_dim: int = 256,
    train_backbone: bool = True,
    pretrained: bool = True,
    freeze_bn: bool = True,
    dilation: bool = False,
) -> Backbone:
    """
    Build a backbone network.
    
    Args:
        name: Backbone name ('resnet50' or 'resnet101')
        hidden_dim: Hidden dimension (not used here, for interface consistency)
        train_backbone: Whether to train the backbone
        pretrained: Load ImageNet pretrained weights
        freeze_bn: Freeze BatchNorm layers
        dilation: Use dilation in last block
    
    Returns:
        Backbone module
    """
    return Backbone(
        name=name,
        train_backbone=train_backbone,
        dilation=dilation,
        pretrained=pretrained,
        freeze_bn=freeze_bn,
    )
