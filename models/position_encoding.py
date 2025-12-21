"""
Positional Encoding for DETR
============================

DETR uses sinusoidal positional encodings to provide spatial information
to the transformer. Unlike NLP where positions are 1D (sequence), images
require 2D positional encodings (height and width).

Why Positional Encoding?
- Transformers have no inherent notion of position
- Without positional info, the model can't distinguish spatial relationships
- Sinusoidal encodings allow the model to learn relative positions

How it works:
1. For each position (y, x) in the feature map
2. Compute sinusoidal encoding based on coordinates
3. Concatenate row (y) and column (x) encodings
4. Result: [hidden_dim] encoding for each spatial position
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from utils.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    Sinusoidal 2D positional encoding.
    
    This is the original positional encoding from the DETR paper.
    It creates unique encodings for each (x, y) position using sine
    and cosine functions at different frequencies.
    
    Mathematical formula:
        PE(pos, 2i) = sin(pos / (10000^(2i/d)))
        PE(pos, 2i+1) = cos(pos / (10000^(2i/d)))
    
    Where:
        - pos: position (x or y coordinate)
        - i: dimension index
        - d: total dimensions (hidden_dim / 2 for each of x and y)
    """
    
    def __init__(
        self,
        num_pos_feats: int = 128,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        """
        Args:
            num_pos_feats: Number of positional features (hidden_dim / 2)
            temperature: Temperature for the sinusoidal encoding (default 10000)
            normalize: Whether to normalize positions to [0, 1]
            scale: Scale factor for normalized positions (default 2*pi)
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    
    def forward(self, tensor_list: NestedTensor) -> torch.Tensor:
        """
        Compute positional encodings for the feature map.
        
        Args:
            tensor_list: NestedTensor with features and mask
                - features: [B, C, H, W]
                - mask: [B, H, W] (True where padding)
        
        Returns:
            Positional encodings of shape [B, hidden_dim, H, W]
        """
        x = tensor_list.tensors
        mask = tensor_list.mask
        
        assert mask is not None, "Mask is required for positional encoding"
        
        # Invert mask: True = valid, False = padding
        not_mask = ~mask
        
        # Cumulative sum to get position indices
        # Y positions: cumsum along height
        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)
        # X positions: cumsum along width
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)
        
        # Normalize positions to [0, scale]
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        
        # Create dimension indices
        dim_t = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
            device=x.device
        )
        # Temperature scaling: 10000^(2i/d)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        # Compute positional encodings
        # Shape: [B, H, W, num_pos_feats]
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        # Apply sin to even indices, cos to odd indices
        # Then interleave them
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4
        ).flatten(3)
        
        # Concatenate x and y encodings
        # Shape: [B, H, W, hidden_dim]
        pos = torch.cat((pos_y, pos_x), dim=3)
        
        # Transpose to [B, hidden_dim, H, W]
        pos = pos.permute(0, 3, 1, 2)
        
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Learned 2D positional embeddings.
    
    Alternative to sinusoidal encodings where positions are learned.
    Creates two embedding tables (one for rows, one for columns) and
    combines them.
    
    Pros:
        - Can learn task-specific positional patterns
    Cons:
        - Limited to max resolution seen during training
        - More parameters to learn
    """
    
    def __init__(self, num_pos_feats: int = 128, max_size: int = 50):
        """
        Args:
            num_pos_feats: Number of positional features (hidden_dim / 2)
            max_size: Maximum feature map size (H or W)
        """
        super().__init__()
        self.row_embed = nn.Embedding(max_size, num_pos_feats)
        self.col_embed = nn.Embedding(max_size, num_pos_feats)
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
    
    def forward(self, tensor_list: NestedTensor) -> torch.Tensor:
        """
        Compute learned positional encodings.
        
        Args:
            tensor_list: NestedTensor with features
        
        Returns:
            Positional encodings of shape [B, hidden_dim, H, W]
        """
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        
        # Get position indices
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        
        # Get embeddings
        x_emb = self.col_embed(i)  # [W, num_pos_feats]
        y_emb = self.row_embed(j)  # [H, num_pos_feats]
        
        # Broadcast and concatenate
        # [H, W, num_pos_feats] + [H, W, num_pos_feats] -> [H, W, hidden_dim]
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1)
        
        # [H, W, hidden_dim] -> [hidden_dim, H, W]
        pos = pos.permute(2, 0, 1)
        
        # [hidden_dim, H, W] -> [B, hidden_dim, H, W]
        pos = pos.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        
        return pos


def build_position_encoding(
    hidden_dim: int = 256,
    position_embedding_type: str = 'sine',
) -> nn.Module:
    """
    Build position encoding module.
    
    Args:
        hidden_dim: Hidden dimension of the model
        position_embedding_type: 'sine' or 'learned'
    
    Returns:
        Position encoding module
    """
    num_pos_feats = hidden_dim // 2
    
    if position_embedding_type == 'sine':
        return PositionEmbeddingSine(num_pos_feats, normalize=True)
    elif position_embedding_type == 'learned':
        return PositionEmbeddingLearned(num_pos_feats)
    else:
        raise ValueError(f"Unknown position embedding type: {position_embedding_type}")
