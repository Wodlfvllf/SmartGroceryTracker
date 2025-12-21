"""
Transformer Module for DETR
===========================

The transformer is the core of DETR, replacing traditional detection heads
(like Region Proposal Networks) with a purely attention-based mechanism.

Architecture:
    Flattened Features [B, H*W, C] + Position Encodings
                    ↓
    ┌─────────────────────────────────────┐
    │         TRANSFORMER ENCODER         │
    │  (Self-attention over image features) │
    │  Global context: every pixel sees   │
    │  every other pixel                  │
    └─────────────────────────────────────┘
                    ↓
           Memory [B, H*W, C]
                    ↓
    ┌─────────────────────────────────────┐
    │         TRANSFORMER DECODER         │
    │  Object Queries [B, N, C] (N=100)   │
    │  Cross-attention with memory        │
    │  Each query learns to find one object │
    └─────────────────────────────────────┘
                    ↓
         Output [B, N, C] → FFN → Predictions

Object Queries Explained:
    - N learnable embeddings (typically N=100)
    - Each query specializes to detect objects in certain positions/scales
    - Queries attend to image features via cross-attention
    - Output: one prediction per query (class + bbox)
    - Hungarian matching assigns queries to ground truth objects
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """
    Full transformer with encoder and decoder for DETR.
    
    The encoder processes flattened image features.
    The decoder uses object queries to extract object representations.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        normalize_before: bool = False,
        return_intermediate_dec: bool = True,
    ):
        """
        Args:
            d_model: Hidden dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: FFN intermediate dimension
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
            normalize_before: Apply LayerNorm before attention (pre-norm)
            return_intermediate_dec: Return outputs from all decoder layers
        """
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        
        # Build encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=encoder_norm,
        )
        
        # Build decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            norm=decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor,
        query_embed: torch.Tensor,
        pos_embed: torch.Tensor,
    ):
        """
        Forward pass through the transformer.
        
        Args:
            src: Flattened feature map [B, C, H, W] -> will be [H*W, B, C]
            mask: Padding mask [B, H, W]
            query_embed: Object queries [N, C]
            pos_embed: Position encodings [B, C, H, W]
        
        Returns:
            hs: Decoder outputs [num_layers, B, N, C] or [1, B, N, C]
            memory: Encoder output [H*W, B, C]
        """
        # Flatten spatial dimensions: [B, C, H, W] -> [H*W, B, C]
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        
        # Flatten mask: [B, H, W] -> [B, H*W]
        mask = mask.flatten(1)
        
        # Prepare object queries: [N, C] -> [N, B, C]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        
        # Initialize target queries (zeros, will be updated by decoder)
        tgt = torch.zeros_like(query_embed)
        
        # Encode image features
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        
        # Decode object queries
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        
        # Transpose: [num_layers, N, B, C] -> [num_layers, B, N, C]
        hs = hs.transpose(1, 2)
        
        return hs, memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""
    
    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            _get_clones(encoder_layer, 1)[0] for _ in range(num_layers)
        ])
        # Actually clone properly
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ):
        """
        Forward through all encoder layers.
        
        Args:
            src: Input features [S, B, C]
            mask: Attention mask [S, S]
            src_key_padding_mask: Padding mask [B, S]
            pos: Position encodings [S, B, C]
        
        Returns:
            Output features [S, B, C]
        """
        output = src
        
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output


class TransformerDecoder(nn.Module):
    """Stack of transformer decoder layers."""
    
    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        return_intermediate: bool = False,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):
        """
        Forward through all decoder layers.
        
        Args:
            tgt: Target (initialized as zeros) [N, B, C]
            memory: Encoder output [S, B, C]
            tgt_mask: Target attention mask
            memory_mask: Memory attention mask
            tgt_key_padding_mask: Target padding mask
            memory_key_padding_mask: Memory padding mask [B, S]
            pos: Position encodings for memory [S, B, C]
            query_pos: Position encodings for queries [N, B, C]
        
        Returns:
            Output [num_layers, N, B, C] if return_intermediate, else [1, N, B, C]
        """
        output = tgt
        intermediate = []
        
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        
        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer.
    
    Architecture:
        Input → Self-Attention → Add & Norm → FFN → Add & Norm → Output
    
    Self-attention allows every position to attend to all other positions,
    giving the model global context over the entire image.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        normalize_before: bool = False,
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
    
    def with_pos_embed(self, tensor: torch.Tensor, pos: Optional[torch.Tensor]):
        """Add positional embedding to tensor."""
        return tensor if pos is None else tensor + pos
    
    def forward_post(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ):
        """Post-norm forward (default in original DETR)."""
        # Add position to query and key, but not value
        q = k = self.with_pos_embed(src, pos)
        
        # Self-attention
        src2 = self.self_attn(
            q, k, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        
        # Add & Norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        
        # Add & Norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src
    
    def forward_pre(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ):
        """Pre-norm forward (norm before attention)."""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        
        src2 = self.self_attn(
            q, k, src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer.
    
    Architecture:
        Object Query → Self-Attention → Add & Norm
                    → Cross-Attention (with encoder memory) → Add & Norm
                    → FFN → Add & Norm → Output
    
    Cross-attention is the key: object queries attend to image features
    to gather information about potential objects.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        normalize_before: bool = False,
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
    
    def with_pos_embed(self, tensor: torch.Tensor, pos: Optional[torch.Tensor]):
        """Add positional embedding to tensor."""
        return tensor if pos is None else tensor + pos
    
    def forward_post(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):
        """Post-norm forward."""
        # Self-attention among object queries
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention: queries attend to image features
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt
    
    def forward_pre(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):
        """Pre-norm forward."""
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        
        return tgt
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos
            )
        return self.forward_post(
            tgt, memory, tgt_mask, memory_mask,
            tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos
        )


def _get_clones(module: nn.Module, n: int):
    """Create n copies of a module."""
    return nn.ModuleList([module.__class__(**module.__dict__) if hasattr(module, '__dict__') 
                          else type(module)(*[getattr(module, attr) for attr in dir(module) if not attr.startswith('_')])
                          for _ in range(n)])


def _get_clones(module, n):
    """Create n deep copies of a module."""
    import copy
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation: str):
    """Get activation function by name."""
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    else:
        raise ValueError(f"Unknown activation: {activation}")


def build_transformer(
    hidden_dim: int = 256,
    num_heads: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
) -> Transformer:
    """
    Build the transformer module.
    
    Args:
        hidden_dim: Hidden dimension (d_model)
        num_heads: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        dim_feedforward: FFN intermediate dimension
        dropout: Dropout rate
    
    Returns:
        Transformer module
    """
    return Transformer(
        d_model=hidden_dim,
        nhead=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        normalize_before=False,
        return_intermediate_dec=True,
    )
