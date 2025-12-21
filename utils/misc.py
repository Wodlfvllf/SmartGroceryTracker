"""
Miscellaneous Utilities for Smart Grocery Tracker
==================================================

General utility functions for:
- Configuration loading
- Checkpoint saving/loading
- Batch collation for variable-length annotations
- NestedTensor for padded image batches
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class NestedTensor:
    """
    Container class for padded tensors with their corresponding masks.
    
    DETR uses NestedTensor because:
    1. Images in a batch may have different sizes
    2. We pad them to the same size for batching
    3. The mask indicates which pixels are padding (True = padding)
    
    This allows the model to ignore padded regions in attention.
    """
    
    def __init__(self, tensors: Tensor, mask: Tensor):
        """
        Args:
            tensors: Batched tensors of shape [B, C, H, W]
            mask: Boolean mask of shape [B, H, W], True where padding
        """
        self.tensors = tensors
        self.mask = mask
    
    def to(self, device: torch.device) -> 'NestedTensor':
        """Move to specified device."""
        return NestedTensor(
            self.tensors.to(device),
            self.mask.to(device) if self.mask is not None else None
        )
    
    def decompose(self) -> Tuple[Tensor, Tensor]:
        """Return tensors and mask."""
        return self.tensors, self.mask
    
    @property
    def device(self) -> torch.device:
        return self.tensors.device
    
    @property
    def dtype(self) -> torch.dtype:
        return self.tensors.dtype


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    """
    Create a NestedTensor from a list of tensors with different sizes.
    
    This function:
    1. Finds the maximum H and W across all tensors
    2. Pads all tensors to that size
    3. Creates a mask indicating padded regions
    
    Args:
        tensor_list: List of tensors of shape [C, H_i, W_i]
    
    Returns:
        NestedTensor with padded tensors and mask
    """
    if len(tensor_list) == 0:
        raise ValueError("Cannot create NestedTensor from empty list")
    
    # Get max dimensions
    max_h = max(t.shape[1] for t in tensor_list)
    max_w = max(t.shape[2] for t in tensor_list)
    
    # Batch size and channels
    batch_size = len(tensor_list)
    channels = tensor_list[0].shape[0]
    
    # Create padded batch tensor
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    batch_tensor = torch.zeros(batch_size, channels, max_h, max_w, dtype=dtype, device=device)
    
    # Create mask (True where padding)
    mask = torch.ones(batch_size, max_h, max_w, dtype=torch.bool, device=device)
    
    # Fill in tensors and update mask
    for i, t in enumerate(tensor_list):
        h, w = t.shape[1], t.shape[2]
        batch_tensor[i, :, :h, :w] = t
        mask[i, :h, :w] = False  # False = not padding
    
    return NestedTensor(batch_tensor, mask)


def collate_fn(batch: List[Tuple[Tensor, Dict]]) -> Tuple[NestedTensor, List[Dict]]:
    """
    Custom collate function for DETR dataloader.
    
    Handles:
    1. Variable-size images → NestedTensor with padding
    2. Variable number of objects → List of target dicts
    
    Args:
        batch: List of (image, target) tuples
    
    Returns:
        NestedTensor of images, list of targets
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Stack images with padding
    nested = nested_tensor_from_tensor_list(images)
    
    return nested, targets


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    scheduler: Optional[Any] = None,
    config: Optional[Dict] = None,
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        epoch: Current epoch number
        loss: Current loss value
        path: Path to save the checkpoint
        scheduler: Optional learning rate scheduler
        config: Optional configuration dict
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Build checkpoint dict
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if config is not None:
        checkpoint['config'] = config
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        path: Path to the checkpoint file
        model: The model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load the checkpoint to
    
    Returns:
        Checkpoint dictionary with epoch, loss, etc.
    """
    checkpoint = torch.load(path, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return checkpoint


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Get the best available device.
    
    Args:
        force_cpu: If True, always return CPU device
    
    Returns:
        torch.device for CUDA if available, else CPU
    """
    if force_cpu:
        return torch.device('cpu')
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    """
    Inverse sigmoid function.
    
    Used in some DETR variants for box refinement.
    
    Args:
        x: Input tensor, values should be in (0, 1)
        eps: Small value to avoid numerical issues
    
    Returns:
        Tensor with inverse sigmoid applied
    """
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


def accuracy(output: Tensor, target: Tensor, topk: Tuple[int, ...] = (1,)) -> List[Tensor]:
    """
    Compute top-k accuracy.
    
    Args:
        output: Model predictions of shape [N, C]
        target: Ground truth labels of shape [N]
        topk: Tuple of k values for top-k accuracy
    
    Returns:
        List of accuracy values for each k
    """
    maxk = max(topk)
    batch_size = target.size(0)
    
    if batch_size == 0:
        return [torch.tensor(0.0) for _ in topk]
    
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    
    return res
