"""
DETR Trainer
============

Training loop for DETR with:
- Mixed precision training (AMP)
- Gradient clipping (essential for DETR stability)
- Learning rate scheduling
- Checkpoint saving
- Validation metrics
- Early stopping
"""

import os
import time
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from training.criterion import SetCriterion
from utils.misc import save_checkpoint


class Trainer:
    """
    DETR Training Manager.
    
    Handles the training loop, validation, checkpointing, and metrics.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: SetCriterion,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional[Any] = None,
        device: torch.device = torch.device('cuda'),
        checkpoint_dir: str = 'checkpoints',
        use_amp: bool = True,
        clip_max_norm: float = 0.1,
        log_interval: int = 50,
    ):
        """
        Args:
            model: DETR model
            criterion: Loss criterion
            optimizer: Optimizer
            train_loader: Training data loader
            val_loader: Validation data loader
            scheduler: Learning rate scheduler
            device: Device to train on
            checkpoint_dir: Directory for checkpoints
            use_amp: Use automatic mixed precision
            clip_max_norm: Max gradient norm for clipping
            log_interval: Logging frequency (batches)
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.use_amp = use_amp
        self.clip_max_norm = clip_max_norm
        self.log_interval = log_interval
        
        # Move model and criterion to device
        self.model.to(device)
        self.criterion.to(device)
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
    
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dict of average losses
        """
        self.model.train()
        self.criterion.train()
        
        total_losses = {}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (samples, targets) in enumerate(pbar):
            # Move to device
            samples = samples.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                outputs = self.model(samples)
                loss_dict = self.criterion(outputs, targets)
                
                # Compute weighted loss
                weight_dict = self.criterion.weight_dict
                losses = sum(
                    loss_dict[k] * weight_dict.get(k, 1.0)
                    for k in loss_dict.keys() if k in weight_dict
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(losses).backward()
                
                # Gradient clipping (after unscaling)
                if self.clip_max_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_max_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses.backward()
                
                if self.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_max_norm
                    )
                
                self.optimizer.step()
            
            # Accumulate losses
            for k, v in loss_dict.items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.item()
            num_batches += 1
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{losses.item():.4f}",
                    'loss_ce': f"{loss_dict.get('loss_ce', 0):.4f}",
                    'loss_bbox': f"{loss_dict.get('loss_bbox', 0):.4f}",
                    'loss_giou': f"{loss_dict.get('loss_giou', 0):.4f}",
                })
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dict of average validation losses
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        self.criterion.eval()
        
        total_losses = {}
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for samples, targets in pbar:
            # Move to device
            samples = samples.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                outputs = self.model(samples)
                loss_dict = self.criterion(outputs, targets)
            
            # Accumulate losses
            for k, v in loss_dict.items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.item()
            num_batches += 1
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def train(
        self,
        num_epochs: int,
        save_every: int = 10,
        validate_every: int = 1,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            num_epochs: Total epochs to train
            save_every: Save checkpoint every N epochs
            validate_every: Validate every N epochs
        
        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"  Device: {self.device}")
        print(f"  AMP: {self.use_amp}")
        print(f"  Gradient clipping: {self.clip_max_norm}")
        print()
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            epoch_start = time.time()
            
            # Train
            train_losses = self.train_one_epoch(epoch)
            train_loss = sum(
                train_losses[k] * self.criterion.weight_dict.get(k, 1.0)
                for k in train_losses if k in self.criterion.weight_dict
            )
            history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = None
            if self.val_loader is not None and (epoch + 1) % validate_every == 0:
                val_losses = self.validate()
                val_loss = sum(
                    val_losses[k] * self.criterion.weight_dict.get(k, 1.0)
                    for k in val_losses if k in self.criterion.weight_dict
                )
                history['val_loss'].append(val_loss)
            
            # Learning rate step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Logging
            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch}/{num_epochs - 1}")
            print(f"  Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"  Val Loss: {val_loss:.4f}")
            print(f"  LR: {lr:.2e}")
            print(f"  Time: {epoch_time:.1f}s")
            
            # Save best model
            current_loss = val_loss if val_loss is not None else train_loss
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    current_loss,
                    os.path.join(self.checkpoint_dir, 'best_model.pth'),
                    scheduler=self.scheduler,
                )
                print(f"  ★ New best model saved!")
            
            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    current_loss,
                    os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'),
                    scheduler=self.scheduler,
                )
            
            self.current_epoch = epoch + 1
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time / 3600:.2f} hours")
        print(f"Best loss: {self.best_loss:.4f}")
        
        # Save final model
        save_checkpoint(
            self.model,
            self.optimizer,
            num_epochs - 1,
            current_loss,
            os.path.join(self.checkpoint_dir, 'final_model.pth'),
            scheduler=self.scheduler,
        )
        
        return history
