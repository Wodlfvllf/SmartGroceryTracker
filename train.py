#!/usr/bin/env python
"""
Training Script for DETR Fine-tuning
====================================

Script B: Fine-tune a pretrained DETR model on custom grocery datasets
(Fruits360, SKU110k, Custom Refrigerator Dataset).

This script:
1. Loads pretrained DETR from Hugging Face
2. Replaces classification head for custom grocery classes
3. Fine-tunes on your COCO-format dataset
4. Uses Hungarian matching loss (bipartite matching)
5. Saves checkpoints and best model

Usage:
    # Basic training
    python train.py --train_images data/train/images \
                    --train_annotations data/train/annotations.json
    
    # With validation
    python train.py --train_images data/train/images \
                    --train_annotations data/train/annotations.json \
                    --val_images data/val/images \
                    --val_annotations data/val/annotations.json
    
    # Custom configuration
    python train.py --config configs/default.yaml --epochs 100 --batch_size 4

Dataset Preparation:
    Ensure your data is in COCO format:
    data/
    ├── train/
    │   ├── images/          # Training images
    │   └── annotations.json # COCO format annotations
    └── val/
        ├── images/          # Validation images
        └── annotations.json # COCO format annotations
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from models.detr import build_detr
from training.criterion import build_criterion
from training.trainer import Trainer
from data.dataset import build_dataloader
from utils.misc import load_config, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train DETR on grocery dataset")
    
    # Data arguments
    parser.add_argument('--train_images', type=str, required=True,
                        help='Path to training images directory')
    parser.add_argument('--train_annotations', type=str, required=True,
                        help='Path to training annotations JSON (COCO format)')
    parser.add_argument('--val_images', type=str, default=None,
                        help='Path to validation images directory')
    parser.add_argument('--val_annotations', type=str, default=None,
                        help='Path to validation annotations JSON')
    
    # Configuration
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    
    # Model arguments
    parser.add_argument('--use_pretrained', action='store_true', default=True,
                        help='Use pretrained DETR (default: True)')
    parser.add_argument('--from_scratch', action='store_true',
                        help='Train from scratch instead of pretrained')
    parser.add_argument('--pretrained_model', type=str, 
                        default='facebook/detr-resnet-50',
                        help='Hugging Face model name for pretrained DETR')
    parser.add_argument('--num_queries', type=int, default=100,
                        help='Number of object queries')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (DETR uses small batches)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_backbone', type=float, default=1e-5,
                        help='Learning rate for backbone')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--clip_max_norm', type=float, default=0.1,
                        help='Maximum gradient norm')
    parser.add_argument('--lr_drop', type=int, default=200,
                        help='Epoch to drop learning rate')
    
    # System arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda, mps, cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Data loader workers')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable AMP')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Override pretrained flag if from_scratch is set
    if args.from_scratch:
        args.use_pretrained = False
    
    # Override AMP if no_amp is set
    if args.no_amp:
        args.use_amp = False
    
    # Get device
    device = get_device() if args.device is None else torch.device(args.device)
    print(f"\n{'='*60}")
    print("DETR Training for Smart Grocery Tracker")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # =================================================================
    # Step 1: Load Configuration
    # =================================================================
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    
    # Get class names from config
    class_names = config.get('classes', [
        'banana', 'apple', 'orange', 'avocado', 'lemon', 'strawberry', 'grape',
        'tomato', 'lettuce', 'carrot', 'cucumber', 'broccoli', 'onion', 'pepper',
        'milk', 'juice', 'yogurt', 'cheese', 'butter',
        'egg_carton', 'bottle', 'can', 'container', 'jar',
    ])
    num_classes = len(class_names)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names[:5]}... (see config for full list)")
    
    # =================================================================
    # Step 2: Build Datasets
    # =================================================================
    print("\nBuilding datasets...")
    
    train_loader, train_dataset = build_dataloader(
        img_folder=args.train_images,
        ann_file=args.train_annotations,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=True,
    )
    print(f"Training: {len(train_dataset)} images, {len(train_loader)} batches")
    
    val_loader = None
    if args.val_images and args.val_annotations:
        val_loader, val_dataset = build_dataloader(
            img_folder=args.val_images,
            ann_file=args.val_annotations,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_train=False,
        )
        print(f"Validation: {len(val_dataset)} images, {len(val_loader)} batches")
    
    # Update num_classes from dataset if different
    if hasattr(train_dataset, 'num_classes') and train_dataset.num_classes > 0:
        num_classes = train_dataset.num_classes
        print(f"Updated num_classes from dataset: {num_classes}")
    
    # =================================================================
    # Step 3: Build Model
    # =================================================================
    print("\nBuilding model...")
    
    model = build_detr(
        num_classes=num_classes,
        use_pretrained=args.use_pretrained,
        pretrained_model=args.pretrained_model,
        num_queries=args.num_queries,
        freeze_backbone=args.freeze_backbone,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # =================================================================
    # Step 4: Build Criterion
    # =================================================================
    print("\nBuilding loss criterion...")
    
    # Loss weights from config
    loss_config = config.get('loss', {})
    criterion = build_criterion(
        num_classes=num_classes,
        cost_class=loss_config.get('cost_class', 1.0),
        cost_bbox=loss_config.get('cost_bbox', 5.0),
        cost_giou=loss_config.get('cost_giou', 2.0),
        loss_ce=loss_config.get('loss_ce', 1.0),
        loss_bbox=loss_config.get('loss_bbox', 5.0),
        loss_giou=loss_config.get('loss_giou', 2.0),
        eos_coef=loss_config.get('eos_coef', 0.1),
    )
    
    print("Loss components:")
    print(f"  - Classification (CE): weight={loss_config.get('loss_ce', 1.0)}")
    print(f"  - Bounding box (L1): weight={loss_config.get('loss_bbox', 5.0)}")
    print(f"  - GIoU: weight={loss_config.get('loss_giou', 2.0)}")
    print(f"  - No-object weight: {loss_config.get('eos_coef', 0.1)}")
    
    # =================================================================
    # Step 5: Build Optimizer
    # =================================================================
    print("\nBuilding optimizer...")
    
    # Separate backbone and other parameters for different learning rates
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters()
                      if "backbone" not in n and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model.named_parameters()
                      if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    
    optimizer = AdamW(
        param_dicts,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    print(f"Optimizer: AdamW")
    print(f"  - LR (transformer): {args.lr}")
    print(f"  - LR (backbone): {args.lr_backbone}")
    print(f"  - Weight decay: {args.weight_decay}")
    
    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.lr_drop, gamma=0.1)
    print(f"Scheduler: StepLR, drop at epoch {args.lr_drop}")
    
    # =================================================================
    # Step 6: Resume from checkpoint if provided
    # =================================================================
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        from utils.misc import load_checkpoint
        ckpt = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # =================================================================
    # Step 7: Build Trainer and Train
    # =================================================================
    print("\nInitializing trainer...")
    
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=args.use_amp,
        clip_max_norm=args.clip_max_norm,
    )
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    history = trainer.train(
        num_epochs=args.epochs,
        save_every=10,
        validate_every=1,
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best model saved to: {args.checkpoint_dir}/best_model.pth")
    print(f"Final model saved to: {args.checkpoint_dir}/final_model.pth")


if __name__ == '__main__':
    main()
