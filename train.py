#!/usr/bin/env python
"""
Training Script for DETR Fine-tuning (Modal Labs Cloud)
========================================================

Script B: Fine-tune a pretrained DETR model on custom grocery datasets.

This script runs on Modal Labs cloud with GPU support.

Usage (Modal Cloud):
    # Basic training (uses SmartFridgeV2 defaults)
    modal run train.py
    
    # With custom paths
    modal run train.py --train-images /data/train \\
                       --train-annotations /data/train/_annotations.coco.json
    
    # Custom epochs and batch size
    modal run train.py --epochs 100 --batch-size 4
"""

import modal

# =============================================================================
# MODAL CONFIGURATION (self-contained)
# =============================================================================

APP_NAME = "smart-grocery-training"
app = modal.App(APP_NAME)

# Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "opencv-python>=4.8.0",
        "Pillow>=9.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "albumentations>=1.3.0",
        "pycocotools>=2.0.6",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.2.0",
    )
)

# Volumes
data_volume = modal.Volume.from_name("grocery-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("grocery-checkpoints", create_if_missing=True)

DATA_PATH = "/data"
CHECKPOINTS_PATH = "/checkpoints"
GPU_TRAINING = "A10G"

# =============================================================================
# MODAL FUNCTION
# =============================================================================

@app.function(
    gpu=GPU_TRAINING,
    image=image,
    volumes={
        DATA_PATH: data_volume,
        CHECKPOINTS_PATH: checkpoints_volume,
    },
    timeout=86400,  # 24 hours
    mounts=[
        modal.Mount.from_local_dir(
            ".",
            remote_path="/app",
            condition=lambda path: not any(
                x in path for x in [".git", "__pycache__", ".pyc", "checkpoints", "output", "SmartFridgeV2", ".zip"]
            ),
        )
    ],
)
def train_detr(
    train_images: str,
    train_annotations: str,
    val_images: str = None,
    val_annotations: str = None,
    config: str = "/app/configs/default.yaml",
    use_pretrained: bool = True,
    pretrained_model: str = "facebook/detr-resnet-50",
    num_queries: int = 100,
    freeze_backbone: bool = False,
    epochs: int = 300,
    batch_size: int = 2,
    lr: float = 1e-4,
    lr_backbone: float = 1e-5,
    weight_decay: float = 1e-4,
    clip_max_norm: float = 0.1,
    lr_drop: int = 200,
    num_workers: int = 4,
    checkpoint_dir: str = CHECKPOINTS_PATH,
    resume: str = None,
    use_amp: bool = True,
) -> dict:
    """Train DETR model on Modal cloud with GPU."""
    import os
    import sys
    
    # Add project to path
    sys.path.insert(0, "/app")
    
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import StepLR
    
    from models.detr import build_detr
    from training.criterion import build_criterion
    from training.trainer import Trainer
    from data.dataset import build_dataloader
    from utils.misc import load_config, get_device
    
    device = get_device()
    print(f"\n{'='*60}")
    print("DETR Training for Smart Grocery Tracker (Modal Cloud)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")
    
    # Load config
    cfg = {}
    if os.path.exists(config):
        cfg = load_config(config)
        print(f"Loaded config: {config}")
    
    class_names = cfg.get('classes', [
        'banana', 'apple', 'orange', 'avocado', 'lemon', 'strawberry', 'grape',
        'tomato', 'lettuce', 'carrot', 'cucumber', 'broccoli', 'onion', 'pepper',
        'milk', 'juice', 'yogurt', 'cheese', 'butter',
        'egg_carton', 'bottle', 'can', 'container', 'jar',
    ])
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")
    
    # Build datasets
    print("\nBuilding datasets...")
    train_loader, train_dataset = build_dataloader(
        img_folder=train_images,
        ann_file=train_annotations,
        batch_size=batch_size,
        num_workers=num_workers,
        is_train=True,
    )
    print(f"Training: {len(train_dataset)} images")
    
    val_loader = None
    if val_images and val_annotations:
        val_loader, val_dataset = build_dataloader(
            img_folder=val_images,
            ann_file=val_annotations,
            batch_size=batch_size,
            num_workers=num_workers,
            is_train=False,
        )
        print(f"Validation: {len(val_dataset)} images")
    
    if hasattr(train_dataset, 'num_classes') and train_dataset.num_classes > 0:
        num_classes = train_dataset.num_classes
    
    # Build model
    print("\nBuilding model...")
    model = build_detr(
        num_classes=num_classes,
        use_pretrained=use_pretrained,
        pretrained_model=pretrained_model,
        num_queries=num_queries,
        freeze_backbone=freeze_backbone,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Build criterion
    loss_cfg = cfg.get('loss', {})
    criterion = build_criterion(
        num_classes=num_classes,
        cost_class=loss_cfg.get('cost_class', 1.0),
        cost_bbox=loss_cfg.get('cost_bbox', 5.0),
        cost_giou=loss_cfg.get('cost_giou', 2.0),
        loss_ce=loss_cfg.get('loss_ce', 1.0),
        loss_bbox=loss_cfg.get('loss_bbox', 5.0),
        loss_giou=loss_cfg.get('loss_giou', 2.0),
        eos_coef=loss_cfg.get('eos_coef', 0.1),
    )
    
    # Build optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad], "lr": lr},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": lr_backbone},
    ]
    optimizer = AdamW(param_dicts, lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_drop, gamma=0.1)
    
    print(f"Optimizer: AdamW, LR={lr}, LR_backbone={lr_backbone}")
    
    # Resume if provided
    if resume:
        from utils.misc import load_checkpoint
        load_checkpoint(resume, model, optimizer, scheduler, device)
    
    # Train
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
        use_amp=use_amp,
        clip_max_norm=clip_max_norm,
    )
    
    history = trainer.train(num_epochs=epochs, save_every=10, validate_every=1)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Models saved to: {checkpoint_dir}")
    print("="*60)
    
    # Commit checkpoints
    checkpoints_volume.commit()
    
    return {
        "epochs": epochs,
        "final_train_loss": history['train_loss'][-1] if history['train_loss'] else None,
        "checkpoint_dir": checkpoint_dir,
    }


@app.local_entrypoint()
def main(
    train_images: str = None,
    train_annotations: str = None,
    val_images: str = None,
    val_annotations: str = None,
    epochs: int = 300,
    batch_size: int = 2,
    lr: float = 1e-4,
    freeze_backbone: bool = False,
):
    """Run training via: modal run train.py"""
    # Defaults
    if train_images is None:
        train_images = f"{DATA_PATH}/SmartFridgeV2_Final.v8i.coco/train"
    if train_annotations is None:
        train_annotations = f"{DATA_PATH}/SmartFridgeV2_Final.v8i.coco/train/_annotations.coco.json"
    if val_images is None:
        val_images = f"{DATA_PATH}/SmartFridgeV2_Final.v8i.coco/valid"
    if val_annotations is None:
        val_annotations = f"{DATA_PATH}/SmartFridgeV2_Final.v8i.coco/valid/_annotations.coco.json"
    
    print("Starting DETR training on Modal cloud...")
    print(f"  Train: {train_images}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}")
    print(f"  GPU: {GPU_TRAINING}")
    
    result = train_detr.remote(
        train_images=train_images,
        train_annotations=train_annotations,
        val_images=val_images,
        val_annotations=val_annotations,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        freeze_backbone=freeze_backbone,
    )
    
    print(f"\nTraining completed!")
    print(f"  Final loss: {result['final_train_loss']}")
    print(f"  Checkpoints: {result['checkpoint_dir']}")
    return result
