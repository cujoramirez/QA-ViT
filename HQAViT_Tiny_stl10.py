"""
HQA-ViT STL-10 Supervised-Only Fine-tuning
Direct transfer from CIFAR-100 pretrained model to STL-10 using only 5k labeled images
Testing robustness without any unsupervised/self-supervised learning
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time 
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import json
import math

try:
    from HQAViT_CIFAR100 import HQAViT, HQAViTConfig
except ImportError:
    print("Error: Cannot import model classes. Make sure HQAViT_CIFAR100.py is in the same directory.")
    raise


@dataclass
class FineTuneConfig:
    """Supervised fine-tuning configuration"""
    # Training parameters
    epochs: int = 50
    batch_size: int = 128
    warmup_epochs: int = 3
    
    # Learning rate
    base_lr: float = 5e-5
    head_lr_multiplier: float = 10.0
    min_lr: float = 1e-6
    
    # Regularization
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    max_grad_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = 'bfloat16'
    
    # Data
    num_workers: int = 2
    pin_memory: bool = True
    img_size: int = 96
    
    # Paths
    pretrained_path: str = "./checkpoints_finetuned/best_finetuned.pth"
    data_root: str = "./data"
    checkpoint_dir: str = "./checkpoints_stl10_supervised"
    log_dir: str = "./logs_stl10_supervised"
    
    # Logging
    print_freq: int = 50
    eval_freq: int = 1
    save_freq: int = 10


class TrainingLogger:
    """Training logger for metrics and plots"""
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'epoch_time': []
        }
        
        self.best_val_acc = 0.0
        self.start_time = time.time()
    
    def log_epoch(self, epoch, metrics):
        self.history['epoch'].append(epoch)
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        if 'val_acc' in metrics and metrics['val_acc'] > self.best_val_acc:
            self.best_val_acc = metrics['val_acc']
    
    def save_metrics(self):
        metrics_file = self.log_dir / 'training_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump({
                'history': self.history,
                'best_val_acc': self.best_val_acc,
                'total_time': time.time() - self.start_time
            }, f, indent=2)
        print(f"[INFO] Metrics saved to: {metrics_file}")
    
    def plot_training_curves(self):
        if len(self.history['epoch']) == 0:
            return
        
        fig = plt.figure(figsize=(20, 6))
        
        # Loss curves
        ax1 = plt.subplot(1, 3, 1)
        ax1.plot(self.history['epoch'], self.history['train_loss'], 'b-', 
                label='Train Loss', linewidth=2, marker='o', markersize=3)
        ax1.plot(self.history['epoch'], self.history['val_loss'], 'r-', 
                label='Val Loss', linewidth=2, marker='s', markersize=3)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2 = plt.subplot(1, 3, 2)
        ax2.plot(self.history['epoch'], self.history['train_acc'], 'b-', 
                label='Train Acc', linewidth=2, marker='o', markersize=3)
        ax2.plot(self.history['epoch'], self.history['val_acc'], 'r-', 
                label='Val Acc', linewidth=2, marker='s', markersize=3)
        ax2.axhline(y=self.best_val_acc, color='g', linestyle='--', 
                   linewidth=2, label=f'Best: {self.best_val_acc:.2f}%')
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Learning rate
        ax3 = plt.subplot(1, 3, 3)
        ax3.plot(self.history['epoch'], self.history['lr'], 'purple', linewidth=2.5)
        ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.log_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Training curves saved to: {plot_path}")


def get_stl10_dataloaders(config: FineTuneConfig):
    """Get STL-10 train and test dataloaders (5k train, 8k test)"""
    mean = (0.4467, 0.4398, 0.4066)
    std = (0.2603, 0.2566, 0.2713)
    
    # Stronger training augmentations
    # Use RandomResizedCrop to vary scale/crop and RandAugment (if available) for stronger
    # automated augmentation policies. Keep ColorJitter and RandomErasing as additional regularizers.
    # Fall back safely if RandAugment/AutoAugment is not present in torchvision version.

    # Build optional augmentation (RandAugment / AutoAugment / fallback)
    if hasattr(transforms, 'RandAugment'):
        strong_policy = transforms.RandAugment(num_ops=2, magnitude=9)
        policy_name = 'RandAugment'
    elif hasattr(transforms, 'AutoAugment'):
        try:
            # AutoAugment may require a policy argument in some versions; default should work.
            strong_policy = transforms.AutoAugment()
        except Exception:
            strong_policy = transforms.Compose([])
        policy_name = 'AutoAugment'
    else:
        strong_policy = transforms.Compose([])
        policy_name = 'None (fallback)'

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.img_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        # apply automated augmentation policy if available
        strong_policy,
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        # occasional Gaussian blur to improve robustness
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])

    # Informational message about augmentation policy (non-fatal)
    try:
        print(f"[AUGMENT] Using strong augmentation policy: {policy_name}")
    except Exception:
        pass
    
    # Validation (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Load datasets
    train_dataset = datasets.STL10(
        root=config.data_root,
        split='train',
        transform=train_transform,
        download=True
    )
    
    val_dataset = datasets.STL10(
        root=config.data_root,
        split='test',
        transform=val_transform,
        download=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=True if config.num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False,
    )
    
    return train_loader, val_loader


def adjust_positional_embedding(model, new_img_size):
    """Adjust positional embedding for different image sizes"""
    patch_size = model.patch_embed.patch_size[0] if hasattr(model.patch_embed, 'patch_size') else 4
    new_num_patches = (new_img_size // patch_size) ** 2
    
    pos_embed = model.pos_embed
    old_num_patches = pos_embed.shape[1]
    
    if new_num_patches != old_num_patches:
        print(f"[INFO] Adjusting positional embedding: {old_num_patches} -> {new_num_patches} patches")
        
        old_size = int(math.sqrt(old_num_patches))
        new_size = int(math.sqrt(new_num_patches))
        
        if old_size * old_size == old_num_patches and new_size * new_size == new_num_patches:
            pos_embed_reshaped = pos_embed.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
            pos_embed_resized = F.interpolate(
                pos_embed_reshaped, 
                size=(new_size, new_size), 
                mode='bicubic', 
                align_corners=False
            )
            pos_embed_new = pos_embed_resized.permute(0, 2, 3, 1).reshape(1, new_num_patches, -1)
            model.pos_embed = nn.Parameter(pos_embed_new)
            print(f"[INFO] Positional embedding adjusted successfully")
        else:
            print(f"[WARNING] Non-square patch grid, using repeat strategy")
            if new_num_patches > old_num_patches:
                repeats = (new_num_patches // old_num_patches) + 1
                pos_embed_new = pos_embed.repeat(1, repeats, 1)[:, :new_num_patches, :]
            else:
                pos_embed_new = pos_embed[:, :new_num_patches, :]
            model.pos_embed = nn.Parameter(pos_embed_new)


def check_for_nan(loss, model, optimizer, epoch, batch_idx):
    """Check for NaN and provide debugging info"""
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"\n[ERROR] NaN/Inf detected at epoch {epoch}, batch {batch_idx}")
        print(f"   Loss value: {loss.item()}")
        print("   Stopping training to prevent further corruption.")
        return True
    return False


def train_epoch(model, loader, optimizer, scheduler, scaler, criterion, config, epoch):
    """Train one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0
    
    amp_dtype = torch.bfloat16 if config.amp_dtype == 'bfloat16' else torch.float16
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        
        with autocast(enabled=config.use_amp, dtype=amp_dtype):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        if check_for_nan(loss, model, optimizer, epoch, batch_idx):
            raise ValueError("Training stopped due to NaN loss")
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        num_batches += 1
        
        if batch_idx % config.print_freq == 0:
            acc = 100. * correct / total
            lr = optimizer.param_groups[0]['lr']
            avg_loss = total_loss / num_batches
            print(f'  Epoch {epoch:3d} [{batch_idx:4d}/{len(loader):4d}] | '
                  f'Loss: {avg_loss:.4f} | Acc: {acc:6.2f}% | LR: {lr:.7f}')
    
    scheduler.step()
    return total_loss / num_batches, 100. * correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    """Validation"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def main():
    print("\n" + "="*100)
    print("HQA-ViT STL-10 SUPERVISED-ONLY FINE-TUNING".center(100))
    print("Direct Transfer from CIFAR-100 to STL-10 (5k labeled images only)".center(100))
    print("="*100)
    
    config = FineTuneConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n[DEVICE] Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"    Precision: {'BF16' if config.amp_dtype == 'bfloat16' else 'FP16'}")
    
    # Check BF16 support
    if torch.cuda.is_available() and config.amp_dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        print("    [WARNING] BF16 not supported, falling back to FP32")
        config.use_amp = False
    
    # Windows-specific settings
    if os.name == 'nt':
        print("    [WARNING] Windows detected - setting num_workers=0 and pin_memory=False")
        config.num_workers = 0
        config.pin_memory = False
    
    # Create directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Load CIFAR-100 pretrained model
    print(f"\n[LOAD] Loading CIFAR-100 pretrained model from: {config.pretrained_path}")
    
    if not os.path.exists(config.pretrained_path):
        raise FileNotFoundError(f"Pretrained model not found at {config.pretrained_path}")
    
    checkpoint = torch.load(config.pretrained_path, map_location='cpu')
    model_config = checkpoint.get('model_config', HQAViTConfig())
    
    # Create model
    model = HQAViT(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    cifar100_acc = checkpoint.get('val_acc', 0.0)
    print(f"   [SUCCESS] CIFAR-100 Validation Accuracy: {cifar100_acc:.2f}%")
    
    # Adjust for STL-10 image size (96x96)
    print(f"\n[INFO] Adjusting model for STL-10 image size (96x96)...")
    adjust_positional_embedding(model, config.img_size)
    
    # Replace classification head for 10 classes
    old_head_dim = model.head.in_features
    model.head = nn.Linear(old_head_dim, 10).cuda()
    print(f"[INFO] Replaced classification head: {old_head_dim} -> 10 classes")
    
    # Load data
    print(f"\n[DATA] Loading STL-10 labeled data...")
    train_loader, val_loader = get_stl10_dataloaders(config)
    print(f"   Train: {len(train_loader.dataset):,} samples ({len(train_loader)} batches)")
    print(f"   Val:   {len(val_loader.dataset):,} samples ({len(val_loader)} batches)")
    
    print(f"\n[DATASET] STL-10 Information:")
    print(f"   Image size: 96x96 (vs CIFAR-100: 32x32)")
    print(f"   Classes: 10 (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck)")
    print(f"   Training: 5,000 labeled images ONLY")
    print(f"   Testing: 8,000 labeled images")
    print(f"   Note: Not using 100k unlabeled images (supervised-only)")
    
    # Setup optimizer with differential learning rates
    head_params = list(model.head.parameters())
    backbone_params = [p for n, p in model.named_parameters() if 'head' not in n]
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config.base_lr},
        {'params': head_params, 'lr': config.base_lr * config.head_lr_multiplier}
    ], betas=(0.9, 0.999), weight_decay=config.weight_decay, eps=1e-8)
    
    # Learning rate schedulers
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs - config.warmup_epochs,
        eta_min=config.min_lr
    )
    
    warmup_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (epoch + 1) / config.warmup_epochs
    )
    
    scaler = GradScaler(enabled=config.use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    logger = TrainingLogger(config.log_dir)
    
    print(f"\n[CONFIG] Training Configuration:")
    print(f"   Epochs:           {config.epochs} (warmup: {config.warmup_epochs})")
    print(f"   Batch size:       {config.batch_size}")
    print(f"   Backbone LR:      {config.base_lr:.6f}")
    print(f"   Head LR:          {config.base_lr * config.head_lr_multiplier:.6f}")
    print(f"   Weight decay:     {config.weight_decay}")
    print(f"   Label smoothing:  {config.label_smoothing}")
    print(f"   Grad clip:        {config.max_grad_norm}")
    
    print(f"\n{'='*100}")
    print("SUPERVISED FINE-TUNING STARTED".center(100))
    print(f"{'='*100}\n")
    
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        
        current_scheduler = warmup_scheduler if epoch <= config.warmup_epochs else main_scheduler
        
        try:
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, current_scheduler,
                scaler, criterion, config, epoch
            )
            
            val_loss, val_acc = validate(model, val_loader, criterion)
        except ValueError as e:
            print(f"\n[ERROR] Training stopped: {e}")
            break
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.log_epoch(epoch, {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr,
            'epoch_time': epoch_time
        })
        
        print(f"\n{'='*100}")
        print(f"EPOCH {epoch}/{config.epochs} SUMMARY".center(100))
        print(f"{'='*100}")
        print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:6.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}  |  Val Acc:   {val_acc:6.2f}%")
        print(f"  Learning Rate: {current_lr:.7f}")
        print(f"  Epoch Time: {epoch_time:.1f}s")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'cifar100_acc': cifar100_acc,
                'config': config,
            }, f"{config.checkpoint_dir}/best_model.pth")
            print(f"  [BEST] NEW BEST! Val Acc: {best_acc:.2f}% (saved)")
        else:
            print(f"  [INFO] Best Val Acc: {best_acc:.2f}% (epoch {best_epoch})")
        
        print(f"{'='*100}\n")
        
        if epoch % config.save_freq == 0:
            logger.plot_training_curves()
            logger.save_metrics()
    
    logger.save_metrics()
    logger.plot_training_curves()
    
    print(f"\n{'='*100}")
    print("SUPERVISED FINE-TUNING COMPLETE".center(100))
    print(f"{'='*100}")
    print(f"\n[RESULTS] Final Results:")
    print(f"   CIFAR-100 Pretrained Accuracy:  {cifar100_acc:.2f}%")
    print(f"   STL-10 Final Accuracy:          {best_acc:.2f}% (epoch {best_epoch})")
    print(f"   Transfer Performance:           {best_acc - cifar100_acc:+.2f}%")
    
    print(f"\n[ANALYSIS] Transfer Learning Analysis:")
    if best_acc > cifar100_acc:
        print(f"   [EXCELLENT] Model improved on STL-10 despite larger images")
        print(f"   [EXCELLENT] Features transfer well across datasets")
    elif best_acc > cifar100_acc - 5:
        print(f"   [GOOD] Model maintained performance on different dataset")
        print(f"   [GOOD] Reasonable robustness to domain shift")
    else:
        print(f"   [MODERATE] Performance drop indicates domain gap")
        print(f"   [MODERATE] May benefit from additional data or pretraining")
    
    print(f"\n[FILES] Saved Files:")
    print(f"   Best Model:       {config.checkpoint_dir}/best_model.pth")
    print(f"   Training Logs:    {config.log_dir}/training_metrics.json")
    print(f"   Training Curves:  {config.log_dir}/training_curves.png")
    
    print(f"\n{'='*100}\n")
    
    return best_acc


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    try:
        final_acc = main()
        print(f"[SUCCESS] Supervised fine-tuning completed successfully!")
        print(f"   Final STL-10 accuracy: {final_acc:.2f}%")
    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Training failed with error: {e}")
        import traceback
        traceback.print_exc()