"""
HQA-ViT Fine-tuning Script for CIFAR-10
Classic transfer learning from CIFAR-100 checkpoint (similar to ImageNet-1K style)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
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
from datetime import datetime

try:
    from HQAViT_CIFAR100 import HQAViT, HQAViTConfig
except ImportError:
    print("Error: Cannot import model classes. Make sure HQAViT_CIFAR100.py is in the same directory.")
    raise


@dataclass
class FineTuneConfig:
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    
    epochs: int = 100
    warmup_epochs: int = 5
    
    # Classic transfer learning rates
    base_lr: float = 1e-4
    head_lr_multiplier: float = 10.0
    min_lr: float = 1e-6
    
    # Standard regularization
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    
    max_grad_norm: float = 1.0
    
    use_amp: bool = True
    amp_dtype: str = 'bfloat16'
    
    # Paths
    pretrained_path: str = "./checkpoints_finetuned/best_finetuned.pth"
    data_root: str = "./data"
    checkpoint_dir: str = "./checkpoints_cifar10"
    log_dir: str = "./logs_cifar10"
    
    print_freq: int = 50
    eval_freq: int = 1
    save_freq: int = 20


class TrainingLogger:
    """Comprehensive training logger with visualization"""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'grad_norm': [],
            'epoch_time': []
        }
        
        self.best_val_acc = 0.0
        self.start_time = time.time()
    
    def log_epoch(self, epoch, metrics):
        """Log epoch metrics"""
        self.history['epoch'].append(epoch)
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        if 'val_acc' in metrics and metrics['val_acc'] > self.best_val_acc:
            self.best_val_acc = metrics['val_acc']
    
    def save_metrics(self):
        """Save metrics to JSON"""
        metrics_file = self.log_dir / 'training_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump({
                'history': self.history,
                'best_val_acc': self.best_val_acc,
                'total_time': time.time() - self.start_time
            }, f, indent=2)
        print(f"📊 Metrics saved to: {metrics_file}")
    
    def plot_training_curves(self):
        """Generate comprehensive training visualization plots"""
        if len(self.history['epoch']) == 0:
            return
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Loss curves
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(self.history['epoch'], self.history['train_loss'], 'b-', 
                label='Train Loss', linewidth=2, marker='o', markersize=3)
        ax1.plot(self.history['epoch'], self.history['val_loss'], 'r-', 
                label='Val Loss', linewidth=2, marker='s', markersize=3)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 2. Accuracy curves
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(self.history['epoch'], self.history['train_acc'], 'b-', 
                label='Train Acc', linewidth=2, marker='o', markersize=3)
        ax2.plot(self.history['epoch'], self.history['val_acc'], 'r-', 
                label='Val Acc', linewidth=2, marker='s', markersize=3)
        ax2.axhline(y=self.best_val_acc, color='g', linestyle='--', 
                   linewidth=2, label=f'Best Val: {self.best_val_acc:.2f}%')
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # 3. Learning rate schedule
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(self.history['epoch'], self.history['lr'], 'purple', linewidth=2.5)
        ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # 4. Gradient norm
        ax4 = plt.subplot(2, 3, 4)
        if self.history['grad_norm']:
            ax4.plot(self.history['epoch'], self.history['grad_norm'], 'orange', 
                    linewidth=2, marker='o', markersize=3)
            ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Gradient Norm', fontsize=12, fontweight='bold')
            ax4.set_title('Gradient Norm Over Time', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3, linestyle='--')
        
        # 5. Overfitting analysis
        ax5 = plt.subplot(2, 3, 5)
        train_val_gap = [t - v for t, v in zip(self.history['train_acc'], self.history['val_acc'])]
        ax5.plot(self.history['epoch'], train_val_gap, 'red', linewidth=2.5)
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
        ax5.fill_between(self.history['epoch'], 0, train_val_gap, 
                        alpha=0.3, color='red' if max(train_val_gap) > 5 else 'green')
        ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Train - Val Acc (%)', fontsize=12, fontweight='bold')
        ax5.set_title('Overfitting Gap Analysis', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3, linestyle='--')
        
        # 6. Training speed
        ax6 = plt.subplot(2, 3, 6)
        if self.history['epoch_time']:
            colors = ['teal' if t < np.mean(self.history['epoch_time']) else 'coral' 
                     for t in self.history['epoch_time']]
            ax6.bar(self.history['epoch'], self.history['epoch_time'], color=colors, alpha=0.7)
            avg_time = np.mean(self.history['epoch_time'])
            ax6.axhline(y=avg_time, color='red', linestyle='--', linewidth=2, 
                       label=f'Avg: {avg_time:.1f}s')
            ax6.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax6.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
            ax6.set_title('Epoch Training Time', fontsize=14, fontweight='bold')
            ax6.legend(fontsize=11, framealpha=0.9)
            ax6.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        plot_path = self.log_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training curves saved to: {plot_path}")
    
    def plot_final_summary(self, config, pretrained_acc):
        """Generate final summary visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Accuracy comparison
        ax1 = axes[0]
        epochs = self.history['epoch']
        
        ax1.plot(epochs, self.history['train_acc'], 'b-', 
                label='Train Acc', linewidth=2.5, marker='o', markersize=4)
        ax1.plot(epochs, self.history['val_acc'], 'r-', 
                label='Val Acc', linewidth=2.5, marker='s', markersize=4)
        ax1.axhline(y=pretrained_acc, color='gray', linestyle='--', 
                   linewidth=2, label=f'Pretrained (CIFAR-100): {pretrained_acc:.2f}%', alpha=0.7)
        ax1.axhline(y=self.best_val_acc, color='green', linestyle='--', 
                   linewidth=2, label=f'Best Val (CIFAR-10): {self.best_val_acc:.2f}%')
        
        ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax1.set_title('Transfer Learning: CIFAR-100 → CIFAR-10', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=11, framealpha=0.9, loc='lower right')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim([min(min(self.history['val_acc']), pretrained_acc) - 5, 
                      max(max(self.history['train_acc']), 100)])
        
        # 2. Performance improvement bar chart
        ax2 = axes[1]
        categories = ['Pretrained\n(CIFAR-100)', 'Final Val\n(CIFAR-10)', 'Best Val\n(CIFAR-10)']
        values = [pretrained_acc, self.history['val_acc'][-1], self.best_val_acc]
        colors = ['gray', 'orange', 'green']
        
        bars = ax2.bar(categories, values, color=colors, alpha=0.7, 
                      edgecolor='black', linewidth=2, width=0.6)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val:.2f}%', ha='center', va='bottom', 
                    fontsize=13, fontweight='bold')
        
        # Add improvement annotations
        improvement_final = values[1] - values[0]
        improvement_best = values[2] - values[0]
        
        ax2.annotate(f'+{improvement_final:.2f}%', 
                    xy=(1, values[1]), xytext=(1, values[0] + (values[1] - values[0])/2),
                    ha='center', fontsize=11, color='darkred', fontweight='bold')
        ax2.annotate(f'+{improvement_best:.2f}%', 
                    xy=(2, values[2]), xytext=(2, values[0] + (values[2] - values[0])/2),
                    ha='center', fontsize=11, color='darkgreen', fontweight='bold')
        
        ax2.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax2.set_title('Performance Comparison', fontsize=16, fontweight='bold')
        ax2.set_ylim([min(values) - 5, max(values) + 5])
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        plot_path = self.log_dir / 'final_summary.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Final summary saved to: {plot_path}")


def get_cifar10_loaders(config: FineTuneConfig):
    """CIFAR-10 data loaders with standard augmentation"""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    # Training augmentation - heavier policy for stronger regularization
    # - RandomCrop + Flip + Rotation
    # - ColorJitter (strong)
    # - RandAugment for diverse photometric/geometric ops
    # - ToTensor + Normalize
    # - RandomErasing to simulate occlusion
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        # RandAugment works on PIL images and applies several random ops
        transforms.RandAugment(num_ops=2, magnitude=11),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # RandomErasing on tensors: helps with occlusion robustness
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
    ])
    
    # Validation transform
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    train_dataset = datasets.CIFAR10(
        root=config.data_root, 
        train=True, 
        transform=train_transform, 
        download=True
    )
    
    val_dataset = datasets.CIFAR10(
        root=config.data_root, 
        train=False, 
        transform=val_transform, 
        download=True
    )
    
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


def get_param_groups(model, config: FineTuneConfig):
    """Create parameter groups with differential learning rates"""
    # Head gets higher learning rate (needs to learn from scratch)
    head_params = []
    backbone_params = []
    
    for name, param in model.named_parameters():
        if 'head' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    return [
        {'params': backbone_params, 'lr': config.base_lr, 'name': 'backbone'},
        {'params': head_params, 'lr': config.base_lr * config.head_lr_multiplier, 'name': 'head'}
    ]


def train_epoch(model, loader, optimizer, scheduler, scaler, criterion, config, epoch):
    """Training epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    grad_norms = []
    
    amp_dtype = torch.bfloat16 if config.amp_dtype == 'bfloat16' else torch.float16
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        
        with autocast(enabled=config.use_amp, dtype=amp_dtype):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        grad_norms.append(grad_norm.item())
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % config.print_freq == 0:
            acc = 100. * correct / total
            lr = optimizer.param_groups[0]['lr']
            avg_loss = total_loss / (batch_idx + 1)
            print(f'  Epoch {epoch:3d} [{batch_idx:4d}/{len(loader):4d}] | '
                  f'Loss: {avg_loss:.4f} | Acc: {acc:6.2f}% | LR: {lr:.7f}')
    
    scheduler.step()
    
    avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
    return total_loss / len(loader), 100. * correct / total, avg_grad_norm


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
    print("HQA-ViT TRANSFER LEARNING: CIFAR-100 → CIFAR-10".center(100))
    print("Classic Fine-tuning (Similar to ImageNet-1K Transfer)".center(100))
    print("="*100)
    
    config = FineTuneConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n🖥️  Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    
    # Create directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = TrainingLogger(config.log_dir)
    
    # Load pretrained model
    print(f"\n📦 Loading pretrained model from: {config.pretrained_path}")
    
    if not os.path.exists(config.pretrained_path):
        raise FileNotFoundError(f"Pretrained model not found at {config.pretrained_path}")
    
    checkpoint = torch.load(config.pretrained_path, map_location='cpu')
    model_config = checkpoint.get('model_config', HQAViTConfig())
    
    # Modify for CIFAR-10 (10 classes instead of 100)
    original_num_classes = model_config.num_classes
    model_config.num_classes = 10
    
    model = HQAViT(model_config).to(device)
    
    # Load weights except head
    pretrained_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    
    # Filter out head weights (head needs to be trained from scratch)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and 'head' not in k}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    pretrained_acc = checkpoint.get('val_acc', 0.0)
    print(f"   ✅ Pretrained on CIFAR-100: {pretrained_acc:.2f}%")
    print(f"   🔄 Transfer to CIFAR-10: {original_num_classes} → 10 classes")
    print(f"   📊 Loaded {len(pretrained_dict)}/{len(model_dict)} layers")
    print(f"   🎯 Classification head initialized randomly for 10 classes")
    
    # Load data
    print(f"\n📁 Loading CIFAR-10...")
    train_loader, val_loader = get_cifar10_loaders(config)
    print(f"   Train: {len(train_loader.dataset):,} samples ({len(train_loader)} batches)")
    print(f"   Val:   {len(val_loader.dataset):,} samples ({len(val_loader)} batches)")
    
    # Optimizer setup
    param_groups = get_param_groups(model, config)
    
    print(f"\n⚙️  Parameter Groups:")
    for group in param_groups:
        num_params = sum(p.numel() for p in group['params'])
        print(f"   {group['name']:<15} LR: {group['lr']:.6f}  |  Params: {num_params:,}")
    
    optimizer = optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay
    )
    
    # Cosine annealing scheduler (after warmup)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs - config.warmup_epochs,
        eta_min=config.min_lr
    )
    
    # Warmup scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config.warmup_epochs
    )
    
    scaler = GradScaler(enabled=config.use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    print(f"\n🔧 Training Configuration:")
    print(f"   Epochs:          {config.epochs} (warmup: {config.warmup_epochs})")
    print(f"   Batch size:      {config.batch_size}")
    print(f"   Base LR:         {config.base_lr:.6f}")
    print(f"   Head LR:         {config.base_lr * config.head_lr_multiplier:.6f} ({config.head_lr_multiplier}x)")
    print(f"   Min LR:          {config.min_lr:.6f}")
    print(f"   Weight decay:    {config.weight_decay}")
    print(f"   Label smoothing: {config.label_smoothing}")
    print(f"   Mixed precision: {config.use_amp} ({config.amp_dtype})")
    print(f"   Gradient clip:   {config.max_grad_norm}")
    
    print(f"\n{'='*100}")
    print("TRAINING STARTED".center(100))
    print(f"{'='*100}\n")
    
    best_acc = 0.0
    best_epoch = 0
    train_start_time = time.time()
    
    # Training loop
    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        
        # Use warmup scheduler for first few epochs, then cosine
        current_scheduler = warmup_scheduler if epoch <= config.warmup_epochs else scheduler
        
        train_loss, train_acc, grad_norm = train_epoch(
            model, train_loader, optimizer, current_scheduler, 
            scaler, criterion, config, epoch
        )
        
        if epoch % config.eval_freq == 0:
            val_loss, val_acc = validate(model, val_loader, criterion)
            
            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log metrics
            logger.log_epoch(epoch, {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': current_lr,
                'grad_norm': grad_norm,
                'epoch_time': epoch_time
            })
            
            # Print summary
            print(f"\n{'='*100}")
            print(f"EPOCH {epoch}/{config.epochs} SUMMARY".center(100))
            print(f"{'='*100}")
            print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:6.2f}%")
            print(f"  Val Loss:   {val_loss:.4f}  |  Val Acc:   {val_acc:6.2f}%")
            print(f"  Learning Rate: {current_lr:.7f}  |  Grad Norm: {grad_norm:.4f}")
            print(f"  Epoch Time: {epoch_time:.1f}s  ({epoch_time/60:.2f} min)")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'pretrained_acc': pretrained_acc,
                    'config': config,
                    'model_config': model_config,
                }, f"{config.checkpoint_dir}/best_cifar10.pth")
                print(f"  🌟 NEW BEST! Val Acc: {best_acc:.2f}% (saved)")
            else:
                print(f"  📊 Best Val Acc: {best_acc:.2f}% (epoch {best_epoch})")
            
            print(f"{'='*100}\n")
            
            # Generate plots periodically
            if epoch % 5 == 0 or epoch == config.epochs:
                logger.plot_training_curves()
        
        # Save periodic checkpoints
        if epoch % config.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f"{config.checkpoint_dir}/checkpoint_epoch_{epoch}.pth")
    
    # Training complete
    total_time = time.time() - train_start_time
    
    # Save final metrics and plots
    logger.save_metrics()
    logger.plot_training_curves()
    logger.plot_final_summary(config, pretrained_acc)
    
    # Final summary
    print(f"\n{'='*100}")
    print("TRAINING COMPLETE".center(100))
    print(f"{'='*100}")
    print(f"\n📊 Final Results:")
    print(f"   Pretrained (CIFAR-100):  {pretrained_acc:.2f}%")
    print(f"   Best Val (CIFAR-10):     {best_acc:.2f}%  (epoch {best_epoch})")
    print(f"   Improvement:             +{best_acc - pretrained_acc:.2f}%")
    print(f"   Final Train Acc:         {logger.history['train_acc'][-1]:.2f}%")
    print(f"   Final Val Acc:           {logger.history['val_acc'][-1]:.2f}%")
    print(f"   Total Training Time:     {total_time/3600:.2f} hours")
    print(f"\n📁 Saved Files:")
    print(f"   Best Model:     {config.checkpoint_dir}/best_cifar10.pth")
    print(f"   Training Plots: {config.log_dir}/training_curves.png")
    print(f"   Final Summary:  {config.log_dir}/final_summary.png")
    print(f"   Metrics JSON:   {config.log_dir}/training_metrics.json")
    print(f"\n{'='*100}\n")
    
    return logger


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
        logger = main()
        print("✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()