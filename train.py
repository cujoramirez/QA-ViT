"""
QAViT Training Script for CIFAR-100
CVPR Paper Implementation - Pretraining from Scratch

Features:
- Mixed precision training (AMP)
- Cosine annealing learning rate schedule
- Checkpointing and resumption
- Logging and visualization
- Support for Tiny and Small models
"""

# Disable TensorFlow and suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import time
import json
import math
from datetime import datetime
from pathlib import Path

from QAViT import (
    qavit_tiny, qavit_small,
    build_dataloader, DATASET_CONFIGS
)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth'):
    """Save checkpoint to disk"""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None):
    """Load checkpoint from disk"""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, 0
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_acc1 = checkpoint.get('best_acc1', 0)
    
    print(f"Loaded checkpoint from epoch {epoch} (best acc1: {best_acc1:.2f}%)")
    return epoch, best_acc1


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, device, args, writer, checkpoint_dir):
    """Train for one epoch"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.train()
    end = time.time()
    
    for i, (images, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # Calculate global step for logging
        global_step = epoch * len(train_loader) + i
        
        # Mixed precision training
        with torch.amp.autocast('cuda', enabled=args.amp):
            output = model(images)
            loss = criterion(output, target)
        
        # Measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))
        
        # Compute gradient and do optimizer step
        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(model.parameters(), args.grad_clip)
                # Log gradient norm every 100 iterations for monitoring
                if i % 100 == 0:
                    writer.add_scalar('Train/GradNorm', grad_norm.item(), global_step)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                grad_norm = clip_grad_norm_(model.parameters(), args.grad_clip)
                if i % 100 == 0:
                    writer.add_scalar('Train/GradNorm', grad_norm.item(), global_step)
            optimizer.step()
        
        # Check for NaN/Inf and abort early
        if not torch.isfinite(loss):
            print(f"\n⚠️  WARNING: Loss became {loss.item()} at epoch {epoch}, iteration {i}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"Saving emergency checkpoint before crash...")
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if args.amp else None,
                'best_acc1': 0,
            }, checkpoint_dir, filename='emergency_nan.pth')
            raise ValueError(f"Training unstable: Loss = {loss.item()}")
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Log to tensorboard
        if i % args.print_freq == 0:
            writer.add_scalar('Train/Loss', losses.val, global_step)
            writer.add_scalar('Train/Acc@1', top1.val, global_step)
            writer.add_scalar('Train/Acc@5', top5.val, global_step)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)
            
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')
    
    return losses.avg, top1.avg, top5.avg


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Compute output
            output = model(images)
            loss = criterion(output, target)
            
            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        print(f'Validation: \t'
              f'Time {batch_time.avg:.3f}\t'
              f'Loss {losses.avg:.4f}\t'
              f'Acc@1 {top1.avg:.3f}\t'
              f'Acc@5 {top5.avg:.3f}')
    
    return losses.avg, top1.avg, top5.avg


def main():
    parser = argparse.ArgumentParser(description='QAViT Training on CIFAR-100')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='tiny', choices=['tiny', 'small'],
                        help='Model size: tiny or small')
    parser.add_argument('--dataset', type=str, default='cifar100', 
                        choices=['cifar100', 'imagenet'],
                        help='Dataset to use')
    parser.add_argument('--data-path', type=str, default='./data',
                        help='Path to dataset')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Initial learning rate (lowered for stability)')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                        help='Number of warmup epochs (reduced for faster convergence)')
    
    # Optimization
    parser.add_argument('--optimizer', type=str, default='adamw', 
                        choices=['adamw', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping max norm')
    
    # Checkpointing
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Logging
    parser.add_argument('--print-freq', type=int, default=50,
                        help='Print frequency (batches)')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Directory for tensorboard logs')
    
    # Hardware
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"qavit_{args.model}_{args.dataset}_{timestamp}"
    log_path = os.path.join(args.log_dir, exp_name)
    checkpoint_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_path)
    
    # Create model
    print(f"\n{'='*80}")
    print(f"Creating QAViT-{args.model.upper()} model...")
    print(f"{'='*80}\n")
    
    if args.model == 'tiny':
        model = qavit_tiny(num_classes=100 if args.dataset == 'cifar100' else 1000)
    else:
        model = qavit_small(num_classes=100 if args.dataset == 'cifar100' else 1000)
    
    model = model.to(device)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {num_params:,}")
    print(f"Trainable Parameters: {num_trainable:,}")
    
    # Create dataloaders
    print(f"\nCreating {args.dataset.upper()} dataloaders...")
    # Get base config and override parameters
    import copy
    dataset_config = copy.copy(DATASET_CONFIGS[args.dataset])
    dataset_config.data_root = args.data_path
    dataset_config.batch_size = args.batch_size
    dataset_config.num_workers = args.num_workers
    train_loader = build_dataloader(dataset_config, is_train=True)
    val_loader = build_dataloader(dataset_config, is_train=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Loss function with label smoothing for stability
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    
    # Optimizer
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    
    # Learning rate scheduler (cosine annealing with warmup)
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return epoch / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        cosine = 0.5 * (1 + math.cos(progress * math.pi))
        return max(args.min_lr / args.lr, cosine)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=args.amp)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc1 = 0
    if args.resume:
        start_epoch, best_acc1 = load_checkpoint(
            args.resume, model, optimizer, scaler
        )
    
    print(f"\n{'='*80}")
    print(f"Starting training from epoch {start_epoch} to {args.epochs}")
    print(f"{'='*80}\n")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            epoch, device, args, writer, checkpoint_dir
        )
        
        # Validate
        val_loss, val_acc1, val_acc5 = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch metrics
        writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
        writer.add_scalar('Epoch/Train_Acc@1', train_acc1, epoch)
        writer.add_scalar('Epoch/Train_Acc@5', train_acc5, epoch)
        writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
        writer.add_scalar('Epoch/Val_Acc@1', val_acc1, epoch)
        writer.add_scalar('Epoch/Val_Acc@5', val_acc5, epoch)
        writer.add_scalar('Epoch/LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        
        if (epoch + 1) % args.save_freq == 0 or is_best:
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_acc1': best_acc1,
                'config': vars(args)
            }
            
            save_checkpoint(state, checkpoint_dir, f'epoch_{epoch+1}.pth')
            
            if is_best:
                save_checkpoint(state, checkpoint_dir, 'best.pth')
                print(f"★ New best validation accuracy: {best_acc1:.2f}%")
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_acc1:.2f}%")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_path}")
    print(f"{'='*80}\n")
    
    writer.close()


if __name__ == '__main__':
    main()
