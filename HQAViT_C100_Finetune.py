"""
HQA-ViT Fine-tuning Script for CIFAR-100
Fine-tune pretrained model with layer-wise learning rates and advanced techniques
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
from copy import deepcopy

try:
    from HQAViT_CIFAR100 import HQAViT, HQAViTConfig, ModelEMA, GradientMonitor
except ImportError:
    print("Error: Cannot import model classes. Make sure HQAViT_CIFAR100.py is in the same directory.")
    raise


# Placeholder for checkpoints that saved a `TrainingConfig` in the original training script
@dataclass
class TrainingConfig:
    pass


@dataclass
class FineTuneConfig:
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    
    epochs: int = 50
    warmup_epochs: int = 5
    
    # MUCH lower learning rates to prevent overfitting
    base_lr: float = 5e-6  # Reduced from 1e-5
    head_lr_multiplier: float = 5.0  # Reduced from 10.0
    layer_lr_decay: float = 0.8  # More aggressive decay
    min_lr: float = 1e-8
    
    # Stronger regularization
    weight_decay: float = 0.05  # Increased from 0.01
    label_smoothing: float = 0.15  # Increased from 0.08
    
    max_grad_norm: float = 0.5  # Tighter clipping
    grad_clip_mode: str = 'norm'
    
    use_amp: bool = True
    amp_dtype: str = 'bfloat16'
    
    use_ema: bool = True
    ema_decay: float = 0.9998  # Slower EMA updates
    
    use_tta: bool = True
    tta_transforms: int = 5
    
    # Mixup/Cutmix for regularization
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.5
    
    pretrained_path: str = "./checkpoints_hqavit/best_model_ema.pth"
    data_root: str = "./data"
    checkpoint_dir: str = "./checkpoints_finetuned"
    
    print_freq: int = 20
    eval_freq: int = 1
    save_freq: int = 10


def get_finetune_loaders(config: FineTuneConfig):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    
    # OPTION 2: Moderate augmentation with light RandAugment
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.08)
        ], p=0.6),
        transforms.RandAugment(num_ops=2, magnitude=8),  # Light RandAugment
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.25), ratio=(0.3, 3.3))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    tta_transforms_list = [
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([
            transforms.RandomCrop(32, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomCrop(32, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    ]
    
    train_dataset = datasets.CIFAR100(
        root=config.data_root, 
        train=True, 
        transform=train_transform, 
        download=True
    )
    
    val_dataset = datasets.CIFAR100(
        root=config.data_root, 
        train=False, 
        transform=val_transform, 
        download=True
    )
    
    tta_datasets = [
        datasets.CIFAR100(root=config.data_root, train=False, transform=t, download=False)
        for t in tta_transforms_list[:config.tta_transforms]
    ]
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2 if config.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2 if config.num_workers > 0 else None
    )
    
    tta_loaders = [
        DataLoader(
            ds,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        for ds in tta_datasets
    ]
    
    return train_loader, val_loader, tta_loaders


def get_layer_wise_params(model, config: FineTuneConfig):
    param_groups = []
    assigned_ids = set()

    def add_group(name: str, params_list: list, lr: float):
        filtered = [p for p in params_list if id(p) not in assigned_ids]
        if not filtered:
            return
        for p in filtered:
            assigned_ids.add(id(p))
        param_groups.append({'params': filtered, 'lr': lr, 'name': name})

    # Head (highest LR)
    head_params = [param for n, param in model.named_parameters() if 'head' in n or ('norm' in n and 'head' in n)]
    add_group('head', head_params, config.base_lr * config.head_lr_multiplier)

    # Transformer stages (layer-wise decay)
    num_stages = 4
    for stage_idx in range(num_stages, 0, -1):
        stage_params = [param for n, param in model.named_parameters() if f'stage{stage_idx}' in n]
        lr_scale = config.layer_lr_decay ** (num_stages - stage_idx)
        add_group(f'stage{stage_idx}', stage_params, config.base_lr * lr_scale)

    # Fusion modules
    fusion_params = [param for n, param in model.named_parameters() if 'fuse' in n or 'rrcv' in n or 'lmfa' in n]
    add_group('fusion', fusion_params, config.base_lr * 0.5)

    # CNN stem (lowest LR)
    cnn_params = [param for n, param in model.named_parameters() if 'cnn_stem' in n]
    add_group('cnn_stem', cnn_params, config.base_lr * 0.1)

    # Embeddings
    embed_params = [param for n, param in model.named_parameters() if 'patch_embed' in n or 'pos_embed' in n or 'global_bank' in n]
    add_group('embeddings', embed_params, config.base_lr * 0.3)

    # Remaining parameters
    remaining_params = [p for p in model.parameters() if id(p) not in assigned_ids]
    add_group('remaining', remaining_params, config.base_lr)

    return param_groups


def rand_bbox(size, lam):
    """Generate random bbox for CutMix"""
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return int(x1), int(y1), int(x2), int(y2)


def train_epoch(model, loader, optimizer, scheduler, scaler, config, epoch, monitor, model_ema=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        
        # Apply Mixup/CutMix for regularization
        use_mix = None
        lam = 1.0
        if config.use_mixup and np.random.rand() < config.mix_prob:
            # CutMix
            rand_index = torch.randperm(inputs.size(0)).cuda()
            lam = np.random.beta(config.cutmix_alpha, config.cutmix_alpha)
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[rand_index, :, bby1:bby2, bbx1:bbx2]
            targets_a, targets_b = targets, targets[rand_index]
            W, H = inputs.size(3), inputs.size(2)
            lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / float(W * H))
            use_mix = 'cutmix'
        elif np.random.rand() < config.mix_prob / 2:
            # Mixup
            rand_index = torch.randperm(inputs.size(0)).cuda()
            lam = np.random.beta(config.mixup_alpha, config.mixup_alpha)
            inputs = lam * inputs + (1 - lam) * inputs[rand_index]
            targets_a, targets_b = targets, targets[rand_index]
            use_mix = 'mixup'
        
        amp_dtype = torch.bfloat16 if config.amp_dtype == 'bfloat16' else torch.float16
        
        with autocast(enabled=config.use_amp, dtype=amp_dtype):
            outputs = model(inputs)
            
            if use_mix is None:
                loss = criterion(outputs, targets)
            else:
                loss = lam * criterion(outputs, targets_a) + (1.0 - lam) * criterion(outputs, targets_b)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        if config.grad_clip_mode == 'norm':
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        if batch_idx % 100 == 0:
            grad_norm, _, _, _ = monitor.log_gradients(model, detailed=False)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        
        if model_ema is not None:
            model_ema.update(model)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % config.print_freq == 0:
            acc = 100. * correct / total
            lr = optimizer.param_groups[0]['lr']
            avg_loss = total_loss / (batch_idx + 1)
            print(f'Epoch {epoch:3d} [{batch_idx:4d}/{len(loader):4d}] | '
                  f'Loss: {avg_loss:.4f} | Acc: {acc:6.2f}% | LR: {lr:.7f}')
    
    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for inputs, targets in loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def validate_tta(model, tta_loaders):
    model.eval()
    
    all_predictions = []
    all_targets = None

    for loader_idx, loader in enumerate(tta_loaders):
        batch_predictions = []
        targets_list = []

        for inputs, targets in loader:
            inputs = inputs.cuda()
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            batch_predictions.append(probs.cpu())

            if loader_idx == 0:
                targets_list.append(targets.cpu())

        all_predictions.append(torch.cat(batch_predictions, dim=0))

        if loader_idx == 0:
            if len(targets_list) > 0:
                all_targets = torch.cat(targets_list, dim=0)
    
    if len(all_predictions) == 0:
        return 0.0

    avg_predictions = torch.stack(all_predictions).mean(dim=0)
    predicted = avg_predictions.argmax(dim=1)

    if all_targets is None:
        return 0.0

    correct = predicted.eq(all_targets).sum().item()
    total = all_targets.size(0)
    acc = 100. * correct / total
    
    return acc


def main():
    print("\n" + "="*100)
    print("HQA-ViT FINE-TUNING ON CIFAR-100".center(100))
    print("="*100)
    
    config = FineTuneConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDevice: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading pretrained model from: {config.pretrained_path}")
    
    if not os.path.exists(config.pretrained_path):
        raise FileNotFoundError(f"Pretrained model not found at {config.pretrained_path}")
    
    checkpoint = torch.load(config.pretrained_path, map_location='cpu')
    model_config = checkpoint.get('model_config', HQAViTConfig())
    
    model = HQAViT(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    pretrained_acc = checkpoint.get('val_acc', 0.0)
    print(f"   Pretrained accuracy: {pretrained_acc:.2f}%")
    
    print(f"\nLoading CIFAR-100...")
    train_loader, val_loader, tta_loaders = get_finetune_loaders(config)
    print(f"   Train: {len(train_loader.dataset):,} samples")
    print(f"   Val: {len(val_loader.dataset):,} samples")
    
    param_groups = get_layer_wise_params(model, config)
    
    print(f"\nParameter Groups:")
    for group in param_groups:
        num_params = sum(p.numel() for p in group['params'])
        print(f"   {group['name']:<20} LR: {group['lr']:.7f}  Params: {num_params:,}")
    
    optimizer = optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay
    )
    
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = steps_per_epoch * config.warmup_epochs
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=total_steps - warmup_steps,
        T_mult=1,
        eta_min=config.min_lr
    )
    
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_steps
    )
    
    scaler = GradScaler(enabled=config.use_amp)
    monitor = GradientMonitor()
    
    model_ema = None
    if config.use_ema:
        print(f"\nEMA enabled (decay={config.ema_decay})")
        model_ema = ModelEMA(model, decay=config.ema_decay, device=device)
    
    print(f"\nConfiguration:")
    print(f"   Epochs: {config.epochs} (warmup: {config.warmup_epochs})")
    print(f"   Base LR: {config.base_lr} → Min LR: {config.min_lr}")
    print(f"   Weight decay: {config.weight_decay}")
    print(f"   Label smoothing: {config.label_smoothing}")
    print(f"   Mixup/CutMix: {config.use_mixup} (prob: {config.mix_prob})")
    print(f"   TTA: {config.use_tta} ({config.tta_transforms} transforms)")
    print(f"\n⚠️  Strategy: Combat overfitting with:")
    print(f"   - Lower LR (5e-6 base)")
    print(f"   - Higher weight decay (0.05)")
    print(f"   - Stronger label smoothing (0.15)")
    print(f"   - Mixup/CutMix regularization")
    print(f"   - Light RandAugment (2 ops, mag 8)")
    
    print(f"\n{'='*100}")
    print("FINE-TUNING STARTED".center(100))
    print(f"{'='*100}\n")
    
    best_acc = pretrained_acc
    best_ema_acc = pretrained_acc
    best_tta_acc = 0
    train_start_time = time.time()
    
    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        
        current_scheduler = warmup_scheduler if epoch <= config.warmup_epochs else scheduler
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, 
            current_scheduler, scaler, config, epoch, monitor, model_ema
        )
        
        if epoch % config.eval_freq == 0:
            val_loss, val_acc = validate(model, val_loader)
            
            ema_val_loss, ema_val_acc = 0, 0
            if model_ema is not None:
                ema_val_loss, ema_val_acc = validate(model_ema.ema, val_loader)
            
            tta_acc = 0
            tta_ema_acc = 0
            if config.use_tta:
                tta_acc = validate_tta(model, tta_loaders)
                if model_ema is not None:
                    tta_ema_acc = validate_tta(model_ema.ema, tta_loaders)
            
            epoch_time = time.time() - epoch_start_time
            
            print(f"\n{'='*100}")
            print(f"EPOCH {epoch}/{config.epochs} SUMMARY".center(100))
            print(f"{'='*100}")
            print(f"{'Metric':<25} {'Train':>12} {'Val':>12} {'EMA':>12} {'TTA':>12} {'TTA+EMA':>12}")
            print("-"*100)
            print(f"{'Loss':<25} {train_loss:>12.4f} {val_loss:>12.4f} {ema_val_loss:>12.4f}")
            print(f"{'Accuracy (%)':<25} {train_acc:>12.2f} {val_acc:>12.2f} {ema_val_acc:>12.2f} {tta_acc:>12.2f} {tta_ema_acc:>12.2f}")
            print(f"{'Best (%)':<25} {'':>12} {max(best_acc, val_acc):>12.2f} {max(best_ema_acc, ema_val_acc):>12.2f} {max(best_tta_acc, tta_acc, tta_ema_acc):>12.2f}")
            print(f"{'Improvement':<25} {'':>12} {val_acc - pretrained_acc:>+12.2f} {ema_val_acc - pretrained_acc:>+12.2f} {tta_acc - pretrained_acc:>+12.2f} {tta_ema_acc - pretrained_acc:>+12.2f}")
            print(f"{'Time (min)':<25} {epoch_time/60:>12.1f}")
            print(f"{'='*100}\n")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'pretrained_acc': pretrained_acc,
                    'improvement': val_acc - pretrained_acc,
                    'config': config,
                    'model_config': model_config,
                }, f"{config.checkpoint_dir}/best_finetuned.pth")
                print(f"Best model saved! Val Acc: {best_acc:.2f}% (+{best_acc - pretrained_acc:.2f}%)\n")
            
            if model_ema is not None and ema_val_acc > best_ema_acc:
                best_ema_acc = ema_val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_ema.ema.state_dict(),
                    'val_acc': ema_val_acc,
                    'pretrained_acc': pretrained_acc,
                    'improvement': ema_val_acc - pretrained_acc,
                    'config': config,
                    'model_config': model_config,
                }, f"{config.checkpoint_dir}/best_finetuned_ema.pth")
                print(f"Best EMA saved! Val Acc: {best_ema_acc:.2f}% (+{best_ema_acc - pretrained_acc:.2f}%)\n")
            
            if config.use_tta:
                best_tta_acc = max(best_tta_acc, tta_acc, tta_ema_acc)
        
        if epoch % config.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{config.checkpoint_dir}/checkpoint_ft_epoch_{epoch}.pth")
    
    total_time = time.time() - train_start_time
    
    print(f"\n{'='*100}")
    print("FINE-TUNING COMPLETE".center(100))
    print(f"{'='*100}")
    print(f"\n{'Results':-^100}")
    print(f"   Pretrained:     {pretrained_acc:.2f}%")
    print(f"   Best Val:       {best_acc:.2f}% (+{best_acc - pretrained_acc:.2f}%)")
    print(f"   Best EMA:       {best_ema_acc:.2f}% (+{best_ema_acc - pretrained_acc:.2f}%)")
    print(f"   Best TTA:       {best_tta_acc:.2f}% (+{best_tta_acc - pretrained_acc:.2f}%)")
    print(f"   Total time:     {total_time/3600:.2f} hours")
    print(f"\n{'Checkpoints':-^100}")
    print(f"   {config.checkpoint_dir}/best_finetuned.pth")
    print(f"   {config.checkpoint_dir}/best_finetuned_ema.pth")
    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    main()