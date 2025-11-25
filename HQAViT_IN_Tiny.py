"""
HQA-ViT: Hybrid Quad-Attention Vision Transformer
Tiny ImageNet Pretraining with Training Visualization
UPGRADED VERSION - Target: 70-80% Accuracy
Integrates: CNNStem + LMFAdapter + RRCV + SplitFusion + TokenLearner + EMA
Based on CTA-Net paper architecture
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import math
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from collections import defaultdict

# ============================================================================
# FlashAttention Check
# ============================================================================
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

# ============================================================================
# Configuration - UPGRADED
# ============================================================================
@dataclass
class HQAViTConfig:
    """HQA-ViT configuration for Tiny ImageNet - UPGRADED"""
    img_size: int = 64
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 200
    embed_dim: int = 192  # Kept at 192 for VRAM constraint
    depth: int = 12  # UPGRADED: 8 -> 12 (hierarchical [2,2,6,2])
    num_heads: int = 4
    compress_ratio: int = 4
    bottleneck_ratio: int = 2
    mlp_ratio: float = 0.5
    global_bank_size: int = 16
    dropout: float = 0.1
    drop_path: float = 0.2  # UPGRADED: 0.1 -> 0.2 for stability
    window_size: int = 4
    dilation_factors: Tuple[int, ...] = (1, 2)
    landmark_pooling_stride: int = 2
    num_channel_groups: int = 6
    linformer_k: int = 32

    # CNN Stem channels for stages
    cnn_c2: int = 64
    cnn_c3: int = 128
    cnn_c4: int = 256

    # RRCV config
    rrcv_channels: int = 64
    rrcv_num_blocks: int = 1

    # TokenLearner config - UPGRADED
    use_token_learner: bool = True
    num_learned_tokens: int = 64  # UPGRADED: 16 -> 64 (8x8 grid, quadrupled!)

    # Fusion config
    fusion_stages: Tuple[int, ...] = (2, 3, 4)


@dataclass
class TrainingConfig:
    """Training configuration - UPGRADED"""
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    pin_memory: bool = True

    epochs: int = 450
    warmup_epochs: int = 30  # UPGRADED: 20 -> 30 for deeper network
    base_lr: float = 5e-4
    min_lr: float = 5e-6
    weight_decay: float = 0.05

    label_smoothing: float = 0.1

    max_grad_norm: float = 1.0
    grad_clip_mode: str = 'norm'

    print_freq: int = 50
    eval_freq: int = 1
    save_freq: int = 10

    use_amp: bool = True
    amp_dtype: str = 'bfloat16'
    use_compile: bool = False
    compile_mode: str = 'default'

    # EMA config
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_decay_warmup: float = 0.99

    data_root: str = "./data/tiny-imagenet-200"
    checkpoint_dir: str = "./checkpoints_hqavit_tinyimagenet"
    plot_dir: str = "./plots_hqavit_tinyimagenet"

    # Mixup / CutMix settings
    use_mixup: bool = True
    mixup_alpha: float = 0.8
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.5


# ============================================================================
# Tiny ImageNet Dataset
# ============================================================================
class TinyImageNetDataset(Dataset):
    """Tiny ImageNet dataset loader"""
    def __init__(self, root_dir, split='train', transform=None, download=False):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        if download:
            self._download()

        self.samples = []
        self.targets = []
        self.class_to_idx = {}

        self._load_data()

    def _download(self):
        """Download Tiny ImageNet if not present"""
        if self.root_dir.exists():
            print(f"✓ Tiny ImageNet already exists at {self.root_dir}")
            return

        print(f"⬇ Downloading Tiny ImageNet to {self.root_dir}...")
        import urllib.request
        import zipfile

        self.root_dir.parent.mkdir(parents=True, exist_ok=True)
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = self.root_dir.parent / "tiny-imagenet-200.zip"

        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root_dir.parent)

        zip_path.unlink()
        print(f"✓ Download complete!")

    def _load_data(self):
        """Load image paths and labels"""
        if self.split == 'train':
            train_dir = self.root_dir / 'train'
            classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

            for cls in classes:
                cls_dir = train_dir / cls / 'images'
                for img_path in cls_dir.glob('*.JPEG'):
                    self.samples.append(img_path)
                    self.targets.append(self.class_to_idx[cls])

        elif self.split == 'val':
            val_dir = self.root_dir / 'val'

            val_annotations = val_dir / 'val_annotations.txt'
            img_to_class = {}
            with open(val_annotations, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_to_class[parts[0]] = parts[1]

            train_dir = self.root_dir / 'train'
            classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

            val_images_dir = val_dir / 'images'
            for img_path in val_images_dir.glob('*.JPEG'):
                cls = img_to_class[img_path.name]
                self.samples.append(img_path)
                self.targets.append(self.class_to_idx[cls])

        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target


# ============================================================================
# Training History Tracker
# ============================================================================
class TrainingHistory:
    """Track and plot training metrics"""
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = defaultdict(list)
        self.epoch_times = []

    def update(self, epoch, **metrics):
        """Update history with metrics"""
        self.history['epoch'].append(epoch)
        for key, value in metrics.items():
            self.history[key].append(value)

    def plot_all(self):
        """Generate all plots"""
        self._plot_loss()
        self._plot_accuracy()
        self._plot_lr()
        self._plot_gradients()
        self._plot_ema_distance()
        self._plot_combined()

    def _plot_loss(self):
        """Plot training and validation loss"""
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = self.history['epoch']
        if 'train_loss' in self.history:
            ax.plot(epochs, self.history['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in self.history:
            ax.plot(epochs, self.history['val_loss'], label='Val Loss', linewidth=2)
        if 'ema_val_loss' in self.history:
            ax.plot(epochs, self.history['ema_val_loss'], label='EMA Val Loss',
                    linewidth=2, linestyle='--', alpha=0.7)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'loss_curve.png', dpi=150)
        plt.close()

    def _plot_accuracy(self):
        """Plot training and validation accuracy"""
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = self.history['epoch']
        if 'train_acc' in self.history:
            ax.plot(epochs, self.history['train_acc'], label='Train Acc', linewidth=2)
        if 'val_acc' in self.history:
            ax.plot(epochs, self.history['val_acc'], label='Val Acc', linewidth=2)
        if 'ema_val_acc' in self.history:
            ax.plot(epochs, self.history['ema_val_acc'], label='EMA Val Acc',
                    linewidth=2, linestyle='--', alpha=0.7)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'accuracy_curve.png', dpi=150)
        plt.close()

    def _plot_lr(self):
        """Plot learning rate schedule"""
        if 'lr' not in self.history:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = self.history['epoch']
        ax.plot(epochs, self.history['lr'], linewidth=2, color='orange')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'lr_schedule.png', dpi=150)
        plt.close()

    def _plot_gradients(self):
        """Plot gradient norms"""
        if 'grad_norm' not in self.history:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = self.history['epoch']
        ax.plot(epochs, self.history['grad_norm'], linewidth=2, color='red', alpha=0.7)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Gradient Norm', fontsize=12)
        ax.set_title('Gradient Norm Over Training', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'gradient_norm.png', dpi=150)
        plt.close()

    def _plot_ema_distance(self):
        """Plot EMA tracking distance"""
        if 'ema_param_dist' not in self.history:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = self.history['epoch']
        ax.plot(epochs, self.history['ema_param_dist'], linewidth=2,
                color='purple', label='Param Distance')
        if 'ema_buffer_dist' in self.history:
            ax.plot(epochs, self.history['ema_buffer_dist'], linewidth=2,
                    color='green', label='Buffer Distance', alpha=0.7)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('L2 Distance', fontsize=12)
        ax.set_title('EMA Model Tracking Distance', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'ema_distance.png', dpi=150)
        plt.close()

    def _plot_combined(self):
        """Combined dashboard plot"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        epochs = self.history['epoch']

        # Loss
        ax1 = fig.add_subplot(gs[0, 0])
        if 'train_loss' in self.history:
            ax1.plot(epochs, self.history['train_loss'], label='Train', linewidth=2)
        if 'val_loss' in self.history:
            ax1.plot(epochs, self.history['val_loss'], label='Val', linewidth=2)
        if 'ema_val_loss' in self.history:
            ax1.plot(epochs, self.history['ema_val_loss'], label='EMA Val',
                    linewidth=2, linestyle='--')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2 = fig.add_subplot(gs[0, 1])
        if 'train_acc' in self.history:
            ax2.plot(epochs, self.history['train_acc'], label='Train', linewidth=2)
        if 'val_acc' in self.history:
            ax2.plot(epochs, self.history['val_acc'], label='Val', linewidth=2)
        if 'ema_val_acc' in self.history:
            ax2.plot(epochs, self.history['ema_val_acc'], label='EMA Val',
                    linewidth=2, linestyle='--')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Curves', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Learning Rate
        ax3 = fig.add_subplot(gs[1, 0])
        if 'lr' in self.history:
            ax3.plot(epochs, self.history['lr'], linewidth=2, color='orange')
        ax3.set_yscale('log')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Gradient Norm
        ax4 = fig.add_subplot(gs[1, 1])
        if 'grad_norm' in self.history:
            ax4.plot(epochs, self.history['grad_norm'], linewidth=2, color='red', alpha=0.7)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Gradient Norm')
        ax4.set_title('Gradient Norm', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # EMA Distance
        ax5 = fig.add_subplot(gs[2, 0])
        if 'ema_param_dist' in self.history:
            ax5.plot(epochs, self.history['ema_param_dist'], linewidth=2,
                    color='purple', label='Param')
        if 'ema_buffer_dist' in self.history:
            ax5.plot(epochs, self.history['ema_buffer_dist'], linewidth=2,
                    color='green', label='Buffer', alpha=0.7)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('L2 Distance')
        ax5.set_title('EMA Tracking Distance', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Best Metrics Summary
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')

        best_metrics = []
        if 'val_acc' in self.history:
            best_val = max(self.history['val_acc'])
            best_metrics.append(f"Best Val Acc: {best_val:.2f}%")
        if 'ema_val_acc' in self.history:
            best_ema = max(self.history['ema_val_acc'])
            best_metrics.append(f"Best EMA Val Acc: {best_ema:.2f}%")
        if 'val_loss' in self.history:
            best_loss = min(self.history['val_loss'])
            best_metrics.append(f"Best Val Loss: {best_loss:.4f}")

        summary_text = "\n\n".join(best_metrics)
        ax6.text(0.5, 0.5, summary_text,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14,
                fontweight='bold',
                transform=ax6.transAxes)
        ax6.set_title('Best Metrics', fontweight='bold', fontsize=14)

        plt.suptitle('HQA-ViT Training Dashboard - UPGRADED', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(self.save_dir / 'training_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================================
# EMA (Exponential Moving Average)
# ============================================================================
class ModelEMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional[str] = None):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.device = device
        if device is not None:
            self.ema.to(device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters and buffers"""
        ema_params = dict(self.ema.named_parameters())
        model_params = dict(model.named_parameters())

        for name, ema_p in ema_params.items():
            if name in model_params:
                model_p = model_params[name].detach()
                ema_p.mul_(self.decay).add_(model_p, alpha=1.0 - self.decay)

        ema_buffers = dict(self.ema.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, ema_buf in ema_buffers.items():
            if name in model_buffers:
                ema_buf.copy_(model_buffers[name])

    @torch.no_grad()
    def compute_distance(self, model: nn.Module):
        """Compute L2 distance between EMA and model parameters"""
        param_dist = 0.0
        buffer_dist = 0.0

        ema_params = dict(self.ema.named_parameters())
        model_params = dict(model.named_parameters())
        for name, ema_p in ema_params.items():
            if name in model_params:
                param_dist += (ema_p - model_params[name]).norm().item() ** 2
        param_dist = param_dist ** 0.5

        ema_buffers = dict(self.ema.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, ema_buf in ema_buffers.items():
            if name in model_buffers and ema_buf.dtype == torch.float32:
                buffer_dist += (ema_buf - model_buffers[name]).norm().item() ** 2
        buffer_dist = buffer_dist ** 0.5

        return param_dist, buffer_dist

    def set_decay(self, decay: float):
        """Update decay rate"""
        self.decay = decay


# ============================================================================
# Gradient Monitoring
# ============================================================================
class GradientMonitor:
    """Enhanced gradient monitoring"""
    def __init__(self):
        self.grad_norms = []
        self.param_norms = []
        self.layer_grad_history = {}
        self.explosion_count = 0

    def log_gradients(self, model, detailed=False):
        total_norm = 0.0
        param_norm = 0.0
        grad_stats = {}
        layer_stats = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                g_norm = param.grad.norm().item()
                total_norm += g_norm ** 2

                p_norm = param.norm().item()
                param_norm += p_norm ** 2

                layer_name = '.'.join(name.split('.')[:2])
                if layer_name not in layer_stats:
                    layer_stats[layer_name] = {'grad_norm': 0, 'param_norm': 0, 'count': 0}
                layer_stats[layer_name]['grad_norm'] += g_norm
                layer_stats[layer_name]['param_norm'] += p_norm
                layer_stats[layer_name]['count'] += 1

                if g_norm > 10.0 or torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    grad_stats[name] = {
                        'grad_norm': g_norm,
                        'grad_mean': param.grad.mean().item(),
                        'grad_max': param.grad.abs().max().item(),
                        'grad_min': param.grad.abs().min().item(),
                        'has_nan': torch.isnan(param.grad).any().item(),
                        'has_inf': torch.isinf(param.grad).any().item(),
                        'param_norm': p_norm,
                    }

        total_norm = total_norm ** 0.5
        param_norm = param_norm ** 0.5

        self.grad_norms.append(total_norm)
        self.param_norms.append(param_norm)

        if detailed:
            for layer, stats in layer_stats.items():
                if layer not in self.layer_grad_history:
                    self.layer_grad_history[layer] = []
                self.layer_grad_history[layer].append(stats['grad_norm'] / max(stats['count'], 1))

        return total_norm, param_norm, grad_stats, layer_stats

    def check_explosion(self, threshold=50.0):
        if len(self.grad_norms) > 0:
            is_exploding = self.grad_norms[-1] > threshold
            if is_exploding:
                self.explosion_count += 1
            return is_exploding
        return False


# ============================================================================
# Core Components
# ============================================================================
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class GlobalTokenBank(nn.Module):
    def __init__(self, bank_size: int, embed_dim: int):
        super().__init__()
        self.bank_size = bank_size
        self.embed_dim = embed_dim

        self.global_k = nn.Parameter(torch.randn(1, bank_size, embed_dim) * 0.02)
        self.global_v = nn.Parameter(torch.randn(1, bank_size, embed_dim) * 0.02)

        self.write_norm = nn.LayerNorm(embed_dim)
        self.write_compression = nn.Linear(embed_dim, embed_dim)
        self.write_gate = nn.Linear(embed_dim, bank_size)

        self.register_buffer('update_count', torch.tensor(0))

    def read(self, batch_size: int):
        return (
            self.global_k.expand(batch_size, -1, -1),
            self.global_v.expand(batch_size, -1, -1)
        )

    def write(self, tokens: torch.Tensor, residual: bool = True):
        if not self.training:
            return

        B, N, C = tokens.shape
        tokens_norm = self.write_norm(tokens)
        compressed = self.write_compression(tokens_norm)

        weights = F.softmax(self.write_gate(tokens_norm), dim=1)

        update_k = torch.bmm(weights.transpose(1, 2), compressed)
        update_v = torch.bmm(weights.transpose(1, 2), tokens_norm)

        if residual:
            update_k = torch.clamp(update_k.mean(0, keepdim=True), -0.05, 0.05)
            update_v = torch.clamp(update_v.mean(0, keepdim=True), -0.05, 0.05)

        update_rate = 0.005 if self.update_count < 1000 else 0.01

        self.global_k.data.add_(update_rate * update_k)
        self.global_v.data.add_(update_rate * update_v)

        self.global_k.data.clamp_(-0.5, 0.5)
        self.global_v.data.clamp_(-0.5, 0.5)

        self.update_count += 1


class LinformerCompression(nn.Module):
    def __init__(self, seq_len: int, compressed_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.compressed_len = compressed_len
        self.E_k = nn.Parameter(torch.randn(seq_len, compressed_len) * 0.02)
        self.E_v = nn.Parameter(torch.randn(seq_len, compressed_len) * 0.02)

    def forward(self, k, v):
        B, H, N, D = k.shape

        if N != self.seq_len:
            if N < self.seq_len:
                k = F.pad(k, (0, 0, 0, self.seq_len - N))
                v = F.pad(v, (0, 0, 0, self.seq_len - N))
            else:
                k = k[:, :, :self.seq_len]
                v = v[:, :, :self.seq_len]

        k_flat = k.reshape(B * H, self.seq_len, D)
        v_flat = v.reshape(B * H, self.seq_len, D)

        k_compressed = torch.matmul(self.E_k.T, k_flat)
        v_compressed = torch.matmul(self.E_v.T, v_flat)

        return (
            k_compressed.reshape(B, H, self.compressed_len, D),
            v_compressed.reshape(B, H, self.compressed_len, D)
        )


def efficient_attention(q, k, v, dropout_p=0.0, training=True):
    if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
        return torch.zeros_like(q)

    use_flash = (
        HAS_FLASH_ATTN
        and q.device.type == 'cuda'
        and training
        and q.dtype in (torch.float16, torch.bfloat16)
    )

    if use_flash:
        try:
            B, H, N_q, D = q.shape
            target_dtype = q.dtype

            if k.dtype != target_dtype:
                k = k.to(target_dtype)
            if v.dtype != target_dtype:
                v = v.to(target_dtype)

            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)

            output = flash_attn_func(q_t, k_t, v_t, dropout_p=dropout_p if training else 0.0, causal=False)
            output = output.transpose(1, 2)
        except:
            if q.dtype == torch.bfloat16:
                q32 = q.to(torch.float32)
                k32 = k.to(torch.float32)
                v32 = v.to(torch.float32)
                output = F.scaled_dot_product_attention(q32, k32, v32, dropout_p=dropout_p if training else 0.0)
                output = output.to(torch.bfloat16)
            else:
                output = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p if training else 0.0)
    else:
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p if training else 0.0)

    if torch.isnan(output).any():
        return torch.zeros_like(output)

    return output


# ============================================================================
# Attention Branches
# ============================================================================
class EfficientSpatialWindowAttention(nn.Module):
    def __init__(self, config, global_bank):
        super().__init__()
        self.config = config
        self.global_bank = global_bank
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.window_size = config.window_size

        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=True)
        self.linformer = LinformerCompression(self.window_size ** 2, config.linformer_k)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.embed_dim)

    def window_partition(self, x, window_size):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.view(B, H, W, C)

        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w

        num_h, num_w = H // window_size, W // window_size
        x = x.view(B, num_h, window_size, num_w, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
        return windows

    def window_reverse(self, windows, window_size, H, W, B):
        num_h, num_w = H // window_size, W // window_size
        x = windows.view(B, num_h, num_w, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1).view(B, H * W, -1)
        return x

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))

        x_windows = self.window_partition(x, self.window_size)
        BW, NW, _ = x_windows.shape

        qkv = self.qkv(x_windows).reshape(BW, NW, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k_compressed, v_compressed = self.linformer(k, v)

        k_bank, v_bank = self.global_bank.read(BW)
        k_bank = k_bank.reshape(BW, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_bank = v_bank.reshape(BW, -1, self.num_heads, self.head_dim).transpose(1, 2)

        k_full = torch.cat([k_compressed, k_bank], dim=2)
        v_full = torch.cat([v_compressed, v_bank], dim=2)

        attn_output = efficient_attention(q, k_full, v_full, self.dropout.p, self.training)
        attn_output = attn_output.transpose(1, 2).reshape(BW, NW, C)

        output = self.proj(attn_output)
        output = self.dropout(output)
        output = self.window_reverse(output, self.window_size, H, W, B)

        self.global_bank.write(self.norm(output))
        return output


class EfficientMultiScaleDilatedAttention(nn.Module):
    def __init__(self, config, global_bank):
        super().__init__()
        self.config = config
        self.global_bank = global_bank
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.dilation_factors = config.dilation_factors

        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=True)
        self.linformer = LinformerCompression(128, config.linformer_k)
        self.landmark_pool = nn.AvgPool1d(config.landmark_pooling_stride, config.landmark_pooling_stride)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.embed_dim)

    def extract_dilated_tokens(self, x, dilation):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.view(B, H, W, C)
        x_dilated = x[:, ::dilation, ::dilation, :]
        return x_dilated.reshape(B, -1, C)

    def forward(self, x):
        B, N, C = x.shape

        multi_scale = [self.extract_dilated_tokens(x, d) for d in self.dilation_factors]
        x_multi = torch.cat(multi_scale, dim=1)
        x_pooled = self.landmark_pool(x_multi.transpose(1, 2)).transpose(1, 2)

        BM, NM, _ = x_pooled.shape
        qkv_pooled = self.qkv(x_pooled).reshape(BM, NM, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        _, k, v = qkv_pooled[0], qkv_pooled[1], qkv_pooled[2]

        if NM < self.linformer.seq_len:
            k = F.pad(k, (0, 0, 0, self.linformer.seq_len - NM))
            v = F.pad(v, (0, 0, 0, self.linformer.seq_len - NM))
        elif NM > self.linformer.seq_len:
            k = k[:, :, :self.linformer.seq_len]
            v = v[:, :, :self.linformer.seq_len]

        k_compressed, v_compressed = self.linformer(k, v)

        k_bank, v_bank = self.global_bank.read(B)
        k_bank = k_bank.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_bank = v_bank.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        k_full = torch.cat([k_compressed, k_bank], dim=2)
        v_full = torch.cat([v_compressed, v_bank], dim=2)

        q_orig = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)[:, :, 0].permute(0, 2, 1, 3)

        attn_output = efficient_attention(q_orig, k_full, v_full, self.dropout.p, self.training)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)

        output = self.proj(attn_output)
        output = self.dropout(output)

        self.global_bank.write(self.norm(output))
        return output


class EfficientChannelGroupAttention(nn.Module):
    def __init__(self, config, global_bank):
        super().__init__()
        self.config = config
        self.global_bank = global_bank
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.num_groups = config.num_channel_groups
        self.channels_per_group = config.embed_dim // self.num_groups

        self.compress_c = config.embed_dim // 2
        self.compress_per_group = self.compress_c // self.num_groups

        self.q_proj = nn.Linear(self.channels_per_group, self.compress_per_group)
        self.k_proj = nn.Linear(self.channels_per_group, self.compress_per_group)
        self.v_proj = nn.Linear(self.channels_per_group, self.compress_per_group)
        self.bank_k_proj = nn.Linear(config.embed_dim, self.compress_per_group)
        self.bank_v_proj = nn.Linear(config.embed_dim, self.compress_per_group)

        self.proj = nn.Linear(self.compress_c, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        B, N, C = x.shape

        x_grouped = x.view(B, N, self.num_groups, self.channels_per_group).permute(0, 2, 1, 3)
        BG = B * self.num_groups
        x_flat = x_grouped.reshape(BG, N, self.channels_per_group)

        q = self.q_proj(x_flat)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        head_dim_compressed = self.compress_per_group // self.num_heads
        q = q.reshape(BG, N, self.num_heads, head_dim_compressed).transpose(1, 2)
        k = k.reshape(BG, N, self.num_heads, head_dim_compressed).transpose(1, 2)
        v = v.reshape(BG, N, self.num_heads, head_dim_compressed).transpose(1, 2)

        k_bank, v_bank = self.global_bank.read(B)
        k_bank_compressed = self.bank_k_proj(k_bank).unsqueeze(1).expand(-1, self.num_groups, -1, -1)
        v_bank_compressed = self.bank_v_proj(v_bank).unsqueeze(1).expand(-1, self.num_groups, -1, -1)
        k_bank_compressed = k_bank_compressed.reshape(BG, -1, self.compress_per_group)
        v_bank_compressed = v_bank_compressed.reshape(BG, -1, self.compress_per_group)

        k_bank = k_bank_compressed.reshape(BG, -1, self.num_heads, head_dim_compressed).transpose(1, 2)
        v_bank = v_bank_compressed.reshape(BG, -1, self.num_heads, head_dim_compressed).transpose(1, 2)

        k_full = torch.cat([k, k_bank], dim=2)
        v_full = torch.cat([v, v_bank], dim=2)

        attn_output = efficient_attention(q, k_full, v_full, self.dropout.p, self.training)
        attn_output = attn_output.transpose(1, 2).reshape(BG, N, -1)
        attn_output = attn_output.view(B, self.num_groups, N, -1).permute(0, 2, 1, 3).reshape(B, N, self.compress_c)

        output = self.proj(attn_output)
        output = self.dropout(output)

        self.global_bank.write(self.norm(output))
        return output


class CrossAttentionBranch(nn.Module):
    def __init__(self, config, global_bank):
        super().__init__()
        self.config = config
        self.global_bank = global_bank
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads

        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k_bank, v_bank = self.global_bank.read(B)
        k = self.k_proj(k_bank).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v_bank).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = efficient_attention(q, k, v, self.dropout.p, self.training)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)

        output = self.proj(attn_output)
        output = self.dropout(output)
        return output


# ============================================================================
# Fusion and FFN
# ============================================================================
class HybridFusion(nn.Module):
    def __init__(self, embed_dim, num_branches=4):
        super().__init__()
        self.fusion_weights = nn.Parameter(torch.ones(num_branches))

    def forward(self, branches):
        weights = F.softmax(self.fusion_weights, dim=0)
        scaled = [b * w for b, w in zip(branches, weights)]
        return torch.cat(scaled, dim=-1)


class BottleneckMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class DepthwiseConv2d(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)

        nn.init.kaiming_normal_(self.dwconv.weight, mode='fan_out', nonlinearity='linear')
        with torch.no_grad():
            self.dwconv.weight.div_(math.sqrt(dim))

        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.1)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x * self.scale
        return x.flatten(2).transpose(1, 2)


class CCFFFN(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=0.5, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)

        self.act = nn.GELU()
        self.dwconv_norm = nn.LayerNorm(hidden_dim)
        self.dwconv = DepthwiseConv2d(hidden_dim, kernel_size=3)
        self.post_dwconv_norm = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))

        x = self.fc1(x)
        x = self.act(x)
        x = self.dwconv_norm(x)
        x = self.dwconv(x, H, W)
        x = self.post_dwconv_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x * self.gamma


# ============================================================================
# CNN Stem
# ============================================================================
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class CNNStemModel(nn.Module):
    """CNN Lateral Backbone for 64×64 input"""
    def __init__(self, in_ch=3, c2=64, c3=128, c4=256):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(32, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            ConvNeXtBlock(c2),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=1),
            nn.BatchNorm2d(c3),
            ConvNeXtBlock(c3),
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=1),
            nn.BatchNorm2d(c4),
            ConvNeXtBlock(c4),
        )

    def forward(self, x):
        x = self.stem(x)
        F2 = self.stage1(x)
        F3 = self.stage2(F2)
        F4 = self.stage3(F3)

        return F2, F3, F4


# ============================================================================
# LMFAdapter, RRCV, SplitFusion, TokenLearner
# ============================================================================
class LMFAdapter(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, target_hw: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.target_hw = target_hw

        self.dwconv_3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.dwconv_5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)

        self.proj = nn.Conv2d(3 * in_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, feat):
        B, C, H, W = feat.shape

        f1 = self.dwconv_3x3(feat)
        f2 = self.dwconv_5x5(feat)
        f3 = feat

        f_cat = torch.cat([f1, f2, f3], dim=1)
        f_proj = self.proj(f_cat)

        if H != self.target_hw or W != self.target_hw:
            f_proj = F.interpolate(f_proj, size=(self.target_hw, self.target_hw),
                                  mode='bilinear', align_corners=False)

        A = f_proj.flatten(2).transpose(1, 2)
        A = self.norm(A)
        A = self.act(A)

        return A


class RRCV(nn.Module):
    def __init__(self, embed_dim: int, rec_channels: int = 64, num_blocks: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.rec_channels = rec_channels

        self.reverse_proj = nn.Conv2d(embed_dim, rec_channels, kernel_size=1)

        self.blocks = nn.ModuleList([
            ConvNeXtBlock(rec_channels) for _ in range(num_blocks)
        ])

        self.reembed_proj = nn.Conv2d(rec_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, A, H: int, W: int):
        B, N, C = A.shape

        X = A.permute(0, 2, 1).view(B, C, H, W)
        R_in = self.reverse_proj(X)

        R_feat = R_in
        for block in self.blocks:
            R_feat = block(R_feat)

        R_proj = self.reembed_proj(R_feat)
        R_tokens = R_proj.flatten(2).transpose(1, 2)
        R_tokens = self.norm(R_tokens)

        R_tokens = A + self.beta * R_tokens

        return R_tokens


class SplitFusion(nn.Module):
    def __init__(self, embed_dim: int, use_learnable_weights: bool = True):
        super().__init__()
        self.embed_dim = embed_dim

        self.gate_norm = nn.LayerNorm(embed_dim)
        self.gate_fc = nn.Linear(embed_dim, embed_dim)

        self.cat_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        if use_learnable_weights:
            self.fusion_weights = nn.Parameter(torch.tensor([0.75, 0.25]))
        else:
            self.register_buffer('fusion_weights', torch.tensor([0.75, 0.25]))

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, T_in, R):
        T_add = T_in + R
        gate = torch.sigmoid(self.gate_fc(self.gate_norm(T_add)))
        A_add_scaled = gate * R
        T_add_out = T_in + A_add_scaled

        T_cat = torch.cat([T_in, R], dim=-1)
        T_cat_out = T_in + self.cat_mlp(T_cat)

        weights = F.softmax(self.fusion_weights, dim=0)
        T_fused = weights[0] * T_add_out + weights[1] * T_cat_out

        T_fused = self.final_norm(T_fused)

        return T_fused


class TokenLearner(nn.Module):
    def __init__(self, in_dim: int, num_out_tokens: int = 64):
        super().__init__()
        self.num_out_tokens = num_out_tokens

        self.attention = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, num_out_tokens),
        )

    def forward(self, x):
        B, N, C = x.shape

        scores = self.attention(x)
        scores = F.softmax(scores, dim=1)

        scores = scores.transpose(1, 2)
        x_compressed = torch.bmm(scores, x)

        return x_compressed


class TokenUpMix(nn.Module):
    def __init__(self, embed_dim: int, num_in_tokens: int, num_out_tokens: int):
        super().__init__()
        self.num_in_tokens = num_in_tokens
        self.num_out_tokens = num_out_tokens

        self.upsample_attn = nn.Linear(num_in_tokens, num_out_tokens)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_compressed):
        B, M, C = x_compressed.shape

        x_t = x_compressed.transpose(1, 2)
        x_up = self.upsample_attn(x_t)
        x = x_up.transpose(1, 2)
        x = self.norm(x)

        return x


# ============================================================================
# QuadAttentionBlock
# ============================================================================
class QuadAttentionBlock(nn.Module):
    def __init__(self, config, global_bank, drop_path=0.):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.compressed_dim = config.embed_dim // config.compress_ratio

        self.norm1 = nn.LayerNorm(config.embed_dim)

        self.swa = EfficientSpatialWindowAttention(config, global_bank)
        self.msda = EfficientMultiScaleDilatedAttention(config, global_bank)
        self.cga = EfficientChannelGroupAttention(config, global_bank)
        self.cross_attn = CrossAttentionBranch(config, global_bank)

        self.norm_swa = nn.LayerNorm(config.embed_dim)
        self.norm_msda = nn.LayerNorm(config.embed_dim)
        self.norm_cga = nn.LayerNorm(config.embed_dim)
        self.norm_cross = nn.LayerNorm(config.embed_dim)

        self.compress_swa = nn.Linear(config.embed_dim, self.compressed_dim)
        self.compress_msda = nn.Linear(config.embed_dim, self.compressed_dim)
        self.compress_cga = nn.Linear(config.embed_dim, self.compressed_dim)
        self.compress_cross = nn.Linear(config.embed_dim, self.compressed_dim)

        self.fusion = HybridFusion(self.compressed_dim, 4)

        bottleneck_hidden = config.embed_dim // config.bottleneck_ratio
        self.bottleneck_mlp = BottleneckMLP(4 * self.compressed_dim, bottleneck_hidden, config.embed_dim, config.dropout)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.ccf_ffn = CCFFFN(config.embed_dim, config.mlp_ratio, config.dropout)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x_norm = self.norm1(x)

        swa_out = self.compress_swa(self.norm_swa(self.swa(x_norm)))
        msda_out = self.compress_msda(self.norm_msda(self.msda(x_norm)))
        cga_out = self.compress_cga(self.norm_cga(self.cga(x_norm)))
        cross_out = self.compress_cross(self.norm_cross(self.cross_attn(x_norm)))

        fused = self.fusion([swa_out, msda_out, cga_out, cross_out])
        mlp_out = self.bottleneck_mlp(fused)

        x = x + self.drop_path1(mlp_out)
        x = x + self.drop_path2(self.ccf_ffn(self.norm2(x)))

        return x


class QuadBlockWithTokenLearner(nn.Module):
    def __init__(self, config, global_bank, drop_path=0., use_token_learner=True):
        super().__init__()
        self.use_token_learner = use_token_learner

        if use_token_learner:
            M = config.num_learned_tokens
            sq = int(math.sqrt(M))
            if sq * sq != M:
                new_M = max(4, sq * sq)
                print(f"[WARNING] config.num_learned_tokens={M} is not a perfect square. Using {new_M} instead.")
                M = new_M

            self.token_learner = TokenLearner(config.embed_dim, M)
            self.token_upmix = TokenUpMix(config.embed_dim, M, (config.img_size // config.patch_size) ** 2)

        self.quad_block = QuadAttentionBlock(config, global_bank, drop_path)

    def forward(self, x):
        if self.use_token_learner:
            x_compressed = self.token_learner(x)
            x_compressed = self.quad_block(x_compressed)
            x = self.token_upmix(x_compressed)
        else:
            x = self.quad_block(x)

        return x


# ============================================================================
# Main HQA-ViT Model - UPGRADED WITH HIERARCHICAL DEPTH [2, 2, 6, 2]
# ============================================================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class HQAViT(nn.Module):
    """HQA-ViT for Tiny ImageNet - UPGRADED VERSION"""
    def __init__(self, config: HQAViTConfig):
        super().__init__()
        self.config = config
        self.num_patches = (config.img_size // config.patch_size) ** 2
        self.H = self.W = config.img_size // config.patch_size

        self.patch_embed = PatchEmbed(config.img_size, config.patch_size, config.in_channels, config.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, config.embed_dim))
        self.pos_drop = nn.Dropout(config.dropout)

        self.global_bank = GlobalTokenBank(config.global_bank_size, config.embed_dim)

        self.cnn_stem = CNNStemModel(
            in_ch=config.in_channels,
            c2=config.cnn_c2,
            c3=config.cnn_c3,
            c4=config.cnn_c4
        )

        self.lmfa2 = LMFAdapter(config.cnn_c2, config.embed_dim, target_hw=self.H)
        self.lmfa3 = LMFAdapter(config.cnn_c3, config.embed_dim, target_hw=self.H)
        self.lmfa4 = LMFAdapter(config.cnn_c4, config.embed_dim, target_hw=self.H)

        self.rrcv2 = RRCV(config.embed_dim, config.rrcv_channels, config.rrcv_num_blocks)
        self.rrcv3 = RRCV(config.embed_dim, config.rrcv_channels, config.rrcv_num_blocks)
        self.rrcv4 = RRCV(config.embed_dim, config.rrcv_channels, config.rrcv_num_blocks)

        self.fuse2 = SplitFusion(config.embed_dim)
        self.fuse3 = SplitFusion(config.embed_dim)
        self.fuse4 = SplitFusion(config.embed_dim)

        # UPGRADED: Hierarchical depth [2, 2, 6, 2] = 12 total layers
        dpr = [x.item() for x in torch.linspace(0, config.drop_path, config.depth)]

        # Stage 1: 2 blocks (depth 0-1)
        self.stage1_blocks = nn.ModuleList([
            QuadBlockWithTokenLearner(config, self.global_bank, dpr[i], config.use_token_learner)
            for i in range(2)
        ])

        # Stage 2: 2 blocks (depth 2-3)
        self.stage2_blocks = nn.ModuleList([
            QuadBlockWithTokenLearner(config, self.global_bank, dpr[i], config.use_token_learner)
            for i in range(2, 4)
        ])

        # Stage 3: 6 blocks - THE HEAVY MIDDLE (depth 4-9)
        self.stage3_blocks = nn.ModuleList([
            QuadBlockWithTokenLearner(config, self.global_bank, dpr[i], config.use_token_learner)
            for i in range(4, 10)
        ])

        # Stage 4: 2 blocks (depth 10-11)
        self.stage4_blocks = nn.ModuleList([
            QuadBlockWithTokenLearner(config, self.global_bank, dpr[i], config.use_token_learner)
            for i in range(10, 12)
        ])

        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

        print(f"\n{'='*80}")
        print("HQA-ViT ARCHITECTURE - UPGRADED".center(80))
        print(f"{'='*80}")
        print(f"  Hierarchical Depth: [2, 2, 6, 2] = {config.depth} layers")
        print(f"  TokenLearner: {config.num_learned_tokens} tokens ({int(math.sqrt(config.num_learned_tokens))}×{int(math.sqrt(config.num_learned_tokens))} grid)")
        print(f"  Drop Path: {config.drop_path}")
        print(f"  Embed Dim: {config.embed_dim}")
        print(f"{'='*80}\n")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        B = x.shape[0]

        F2, F3, F4 = self.cnn_stem(x)

        A2 = self.lmfa2(F2)
        A3 = self.lmfa3(F3)
        A4 = self.lmfa4(F4)

        R2 = self.rrcv2(A2, self.H, self.W)
        R3 = self.rrcv3(A3, self.H, self.W)
        R4 = self.rrcv4(A4, self.H, self.W)

        T = self.patch_embed(x)
        T = T + self.pos_embed
        T = self.pos_drop(T)

        # Stage 1: 2 blocks
        for block in self.stage1_blocks:
            T = block(T)

        # Fuse with CNN stage 2
        T = self.fuse2(T, R2)

        # Stage 2: 2 blocks
        for block in self.stage2_blocks:
            T = block(T)

        # Fuse with CNN stage 3
        T = self.fuse3(T, R3)

        # Stage 3: 6 blocks - THE HEAVY MIDDLE
        for block in self.stage3_blocks:
            T = block(T)

        # Fuse with CNN stage 4
        T = self.fuse4(T, R4)

        # Stage 4: 2 blocks
        for block in self.stage4_blocks:
            T = block(T)

        T = self.norm(T)
        T = T.mean(dim=1)
        logits = self.head(T)

        return logits


# ============================================================================
# Data Loading - UPGRADED
# ============================================================================
def get_tinyimagenet_loaders(config: TrainingConfig):
    """Tiny ImageNet data loading with UPGRADED augmentation"""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # UPGRADED: Softer augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandAugment(num_ops=2, magnitude=6),  # UPGRADED: 9 -> 6
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # REMOVED: RandomErasing (too aggressive)
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = TinyImageNetDataset(root_dir=config.data_root, split='train',
                                       transform=train_transform, download=True)
    val_dataset = TinyImageNetDataset(root_dir=config.data_root, split='val',
                                     transform=val_transform, download=False)

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

    return train_loader, val_loader


# ============================================================================
# Training Functions
# ============================================================================
def rand_bbox(size, lam):
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

        use_mix = None
        lam = 1.0
        if config.use_cutmix and np.random.rand() < config.mix_prob:
            rand_index = torch.randperm(inputs.size(0)).cuda()
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), np.random.beta(config.cutmix_alpha, config.cutmix_alpha))
            inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[rand_index, :, bby1:bby2, bbx1:bbx2]
            targets_a, targets_b = targets, targets[rand_index]
            W = inputs.size(3)
            H = inputs.size(2)
            lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / float(W * H))
            use_mix = 'cutmix'
        elif config.use_mixup and np.random.rand() < config.mix_prob:
            rand_index = torch.randperm(inputs.size(0)).cuda()
            lam = np.random.beta(config.mixup_alpha, config.mixup_alpha)
            inputs = (lam * inputs + (1 - lam) * inputs[rand_index])
            targets_a, targets_b = targets, targets[rand_index]
            use_mix = 'mixup'

        amp_dtype = torch.bfloat16 if config.amp_dtype == 'bfloat16' else torch.float16
        with autocast(enabled=config.use_amp, dtype=amp_dtype):
            outputs = model(inputs)
            if use_mix is None:
                loss = criterion(outputs, targets) / config.gradient_accumulation_steps
            else:
                loss = lam * criterion(outputs, targets_a) + (1.0 - lam) * criterion(outputs, targets_b)
                loss = loss / config.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)

            for name, param in model.named_parameters():
                if ('cnn_stem' in name or 'dwconv' in name) and param.grad is not None:
                    torch.nn.utils.clip_grad_norm_([param], max_norm=0.1)

            detailed = (batch_idx % 200 == 0)
            grad_norm, param_norm, grad_stats, layer_stats = monitor.log_gradients(model, detailed=detailed)

            if monitor.check_explosion(threshold=50.0):
                print(f"\n{'='*100}")
                print(f"🚨 GRADIENT EXPLOSION DETECTED".center(100))
                print(f"{'='*100}\n")

            if config.grad_clip_mode == 'norm':
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            elif config.grad_clip_mode == 'value':
                torch.nn.utils.clip_grad_value_(model.parameters(), config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            if model_ema is not None:
                model_ema.update(model)

        total_loss += loss.item() * config.gradient_accumulation_steps
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % config.print_freq == 0:
            acc = 100. * correct / total
            lr = optimizer.param_groups[0]['lr']
            avg_loss = total_loss / (batch_idx + 1)
            print(f'Epoch {epoch:3d} [{batch_idx:4d}/{len(loader):4d}] | '
                  f'Loss: {avg_loss:.4f} | Acc: {acc:6.2f}% | '
                  f'LR: {lr:.6f} | Grad: {grad_norm:.4f}')

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


# ============================================================================
# Main Training Script
# ============================================================================
def main():
    print("\n" + "="*100)
    print("HQA-ViT TINY IMAGENET PRETRAINING - UPGRADED VERSION".center(100))
    print("Target: 70-80% Accuracy".center(100))
    print("="*100)

    model_config = HQAViTConfig()
    train_config = TrainingConfig()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ Hardware Configuration:")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA: {torch.version.cuda}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   FlashAttention: {'Available ✓' if HAS_FLASH_ATTN else 'Not Available ✗'}")

    Path(train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(train_config.plot_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*100}")
    print("MODEL INITIALIZATION".center(100))
    print(f"{'='*100}")
    model = HQAViT(model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📊 Model Statistics:")
    print(f"   Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   Trainable Parameters: {trainable_params:,}")

    print(f"\n{'='*100}")
    print("DATA PREPARATION".center(100))
    print(f"{'='*100}")
    print(f"\n📚 Loading Tiny ImageNet...")
    train_loader, val_loader = get_tinyimagenet_loaders(train_config)
    print(f"   Train samples: {len(train_loader.dataset):,}")
    print(f"   Val samples: {len(val_loader.dataset):,}")
    print(f"   Batch size: {train_config.batch_size}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.base_lr,
        betas=(0.9, 0.999),
        weight_decay=train_config.weight_decay
    )

    steps_per_epoch = len(train_loader) // train_config.gradient_accumulation_steps
    total_steps = steps_per_epoch * train_config.epochs
    warmup_steps = steps_per_epoch * train_config.warmup_epochs

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_config.base_lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1e4
    )

    scaler = GradScaler(enabled=train_config.use_amp)
    monitor = GradientMonitor()

    model_ema = None
    if train_config.use_ema:
        print(f"\n✓ Initializing EMA (decay={train_config.ema_decay})")
        model_ema = ModelEMA(model, decay=train_config.ema_decay, device=device)

    history = TrainingHistory(train_config.plot_dir)

    print(f"\n{'='*100}")
    print("TRAINING STARTED".center(100))
    print(f"{'='*100}\n")

    best_acc = 0
    best_ema_acc = 0
    train_start_time = time.time()

    for epoch in range(1, train_config.epochs + 1):
        epoch_start_time = time.time()

        if train_config.use_ema and epoch <= train_config.warmup_epochs:
            current_decay = train_config.ema_decay_warmup + \
                          (train_config.ema_decay - train_config.ema_decay_warmup) * \
                          (epoch / train_config.warmup_epochs)
            model_ema.set_decay(current_decay)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            train_config, epoch, monitor, model_ema
        )

        if epoch % train_config.eval_freq == 0:
            val_loss, val_acc = validate(model, val_loader)

            ema_val_loss, ema_val_acc = 0, 0
            param_dist, buffer_dist = 0.0, 0.0
            if model_ema is not None:
                ema_val_loss, ema_val_acc = validate(model_ema.ema, val_loader)
                param_dist, buffer_dist = model_ema.compute_distance(model)

            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - train_start_time

            history.update(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                ema_val_loss=ema_val_loss,
                ema_val_acc=ema_val_acc,
                lr=optimizer.param_groups[0]['lr'],
                grad_norm=monitor.grad_norms[-1] if monitor.grad_norms else 0,
                ema_param_dist=param_dist,
                ema_buffer_dist=buffer_dist
            )

            if epoch % 10 == 0:
                print(f"\n📊 Generating training plots...")
                history.plot_all()
                print(f"   Plots saved to {train_config.plot_dir}")

            print(f"\n{'='*100}")
            print(f"EPOCH {epoch}/{train_config.epochs} SUMMARY".center(100))
            print(f"{'='*100}")
            print(f"{'Metric':<30} {'Train':>15} {'Val':>15} {'EMA Val':>15}")
            print("-"*100)
            print(f"{'Loss':<30} {train_loss:>15.4f} {val_loss:>15.4f} {ema_val_loss:>15.4f}")
            print(f"{'Accuracy (%)':<30} {train_acc:>15.2f} {val_acc:>15.2f} {ema_val_acc:>15.2f}")
            print(f"{'Best Acc (%)':<30} {'-':>15} {max(best_acc, val_acc):>15.2f} {max(best_ema_acc, ema_val_acc):>15.2f}")
            print(f"{'Epoch Time (min)':<30} {epoch_time/60:>15.1f}")
            print(f"{'Learning Rate':<30} {optimizer.param_groups[0]['lr']:>15.6f}")
            print(f"{'='*100}\n")

            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'best_acc': best_acc,
                }
                torch.save(checkpoint, f"{train_config.checkpoint_dir}/best_model.pth")
                print(f"✅ Best model saved! Val Acc: {best_acc:.2f}%\n")

            if model_ema is not None and ema_val_acc > best_ema_acc:
                best_ema_acc = ema_val_acc
                checkpoint_ema = {
                    'epoch': epoch,
                    'model_state_dict': model_ema.ema.state_dict(),
                    'val_acc': ema_val_acc,
                    'best_ema_acc': best_ema_acc,
                }
                torch.save(checkpoint_ema, f"{train_config.checkpoint_dir}/best_model_ema.pth")
                print(f"✅ Best EMA model saved! EMA Val Acc: {best_ema_acc:.2f}%\n")

            if epoch % train_config.save_freq == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint, f"{train_config.checkpoint_dir}/checkpoint_epoch_{epoch}.pth")
                print(f"💾 Checkpoint saved: epoch_{epoch}.pth\n")

    print(f"\n📊 Generating final training plots...")
    history.plot_all()

    total_training_time = time.time() - train_start_time

    print(f"\n{'='*100}")
    print("TRAINING COMPLETE".center(100))
    print(f"{'='*100}")
    print(f"\n{'Final Results':-^100}")
    print(f"   Best Validation Accuracy: {best_acc:.2f}%")
    if model_ema is not None:
        print(f"   Best EMA Validation Accuracy: {best_ema_acc:.2f}%")
    print(f"   Total Training Time: {total_training_time/3600:.2f} hours")
    print(f"   Plots saved to: {train_config.plot_dir}")
    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    main()