"""
QAViT for CIFAR-100 - Optimized Training Script
Includes gradient explosion debugging and monitoring
Optimized for RTX 3060 6GB and RTX 5070 12GB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

# ============================================================================
# FlashAttention Check
# ============================================================================
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("⚠️ FlashAttention2 not available, using PyTorch SDPA")


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class QAViTConfig:
    """Optimized config for CIFAR-100"""
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 100
    embed_dim: int = 192  # Reduced from 256 for speed
    depth: int = 8  # Reduced from 12 for speed
    num_heads: int = 4
    compress_ratio: int = 4
    bottleneck_ratio: int = 2
    mlp_ratio: float = 0.5
    global_bank_size: int = 16
    dropout: float = 0.1
    drop_path: float = 0.1
    window_size: int = 7
    dilation_factors: Tuple[int, ...] = (1, 2, 3)
    landmark_pooling_stride: int = 2
    num_channel_groups: int = 6  # Divisible by 192
    linformer_k: int = 64


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Hardware-specific
    batch_size: int = 128  # Start conservative for 6GB
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    
    # Training
    epochs: int = 200
    warmup_epochs: int = 10
    base_lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 0.05
    
    # Regularization
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Gradient control (CRITICAL for stability)
    max_grad_norm: float = 1.0  # Gradient clipping
    grad_clip_mode: str = 'norm'  # 'norm' or 'value'
    
    # Monitoring
    print_freq: int = 50
    eval_freq: int = 1
    save_freq: int = 10
    
    # AMP
    use_amp: bool = True
    
    # Paths
    data_root: str = "./data"
    checkpoint_dir: str = "./checkpoints"


# ============================================================================
# Gradient & Activation Monitoring
# ============================================================================
class GradientMonitor:
    """Monitor gradients and activations for debugging"""
    
    def __init__(self):
        self.grad_norms = []
        self.param_norms = []
        self.activation_stats = {}
        
    def log_gradients(self, model):
        """Log gradient statistics"""
        total_norm = 0.0
        param_norm = 0.0
        grad_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Gradient norm
                g_norm = param.grad.norm().item()
                total_norm += g_norm ** 2
                
                # Parameter norm
                p_norm = param.norm().item()
                param_norm += p_norm ** 2
                
                # Track problematic layers
                if g_norm > 10.0 or torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    grad_stats[name] = {
                        'grad_norm': g_norm,
                        'grad_mean': param.grad.mean().item(),
                        'grad_max': param.grad.max().item(),
                        'grad_min': param.grad.min().item(),
                        'has_nan': torch.isnan(param.grad).any().item(),
                        'has_inf': torch.isinf(param.grad).any().item(),
                    }
        
        total_norm = total_norm ** 0.5
        param_norm = param_norm ** 0.5
        
        self.grad_norms.append(total_norm)
        self.param_norms.append(param_norm)
        
        return total_norm, param_norm, grad_stats
    
    def check_explosion(self, threshold=100.0):
        """Check if gradients are exploding"""
        if len(self.grad_norms) > 0:
            return self.grad_norms[-1] > threshold
        return False
    
    def print_stats(self, epoch, step):
        """Print monitoring statistics"""
        if len(self.grad_norms) > 0:
            print(f"\n📊 [Epoch {epoch}, Step {step}] Gradient Stats:")
            print(f"  Grad Norm: {self.grad_norms[-1]:.4f}")
            print(f"  Param Norm: {self.param_norms[-1]:.4f}")
            print(f"  Grad/Param Ratio: {self.grad_norms[-1]/max(self.param_norms[-1], 1e-8):.4f}")


# ============================================================================
# Helper Functions
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


# ============================================================================
# Core Components (Optimized)
# ============================================================================
class GlobalTokenBank(nn.Module):
    """Global Token Bank with stability improvements"""
    def __init__(self, bank_size: int, embed_dim: int):
        super().__init__()
        self.bank_size = bank_size
        self.embed_dim = embed_dim
        
        self.global_k = nn.Parameter(torch.randn(1, bank_size, embed_dim) * 0.02)
        self.global_v = nn.Parameter(torch.randn(1, bank_size, embed_dim) * 0.02)
        
        self.write_norm = nn.LayerNorm(embed_dim)
        self.write_compression = nn.Linear(embed_dim, embed_dim)
        self.write_gate = nn.Linear(embed_dim, bank_size)
        
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
        
        # Smaller update rate and clamping for stability
        if residual:
            update_k = torch.clamp(update_k.mean(0, keepdim=True), -0.1, 0.1)
            update_v = torch.clamp(update_v.mean(0, keepdim=True), -0.1, 0.1)
            self.global_k.data.add_(0.01 * update_k)  # Reduced from 0.1
            self.global_v.data.add_(0.01 * update_v)
            # Clamp after update
            self.global_k.data.clamp_(-1.0, 1.0)
            self.global_v.data.clamp_(-1.0, 1.0)


class LinformerCompression(nn.Module):
    """Linformer with improved initialization"""
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
    """Unified attention with stability checks"""
    # Check for NaN/Inf before attention
    if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
        print("⚠️ NaN detected in attention inputs!")
        return torch.zeros_like(q)
    
    use_flash = HAS_FLASH_ATTN and q.device.type == 'cuda' and training
    
    if use_flash:
        B, H, N_q, D = q.shape
        target_dtype = q.dtype
        
        if k.dtype != target_dtype:
            k = k.to(target_dtype)
        if v.dtype != target_dtype:
            v = v.to(target_dtype)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        output = flash_attn_func(q, k, v, dropout_p=dropout_p if training else 0.0, causal=False)
        output = output.transpose(1, 2)
    else:
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p if training else 0.0)
    
    # Check output
    if torch.isnan(output).any():
        print("⚠️ NaN detected in attention output!")
        return torch.zeros_like(output)
    
    return output


# ============================================================================
# Attention Branches (Simplified for speed)
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
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class CCFFFN(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=0.5, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.dwconv = DepthwiseConv2d(hidden_dim, 3)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = self.fc1(x)
        x = self.act(x)
        x = self.dwconv(x, H, W)
        x = self.fc2(x)
        return self.dropout(x)


# ============================================================================
# Transformer Block
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


# ============================================================================
# Main Model
# ============================================================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class QAViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_patches = (config.img_size // config.patch_size) ** 2
        
        self.patch_embed = PatchEmbed(config.img_size, config.patch_size, config.in_channels, config.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, config.embed_dim))
        self.pos_drop = nn.Dropout(config.dropout)
        
        self.global_bank = GlobalTokenBank(config.global_bank_size, config.embed_dim)
        
        dpr = [x.item() for x in torch.linspace(0, config.drop_path, config.depth)]
        self.blocks = nn.ModuleList([
            QuadAttentionBlock(config, self.global_bank, dpr[i]) for i in range(config.depth)
        ])
        
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        
        # Improved initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
    
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
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


# ============================================================================
# Data Loading
# ============================================================================
def get_cifar100_loaders(config: TrainingConfig):
    """Optimized CIFAR-100 data loading"""
    
    # Calculate stats from CIFAR-100
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    
    # Training transforms - optimized
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.RandomCrop(224, padding=28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    train_dataset = datasets.CIFAR100(root=config.data_root, train=True, transform=train_transform, download=True)
    val_dataset = datasets.CIFAR100(root=config.data_root, train=False, transform=val_transform, download=True)
    
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
def train_epoch(model, loader, optimizer, scheduler, scaler, config, epoch, monitor):
    """Training loop with gradient monitoring"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        
        # Forward with AMP
        with autocast(enabled=config.use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets) / config.gradient_accumulation_steps
        
        # Backward
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            # Unscale before clipping
            scaler.unscale_(optimizer)
            
            # Monitor gradients
            grad_norm, param_norm, grad_stats = monitor.log_gradients(model)
            
            # Check for explosion
            if monitor.check_explosion(threshold=50.0):
                print(f"\n🚨 GRADIENT EXPLOSION DETECTED at epoch {epoch}, step {batch_idx}")
                monitor.print_stats(epoch, batch_idx)
                if grad_stats:
                    print("\n⚠️ Problematic layers:")
                    for name, stats in grad_stats.items():
                        print(f"  {name}: norm={stats['grad_norm']:.4f}, has_nan={stats['has_nan']}, has_inf={stats['has_inf']}")
            
            # Gradient clipping
            if config.grad_clip_mode == 'norm':
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            elif config.grad_clip_mode == 'value':
                torch.nn.utils.clip_grad_value_(model.parameters(), config.max_grad_norm)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        # Statistics
        total_loss += loss.item() * config.gradient_accumulation_steps
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress
        if batch_idx % config.print_freq == 0:
            acc = 100. * correct / total
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch: {epoch} [{batch_idx}/{len(loader)}] | '
                  f'Loss: {total_loss/(batch_idx+1):.4f} | '
                  f'Acc: {acc:.2f}% | '
                  f'LR: {lr:.6f} | '
                  f'Grad: {grad_norm:.4f}')
    
    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def validate(model, loader):
    """Validation loop"""
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
    print("="*80)
    print("QAViT CIFAR-100 Training with Gradient Monitoring")
    print("="*80)
    
    # Configs
    model_config = QAViTConfig()
    train_config = TrainingConfig()
    
    # Create checkpoint directory
    Path(train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Build model
    model = QAViT(model_config).cuda()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model Statistics:")
    print(f"  Parameters: {num_params:,}")
    print(f"  Embed dim: {model_config.embed_dim}")
    print(f"  Depth: {model_config.depth}")
    print(f"  Heads: {model_config.num_heads}")
    
    # Data loaders
    print(f"\n📚 Loading CIFAR-100...")
    train_loader, val_loader = get_cifar100_loaders(train_config)
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Batch size: {train_config.batch_size}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.base_lr,
        betas=(0.9, 0.999),
        weight_decay=train_config.weight_decay
    )
    
    # Scheduler with warmup
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
    
    # AMP scaler
    scaler = GradScaler(enabled=train_config.use_amp)
    
    # Gradient monitor
    monitor = GradientMonitor()
    
    print(f"\n⚙️ Training Configuration:")
    print(f"  Epochs: {train_config.epochs}")
    print(f"  Warmup epochs: {train_config.warmup_epochs}")
    print(f"  Base LR: {train_config.base_lr}")
    print(f"  Weight decay: {train_config.weight_decay}")
    print(f"  Gradient clip: {train_config.max_grad_norm} ({train_config.grad_clip_mode})")
    print(f"  AMP: {train_config.use_amp}")
    print(f"  Gradient accumulation: {train_config.gradient_accumulation_steps}")
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print("="*80)
    
    best_acc = 0
    
    for epoch in range(1, train_config.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, train_config, epoch, monitor
        )
        
        # Validate
        if epoch % train_config.eval_freq == 0:
            val_loss, val_acc = validate(model, val_loader)
            
            epoch_time = time.time() - start_time
            
            print(f"\n{'='*80}")
            print(f"Epoch {epoch} Summary ({epoch_time:.1f}s):")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            monitor.print_stats(epoch, len(train_loader))
            print(f"{'='*80}\n")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'train_config': train_config,
                    'model_config': model_config,
                }, f"{train_config.checkpoint_dir}/best_model.pth")
                print(f"✅ Best model saved! Val Acc: {best_acc:.2f}%\n")
        
        # Save checkpoint
        if epoch % train_config.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f"{train_config.checkpoint_dir}/checkpoint_epoch_{epoch}.pth")
    
    print(f"\n🎉 Training complete! Best Val Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    main()