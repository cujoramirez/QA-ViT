"""
HQA-ViT: Hybrid Quad-Attention Vision Transformer
Complete implementation with CNN-Transformer Aggregation for CIFAR-100
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np
from collections import OrderedDict
from copy import deepcopy

# ============================================================================
# FlashAttention Check
# ============================================================================
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

# ============================================================================
# Configuration
# ============================================================================
@dataclass
class HQAViTConfig:
    """HQA-ViT configuration for CIFAR-100"""
    img_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 100
    embed_dim: int = 192  # d
    depth: int = 12
    num_heads: int = 4
    compress_ratio: int = 4
    bottleneck_ratio: int = 2
    mlp_ratio: float = 0.5
    global_bank_size: int = 16
    dropout: float = 0.1
    drop_path: float = 0.1
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
    
    # TokenLearner config
    use_token_learner: bool = True
    num_learned_tokens: int = 16
    
    # Fusion config
    fusion_stages: Tuple[int, ...] = (2, 3, 4)  # Which stages get CNN fusion


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 256
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    
    epochs: int = 300
    warmup_epochs: int = 10
    base_lr: float = 5e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.05
    
    label_smoothing: float = 0.1
    
    max_grad_norm: float = 0.5
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
    # Make EMA track model more responsively: lower final decay and stronger warmup
    ema_decay: float = 0.9995
    ema_decay_warmup: float = 0.99
    
    data_root: str = "./data"
    checkpoint_dir: str = "./checkpoints_hqavit"
    # Mixup / CutMix settings
    use_mixup: bool = True
    mixup_alpha: float = 0.8
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.5


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
        """Update EMA parameters"""
        ema_params = dict(self.ema.named_parameters())
        model_params = dict(model.named_parameters())
        
        for name, ema_p in ema_params.items():
            if name in model_params:
                model_p = model_params[name].detach()
                ema_p.mul_(self.decay).add_(model_p, alpha=1.0 - self.decay)
    
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
# Core QAViT Components (from baseline)
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
# Attention Branches (from baseline - keeping for completeness)
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
    """Stabilized CCF-FFN with normalization"""
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
# NEW: CNN Stem (Lateral Backbone)
# ============================================================================
class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block for CNN stem"""
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
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class CNNStemModel(nn.Module):
    """
    CNN Lateral Backbone for HQA-ViT
    Produces features at 8×8 resolution for stages 2, 3, 4
    """
    def __init__(self, in_ch=3, c2=64, c3=128, c4=256, norm_layer=nn.LayerNorm):
        super().__init__()
        
        # Stem: 32×32 -> 16×16
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        
        # Stage 1: 16×16 -> 8×8
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            ConvNeXtBlock(c2),
        )
        
        # Stage 2: 8×8 (project to c3)
        self.stage2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=1),
            nn.BatchNorm2d(c3),
            ConvNeXtBlock(c3),
        )
        
        # Stage 3: 8×8 (project to c4)
        self.stage3 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=1),
            nn.BatchNorm2d(c4),
            ConvNeXtBlock(c4),
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, 32, 32]
        Returns:
            F2: [B, c2, 8, 8]
            F3: [B, c3, 8, 8]
            F4: [B, c4, 8, 8]
        """
        x = self.stem(x)      # [B, 32, 16, 16]
        F2 = self.stage1(x)   # [B, c2, 8, 8]
        F3 = self.stage2(F2)  # [B, c3, 8, 8]
        F4 = self.stage3(F3)  # [B, c4, 8, 8]
        
        return F2, F3, F4


# ============================================================================
# NEW: LMFAdapter (Local Multi-scale Feature Adapter)
# ============================================================================
class LMFAdapter(nn.Module):
    """
    Local Multi-scale Feature Adapter
    Converts CNN features to tokens
    """
    def __init__(self, in_channels: int, embed_dim: int, target_hw: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.target_hw = target_hw
        
        # Multi-scale depthwise convolutions
        self.dwconv_3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.dwconv_5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        
        # Projection: 3*in_channels -> embed_dim
        self.proj = nn.Conv2d(3 * in_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()
    
    def forward(self, F):
        """
        Args:
            F: [B, C, H, W] CNN feature map
        Returns:
            A: [B, N, embed_dim] adapted tokens
        """
        B, C, H, W = F.shape
        
        # Multi-scale branches
        F1 = self.dwconv_3x3(F)  # local
        F2 = self.dwconv_5x5(F)  # broader
        F3 = F                    # identity
        
        # Concatenate
        F_cat = torch.cat([F1, F2, F3], dim=1)  # [B, 3C, H, W]
        
        # Project to embed_dim
        F_proj = self.proj(F_cat)  # [B, embed_dim, H, W]
        
        # Resize if needed
        if H != self.target_hw or W != self.target_hw:
            F_proj = F.interpolate(F_proj, size=(self.target_hw, self.target_hw), 
                                   mode='bilinear', align_corners=False)
        
        # Reshape to tokens
        A = F_proj.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        A = self.norm(A)
        A = self.act(A)
        
        return A


# ============================================================================
# NEW: RRCV (Reverse Reconstruction & Refinement)
# ============================================================================
class RRCV(nn.Module):
    """
    Reverse Reconstruction CNN-Variants
    Refines tokens through CNN operations
    """
    def __init__(self, embed_dim: int, rec_channels: int = 64, num_blocks: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.rec_channels = rec_channels
        
        # Reverse embedding: tokens -> feature map
        self.reverse_proj = nn.Conv2d(embed_dim, rec_channels, kernel_size=1)
        
        # Reconstruction blocks
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(rec_channels) for _ in range(num_blocks)
        ])
        
        # Re-embedding: feature map -> tokens
        self.reembed_proj = nn.Conv2d(rec_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Residual scaling
        self.beta = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, A, H: int, W: int):
        """
        Args:
            A: [B, N, embed_dim] tokens
            H, W: spatial dimensions
        Returns:
            R: [B, N, embed_dim] refined tokens
        """
        B, N, C = A.shape
        
        # Reverse embedding: tokens -> feature map
        X = A.permute(0, 2, 1).view(B, C, H, W)  # [B, embed_dim, H, W]
        R_in = self.reverse_proj(X)              # [B, rec_channels, H, W]
        
        # Reconstruction through CNN blocks
        R_feat = R_in
        for block in self.blocks:
            R_feat = block(R_feat)
        
        # Re-embedding: feature map -> tokens
        R_proj = self.reembed_proj(R_feat)                    # [B, embed_dim, H, W]
        R_tokens = R_proj.flatten(2).transpose(1, 2)          # [B, N, embed_dim]
        R_tokens = self.norm(R_tokens)
        
        # Residual connection
        R_tokens = A + self.beta * R_tokens
        
        return R_tokens


# ============================================================================
# NEW: SplitFusion (75% Additive / 25% Concatenative)
# ============================================================================
class SplitFusion(nn.Module):
    """
    SplitFusion: 75% additive + 25% concatenative fusion
    """
    def __init__(self, embed_dim: int, use_learnable_weights: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Additive branch gating
        self.gate_norm = nn.LayerNorm(embed_dim)
        self.gate_fc = nn.Linear(embed_dim, embed_dim)
        
        # Concatenative branch
        self.cat_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Learnable fusion weights
        if use_learnable_weights:
            self.fusion_weights = nn.Parameter(torch.tensor([0.75, 0.25]))
        else:
            self.register_buffer('fusion_weights', torch.tensor([0.75, 0.25]))
        
        self.final_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, T_in, R):
        """
        Args:
            T_in: [B, N, embed_dim] transformer tokens
            R: [B, N, embed_dim] CNN-adapted tokens
        Returns:
            T_fused: [B, N, embed_dim] fused tokens
        """
        # Additive branch with gating
        T_add = T_in + R
        gate = torch.sigmoid(self.gate_fc(self.gate_norm(T_add)))
        A_add_scaled = gate * R
        T_add_out = T_in + A_add_scaled
        
        # Concatenative branch
        T_cat = torch.cat([T_in, R], dim=-1)  # [B, N, 2*embed_dim]
        T_cat_out = T_in + self.cat_mlp(T_cat)
        
        # Weighted fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        T_fused = weights[0] * T_add_out + weights[1] * T_cat_out
        
        T_fused = self.final_norm(T_fused)
        
        return T_fused


# ============================================================================
# NEW: TokenLearner (Token Compression)
# ============================================================================
class TokenLearner(nn.Module):
    """
    TokenLearner: Compress N tokens to M learned tokens
    """
    def __init__(self, in_dim: int, num_out_tokens: int = 16):
        super().__init__()
        self.num_out_tokens = num_out_tokens
        
        # Learnable attention for token selection
        self.attention = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, num_out_tokens),
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, N, embed_dim]
        Returns:
            x_compressed: [B, M, embed_dim] where M = num_out_tokens
        """
        B, N, C = x.shape
        
        # Compute attention scores
        scores = self.attention(x)  # [B, N, M]
        scores = F.softmax(scores, dim=1)  # softmax over N
        
        # Weighted aggregation
        scores = scores.transpose(1, 2)  # [B, M, N]
        x_compressed = torch.bmm(scores, x)  # [B, M, C]
        
        return x_compressed


class TokenUpMix(nn.Module):
    """Reconstruct from M tokens back to N tokens - FIXED lightweight version"""
    def __init__(self, embed_dim: int, num_in_tokens: int, num_out_tokens: int):
        super().__init__()
        self.num_in_tokens = num_in_tokens
        self.num_out_tokens = num_out_tokens
        
        # Use transposed attention instead of massive linear layer
        self.upsample_attn = nn.Linear(num_in_tokens, num_out_tokens)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x_compressed):
        """
        Args:
            x_compressed: [B, M, embed_dim]
        Returns:
            x: [B, N, embed_dim]
        """
        B, M, C = x_compressed.shape
        
        # Transpose and upsample via attention weights
        x_t = x_compressed.transpose(1, 2)  # [B, C, M]
        x_up = self.upsample_attn(x_t)      # [B, C, N]
        x = x_up.transpose(1, 2)            # [B, N, C]
        x = self.norm(x)
        
        return x


# ============================================================================
# QuadAttentionBlock (unchanged from baseline)
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
# QuadAttentionBlock with TokenLearner Wrapper
# ============================================================================
class QuadBlockWithTokenLearner(nn.Module):
    """Wrapper that adds TokenLearner compression before QuadAttention"""
    def __init__(self, config, global_bank, drop_path=0., use_token_learner=True):
        super().__init__()
        self.use_token_learner = use_token_learner
        
        if use_token_learner:
            self.token_learner = TokenLearner(config.embed_dim, config.num_learned_tokens)
            self.token_upmix = TokenUpMix(config.embed_dim, config.num_learned_tokens, 
                                          (config.img_size // config.patch_size) ** 2)
        
        self.quad_block = QuadAttentionBlock(config, global_bank, drop_path)
    
    def forward(self, x):
        """
        Args:
            x: [B, N, embed_dim]
        Returns:
            x: [B, N, embed_dim]
        """
        if self.use_token_learner:
            # Compress
            x_compressed = self.token_learner(x)  # [B, M, embed_dim]
            
            # Apply QuadAttention in compressed space
            x_compressed = self.quad_block(x_compressed)
            
            # Reconstruct
            x = self.token_upmix(x_compressed)  # [B, N, embed_dim]
        else:
            x = self.quad_block(x)
        
        return x


# ============================================================================
# Main HQA-ViT Model
# ============================================================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class HQAViT(nn.Module):
    """
    HQA-ViT: Hybrid Quad-Attention Vision Transformer
    Integrates CNN lateral features into Transformer stages
    """
    def __init__(self, config: HQAViTConfig):
        super().__init__()
        self.config = config
        self.num_patches = (config.img_size // config.patch_size) ** 2
        self.H = self.W = config.img_size // config.patch_size  # 8 for 32×32
        
        # ViT components
        self.patch_embed = PatchEmbed(config.img_size, config.patch_size, config.in_channels, config.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, config.embed_dim))
        self.pos_drop = nn.Dropout(config.dropout)
        
        self.global_bank = GlobalTokenBank(config.global_bank_size, config.embed_dim)
        
        # CNN Stem (lateral backbone)
        self.cnn_stem = CNNStemModel(
            in_ch=config.in_channels,
            c2=config.cnn_c2,
            c3=config.cnn_c3,
            c4=config.cnn_c4
        )
        
        # LMF Adapters for each stage
        self.lmfa2 = LMFAdapter(config.cnn_c2, config.embed_dim, target_hw=self.H)
        self.lmfa3 = LMFAdapter(config.cnn_c3, config.embed_dim, target_hw=self.H)
        self.lmfa4 = LMFAdapter(config.cnn_c4, config.embed_dim, target_hw=self.H)
        
        # RRCV modules for each stage
        self.rrcv2 = RRCV(config.embed_dim, config.rrcv_channels, config.rrcv_num_blocks)
        self.rrcv3 = RRCV(config.embed_dim, config.rrcv_channels, config.rrcv_num_blocks)
        self.rrcv4 = RRCV(config.embed_dim, config.rrcv_channels, config.rrcv_num_blocks)
        
        # SplitFusion modules
        self.fuse2 = SplitFusion(config.embed_dim)
        self.fuse3 = SplitFusion(config.embed_dim)
        self.fuse4 = SplitFusion(config.embed_dim)
        
        # Transformer blocks divided into stages
        # Stage 1: blocks 0-1 (2 blocks, no fusion)
        # Stage 2: blocks 2-3 (2 blocks, fusion with F2)
        # Stage 3: blocks 4-7 (4 blocks, fusion with F3)
        # Stage 4: blocks 8-9 (2 blocks, fusion with F4)
        dpr = [x.item() for x in torch.linspace(0, config.drop_path, config.depth)]
        
        self.stage1_blocks = nn.ModuleList([
            QuadBlockWithTokenLearner(config, self.global_bank, dpr[i], config.use_token_learner)
            for i in range(0, 2)  # blocks 0-1
        ])
        
        self.stage2_blocks = nn.ModuleList([
            QuadBlockWithTokenLearner(config, self.global_bank, dpr[i], config.use_token_learner)
            for i in range(2, 4)  # blocks 2-3
        ])
        
        self.stage3_blocks = nn.ModuleList([
            QuadBlockWithTokenLearner(config, self.global_bank, dpr[i], config.use_token_learner)
            for i in range(4, 10)  # blocks 4-7 (4 blocks!)
        ])
        
        self.stage4_blocks = nn.ModuleList([
            QuadBlockWithTokenLearner(config, self.global_bank, dpr[i], config.use_token_learner)
            for i in range(10, 12)  # blocks 8-9
        ])
        
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        
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
        """
        Args:
            x: [B, 3, 32, 32]
        Returns:
            logits: [B, num_classes]
        """
        B = x.shape[0]
        
        # Extract CNN lateral features
        F2, F3, F4 = self.cnn_stem(x)  # Each [B, C, 8, 8]
        
        # Adapt CNN features to tokens
        A2 = self.lmfa2(F2)  # [B, 64, embed_dim]
        A3 = self.lmfa3(F3)  # [B, 64, embed_dim]
        A4 = self.lmfa4(F4)  # [B, 64, embed_dim]
        
        # Refine through RRCV
        R2 = self.rrcv2(A2, self.H, self.W)
        R3 = self.rrcv3(A3, self.H, self.W)
        R4 = self.rrcv4(A4, self.H, self.W)
        
        # ViT path: Patch embedding
        T = self.patch_embed(x)  # [B, 64, embed_dim]
        T = T + self.pos_embed
        T = self.pos_drop(T)
        
        # Stage 1: Pure ViT (no fusion)
        for block in self.stage1_blocks:
            T = block(T)
        
        # Stage 2: Fusion with F2
        T = self.fuse2(T, R2)
        for block in self.stage2_blocks:
            T = block(T)
        
        # Stage 3: Fusion with F3
        T = self.fuse3(T, R3)
        for block in self.stage3_blocks:
            T = block(T)
        
        # Stage 4: Fusion with F4
        T = self.fuse4(T, R4)
        for block in self.stage4_blocks:
            T = block(T)
        
        # Classification
        T = self.norm(T)
        T = T.mean(dim=1)
        logits = self.head(T)
        
        return logits


# ============================================================================
# Data Loading
# ============================================================================
def get_cifar100_loaders(config: TrainingConfig):
    """CIFAR-100 data loading"""
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    
    # Stronger DeiT-style augmentation: keep spatial jitter + RandAugment,
    # color jitter and random erasing for robustness on CIFAR-sized images.
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), value='random')
    ])
    
    val_transform = transforms.Compose([
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
def rand_bbox(size, lam):
    """Generate random bbox for CutMix

    Args:
        size: tensor size tuple (B, C, H, W)
        lam: lambda sampled from beta
    Returns:
        x1, y1, x2, y2 (int coords)
    """
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    # convert to int
    return int(x1), int(y1), int(x2), int(y2)

def train_epoch(model, loader, optimizer, scheduler, scaler, config, epoch, monitor, model_ema=None):
    """Training loop with EMA"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # Apply MixUp / CutMix augmentation on the batch with configured probability
        use_mix = None
        lam = 1.0
        if config.use_cutmix and np.random.rand() < config.mix_prob:
            # CutMix
            rand_index = torch.randperm(inputs.size(0)).cuda()
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), np.random.beta(config.cutmix_alpha, config.cutmix_alpha))
            # x: [B, C, H, W]
            inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[rand_index, :, bby1:bby2, bbx1:bbx2]
            targets_a, targets_b = targets, targets[rand_index]
            # adjust lambda to exactly match pixel ratio
            W = inputs.size(3)
            H = inputs.size(2)
            lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / float(W * H))
            use_mix = 'cutmix'
        elif config.use_mixup and np.random.rand() < config.mix_prob:
            # MixUp
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
            
            # Per-layer clipping for CNN and DWConv
            for name, param in model.named_parameters():
                if ('cnn_stem' in name or 'dwconv' in name) and param.grad is not None:
                    torch.nn.utils.clip_grad_norm_([param], max_norm=0.1)
            
            # Monitor gradients
            detailed = (batch_idx % 200 == 0)
            grad_norm, param_norm, grad_stats, layer_stats = monitor.log_gradients(model, detailed=detailed)
            
            # Check for explosion
            if monitor.check_explosion(threshold=50.0):
                print(f"\n{'='*100}")
                print(f"🚨 GRADIENT EXPLOSION DETECTED".center(100))
                print(f"{'='*100}\n")
            
            # Global gradient clipping
            if config.grad_clip_mode == 'norm':
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            elif config.grad_clip_mode == 'value':
                torch.nn.utils.clip_grad_value_(model.parameters(), config.max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            # Update EMA
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
def validate(model, loader, use_ema=False):
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
    print("\n" + "="*100)
    print("HQA-ViT CIFAR-100 TRAINING - HYBRID QUAD-ATTENTION WITH CNN AGGREGATION".center(100))
    print("="*100)
    
    # Configs
    model_config = HQAViTConfig()
    train_config = TrainingConfig()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Hardware Configuration:")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA: {torch.version.cuda}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
    else:
        print(f"   Device: CPU")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   FlashAttention: {'Available ✓' if HAS_FLASH_ATTN else 'Not Available ✗'}")
    
    # Create checkpoint directory
    Path(train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Build model
    print(f"\n{'='*100}")
    print("MODEL INITIALIZATION".center(100))
    print(f"{'='*100}")
    model = HQAViT(model_config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Memory (FP32): {total_params*4/1024**2:.2f} MB")
    print(f"   Memory (FP16): {total_params*2/1024**2:.2f} MB")
    
    # Component breakdown
    print(f"\n📦 Component Breakdown:")
    cnn_params = sum(p.numel() for p in model.cnn_stem.parameters())
    lmfa_params = sum(p.numel() for p in model.lmfa2.parameters()) + \
                  sum(p.numel() for p in model.lmfa3.parameters()) + \
                  sum(p.numel() for p in model.lmfa4.parameters())
    rrcv_params = sum(p.numel() for p in model.rrcv2.parameters()) + \
                  sum(p.numel() for p in model.rrcv3.parameters()) + \
                  sum(p.numel() for p in model.rrcv4.parameters())
    fusion_params = sum(p.numel() for p in model.fuse2.parameters()) + \
                    sum(p.numel() for p in model.fuse3.parameters()) + \
                    sum(p.numel() for p in model.fuse4.parameters())
    vit_params = total_params - cnn_params - lmfa_params - rrcv_params - fusion_params
    
    print(f"   CNN Stem: {cnn_params:,} ({100*cnn_params/total_params:.1f}%)")
    print(f"   LMFAdapters: {lmfa_params:,} ({100*lmfa_params/total_params:.1f}%)")
    print(f"   RRCV Modules: {rrcv_params:,} ({100*rrcv_params/total_params:.1f}%)")
    print(f"   SplitFusion: {fusion_params:,} ({100*fusion_params/total_params:.1f}%)")
    print(f"   ViT Core: {vit_params:,} ({100*vit_params/total_params:.1f}%)")
    
    # Data loaders
    print(f"\n{'='*100}")
    print("DATA PREPARATION".center(100))
    print(f"{'='*100}")
    print(f"\n📚 Loading CIFAR-100...")
    train_loader, val_loader = get_cifar100_loaders(train_config)
    print(f"   Train samples: {len(train_loader.dataset):,}")
    print(f"   Val samples: {len(val_loader.dataset):,}")
    print(f"   Batch size: {train_config.batch_size}")
    print(f"   Train batches per epoch: {len(train_loader)}")
    print(f"   Val batches per epoch: {len(val_loader)}")
    
    # Optimizer
    print(f"\n{'='*100}")
    print("TRAINING CONFIGURATION".center(100))
    print(f"{'='*100}")
    
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
    
    # EMA
    model_ema = None
    if train_config.use_ema:
        print(f"\n✓ Initializing EMA (decay={train_config.ema_decay})")
        model_ema = ModelEMA(model, decay=train_config.ema_decay, device=device)
    
    print(f"\n{'Training Hyperparameters':-^100}")
    print(f"   Epochs: {train_config.epochs}")
    print(f"   Warmup epochs: {train_config.warmup_epochs}")
    print(f"   Base learning rate: {train_config.base_lr}")
    print(f"   Weight decay: {train_config.weight_decay}")
    print(f"   Label smoothing: {train_config.label_smoothing}")
    print(f"   Gradient clip: {train_config.max_grad_norm} ({train_config.grad_clip_mode})")
    print(f"   Mixed precision (AMP): {train_config.use_amp} ({train_config.amp_dtype})")
    print(f"   EMA: {train_config.use_ema}")
    print(f"   TokenLearner: {model_config.use_token_learner}")
    
    print(f"\n{'Architecture Configuration':-^100}")
    print(f"   Embedding dim: {model_config.embed_dim}")
    print(f"   Depth: {model_config.depth}")
    print(f"   Heads: {model_config.num_heads}")
    print(f"   CNN channels: [{model_config.cnn_c2}, {model_config.cnn_c3}, {model_config.cnn_c4}]")
    print(f"   RRCV channels: {model_config.rrcv_channels}")
    print(f"   Fusion stages: {model_config.fusion_stages}")
    if model_config.use_token_learner:
        print(f"   Learned tokens: {model_config.num_learned_tokens}")
    
    # Training loop
    print(f"\n{'='*100}")
    print("TRAINING STARTED".center(100))
    print(f"{'='*100}\n")
    
    best_acc = 0
    best_ema_acc = 0
    train_start_time = time.time()
    
    for epoch in range(1, train_config.epochs + 1):
        epoch_start_time = time.time()
        
        # Update EMA decay (warmup)
        if train_config.use_ema and epoch <= train_config.warmup_epochs:
            current_decay = train_config.ema_decay_warmup + \
                           (train_config.ema_decay - train_config.ema_decay_warmup) * \
                           (epoch / train_config.warmup_epochs)
            model_ema.set_decay(current_decay)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, 
            train_config, epoch, monitor, model_ema
        )
        
        # Validate
        if epoch % train_config.eval_freq == 0:
            val_loss, val_acc = validate(model, val_loader)
            
            # Validate EMA model
            ema_val_loss, ema_val_acc = 0, 0
            if model_ema is not None:
                ema_val_loss, ema_val_acc = validate(model_ema.ema, val_loader, use_ema=True)
            
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - train_start_time
            
            print(f"\n{'='*100}")
            print(f"EPOCH {epoch}/{train_config.epochs} SUMMARY".center(100))
            print(f"{'='*100}")
            print(f"{'Metric':<30} {'Train':>15} {'Val':>15} {'EMA Val':>15} {'Details':>20}")
            print("-"*100)
            print(f"{'Loss':<30} {train_loss:>15.4f} {val_loss:>15.4f} {ema_val_loss:>15.4f}")
            print(f"{'Accuracy (%)':<30} {train_acc:>15.2f} {val_acc:>15.2f} {ema_val_acc:>15.2f}")
            print(f"{'Time (seconds)':<30} {epoch_time:>15.1f} {'':>15} {'':>15} {f'{epoch_time/60:.1f} min':>20}")
            print(f"{'Total Time':<30} {total_time:>15.1f} {'':>15} {'':>15} {f'{total_time/3600:.2f} hrs':>20}")
            print(f"{'Best Val Acc (%)':<30} {max(best_acc, val_acc):>15.2f} {max(best_ema_acc, ema_val_acc):>15.2f}")
            print(f"{'Learning Rate':<30} {optimizer.param_groups[0]['lr']:>15.6f}")
            print(f"{'Gradient Norm':<30} {monitor.grad_norms[-1]:>15.4f}")
            print(f"{'Gradient Explosions':<30} {monitor.explosion_count:>15}")
            
            if device.type == 'cuda':
                current_vram = torch.cuda.max_memory_allocated() / 1024**3
                print(f"{'Peak VRAM (GB)':<30} {current_vram:>15.2f}")
            
            print(f"{'='*100}\n")
            
            # Save best model (regular)
            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'train_config': train_config,
                    'model_config': model_config,
                    'best_acc': best_acc,
                }
                torch.save(checkpoint, f"{train_config.checkpoint_dir}/best_model.pth")
                print(f"✅ Best model saved! Val Acc: {best_acc:.2f}%")
                print(f"   Location: {train_config.checkpoint_dir}/best_model.pth\n")
            
            # Save best EMA model
            if model_ema is not None and ema_val_acc > best_ema_acc:
                best_ema_acc = ema_val_acc
                checkpoint_ema = {
                    'epoch': epoch,
                    'model_state_dict': model_ema.ema.state_dict(),
                    'val_acc': ema_val_acc,
                    'val_loss': ema_val_loss,
                    'train_config': train_config,
                    'model_config': model_config,
                    'best_ema_acc': best_ema_acc,
                }
                torch.save(checkpoint_ema, f"{train_config.checkpoint_dir}/best_model_ema.pth")
                print(f"✅ Best EMA model saved! EMA Val Acc: {best_ema_acc:.2f}%")
                print(f"   Location: {train_config.checkpoint_dir}/best_model_ema.pth\n")
        
        # Save checkpoint
        if epoch % train_config.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_acc': val_acc if epoch % train_config.eval_freq == 0 else None,
            }
            torch.save(checkpoint, f"{train_config.checkpoint_dir}/checkpoint_epoch_{epoch}.pth")
            
            if model_ema is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_ema.ema.state_dict(),
                }, f"{train_config.checkpoint_dir}/checkpoint_epoch_{epoch}_ema.pth")
            
            print(f"💾 Checkpoint saved: epoch_{epoch}.pth\n")
    
    # Training complete
    total_training_time = time.time() - train_start_time
    
    print(f"\n{'='*100}")
    print("TRAINING COMPLETE".center(100))
    print(f"{'='*100}")
    print(f"\n{'Final Results':-^100}")
    print(f"   Best Validation Accuracy: {best_acc:.2f}%")
    if model_ema is not None:
        print(f"   Best EMA Validation Accuracy: {best_ema_acc:.2f}%")
    print(f"   Total Training Time: {total_training_time/3600:.2f} hours")
    print(f"   Average Time per Epoch: {total_training_time/train_config.epochs/60:.1f} minutes")
    print(f"   Total Gradient Explosions: {monitor.explosion_count}")
    
    print(f"\n{'Saved Models':-^100}")
    print(f"   Best model: {train_config.checkpoint_dir}/best_model.pth")
    if model_ema is not None:
        print(f"   Best EMA model: {train_config.checkpoint_dir}/best_model_ema.pth")
    print(f"   Latest checkpoint: {train_config.checkpoint_dir}/checkpoint_epoch_{train_config.epochs}.pth")
    
    print(f"\n{'='*100}")
    print("🎉 HQA-ViT TRAINING COMPLETE!".center(100))
    print(f"{'='*100}\n")


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print("\n" + "="*100)
        print("CUDA OPTIMIZATIONS".center(100))
        print("="*100)
        print(f"   cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        print(f"   TF32 (Matmul): {torch.backends.cuda.matmul.allow_tf32}")
        print(f"   TF32 (cuDNN): {torch.backends.cudnn.allow_tf32}")
        print("="*100)
    
    main()