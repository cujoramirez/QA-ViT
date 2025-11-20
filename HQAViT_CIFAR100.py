"""
HQA-ViT CIFAR-100 - Hybrid Quad-Attention Vision Transformer
Integrates CNN lateral features via LMFA + RRCV + SplitFusion + TokenLearner
Includes comprehensive model compilation, architecture visualization, and gradient debugging
Optimized for RTX 3060 6GB and RTX 5070 12GB
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
class QAViTConfig:
    """HQA-ViT Tiny Config optimized for CIFAR-100"""
    img_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 100
    embed_dim: int = 192  # d
    
    # Stage Depths: [Stage 1 (Pure), Stage 2 (Hybrid), Stage 3 (Hybrid), Stage 4 (Hybrid)]
    # Total depth approx 9-10 blocks for Tiny (~5M params target)
    depths: Tuple[int, ...] = (2, 2, 3, 2) 
    
    num_heads: int = 4
    compress_ratio: int = 4  # d' = d/4 = 48
    bottleneck_ratio: int = 2  # r = d/2 = 96
    mlp_ratio: float = 2.0  # Reduced slightly for Hybrid to save params
    global_bank_size: int = 16
    dropout: float = 0.1
    drop_path: float = 0.1
    window_size: int = 4  # for SWA
    dilation_factors: Tuple[int, ...] = (1, 2)  # for MSDA
    landmark_pooling_stride: int = 2
    num_channel_groups: int = 6
    linformer_k: int = 32  # compressed length for Linformer
    
    # Hybrid Configuration
    use_hybrid: bool = True
    cnn_channels: Tuple[int, ...] = (64, 128, 256) # C2, C3, C4
    rrcv_channels: int = 64
    rrcv_blocks: int = 1
    
    # TokenLearner Configuration
    use_token_learner: bool = True
    token_learner_k: int = 16  # Compress 64 tokens -> 16 tokens for attention


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Hardware-specific
    batch_size: int = 256
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    
    # Training
    epochs: int = 300
    warmup_epochs: int = 20
    base_lr: float = 5e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.05
    
    # Regularization
    label_smoothing: float = 0.1
    
    # Gradient control
    max_grad_norm: float = 0.5
    grad_clip_mode: str = 'norm'
    
    # Monitoring
    print_freq: int = 50
    eval_freq: int = 1
    save_freq: int = 10
    
    # AMP & Compilation
    use_amp: bool = True
    amp_dtype: str = 'bfloat16'  # 'float16' or 'bfloat16'
    use_compile: bool = False
    compile_mode: str = 'default'
    
    # Stabilization
    use_ema: bool = True
    ema_decay: float = 0.9999
    
    # Paths
    data_root: str = "./data"
    checkpoint_dir: str = "./checkpoints_hqa"


# ============================================================================
# HQA Hybrid Modules (CTA-Net Implementation)
# ============================================================================

class ConvNeXtBlock(nn.Module):
    """Lightweight ConvNeXt-style block for CNN Stem"""
    def __init__(self, dim, kernel_size=3, mlp_ratio=2.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, 
                                padding=kernel_size//2, groups=dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim) * 0.1)
        
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # [B, C, H, W]
        return input + x

class CNNStemModel(nn.Module):
    """Lateral CNN Backbone (Stages 2, 3, 4 features)"""
    def __init__(self, in_ch=3, c2=64, c3=128, c4=256):
        super().__init__()
        # Initial Stem: 32x32
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        # Downsample to 16x16
        self.downsample1 = nn.Conv2d(64, 64, kernel_size=2, stride=2)
        
        # Stage 2: 16x16 processing -> downsample to 8x8 -> F2
        self.stage2_block = ConvNeXtBlock(64, kernel_size=3)
        self.to_c2 = nn.Sequential(
            nn.Conv2d(64, c2, kernel_size=2, stride=2), # Downsample to 8x8
        )
        
        # Stage 3: 8x8 processing -> F3
        self.stage3_proj = nn.Conv2d(c2, c3, kernel_size=1)
        self.stage3_block = ConvNeXtBlock(c3, kernel_size=3)
        
        # Stage 4: 8x8 processing -> F4
        self.stage4_proj = nn.Conv2d(c3, c4, kernel_size=1)
        self.stage4_block = ConvNeXtBlock(c4, kernel_size=3)
        
    def forward(self, x):
        # x: [B, 3, 32, 32]
        x = self.stem(x)         # [B, 64, 32, 32]
        x = self.downsample1(x)  # [B, 64, 16, 16]
        
        # Stage 2 Path
        s2 = self.stage2_block(x)
        F2 = self.to_c2(s2)      # [B, C2, 8, 8]
        
        # Stage 3 Path
        s3 = self.stage3_proj(F2)
        F3 = self.stage3_block(s3) # [B, C3, 8, 8]
        
        # Stage 4 Path
        s4 = self.stage4_proj(F3)
        F4 = self.stage4_block(s4) # [B, C4, 8, 8]
        
        return F2, F3, F4

class LMFAdapter(nn.Module):
    """Local Multi-scale Feature Adapter"""
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        # Multi-scale branches
        self.dwconv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.dwconv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        # Identity branch implicitly exists
        
        # Fusion and Projection
        self.proj = nn.Conv2d(in_channels * 3, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()
        
    def forward(self, F):
        # F: [B, C, H, W]
        f3 = self.dwconv3(F)
        f5 = self.dwconv5(F)
        f_cat = torch.cat([F, f3, f5], dim=1) # [B, 3C, H, W]
        
        x = self.proj(f_cat) # [B, d, H, W]
        
        # Flatten to tokens
        B, d, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # [B, N, d]
        x = self.norm(x)
        x = self.act(x)
        return x

class RRCV(nn.Module):
    """Reverse Reconstruction & Refinement Module"""
    def __init__(self, embed_dim, rec_channels=64, num_blocks=1):
        super().__init__()
        self.rec_channels = rec_channels
        
        # Reverse Embedding
        self.reverse_proj = nn.Conv2d(embed_dim, rec_channels, kernel_size=1)
        
        # Reconstruction Blocks (Denoising)
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(rec_channels, kernel_size=3) for _ in range(num_blocks)
        ])
        
        # Patch Re-embedding
        self.re_embed = nn.Conv2d(rec_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Stabilization scale
        self.beta = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, A, H, W):
        # A: [B, N, d]
        B, N, d = A.shape
        
        # Reshape to spatial
        X = A.transpose(1, 2).view(B, d, H, W)
        
        # Reverse Project
        R = self.reverse_proj(X)
        
        # Refine
        for blk in self.blocks:
            R = blk(R)
            
        # Re-embed
        R_out = self.re_embed(R)
        R_out = R_out.flatten(2).transpose(1, 2) # [B, N, d]
        R_out = self.norm(R_out)
        
        return A + self.beta * R_out

class SplitFusion(nn.Module):
    """75% Additive / 25% Concatenative Fusion"""
    def __init__(self, embed_dim):
        super().__init__()
        # Additive Gate
        self.add_gate_ln = nn.LayerNorm(embed_dim)
        self.add_gate_linear = nn.Linear(embed_dim, embed_dim)
        
        # Concatenative MLP
        self.cat_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Fusion Weights (Learnable logits for softmax)
        # Initialized to approx 0.75 / 0.25
        self.fusion_logits = nn.Parameter(torch.tensor([1.1, -1.1])) 
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, T_in, R):
        # T_in: Transformer Tokens, R: RRCV Tokens
        
        # Additive Branch (75%)
        T_sum = T_in + R
        gate = torch.sigmoid(self.add_gate_linear(self.add_gate_ln(T_sum)))
        T_add_out = T_in + (gate * R)
        
        # Concatenative Branch (25%)
        T_cat = torch.cat([T_in, R], dim=-1)
        T_cat_res = self.cat_mlp(T_cat)
        T_cat_out = T_in + T_cat_res
        
        # Weighted Sum
        weights = F.softmax(self.fusion_logits, dim=0)
        T_fused = weights[0] * T_add_out + weights[1] * T_cat_out
        
        return self.final_norm(T_fused)

class TokenLearner(nn.Module):
    """Compress N tokens to M salient tokens"""
    def __init__(self, in_dim, num_tokens=16):
        super().__init__()
        self.num_tokens = num_tokens
        # Attention map generator: Project to M maps
        self.attn_gen = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, num_tokens),
            nn.Softmax(dim=1) # Softmax over Spatial dim N (done in fwd)
        )
        
    def forward(self, x):
        # x: [B, N, d]
        # map: [B, N, M]
        attn_maps = self.attn_gen(x) 
        # Transpose to [B, M, N] for multiplication
        attn_maps = attn_maps.transpose(1, 2)
        
        # [B, M, N] @ [B, N, d] -> [B, M, d]
        return torch.matmul(attn_maps, x)

class TokenUpsampler(nn.Module):
    """Reconstruct N tokens from M compressed tokens"""
    def __init__(self, in_dim, out_tokens=64):
        super().__init__()
        self.out_tokens = out_tokens
        self.proj = nn.Linear(in_dim, in_dim) # Mixing
        self.norm = nn.LayerNorm(in_dim)
        
    def forward(self, x_compressed, N_target):
        # x_compressed: [B, M, d]
        # Simple interpolation for efficiency
        B, M, d = x_compressed.shape
        # Reshape to "spatial" 1D for interpolation
        x = x_compressed.transpose(1, 2) # [B, d, M]
        x = F.interpolate(x, size=N_target, mode='linear', align_corners=False)
        x = x.transpose(1, 2) # [B, N, d]
        return self.norm(self.proj(x))

class ModelEMA:
    """Exponential Moving Average for Model Weights"""
    def __init__(self, model, decay=0.9999):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)
            
    @torch.no_grad()
    def update(self, model):
        for ema_v, model_v in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if model_v.dtype.is_floating_point:
                ema_v.copy_(ema_v * self.decay + model_v * (1.0 - self.decay))

# ============================================================================
# Architecture Analysis & Monitoring (Preserved)
# ============================================================================
class ArchitectureAnalyzer:
    """Detailed architecture analysis and visualization"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.layer_info = OrderedDict()
        
    def analyze_architecture(self):
        """Comprehensive architecture analysis"""
        print("\n" + "="*100)
        print("HQA-ViT ARCHITECTURE ANALYSIS".center(100))
        print("="*100)
        self._print_overall_stats()
        self._print_component_breakdown()
        self._print_computational_complexity()
        
    def _print_overall_stats(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n{'OVERALL STATISTICS':-^100}")
        print(f"{'Total Parameters':<40} {total_params:>20,} {f'{total_params/1e6:.2f}M params':>35}")
        print(f"{'Embed Dim':<40} {self.config.embed_dim:>20}")
        print(f"{'Depths':<40} {str(self.config.depths):>20}")
        
    def _print_component_breakdown(self):
        print(f"\n{'COMPONENT BREAKDOWN':-^100}")
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Identify main parts
        parts = {
            'Stem (CNN)': self.model.cnn_stem,
            'Patch Embed': self.model.patch_embed,
            'Global Bank': self.model.global_bank,
            'Stage 1 (Pure)': self.model.stage1,
            'Stage 2 (Hybrid)': self.model.stage2,
            'Stage 3 (Hybrid)': self.model.stage3,
            'Stage 4 (Hybrid)': self.model.stage4,
            'Head': self.model.head
        }
        
        for name, module in parts.items():
            params = sum(p.numel() for p in module.parameters())
            pct = 100 * params / total_params
            print(f"{name:<40} {params:>20,} {pct:>14.2f}%")

    def _print_computational_complexity(self):
        # Simplified FLOPs estimation
        print(f"\n{'COMPUTATIONAL COMPLEXITY ESTIMATE':-^100}")
        print("Hybrid Architecture adds overhead via CNN Stem and Fusion.")
        print("TokenLearner reduces complexity in Quad-Attention blocks.")
        
class GradientMonitor:
    """Enhanced gradient monitoring"""
    def __init__(self):
        self.grad_norms = []
        self.explosion_count = 0
        
    def log_gradients(self, model, detailed=False):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.grad_norms.append(total_norm)
        return total_norm

    def check_explosion(self, threshold=50.0):
        if len(self.grad_norms) > 0 and self.grad_norms[-1] > threshold:
            self.explosion_count += 1
            return True
        return False

# ============================================================================
# Core Components
# ============================================================================
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training: return x
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
        self.global_k = nn.Parameter(torch.randn(1, bank_size, embed_dim) * 0.02)
        self.global_v = nn.Parameter(torch.randn(1, bank_size, embed_dim) * 0.02)
        self.write_norm = nn.LayerNorm(embed_dim)
        self.write_compression = nn.Linear(embed_dim, embed_dim)
        self.write_gate = nn.Linear(embed_dim, bank_size)
        self.register_buffer('update_count', torch.tensor(0))
        
    def read(self, batch_size: int):
        return (self.global_k.expand(batch_size, -1, -1),
                self.global_v.expand(batch_size, -1, -1))
    
    def write(self, tokens: torch.Tensor, residual: bool = True):
        if not self.training: return
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
            # Dynamic pad/slice if seq_len changes (e.g. due to TokenLearner)
            # For simplicity in HQA, we might bypass Linformer if N is small
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
        return (k_compressed.reshape(B, H, self.compressed_len, D),
                v_compressed.reshape(B, H, self.compressed_len, D))

def efficient_attention(q, k, v, dropout_p=0.0, training=True):
    if torch.isnan(q).any(): return torch.zeros_like(q)
    use_flash = (HAS_FLASH_ATTN and q.device.type == 'cuda' and training and q.dtype in (torch.float16, torch.bfloat16))
    if use_flash:
        try:
            q_t, k_t, v_t = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            output = flash_attn_func(q_t, k_t, v_t, dropout_p=dropout_p if training else 0.0, causal=False)
            return output.transpose(1, 2)
        except: pass
    
    # Handle bfloat16 fallback
    if q.dtype == torch.bfloat16:
        q, k, v = q.float(), k.float(), v.float()
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p if training else 0.0)
        return out.bfloat16()
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p if training else 0.0)

# ============================================================================
# Attention Branches
# ============================================================================
# (Modified slightly to handle variable sequence lengths if TokenLearner is active)

class EfficientSpatialWindowAttention(nn.Module):
    def __init__(self, config, global_bank):
        super().__init__()
        self.config = config
        self.global_bank = global_bank
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=True)
        self.linformer = LinformerCompression(config.window_size ** 2, config.linformer_k)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        # If N is small (TokenLearner compressed), skip windowing and do global attn
        if N <= self.config.window_size ** 2:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            # Skip Linformer for small seq
            k_bank, v_bank = self.global_bank.read(B)
            k_bank = k_bank.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v_bank = v_bank.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k_full = torch.cat([k.transpose(1,2), k_bank], dim=2)
            v_full = torch.cat([v.transpose(1,2), v_bank], dim=2)
            output = efficient_attention(q.transpose(1,2), k_full, v_full, self.dropout.p, self.training)
            output = output.transpose(1, 2).reshape(B, N, C)
            self.global_bank.write(self.norm(self.proj(output)))
            return output

        # Standard Window Logic (original)
        H = W = int(math.sqrt(N))
        x = x.view(B, H, W, C)
        # ... (Simplified for brevity, assuming 8x8 tokens fits in window logic or global fallback)
        # Since 8x8 tokens is small, we just treat it as global to avoid complex window partitioning on tiny grids
        x_flat = x.flatten(1, 2)
        qkv = self.qkv(x_flat).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_c, v_c = self.linformer(k.transpose(1,2), v.transpose(1,2)) # Compress
        k_bank, v_bank = self.global_bank.read(B)
        k_bank = k_bank.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_bank = v_bank.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_full = torch.cat([k_c, k_bank], dim=2)
        v_full = torch.cat([v_c, v_bank], dim=2)
        attn = efficient_attention(q.transpose(1,2), k_full, v_full, self.dropout.p, self.training)
        return self.proj(attn.transpose(1,2).reshape(B, N, C))

class EfficientMultiScaleDilatedAttention(nn.Module):
    # Simplified: if N is small, standard attn. Else apply dilated.
    def __init__(self, config, global_bank):
        super().__init__()
        self.config = config
        self.global_bank = global_bank
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        # For Tiny CIFAR (8x8 tokens), MSDA is approximated by global attn to save overhead
        qkv = self.qkv(x).reshape(B, N, 3, self.config.num_heads, C // self.config.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        k_bank, v_bank = self.global_bank.read(B)
        k_bank = k_bank.reshape(B, -1, self.config.num_heads, C // self.config.num_heads).transpose(1, 2)
        v_bank = v_bank.reshape(B, -1, self.config.num_heads, C // self.config.num_heads).transpose(1, 2)
        
        k_full = torch.cat([k.transpose(1,2), k_bank], dim=2)
        v_full = torch.cat([v.transpose(1,2), v_bank], dim=2)
        
        x = efficient_attention(q.transpose(1,2), k_full, v_full, self.dropout.p, self.training)
        x = self.proj(x.transpose(1, 2).reshape(B, N, C))
        self.global_bank.write(self.norm(x))
        return x

class EfficientChannelGroupAttention(nn.Module):
    def __init__(self, config, global_bank):
        super().__init__()
        self.config = config
        self.global_bank = global_bank
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.norm = nn.LayerNorm(config.embed_dim)
        # Simplified CGA for HQA-ViT (Param efficiency)
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        
    def forward(self, x):
        # Simplified to standard attn for stability in Hybrid mode
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.config.num_heads, C//self.config.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = efficient_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), 0.0, self.training)
        x = self.proj(x.transpose(1,2).reshape(B, N, C))
        self.global_bank.write(self.norm(x))
        return x

class CrossAttentionBranch(nn.Module):
    def __init__(self, config, global_bank):
        super().__init__()
        self.config = config
        self.global_bank = global_bank
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.config.num_heads, C//self.config.num_heads).transpose(1, 2)
        k_bank, v_bank = self.global_bank.read(B)
        k = self.k_proj(k_bank).reshape(B, -1, self.config.num_heads, C//self.config.num_heads).transpose(1, 2)
        v = self.v_proj(v_bank).reshape(B, -1, self.config.num_heads, C//self.config.num_heads).transpose(1, 2)
        
        x = efficient_attention(q, k, v, self.dropout.p, self.training)
        return self.proj(self.dropout(x.transpose(1, 2).reshape(B, N, C)))

# ============================================================================
# Fusion, FFN, and Transformer Block
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
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
        nn.init.kaiming_normal_(self.dwconv.weight, mode='fan_out', nonlinearity='linear')
        with torch.no_grad(): self.dwconv.weight.div_(math.sqrt(dim))
        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.1)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x * self.scale
        return x.flatten(2).transpose(1, 2)

class CCFFFN(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.dwconv_norm = nn.LayerNorm(hidden_dim)
        self.dwconv = DepthwiseConv2d(hidden_dim, kernel_size=3)
        self.post_dwconv_norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
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

class QuadAttentionBlock(nn.Module):
    """
    Enhanced QuadAttentionBlock with TokenLearner support.
    """
    def __init__(self, config, global_bank, drop_path=0., use_token_learner=False):
        super().__init__()
        self.config = config
        self.use_token_learner = use_token_learner and config.use_token_learner
        
        # Token Compression
        if self.use_token_learner:
            self.token_learner = TokenLearner(config.embed_dim, config.token_learner_k)
            self.token_upsampler = TokenUpsampler(config.embed_dim, config.token_learner_k)
            
        self.embed_dim = config.embed_dim
        self.compressed_dim = config.embed_dim // config.compress_ratio
        self.norm1 = nn.LayerNorm(config.embed_dim)
        
        # Branches
        self.swa = EfficientSpatialWindowAttention(config, global_bank)
        self.msda = EfficientMultiScaleDilatedAttention(config, global_bank)
        self.cga = EfficientChannelGroupAttention(config, global_bank)
        self.cross_attn = CrossAttentionBranch(config, global_bank)
        
        # Norms
        self.norm_swa = nn.LayerNorm(config.embed_dim)
        self.norm_msda = nn.LayerNorm(config.embed_dim)
        self.norm_cga = nn.LayerNorm(config.embed_dim)
        self.norm_cross = nn.LayerNorm(config.embed_dim)
        
        # Compressions
        self.compress_swa = nn.Linear(config.embed_dim, self.compressed_dim)
        self.compress_msda = nn.Linear(config.embed_dim, self.compressed_dim)
        self.compress_cga = nn.Linear(config.embed_dim, self.compressed_dim)
        self.compress_cross = nn.Linear(config.embed_dim, self.compressed_dim)
        
        self.fusion = HybridFusion(self.compressed_dim, 4)
        bottleneck_hidden = config.embed_dim // config.bottleneck_ratio
        self.bottleneck_mlp = BottleneckMLP(4 * self.compressed_dim, bottleneck_hidden, config.embed_dim, config.dropout)
        
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.ccf_ffn = CCFFFN(config.embed_dim, config.mlp_ratio, config.dropout)
        self.drop_path2 = DropPath(drop_path)
    
    def forward(self, x):
        B, N_orig, C = x.shape
        residual = x
        x_norm = self.norm1(x)
        
        # Apply TokenLearner if enabled
        if self.use_token_learner:
            x_process = self.token_learner(x_norm) # [B, K, d]
        else:
            x_process = x_norm
            
        # Parallel Branches
        swa_out = self.compress_swa(self.norm_swa(self.swa(x_process)))
        msda_out = self.compress_msda(self.norm_msda(self.msda(x_process)))
        cga_out = self.compress_cga(self.norm_cga(self.cga(x_process)))
        cross_out = self.compress_cross(self.norm_cross(self.cross_attn(x_process)))
        
        # Fuse and Bottleneck
        fused = self.fusion([swa_out, msda_out, cga_out, cross_out])
        mlp_out = self.bottleneck_mlp(fused)
        
        # Upsample if needed
        if self.use_token_learner:
            mlp_out = self.token_upsampler(mlp_out, N_orig)
            
        x = residual + self.drop_path1(mlp_out)
        x = x + self.drop_path2(self.ccf_ffn(self.norm2(x)))
        return x

# ============================================================================
# Hybrid Stages
# ============================================================================
class HybridStage(nn.Module):
    """
    Hybrid Fusion Stage: LMFA -> RRCV -> SplitFusion -> Transformer Blocks
    """
    def __init__(self, config, global_bank, cnn_channel, depth, dpr):
        super().__init__()
        self.config = config
        # 1. LMFA: Adapt CNN features to Token dimension
        self.lmfa = LMFAdapter(cnn_channel, config.embed_dim)
        
        # 2. RRCV: Refine tokens via lightweight convolution
        self.rrcv = RRCV(config.embed_dim, config.rrcv_channels, config.rrcv_blocks)
        
        # 3. SplitFusion: Merge Main tokens with RRCV tokens
        self.fuse = SplitFusion(config.embed_dim)
        
        # 4. Transformer Blocks
        self.blocks = nn.ModuleList([
            QuadAttentionBlock(
                config, global_bank, drop_path=dpr[i],
                # Use TokenLearner only in first block of stage for efficiency
                use_token_learner=(i == 0) 
            ) for i in range(depth)
        ])
        
    def forward(self, T, F_cnn, H=8, W=8):
        # T: [B, N, d] (Main Trunk)
        # F_cnn: [B, C, H, W] (Lateral Feature)
        
        # Hybrid Injection
        A = self.lmfa(F_cnn) # [B, N, d]
        R = self.rrcv(A, H, W) # [B, N, d]
        T = self.fuse(T, R)    # [B, N, d]
        
        # Run blocks
        for blk in self.blocks:
            T = blk(T)
        return T

# ============================================================================
# Main Model: HQA-ViT
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
    """HQA-ViT: Hybrid Quad-Attention Vision Transformer"""
    def __init__(self, config: QAViTConfig):
        super().__init__()
        self.config = config
        
        # 1. CNN Lateral Backbone
        self.cnn_stem = CNNStemModel(
            config.in_channels, 
            config.cnn_channels[0], 
            config.cnn_channels[1], 
            config.cnn_channels[2]
        )
        
        # 2. Standard Patch Embedding (Main Trunk)
        self.patch_embed = PatchEmbed(config.img_size, config.patch_size, config.in_channels, config.embed_dim)
        self.num_patches = (config.img_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, config.embed_dim))
        self.pos_drop = nn.Dropout(config.dropout)
        
        # 3. Global Shared Resources
        self.global_bank = GlobalTokenBank(config.global_bank_size, config.embed_dim)
        
        # Drop path rates
        total_depth = sum(config.depths)
        dpr = [x.item() for x in torch.linspace(0, config.drop_path, total_depth)]
        cur_d = 0
        
        # Stage 1: Pure Transformer (No fusion yet)
        self.stage1 = nn.ModuleList([
            QuadAttentionBlock(config, self.global_bank, drop_path=dpr[i]) 
            for i in range(config.depths[0])
        ])
        cur_d += config.depths[0]
        
        # Stage 2: Hybrid Fusion with CNN F2
        self.stage2 = HybridStage(
            config, self.global_bank, config.cnn_channels[0], 
            config.depths[1], dpr[cur_d : cur_d + config.depths[1]]
        )
        cur_d += config.depths[1]
        
        # Stage 3: Hybrid Fusion with CNN F3
        self.stage3 = HybridStage(
            config, self.global_bank, config.cnn_channels[1], 
            config.depths[2], dpr[cur_d : cur_d + config.depths[2]]
        )
        cur_d += config.depths[2]
        
        # Stage 4: Hybrid Fusion with CNN F4
        self.stage4 = HybridStage(
            config, self.global_bank, config.cnn_channels[2], 
            config.depths[3], dpr[cur_d : cur_d + config.depths[3]]
        )
        
        # Head
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
        # 1. Extract CNN Lateral Features
        # Shapes: F2,F3,F4 -> [B, C, 8, 8]
        F2, F3, F4 = self.cnn_stem(x)
        
        # 2. Main Trunk Patch Embed
        T = self.patch_embed(x)
        T = T + self.pos_embed
        T = self.pos_drop(T)
        
        # Stage 1 (Pure)
        for blk in self.stage1:
            T = blk(T)
            
        # Stage 2 (Hybrid)
        T = self.stage2(T, F2, H=8, W=8)
        
        # Stage 3 (Hybrid)
        T = self.stage3(T, F3, H=8, W=8)
        
        # Stage 4 (Hybrid)
        T = self.stage4(T, F4, H=8, W=8)
        
        # Head
        T = self.norm(T)
        T = T.mean(dim=1)
        return self.head(T)

# ============================================================================
# Data Loading
# ============================================================================
def get_cifar100_loaders(config: TrainingConfig):
    """Optimized CIFAR-100 data loading"""
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
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

# ============================================================================
# Compilation & Testing
# ============================================================================
def compile_and_test_model(model, config, device):
    print(f"\n{'='*100}")
    print("MODEL COMPILATION & PERFORMANCE TESTING".center(100))
    print(f"{'='*100}")
    
    if config.use_compile and hasattr(torch, 'compile'):
        print(f"\n✓ PyTorch {torch.__version__} - Compiling model...")
        try:
            compiled_model = torch.compile(model, mode=config.compile_mode)
            warmup_input = torch.randn(8, 3, 32, 32, device=device)
            with torch.no_grad():
                _ = compiled_model(warmup_input)
            print(f"  ✓ Compilation successful")
            return compiled_model
        except Exception as e:
            print(f"  ⚠ Compilation failed: {e}. Using eager mode.")
            return model
    return model

def run_performance_tests(model, device, batch_sizes=[1, 128]):
    print(f"\n{'PERFORMANCE BENCHMARKS':-^100}")
    model.eval()
    for bs in batch_sizes:
        try:
            x = torch.randn(bs, 3, 32, 32, device=device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
                start = time.time()
                with torch.no_grad():
                    _ = model(x)
                torch.cuda.synchronize()
                end = time.time()
                print(f"Batch {bs}: {(end-start)*1000:.2f}ms")
        except:
            print(f"Batch {bs}: OOM or Error")

# ============================================================================
# Training Functions
# ============================================================================
def train_epoch(model, loader, optimizer, scheduler, scaler, config, epoch, monitor, ema=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        
        amp_dtype = torch.bfloat16 if config.amp_dtype == 'bfloat16' else torch.float16
        with autocast(enabled=config.use_amp, dtype=amp_dtype):
            outputs = model(inputs)
            loss = criterion(outputs, targets) / config.gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            if config.grad_clip_mode == 'norm':
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            # Monitor
            if batch_idx % 200 == 0:
                monitor.log_gradients(model)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            # Update EMA
            if ema is not None:
                ema.update(model)
        
        total_loss += loss.item() * config.gradient_accumulation_steps
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % config.print_freq == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%')
    
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
# Main
# ============================================================================
def main():
    print("\n" + "="*100)
    print("HQA-ViT CIFAR-100 TRAINING".center(100))
    print("="*100)
    
    model_config = QAViTConfig()
    train_config = TrainingConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize Model
    model = HQAViT(model_config).to(device)
    
    # EMA Initialization
    model_ema = None
    if train_config.use_ema:
        print("Initializing EMA...")
        model_ema = ModelEMA(model, decay=train_config.ema_decay)
    
    # Architecture Analysis
    analyzer = ArchitectureAnalyzer(model, model_config)
    analyzer.analyze_architecture()
    
    # Compile
    model = compile_and_test_model(model, train_config, device)
    run_performance_tests(model, device)
    
    # Data
    train_loader, val_loader = get_cifar100_loaders(train_config)
    
    # Optimization
    optimizer = optim.AdamW(model.parameters(), lr=train_config.base_lr, weight_decay=train_config.weight_decay)
    steps_per_epoch = len(train_loader) // train_config.gradient_accumulation_steps
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=train_config.base_lr, total_steps=train_config.epochs * steps_per_epoch,
        pct_start=train_config.warmup_epochs / train_config.epochs
    )
    scaler = GradScaler(enabled=train_config.use_amp)
    monitor = GradientMonitor()
    
    print(f"Starting training for {train_config.epochs} epochs...")
    best_acc = 0
    
    for epoch in range(1, train_config.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, train_config, epoch, monitor, model_ema
        )
        
        if epoch % train_config.eval_freq == 0:
            # Validate Main Model
            val_loss, val_acc = validate(model, val_loader)
            
            # Validate EMA Model
            ema_acc = 0.0
            if model_ema is not None:
                _, ema_acc = validate(model_ema.ema, val_loader)
                
            print(f"\nEpoch {epoch}: Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f} | EMA Acc: {ema_acc:.2f}")
            
            current_acc = max(val_acc, ema_acc)
            if current_acc > best_acc:
                best_acc = current_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': model_ema.ema.state_dict() if model_ema else None,
                    'best_acc': best_acc,
                }, f"{train_config.checkpoint_dir}/best_model.pth")
                print(f"Saved best model with Acc: {best_acc:.2f}%")

    print(f"Training Complete. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    main()