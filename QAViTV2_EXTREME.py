"""
QAViT CIFAR-100 - Integrated Training with Detailed Architecture Analysis
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
    """Optimized config for CIFAR-100"""
    img_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 100
    embed_dim: int = 192  # d
    depth: int = 8
    num_heads: int = 4
    compress_ratio: int = 4  # d' = d/4 = 48
    bottleneck_ratio: int = 2  # r = d/2 = 96
    mlp_ratio: float = 0.5
    global_bank_size: int = 16
    dropout: float = 0.1
    drop_path: float = 0.1
    window_size: int = 4  # for SWA
    dilation_factors: Tuple[int, ...] = (1, 2)  # for MSDA
    landmark_pooling_stride: int = 2
    num_channel_groups: int = 6
    linformer_k: int = 32  # compressed length for Linformer


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
    mixup_alpha: float = 0.8       # DeiT-style
    cutmix_alpha: float = 1.0      # DeiT-style
    mixup_prob: float = 0.8        # probability to apply mixup / cutmix
    cutmix_prob: float = 0.2       # probability to use cutmix instead of mixup
    random_erasing_prob: float = 0.25  # DeiT uses 0.25
    
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
    
    # Paths
    data_root: str = "./data"
    checkpoint_dir: str = "./checkpoints"



# ============================================================================
# Architecture Analysis & Monitoring
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
        print("QAVIT ARCHITECTURE ANALYSIS".center(100))
        print("="*100)
        
        # Overall statistics
        self._print_overall_stats()
        
        # Component breakdown
        self._print_component_breakdown()
        
        # Attention branch details
        self._print_attention_details()
        
        # Memory analysis
        self._print_memory_analysis()
        
        # Computational complexity
        self._print_computational_complexity()
        
        # Layer-by-layer details
        self._print_layer_details()
        
    def _print_overall_stats(self):
        """Print overall model statistics"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n{'OVERALL STATISTICS':-^100}")
        print(f"{'Metric':<40} {'Value':>20} {'Details':>35}")
        print("-"*100)
        print(f"{'Total Parameters':<40} {total_params:>20,} {f'{total_params/1e6:.2f}M params':>35}")
        print(f"{'Trainable Parameters':<40} {trainable_params:>20,} {f'{trainable_params/1e6:.2f}M params':>35}")
        print(f"{'Non-trainable Parameters':<40} {total_params-trainable_params:>20,}")
        print(f"{'Memory Footprint (FP32)':<40} {f'{total_params*4/1024**2:.2f} MB':>20}")
        print(f"{'Memory Footprint (FP16/AMP)':<40} {f'{total_params*2/1024**2:.2f} MB':>20}")
        print(f"{'Model Depth':<40} {self.config.depth:>20} {'transformer blocks':>35}")
        print(f"{'Embedding Dimension (d)':<40} {self.config.embed_dim:>20}")
        print(f"{'Number of Attention Heads':<40} {self.config.num_heads:>20}")
        print(f"{'Head Dimension':<40} {self.config.embed_dim//self.config.num_heads:>20}")
        
    def _print_component_breakdown(self):
        """Breakdown by major components"""
        print(f"\n{'COMPONENT BREAKDOWN':-^100}")
        print(f"{'Component':<40} {'Parameters':>20} {'% of Total':>15} {'Memory (MB)':>20}")
        print("-"*100)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        components = {
            'Patch Embedding': self.model.patch_embed,
            'Positional Embedding': {'params': self.model.pos_embed.numel()},
            'Global Token Bank': self.model.global_bank,
            'Transformer Blocks': self.model.blocks,
            'Final Norm': self.model.norm,
            'Classification Head': self.model.head,
        }
        
        for name, component in components.items():
            if isinstance(component, dict):
                params = component['params']
            else:
                params = sum(p.numel() for p in component.parameters())
            
            pct = 100 * params / total_params
            mem = params * 4 / 1024**2
            print(f"{name:<40} {params:>20,} {pct:>14.2f}% {mem:>19.2f}")
        
    def _print_attention_details(self):
        """Detailed attention branch analysis"""
        print(f"\n{'QUAD-ATTENTION BRANCH DETAILS':-^100}")
        print(f"{'Branch':<25} {'Type':<20} {'Compression':<25} {'Key Features':>25}")
        print("-"*100)
        
        branches = [
            ('SWA', 'Spatial Window', f'd → d\' → d ({self.config.embed_dim}→{self.config.embed_dim//4}→{self.config.embed_dim})', 
             f'Window={self.config.window_size}×{self.config.window_size}'),
            ('MSDA', 'Multi-Scale Dilated', f'd → d\' → d ({self.config.embed_dim}→{self.config.embed_dim//4}→{self.config.embed_dim})', 
             f'Dilations={self.config.dilation_factors}'),
            ('CGA', 'Channel Group', f'd → d\' → d ({self.config.embed_dim}→{self.config.embed_dim//4}→{self.config.embed_dim})', 
             f'Groups={self.config.num_channel_groups}'),
            ('Cross-Attn', 'Global Bank Cross', f'd → d\' → d ({self.config.embed_dim}→{self.config.embed_dim//4}→{self.config.embed_dim})', 
             f'Bank={self.config.global_bank_size}'),
        ]
        
        for name, attn_type, compression, features in branches:
            print(f"{name:<25} {attn_type:<20} {compression:<25} {features:>25}")
        
        print(f"\n{'Fusion & Bottleneck':<25} {'Hybrid Fusion':<20} "
              f"{'4d\' → r → d (192→96→192)':<25} {'Learnable weights':>25}")
        
    def _print_memory_analysis(self):
        """Memory usage analysis"""
        print(f"\n{'MEMORY ANALYSIS (Per Forward Pass)':-^100}")
        print(f"{'Component':<40} {'Activations (MB)':>25} {'Gradients (MB)':>30}")
        print("-"*100)
        
        batch_size = 128
        num_patches = (self.config.img_size // self.config.patch_size) ** 2
        
        # Estimates
        components_mem = {
            'Input Image (B×3×32×32)': (batch_size * 3 * 32 * 32 * 4 / 1024**2, 0),
            'Patch Embeddings (B×N×d)': (batch_size * num_patches * self.config.embed_dim * 4 / 1024**2,
                                          batch_size * num_patches * self.config.embed_dim * 4 / 1024**2),
            'Per Transformer Block': (batch_size * num_patches * self.config.embed_dim * 4 * 4 / 1024**2,
                                       batch_size * num_patches * self.config.embed_dim * 4 * 4 / 1024**2),
            'All Transformer Blocks': (batch_size * num_patches * self.config.embed_dim * 4 * 4 * self.config.depth / 1024**2,
                                        batch_size * num_patches * self.config.embed_dim * 4 * 4 * self.config.depth / 1024**2),
            'Classification Head': (batch_size * self.config.num_classes * 4 / 1024**2,
                                     batch_size * self.config.num_classes * 4 / 1024**2),
        }
        
        for name, (act_mem, grad_mem) in components_mem.items():
            print(f"{name:<40} {act_mem:>24.2f} {grad_mem:>29.2f}")
        
        total_act = sum(v[0] for v in components_mem.values())
        total_grad = sum(v[1] for v in components_mem.values())
        print("-"*100)
        print(f"{'TOTAL (Estimated)':<40} {total_act:>24.2f} {total_grad:>29.2f}")
        print(f"{'Peak Memory (Act + Grad + Params)':<40} {total_act + total_grad + sum(p.numel() for p in self.model.parameters())*4/1024**2:>24.2f}")
        
    def _print_computational_complexity(self):
        """Computational complexity analysis"""
        print(f"\n{'COMPUTATIONAL COMPLEXITY':-^100}")
        print(f"{'Operation':<50} {'FLOPs':>25} {'Details':>20}")
        print("-"*100)
        
        N = (self.config.img_size // self.config.patch_size) ** 2  # num patches
        d = self.config.embed_dim
        h = self.config.num_heads
        
        # Patch embedding
        patch_flops = self.config.in_channels * (self.config.patch_size ** 2) * d * N
        print(f"{'Patch Embedding':<50} {f'{patch_flops/1e9:.2f}G':>25}")
        
        # Single transformer block (approximate)
        # QKV projection
        qkv_flops = 3 * N * d * d
        # Attention (simplified, per branch)
        attn_flops = 2 * N * d * (self.config.linformer_k + self.config.global_bank_size)
        # Projection
        proj_flops = N * d * d
        # FFN
        ffn_flops = 2 * N * d * (d * self.config.mlp_ratio)
        # Fusion
        fusion_flops = 4 * N * (d // 4) * d
        
        block_flops = (qkv_flops * 4 + attn_flops * 4 + proj_flops * 4 + ffn_flops + fusion_flops)
        
        print(f"{'  - QKV Projections (4 branches)':<50} {f'{qkv_flops*4/1e9:.2f}G':>25}")
        print(f"{'  - Attention (4 branches, compressed)':<50} {f'{attn_flops*4/1e9:.2f}G':>25}")
        print(f"{'  - Output Projections':<50} {f'{proj_flops*4/1e9:.2f}G':>25}")
        print(f"{'  - Hybrid Fusion':<50} {f'{fusion_flops/1e9:.2f}G':>25}")
        print(f"{'  - CCF-FFN':<50} {f'{ffn_flops/1e9:.2f}G':>25}")
        print(f"{'Single Transformer Block (Total)':<50} {f'{block_flops/1e9:.2f}G':>25}")
        
        # All blocks
        all_blocks_flops = block_flops * self.config.depth
        print(f"{'All Transformer Blocks':<50} {f'{all_blocks_flops/1e9:.2f}G':>25} {f'×{self.config.depth} blocks':>20}")
        
        # Classification head
        head_flops = d * self.config.num_classes
        print(f"{'Classification Head':<50} {f'{head_flops/1e9:.4f}G':>25}")
        
        # Total
        total_flops = patch_flops + all_blocks_flops + head_flops
        print("-"*100)
        print(f"{'TOTAL FLOPs (per image)':<50} {f'{total_flops/1e9:.2f}G':>25}")
        print(f"{'TOTAL FLOPs (batch={128})':<50} {f'{total_flops*128/1e9:.2f}G':>25}")
        
    def _print_layer_details(self):
        """Layer-by-layer parameter details"""
        print(f"\n{'LAYER-BY-LAYER BREAKDOWN':-^100}")
        print(f"{'Layer Name':<60} {'Parameters':>20} {'Shape':>15}")
        print("-"*100)
        
        # Group by transformer blocks
        block_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'blocks.' in name:
                block_idx = int(name.split('.')[1])
                layer_name = '.'.join(name.split('.')[2:])
                block_params.append((block_idx, layer_name, param.numel(), tuple(param.shape)))
            else:
                other_params.append((name, param.numel(), tuple(param.shape)))
        
        # Print other components
        print(f"{'[PATCH EMBEDDING & POSITIONAL]':<60}")
        for name, num, shape in other_params:
            if 'patch_embed' in name or 'pos_embed' in name:
                print(f"  {name:<58} {num:>20,} {str(shape):>15}")
        
        # Print global bank
        print(f"\n{'[GLOBAL TOKEN BANK - Shared across all blocks]':<60}")
        for name, num, shape in other_params:
            if 'global_bank' in name:
                print(f"  {name:<58} {num:>20,} {str(shape):>15}")
        
        # Print first transformer block in detail
        print(f"\n{'[TRANSFORMER BLOCK 0 - Detailed]':<60}")
        block_0_params = [(n, p, s) for i, n, p, s in block_params if i == 0]
        
        branch_names = {
            'swa': 'SWA (Spatial Window Attention)',
            'msda': 'MSDA (Multi-Scale Dilated Attention)',
            'cga': 'CGA (Channel Group Attention)',
            'cross_attn': 'Cross-Attention with Global Bank',
        }
        
        for branch_key, branch_full in branch_names.items():
            print(f"\n  {branch_full}:")
            for name, num, shape in block_0_params:
                if branch_key in name:
                    print(f"    {name:<56} {num:>20,} {str(shape):>15}")
        
        print(f"\n  Fusion & MLP:")
        for name, num, shape in block_0_params:
            if any(x in name for x in ['fusion', 'compress_', 'bottleneck', 'ccf_ffn', 'norm']):
                print(f"    {name:<56} {num:>20,} {str(shape):>15}")
        
        # Summary for remaining blocks
        print(f"\n{'[TRANSFORMER BLOCKS 1-7 - Summary]':<60}")
        for i in range(1, self.config.depth):
            block_i_params = sum(p for idx, n, p, s in block_params if idx == i)
            print(f"  Block {i}: {block_i_params:>20,} parameters (same structure as Block 0)")
        
        # Classification head
        print(f"\n{'[CLASSIFICATION HEAD]':<60}")
        for name, num, shape in other_params:
            if 'head' in name or 'norm' in name and 'blocks' not in name:
                print(f"  {name:<58} {num:>20,} {str(shape):>15}")


class GradientMonitor:
    """Enhanced gradient monitoring with detailed tracking"""
    
    def __init__(self):
        self.grad_norms = []
        self.param_norms = []
        self.layer_grad_history = {}
        self.explosion_count = 0
        
    def log_gradients(self, model, detailed=False):
        """Log gradient statistics with optional detailed tracking"""
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
                
                # Track layer-wise statistics
                layer_name = '.'.join(name.split('.')[:2])
                if layer_name not in layer_stats:
                    layer_stats[layer_name] = {'grad_norm': 0, 'param_norm': 0, 'count': 0}
                layer_stats[layer_name]['grad_norm'] += g_norm
                layer_stats[layer_name]['param_norm'] += p_norm
                layer_stats[layer_name]['count'] += 1
                
                # Detect problematic layers
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
        
        # Store layer history
        if detailed:
            for layer, stats in layer_stats.items():
                if layer not in self.layer_grad_history:
                    self.layer_grad_history[layer] = []
                self.layer_grad_history[layer].append(stats['grad_norm'] / max(stats['count'], 1))
        
        return total_norm, param_norm, grad_stats, layer_stats
    
    def check_explosion(self, threshold=50.0):
        """Check for gradient explosion"""
        if len(self.grad_norms) > 0:
            is_exploding = self.grad_norms[-1] > threshold
            if is_exploding:
                self.explosion_count += 1
            return is_exploding
        return False
    
    def print_detailed_stats(self, epoch, step, grad_stats, layer_stats):
        """Print detailed gradient statistics"""
        print(f"\n{'='*100}")
        print(f"📊 GRADIENT ANALYSIS - Epoch {epoch}, Step {step}")
        print(f"{'='*100}")
        
        # Overall stats
        print(f"\n{'Overall Statistics':-^100}")
        print(f"  Total Gradient Norm:     {self.grad_norms[-1]:.6f}")
        print(f"  Total Parameter Norm:    {self.param_norms[-1]:.6f}")
        print(f"  Grad/Param Ratio:        {self.grad_norms[-1]/max(self.param_norms[-1], 1e-8):.6f}")
        print(f"  Explosion Count:         {self.explosion_count}")
        
        # Layer-wise stats (top 10 by gradient norm)
        print(f"\n{'Layer-wise Gradient Norms (Top 10)':-^100}")
        print(f"{'Layer':<50} {'Grad Norm':>15} {'Param Norm':>15} {'Ratio':>15}")
        print("-"*100)
        
        sorted_layers = sorted(layer_stats.items(), 
                              key=lambda x: x[1]['grad_norm'], 
                              reverse=True)[:10]
        
        for layer, stats in sorted_layers:
            avg_grad = stats['grad_norm'] / max(stats['count'], 1)
            avg_param = stats['param_norm'] / max(stats['count'], 1)
            ratio = avg_grad / max(avg_param, 1e-8)
            print(f"{layer:<50} {avg_grad:>15.6f} {avg_param:>15.6f} {ratio:>15.6f}")
        
        # Problematic layers
        if grad_stats:
            print(f"\n{'⚠️  PROBLEMATIC LAYERS DETECTED':-^100}")
            print(f"{'Layer':<50} {'Grad Norm':>12} {'Max Value':>12} {'NaN':>8} {'Inf':>8}")
            print("-"*100)
            
            for name, stats in list(grad_stats.items())[:10]:  # Top 10 problematic
                print(f"{name:<50} {stats['grad_norm']:>12.4f} {stats['grad_max']:>12.4f} "
                      f"{'Yes' if stats['has_nan'] else 'No':>8} "
                      f"{'Yes' if stats['has_inf'] else 'No':>8}")


# ============================================================================
# Core Components (Same as before, keeping for completeness)
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
        
        # CRITICAL: Add gradient clipping to bank parameters
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
        
        # CRITICAL: More aggressive clamping and slower updates
        if residual:
            update_k = torch.clamp(update_k.mean(0, keepdim=True), -0.05, 0.05)  # Reduced from 0.1
            update_v = torch.clamp(update_v.mean(0, keepdim=True), -0.05, 0.05)
            
            # Even slower update rate: 0.005 instead of 0.01
            update_rate = 0.005 if self.update_count < 1000 else 0.01
            
            self.global_k.data.add_(update_rate * update_k)
            self.global_v.data.add_(update_rate * update_v)
            
            # Aggressive clamping
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
    # Quick NaN guard
    if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
        return torch.zeros_like(q)

    # Only use FlashAttention when available, on CUDA, during training, AND
    # when tensors are in a supported dtype (fp16 or bfloat16). FlashAttention
    # extensions typically do not support fp32 and will raise runtime errors.
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

        except Exception as e:
            # If FlashAttention fails for any reason, fall back to PyTorch's
            # scaled_dot_product_attention implementation. If tensors are in
            # bf16, cast to fp32 for the fallback and then cast back.
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
# Attention Branches (Abbreviated - same as previous script)
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
    """Stabilized Depthwise Convolution with proper initialization"""
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            groups=dim,
            bias=False  # No bias - prevents explosion, LayerNorms handle it
        )
        
        # CRITICAL: Proper initialization for depthwise conv
        # Use smaller std for depthwise to prevent explosion
        nn.init.kaiming_normal_(self.dwconv.weight, mode='fan_out', nonlinearity='linear')
        # Scale down by sqrt(groups) to account for depthwise structure
        with torch.no_grad():
            self.dwconv.weight.div_(math.sqrt(dim))
        
        # Learnable scale parameter for gradual activation
        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.1)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        
        # Apply depthwise conv with scaling
        x = self.dwconv(x)
        x = x * self.scale  # Learnable scaling to control magnitude
        
        return x.flatten(2).transpose(1, 2)


class CCFFFN(nn.Module):
    """Stabilized CCF-FFN with normalization and gradient control"""
    def __init__(self, embed_dim, mlp_ratio=0.5, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        
        # First linear with proper init
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        
        self.act = nn.GELU()
        
        # CRITICAL: Add LayerNorm BEFORE depthwise conv
        self.dwconv_norm = nn.LayerNorm(hidden_dim)
        self.dwconv = DepthwiseConv2d(hidden_dim, kernel_size=3)
        
        # CRITICAL: Add LayerNorm AFTER depthwise conv
        self.post_dwconv_norm = nn.LayerNorm(hidden_dim)
        
        # Second linear with proper init
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual scaling to prevent explosion
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        
        # First linear + activation
        x = self.fc1(x)
        x = self.act(x)
        
        # CRITICAL: Normalize before depthwise conv
        x = self.dwconv_norm(x)
        
        # Depthwise conv
        x = self.dwconv(x, H, W)
        
        # CRITICAL: Normalize after depthwise conv
        x = self.post_dwconv_norm(x)
        
        # Second linear
        x = self.fc2(x)
        x = self.dropout(x)
        
        # Scale output for stability
        x = x * self.gamma
        
        return x


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
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
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
# ============================================================================
# Data Loading
# ============================================================================
def get_cifar100_loaders(config: TrainingConfig):
    """CIFAR-100 data loading with DeiT-style strong augmentations"""
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # Strong color / geometric aug (DeiT uses RandAugment)
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # Random Erasing in pixel space, like DeiT
        transforms.RandomErasing(
            p=config.random_erasing_prob,
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3),
            inplace=True,
        ),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    train_dataset = datasets.CIFAR100(
        root=config.data_root,
        train=True,
        transform=train_transform,
        download=True,
    )
    val_dataset = datasets.CIFAR100(
        root=config.data_root,
        train=False,
        transform=val_transform,
        download=True,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2 if config.num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2 if config.num_workers > 0 else None,
    )
    
    return train_loader, val_loader



# ============================================================================
# Model Compilation & Testing
# ============================================================================
def compile_and_test_model(model, config, device):
    """Compile model with torch.compile and run performance tests"""
    print(f"\n{'='*100}")
    print("MODEL COMPILATION & PERFORMANCE TESTING".center(100))
    print(f"{'='*100}")
    
    # Check PyTorch version
    pytorch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
    
    if config.use_compile and pytorch_version >= (2, 0):
        print(f"\n✓ PyTorch {torch.__version__} detected - torch.compile available")
        print(f"  Compilation mode: '{config.compile_mode}'")
        print(f"  Expected benefits: 1.2-2x speedup, reduced memory overhead")
        
        try:
            print(f"\n  Compiling model...")
            compiled_model = torch.compile(model, mode=config.compile_mode)
            print(f"  ✓ Model compiled successfully!")
            
            # Warmup compiled model
            print(f"\n  Running warmup passes (this may take a minute)...")
            warmup_input = torch.randn(8, 3, 32, 32, device=device)
            with torch.no_grad():
                for _ in range(3):
                    _ = compiled_model(warmup_input)
            print(f"  ✓ Warmup complete")
            
            return compiled_model
            
        except Exception as e:
            print(f"  ⚠ Compilation failed: {e}")
            print(f"  → Falling back to eager mode")
            return model
    else:
        if not config.use_compile:
            print(f"\n  Compilation disabled in config")
        else:
            print(f"\n  PyTorch {torch.__version__} detected")
            print(f"  torch.compile requires PyTorch 2.0+")
        print(f"  → Running in eager mode")
        return model


def run_performance_tests(model, device, batch_sizes=[1, 8, 32, 64, 128]):
    """Run performance benchmarks"""
    print(f"\n{'PERFORMANCE BENCHMARKS':-^100}")
    print(f"{'Batch Size':<15} {'Output Shape':<25} {'Time (ms)':>15} {'Throughput (img/s)':>20} {'VRAM (GB)':>15}")
    print("-"*100)
    
    model.eval()
    
    for bs in batch_sizes:
        try:
            x = torch.randn(bs, 3, 32, 32, device=device)
            
            with torch.no_grad():
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    
                    start.record()
                    output = model(x)
                    end.record()
                    
                    torch.cuda.synchronize()
                    elapsed = start.elapsed_time(end)
                    vram = torch.cuda.max_memory_allocated() / 1024**3
                else:
                    start = time.time()
                    output = model(x)
                    elapsed = (time.time() - start) * 1000
                    vram = 0
            
            throughput = bs / (elapsed / 1000)
            
            if device.type == 'cuda':
                print(f"{bs:<15} {str(output.shape):<25} {elapsed:>15.2f} {throughput:>20.1f} {vram:>15.2f}")
            else:
                print(f"{bs:<15} {str(output.shape):<25} {elapsed:>15.2f} {throughput:>20.1f} {'N/A':>15}")
            
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{bs:<15} {'OOM':^70}")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                print(f"{bs:<15} Error: {str(e)[:60]}")


# ============================================================================
# Training Functions
# ============================================================================
# ============================================================================
# Mixup / CutMix Utilities (DeiT-style)
# ============================================================================
def rand_bbox(size, lam):
    """Generate random CutMix bounding box."""
    W = size[3]
    H = size[2]
    cut_ratio = math.sqrt(1. - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    # uniform center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2


def apply_mixup_cutmix(inputs, targets, config: TrainingConfig):
    """
    Apply Mixup or CutMix to a batch.
    Returns: inputs, targets_a, targets_b, lam, mode
    mode in {'mixup', 'cutmix', 'none'}
    """
    mixup_alpha = config.mixup_alpha
    cutmix_alpha = config.cutmix_alpha
    mixup_prob = config.mixup_prob
    cutmix_prob = config.cutmix_prob

    if (mixup_alpha <= 0 and cutmix_alpha <= 0) or (mixup_prob <= 0 and cutmix_prob <= 0):
        return inputs, targets, None, None, 'none'

    r = np.random.rand()
    # Choose operation
    use_mixup = (r < mixup_prob) and (mixup_alpha > 0)
    use_cutmix = (not use_mixup) and (r < mixup_prob + cutmix_prob) and (cutmix_alpha > 0)

    if not (use_mixup or use_cutmix):
        return inputs, targets, None, None, 'none'

    batch_size = inputs.size(0)
    indices = torch.randperm(batch_size, device=inputs.device)
    targets_a = targets
    targets_b = targets[indices]

    if use_mixup:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        mixed_inputs = lam * inputs + (1. - lam) * inputs[indices]
        return mixed_inputs, targets_a, targets_b, float(lam), 'mixup'

    # CutMix
    lam = np.random.beta(cutmix_alpha, cutmix_alpha)
    x1, y1, x2, y2 = rand_bbox(inputs.size(), lam)
    mixed_inputs = inputs.clone()
    mixed_inputs[:, :, y1:y2, x1:x2] = inputs[indices, :, y1:y2, x1:x2]

    # Adjust lambda based on the actually cut area
    cut_area = (x2 - x1) * (y2 - y1)
    lam_adjusted = 1. - float(cut_area) / float(inputs.size(2) * inputs.size(3))

    return mixed_inputs, targets_a, targets_b, lam_adjusted, 'cutmix'

def train_epoch(model, loader, optimizer, scheduler, scaler, config, epoch, monitor):
    """Training loop with detailed gradient monitoring + Mixup/CutMix"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        targets_orig = targets  # for reporting accuracy
        
        # Apply Mixup / CutMix in input & label space (DeiT-style)
        inputs, targets_a, targets_b, lam, mix_mode = apply_mixup_cutmix(
            inputs, targets, config
        )
        
        # Use configured AMP dtype (bfloat16 for better stability)
        amp_dtype = torch.bfloat16 if config.amp_dtype == 'bfloat16' else torch.float16
        with autocast(enabled=config.use_amp, dtype=amp_dtype):
            outputs = model(inputs)
            
            if mix_mode == 'none':
                loss = criterion(outputs, targets_orig)
            else:
                loss = lam * criterion(outputs, targets_a) + (1. - lam) * criterion(outputs, targets_b)
            
            loss = loss / config.gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            
            # Per-layer gradient clipping for dwconv
            for name, param in model.named_parameters():
                if 'dwconv' in name and param.grad is not None:
                    torch.nn.utils.clip_grad_norm_([param], max_norm=0.1)
            
            # Monitor gradients (detailed every 200 steps)
            detailed = (batch_idx % 200 == 0)
            grad_norm, param_norm, grad_stats, layer_stats = monitor.log_gradients(model, detailed=detailed)
            
            # Check for explosion
            if monitor.check_explosion(threshold=50.0):
                print(f"\n{'='*100}")
                print(f"🚨 GRADIENT EXPLOSION DETECTED".center(100))
                print(f"{'='*100}")
                monitor.print_detailed_stats(epoch, batch_idx, grad_stats, layer_stats)
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
        
        total_loss += loss.item() * config.gradient_accumulation_steps
        
        # For training accuracy, compare to original (unmixed) labels
        _, predicted = outputs.max(1)
        total += targets_orig.size(0)
        correct += predicted.eq(targets_orig).sum().item()
        
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
    print("QAVIT CIFAR-100 TRAINING - INTEGRATED WITH ARCHITECTURE ANALYSIS".center(100))
    print("="*100)
    
    # Configs
    model_config = QAViTConfig()
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
    model = QAViT(model_config).to(device)

    # Quick gradient-flow sanity test for CCF-FFN (runs immediately after model creation)
    try:
        print("\n Testing gradient flow through CCF-FFN...")
        model.train()
        # Create a small test batch on the correct device
        test_input = torch.randn(2, 3, 32, 32, device=device)
        # Forward and reduce to scalar
        test_output = model(test_input)
        test_loss = test_output.sum()
        # Backprop
        test_loss.backward()

        # Inspect depthwise conv gradients
        for name, param in model.named_parameters():
            if 'dwconv' in name and param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                print(f"  {name}: grad_norm={grad_norm:.4f}, param_norm={param_norm:.4f}")
                if grad_norm > 10 or np.isnan(grad_norm) or np.isinf(grad_norm):
                    print(f"    WARNING: Problematic gradient detected!")
                else:
                    print(f"    ✓  Gradient looks healthy")

        model.zero_grad()
        print("✓ Gradient flow test complete\n")
    except RuntimeError as e:
        print(f"Gradient test failed (RuntimeError): {e}")
    except Exception as e:
        print(f"Gradient test failed: {e}")

    # Architecture Analysis
    analyzer = ArchitectureAnalyzer(model, model_config)
    analyzer.analyze_architecture()
    
    # Compile model
    model = compile_and_test_model(model, train_config, device)
    
    # Performance tests
    print(f"\n  Running performance benchmarks...")
    run_performance_tests(model, device, batch_sizes=[1, 8, 32, 64, 128, 256])
    
    # Data loaders
    print(f"\n{'='*100}")
    print("DATA PREPARATION".center(100))
    print(f"{'='*100}")
    print(f"\n Loading CIFAR-100...")
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
    
    print(f"\n{'Training Hyperparameters':-^100}")
    print(f"   Epochs: {train_config.epochs}")
    print(f"   Warmup epochs: {train_config.warmup_epochs}")
    print(f"   Base learning rate: {train_config.base_lr}")
    print(f"   Min learning rate: {train_config.min_lr}")
    print(f"   Weight decay: {train_config.weight_decay}")
    print(f"   Label smoothing: {train_config.label_smoothing}")
    print(f"   Gradient clip ({train_config.grad_clip_mode}): {train_config.max_grad_norm}")
    print(f"   Gradient accumulation steps: {train_config.gradient_accumulation_steps}")
    print(f"   Mixed precision (AMP): {train_config.use_amp}")
    print(f"   Model compilation: {train_config.use_compile}")
    
    print(f"\n{'Training Schedule':-^100}")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Steps per epoch: {steps_per_epoch:,}")
    print(f"   Warmup steps: {warmup_steps:,}")
    print(f"   Evaluation frequency: Every {train_config.eval_freq} epoch(s)")
    print(f"   Checkpoint frequency: Every {train_config.save_freq} epoch(s)")
    
    # Calculate estimated training time
    # Rough estimate: 0.5s per step for batch_size=128
    est_time_per_epoch = steps_per_epoch * 0.5 / 60  # minutes
    est_total_time = est_time_per_epoch * train_config.epochs / 60  # hours
    print(f"\n{'Estimated Training Time':-^100}")
    print(f"   Per epoch: ~{est_time_per_epoch:.1f} minutes")
    print(f"   Total ({train_config.epochs} epochs): ~{est_total_time:.1f} hours")
    
    # Training loop
    print(f"\n{'='*100}")
    print("TRAINING STARTED".center(100))
    print(f"{'='*100}\n")
    
    best_acc = 0
    train_start_time = time.time()
    
    for epoch in range(1, train_config.epochs + 1):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, train_config, epoch, monitor
        )
        
        # Validate
        if epoch % train_config.eval_freq == 0:
            val_loss, val_acc = validate(model, val_loader)
            
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - train_start_time
            
            print(f"\n{'='*100}")
            print(f"EPOCH {epoch}/{train_config.epochs} SUMMARY".center(100))
            print(f"{'='*100}")
            print(f"{'Metric':<30} {'Train':>15} {'Val':>15} {'Details':>35}")
            print("-"*100)
            print(f"{'Loss':<30} {train_loss:>15.4f} {val_loss:>15.4f}")
            print(f"{'Accuracy (%)':<30} {train_acc:>15.2f} {val_acc:>15.2f}")
            print(f"{'Time (seconds)':<30} {epoch_time:>15.1f} {'':>15} {f'{epoch_time/60:.1f} min':>35}")
            print(f"{'Total Time':<30} {total_time:>15.1f} {'':>15} {f'{total_time/3600:.2f} hrs':>35}")
            print(f"{'Best Val Acc (%)':<30} {max(best_acc, val_acc):>15.2f} {'':>15}")
            print(f"{'Learning Rate':<30} {optimizer.param_groups[0]['lr']:>15.6f}")
            print(f"{'Gradient Norm':<30} {monitor.grad_norms[-1]:>15.4f}")
            print(f"{'Gradient Explosions':<30} {monitor.explosion_count:>15}")
            
            if device.type == 'cuda':
                current_vram = torch.cuda.max_memory_allocated() / 1024**3
                print(f"{'Peak VRAM (GB)':<30} {current_vram:>15.2f}")
            
            print(f"{'='*100}\n")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict(),
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
        
        # Save checkpoint
        if epoch % train_config.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_acc': val_acc if epoch % train_config.eval_freq == 0 else None,
            }
            torch.save(checkpoint, f"{train_config.checkpoint_dir}/checkpoint_epoch_{epoch}.pth")
            print(f"💾 Checkpoint saved: epoch_{epoch}.pth\n")
    
    # Training complete
    total_training_time = time.time() - train_start_time
    
    print(f"\n{'='*100}")
    print("TRAINING COMPLETE".center(100))
    print(f"{'='*100}")
    print(f"\n{'Final Results':-^100}")
    print(f"   Best Validation Accuracy: {best_acc:.2f}%")
    print(f"   Total Training Time: {total_training_time/3600:.2f} hours")
    print(f"   Average Time per Epoch: {total_training_time/train_config.epochs/60:.1f} minutes")
    print(f"   Total Gradient Explosions: {monitor.explosion_count}")
    print(f"\n{'Saved Models':-^100}")
    print(f"   Best model: {train_config.checkpoint_dir}/best_model.pth")
    print(f"   Latest checkpoint: {train_config.checkpoint_dir}/checkpoint_epoch_{train_config.epochs}.pth")
    
    # Print layer gradient history summary
    if monitor.layer_grad_history:
        print(f"\n{'Layer Gradient History (Average over training)':-^100}")
        print(f"{'Layer':<50} {'Avg Grad Norm':>20} {'Max Grad Norm':>20}")
        print("-"*100)
        
        for layer, history in sorted(monitor.layer_grad_history.items())[:15]:
            avg_norm = np.mean(history)
            max_norm = np.max(history)
            print(f"{layer:<50} {avg_norm:>20.6f} {max_norm:>20.6f}")
    
    print(f"\n{'='*100}")
    print("🎉 ALL DONE!".center(100))
    print(f"{'='*100}\n")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Print CUDA optimization status
        print("\n" + "="*100)
        print("CUDA OPTIMIZATIONS".center(100))
        print("="*100)
        print(f"   cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        print(f"   TF32 (Matmul): {torch.backends.cuda.matmul.allow_tf32}")
        print(f"   TF32 (cuDNN): {torch.backends.cudnn.allow_tf32}")
        print("="*100)
    
    main()