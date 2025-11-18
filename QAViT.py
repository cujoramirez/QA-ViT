"""
Quad-Attention Vision Transformer (QAViT)
Implementation based on architectural diagrams showing:
- Three attention branches: CGA, MSDA, SWA
- Cross-Attention with Global Token Bank
- Hybrid Fusion + Bottleneck MLP
- CCF-FFN with DWConv
- FlashAttention2 integration throughout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from dataclasses import dataclass

try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("Warning: FlashAttention2 not available, falling back to PyTorch SDPA")


def supports_flash_attention(device):
    """Check if FlashAttention2 is available and device is CUDA"""
    return HAS_FLASH_ATTN and device.type == 'cuda'


@dataclass
class QAViTConfig:
    """Configuration for QAViT models"""
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 1000
    embed_dim: int = 256  # d
    depth: int = 12
    num_heads: int = 4
    compress_ratio: int = 4  # d/d' ratio
    bottleneck_ratio: int = 2  # d/r ratio
    mlp_ratio: float = 0.5  # FFN compression
    global_bank_size: int = 16
    dropout: float = 0.1
    drop_path: float = 0.1
    # SWA specific
    window_size: int = 7
    num_windows: Optional[int] = None
    # MSDA specific
    dilation_factors: Tuple[int, ...] = (1, 2, 3)
    landmark_pooling_stride: int = 2
    # CGA specific
    num_channel_groups: int = 8
    # Linformer compression
    linformer_k: int = 64  # compressed sequence length


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class GlobalTokenBank(nn.Module):
    """
    Global Token Bank with MLA Gateway
    Shared across all transformer blocks
    Supports read/write operations with compression
    """
    def __init__(self, bank_size: int, embed_dim: int):
        super().__init__()
        self.bank_size = bank_size
        self.embed_dim = embed_dim
        
        # Learnable global tokens (shared K, V)
        self.global_k = nn.Parameter(torch.randn(1, bank_size, embed_dim))
        self.global_v = nn.Parameter(torch.randn(1, bank_size, embed_dim))
        
        # MLA Gateway components for compression
        self.write_norm = nn.LayerNorm(embed_dim)
        self.write_compression = nn.Linear(embed_dim, embed_dim)
        self.write_gate = nn.Linear(embed_dim, bank_size)
        
        # Initialize
        nn.init.trunc_normal_(self.global_k, std=0.02)
        nn.init.trunc_normal_(self.global_v, std=0.02)
    
    def read(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read K, V from global bank
        Returns: (K_bank, V_bank) each of shape (B, bank_size, embed_dim)
        """
        k_bank = self.global_k.expand(batch_size, -1, -1)
        v_bank = self.global_v.expand(batch_size, -1, -1)
        return k_bank, v_bank
    
    def write(self, tokens: torch.Tensor, residual: bool = True):
        """
        Write/update global bank via MLA Gateway
        Args:
            tokens: (B, N, embed_dim) - output from attention branch
            residual: whether to use residual update
        """
        # Only update bank during training
        if not self.training:
            return
            
        B, N, C = tokens.shape
        
        # MLA Gateway: compress and weight
        tokens_norm = self.write_norm(tokens)
        compressed = self.write_compression(tokens_norm)  # (B, N, C)
        
        # Compute attention weights for bank update
        # Shape: (B, N, bank_size)
        weights = self.write_gate(tokens_norm)
        weights = F.softmax(weights, dim=1)  # softmax over sequence
        
        # Weighted aggregation: (B, bank_size, C)
        update_k = torch.bmm(weights.transpose(1, 2), compressed)
        update_v = torch.bmm(weights.transpose(1, 2), tokens_norm)
        
        # Residual update with clamping to prevent explosion
        if residual:
            # Clamp updates to [-0.5, 0.5] range to prevent parameter drift
            update_k_clamped = torch.clamp(update_k.mean(0, keepdim=True), -0.5, 0.5)
            update_v_clamped = torch.clamp(update_v.mean(0, keepdim=True), -0.5, 0.5)
            self.global_k.data = self.global_k.data + 0.1 * update_k_clamped
            self.global_v.data = self.global_v.data + 0.1 * update_v_clamped
        else:
            self.global_k.data = torch.clamp(update_k.mean(0, keepdim=True), -2.0, 2.0)
            self.global_v.data = torch.clamp(update_v.mean(0, keepdim=True), -2.0, 2.0)


class LinformerCompression(nn.Module):
    """
    Linformer-style compression for K and V
    Projects sequence length: N → k
    E_k and E_v are shared across heads but applied per-sequence
    """
    def __init__(self, seq_len: int, compressed_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.compressed_len = compressed_len
        # Projection matrices: (seq_len, compressed_len)
        self.E_k = nn.Parameter(torch.randn(seq_len, compressed_len))
        self.E_v = nn.Parameter(torch.randn(seq_len, compressed_len))
        nn.init.xavier_uniform_(self.E_k)
        nn.init.xavier_uniform_(self.E_v)
    
    def forward(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            k, v: (B, num_heads, N, head_dim)
        Returns:
            k', v': (B, num_heads, compressed_len, head_dim)
        """
        B, H, N, D = k.shape
        
        # Handle variable sequence lengths
        if N != self.seq_len:
            if N < self.seq_len:
                # Pad sequence
                pad_len = self.seq_len - N
                k = F.pad(k, (0, 0, 0, pad_len), value=0)
                v = F.pad(v, (0, 0, 0, pad_len), value=0)
            else:
                # Truncate sequence
                k = k[:, :, :self.seq_len, :]
                v = v[:, :, :self.seq_len, :]
        
        # Apply compression: E^T @ K
        # (seq_len, k) @ (B, H, seq_len, D) -> (B, H, k, D)
        # Reshape k,v: (B, H, seq_len, D) -> (B*H, seq_len, D)
        k_flat = k.reshape(B * H, self.seq_len, D)
        v_flat = v.reshape(B * H, self.seq_len, D)
        
        # Matrix multiply: (k, seq_len) @ (B*H, seq_len, D) -> (B*H, k, D)
        k_compressed = torch.matmul(self.E_k.T, k_flat)
        v_compressed = torch.matmul(self.E_v.T, v_flat)
        
        # Reshape back: (B*H, k, D) -> (B, H, k, D)
        k_compressed = k_compressed.reshape(B, H, self.compressed_len, D)
        v_compressed = v_compressed.reshape(B, H, self.compressed_len, D)
        
        return k_compressed, v_compressed


def efficient_attention_with_flash(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    dropout_p: float = 0.0,
    training: bool = True
) -> torch.Tensor:
    """
    Unified attention function with FlashAttention2 when available
    Args:
        q, k, v: (B, num_heads, N, head_dim)
    Returns:
        output: (B, num_heads, N, head_dim)
    """
    use_flash = supports_flash_attention(q.device) and training

    if use_flash:
        # FlashAttention requires specific format
        # Reshape to (B, N, num_heads, head_dim)
        B, H, N_q, D = q.shape
        N_kv = k.shape[2]
        
        # CRITICAL: Ensure all tensors have same dtype (FlashAttention requirement)
        # When using AMP, tensors may have different dtypes if coming from different sources
        target_dtype = q.dtype
        if k.dtype != target_dtype:
            k = k.to(target_dtype)
        if v.dtype != target_dtype:
            v = v.to(target_dtype)
        
        q = q.transpose(1, 2)  # (B, N, H, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Use FlashAttention2
        output = flash_attn_func(
            q, k, v,
            dropout_p=dropout_p if training else 0.0,
            softmax_scale=None,
            causal=False
        )
        
        # Reshape back: (B, N, H, D) -> (B, H, N, D)
        output = output.transpose(1, 2)
    else:
        # Fallback to PyTorch SDPA
        output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=dropout_p if training else 0.0,
            is_causal=False
        )
    
    return output


class EfficientSpatialWindowAttention(nn.Module):
    """
    SWA - Efficient Spatial Window Multi-Head Attention
    Matches diagram (c) with:
    - Window partitioning
    - Linformer compression
    - Global Token Bank integration
    - FlashAttention2
    """
    def __init__(self, config: QAViTConfig, global_bank: GlobalTokenBank):
        super().__init__()
        self.config = config
        self.global_bank = global_bank
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.window_size = config.window_size
        
        # Q, K, V projections
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=True)
        
        # Linformer compression
        # Estimate sequence length per window
        self.linformer = LinformerCompression(
            seq_len=self.window_size * self.window_size,
            compressed_len=config.linformer_k
        )
        
        # Output projection
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Layer norm for MLA Gateway
        self.norm = nn.LayerNorm(config.embed_dim)
    
    def window_partition(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        """
        Partition tokens into windows
        Args:
            x: (B, N, C)
        Returns:
            windows: (B*num_windows, window_size^2, C)
        """
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        
        x = x.view(B, H, W, C)
        
        # Pad if needed
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w
        
        # Partition
        num_h = H // window_size
        num_w = W // window_size
        
        x = x.view(B, num_h, window_size, num_w, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size * window_size, C)
        
        return windows
    
    def window_reverse(self, windows: torch.Tensor, window_size: int, H: int, W: int, B: int) -> torch.Tensor:
        """
        Reverse window partition
        """
        num_h = H // window_size
        num_w = W // window_size
        
        x = windows.view(B, num_h, num_w, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, -1)
        x = x.view(B, H * W, -1)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim)
        Returns:
            output: (B, N, embed_dim)
        """
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        
        # Window partition
        x_windows = self.window_partition(x, self.window_size)  # (B*nW, wsize^2, C)
        BW, NW, _ = x_windows.shape
        
        # QKV projection
        qkv = self.qkv(x_windows).reshape(BW, NW, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, BW, H, NW, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Linformer compression on K, V
        k_compressed, v_compressed = self.linformer(k, v)
        
        # Read from Global Token Bank
        k_bank, v_bank = self.global_bank.read(BW)  # (BW, bank_size, C)
        
        # Reshape bank tokens: (BW, bank_size, C) -> (BW, H, bank_size, D)
        k_bank = k_bank.reshape(BW, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_bank = v_bank.reshape(BW, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Concatenate bank tokens with compressed K, V
        k_full = torch.cat([k_compressed, k_bank], dim=2)  # (BW, H, Nk+bank_size, D)
        v_full = torch.cat([v_compressed, v_bank], dim=2)
        
        # FlashAttention2
        attn_output = efficient_attention_with_flash(
            q, k_full, v_full,
            dropout_p=self.dropout.p,
            training=self.training
        )  # (BW, H, NW, D)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(BW, NW, C)
        output = self.proj(attn_output)
        output = self.dropout(output)
        
        # Window reverse
        output = self.window_reverse(output, self.window_size, H, W, B)
        
        # MLA Gateway write
        self.global_bank.write(self.norm(output))
        
        return output


class EfficientMultiScaleDilatedAttention(nn.Module):
    """
    MSDA - Efficient Multi-Scale Dilated Attention
    Matches diagram (b) with:
    - Multi-scale windows via dilation/landmarks
    - Linformer compression
    - Landmark-Stride pooling
    - Global Token Bank integration
    - FlashAttention2
    """
    def __init__(self, config: QAViTConfig, global_bank: GlobalTokenBank):
        super().__init__()
        self.config = config
        self.global_bank = global_bank
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.dilation_factors = config.dilation_factors
        
        # Q, K, V projections
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=True)
        
        # Estimate multi-scale sequence length
        # For 14x14 patches with dilations [1,2,3]: ~196 + 49 + 25 = 270, then pooled by stride 2
        estimated_seq_len = max(config.linformer_k, 128)  # Safe upper bound
        
        # Linformer compression
        self.linformer = LinformerCompression(
            seq_len=estimated_seq_len,
            compressed_len=config.linformer_k
        )
        
        # Landmark pooling
        self.landmark_pool = nn.AvgPool1d(
            kernel_size=config.landmark_pooling_stride,
            stride=config.landmark_pooling_stride
        )
        
        # Output projection
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Layer norm for MLA Gateway
        self.norm = nn.LayerNorm(config.embed_dim)
    
    def extract_dilated_tokens(self, x: torch.Tensor, dilation: int) -> torch.Tensor:
        """
        Extract tokens with dilation factor
        Args:
            x: (B, N, C)
            dilation: dilation factor
        Returns:
            dilated tokens: (B, N//dilation, C)
        """
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        
        x = x.view(B, H, W, C)
        
        # Sample with dilation
        x_dilated = x[:, ::dilation, ::dilation, :]
        x_dilated = x_dilated.reshape(B, -1, C)
        
        return x_dilated
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim)
        Returns:
            output: (B, N, embed_dim)
        """
        B, N, C = x.shape
        
        # Extract multi-scale features
        multi_scale_features = []
        for dilation in self.dilation_factors:
            dilated = self.extract_dilated_tokens(x, dilation)
            multi_scale_features.append(dilated)
        
        # Concatenate multi-scale features
        x_multi = torch.cat(multi_scale_features, dim=1)  # (B, N_total, C)
        
        # Apply landmark pooling for further compression
        x_pooled = self.landmark_pool(x_multi.transpose(1, 2)).transpose(1, 2)
        
        # QKV projection on pooled features for K, V
        BM, NM, _ = x_pooled.shape
        qkv_pooled = self.qkv(x_pooled).reshape(BM, NM, 3, self.num_heads, self.head_dim)
        qkv_pooled = qkv_pooled.permute(2, 0, 3, 1, 4)  # (3, B, H, NM, D)
        _, k, v = qkv_pooled[0], qkv_pooled[1], qkv_pooled[2]
        
        # Pad or truncate to match Linformer expected seq_len if needed
        if NM < self.linformer.E_k.shape[0]:
            # Pad K, V to match expected sequence length
            pad_len = self.linformer.E_k.shape[0] - NM
            k = F.pad(k, (0, 0, 0, pad_len), value=0)
            v = F.pad(v, (0, 0, 0, pad_len), value=0)
        elif NM > self.linformer.E_k.shape[0]:
            # Truncate to match
            k = k[:, :, :self.linformer.E_k.shape[0], :]
            v = v[:, :, :self.linformer.E_k.shape[0], :]
        
        # Linformer compression on K, V
        k_compressed, v_compressed = self.linformer(k, v)
        
        # Read from Global Token Bank
        k_bank, v_bank = self.global_bank.read(B)  # (B, bank_size, C)
        
        # Reshape bank tokens
        k_bank = k_bank.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_bank = v_bank.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Concatenate bank tokens
        k_full = torch.cat([k_compressed, k_bank], dim=2)
        v_full = torch.cat([v_compressed, v_bank], dim=2)
        
        # For queries, use original sequence
        q_orig = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)[:, :, 0]
        q_orig = q_orig.permute(0, 2, 1, 3)  # (B, H, N, D)
        
        # FlashAttention2 - cross attention from full sequence to multi-scale
        attn_output = efficient_attention_with_flash(
            q_orig, k_full, v_full,
            dropout_p=self.dropout.p,
            training=self.training
        )  # (B, H, N, D)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        output = self.proj(attn_output)
        output = self.dropout(output)
        
        # MLA Gateway write
        self.global_bank.write(self.norm(output))
        
        return output


class EfficientChannelGroupAttention(nn.Module):
    """
    CGA - Efficient Channel Group Attention
    Matches diagram (a) with:
    - Channel grouping and reshaping
    - CGA compression (C -> C')
    - Global Token Bank integration
    - FlashAttention2
    """
    def __init__(self, config: QAViTConfig, global_bank: GlobalTokenBank):
        super().__init__()
        self.config = config
        self.global_bank = global_bank
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.num_groups = config.num_channel_groups
        self.channels_per_group = config.embed_dim // self.num_groups
        
        # CGA compression: C -> C'
        self.compress_c = config.embed_dim // 2
        self.compress_per_group = self.compress_c // self.num_groups
        
        # Q, K, V projections with compression
        self.q_proj = nn.Linear(self.channels_per_group, self.compress_per_group)
        self.k_proj = nn.Linear(self.channels_per_group, self.compress_per_group)
        self.v_proj = nn.Linear(self.channels_per_group, self.compress_per_group)
        
        # Bank projection to compressed space
        self.bank_k_proj = nn.Linear(config.embed_dim, self.compress_per_group)
        self.bank_v_proj = nn.Linear(config.embed_dim, self.compress_per_group)
        
        # Output projection
        self.proj = nn.Linear(self.compress_c, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Layer norm for MLA Gateway
        self.norm = nn.LayerNorm(config.embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim)
        Returns:
            output: (B, N, embed_dim)
        """
        B, N, C = x.shape
        
        # Reshape to channel groups: (B, N, C) -> (B, N, num_groups, C_per_group)
        x_grouped = x.view(B, N, self.num_groups, self.channels_per_group)
        
        # Transpose for channel attention: (B, num_groups, N, C_per_group)
        x_grouped = x_grouped.permute(0, 2, 1, 3)
        
        # Reshape for processing: (B*num_groups, N, C_per_group)
        BG = B * self.num_groups
        x_flat = x_grouped.reshape(BG, N, self.channels_per_group)
        
        # Q, K, V projection with compression
        q = self.q_proj(x_flat)  # (BG, N, C'//num_groups)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)
        
        # Reshape for multi-head: (BG, H, N, D')
        head_dim_compressed = self.compress_per_group // self.num_heads
        q = q.reshape(BG, N, self.num_heads, head_dim_compressed).transpose(1, 2)
        k = k.reshape(BG, N, self.num_heads, head_dim_compressed).transpose(1, 2)
        v = v.reshape(BG, N, self.num_heads, head_dim_compressed).transpose(1, 2)
        
        # Read from Global Token Bank
        k_bank, v_bank = self.global_bank.read(B)  # (B, bank_size, embed_dim)
        
        # Project bank to compressed space and expand for groups
        k_bank_compressed = self.bank_k_proj(k_bank)  # (B, bank_size, compress_per_group)
        v_bank_compressed = self.bank_v_proj(v_bank)  # (B, bank_size, compress_per_group)
        
        # Expand for all groups: (B, bank_size, C') -> (BG, bank_size, C'//num_groups)
        k_bank_compressed = k_bank_compressed.unsqueeze(1).expand(-1, self.num_groups, -1, -1)
        v_bank_compressed = v_bank_compressed.unsqueeze(1).expand(-1, self.num_groups, -1, -1)
        k_bank_compressed = k_bank_compressed.reshape(BG, -1, self.compress_per_group)
        v_bank_compressed = v_bank_compressed.reshape(BG, -1, self.compress_per_group)
        
        # Reshape to multi-head format
        k_bank = k_bank_compressed.reshape(BG, -1, self.num_heads, head_dim_compressed).transpose(1, 2)
        v_bank = v_bank_compressed.reshape(BG, -1, self.num_heads, head_dim_compressed).transpose(1, 2)
        
        # Concatenate bank tokens
        k_full = torch.cat([k, k_bank], dim=2)
        v_full = torch.cat([v, v_bank], dim=2)
        
        # FlashAttention2
        attn_output = efficient_attention_with_flash(
            q, k_full, v_full,
            dropout_p=self.dropout.p,
            training=self.training
        )  # (BG, H, N, D')
        
        # Reshape: (BG, H, N, D') -> (BG, N, C'//num_groups)
        attn_output = attn_output.transpose(1, 2).reshape(BG, N, -1)
        
        # Reshape back: (BG, N, C'//num_groups) -> (B, num_groups, N, C'//num_groups)
        attn_output = attn_output.view(B, self.num_groups, N, -1)
        
        # Concatenate groups: (B, N, num_groups, C'//num_groups) -> (B, N, C')
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, self.compress_c)
        
        # Output projection to original dim
        output = self.proj(attn_output)
        output = self.dropout(output)
        
        # MLA Gateway write
        self.global_bank.write(self.norm(output))
        
        return output


class CrossAttentionBranch(nn.Module):
    """
    Cross-Attention branch with Global Token Bank Read
    Queries from local tokens, Keys/Values from global bank
    Uses MLA (Multi-Latent Attention) mechanism
    """
    def __init__(self, config: QAViTConfig, global_bank: GlobalTokenBank):
        super().__init__()
        self.config = config
        self.global_bank = global_bank
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        
        # Query projection (from local tokens)
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        
        # K, V projection for bank tokens
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        
        # Output projection
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim) - local tokens
        Returns:
            output: (B, N, embed_dim)
        """
        B, N, C = x.shape
        
        # Queries from local tokens
        q = self.q_proj(x)  # (B, N, C)
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        
        # Keys and Values from Global Token Bank
        k_bank, v_bank = self.global_bank.read(B)  # (B, bank_size, C)
        
        k = self.k_proj(k_bank)
        v = self.v_proj(v_bank)
        
        k = k.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, bank_size, D)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # FlashAttention2 - cross attention
        attn_output = efficient_attention_with_flash(
            q, k, v,
            dropout_p=self.dropout.p,
            training=self.training
        )  # (B, H, N, D)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        output = self.proj(attn_output)
        output = self.dropout(output)
        
        return output


class HybridFusion(nn.Module):
    """
    Hybrid Fusion module from diagram (b)
    - Element-wise scale via learnable softmax weights
    - Concatenate branches
    - Applies after per-branch compression
    """
    def __init__(self, embed_dim: int, num_branches: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_branches = num_branches
        
        # Learnable fusion weights (softmax)
        self.fusion_weights = nn.Parameter(torch.ones(num_branches))
    
    def forward(self, branches: list) -> torch.Tensor:
        """
        Args:
            branches: list of (B, N, d') tensors from each branch
        Returns:
            fused: (B, N, 4*d') concatenated and weighted
        """
        # Softmax over fusion weights
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # Apply element-wise scaling
        scaled_branches = []
        for i, branch in enumerate(branches):
            scaled = branch * weights[i]
            scaled_branches.append(scaled)
        
        # Concatenate
        fused = torch.cat(scaled_branches, dim=-1)  # (B, N, 4*d')
        
        return fused


class BottleneckMLP(nn.Module):
    """
    Bottleneck MLP from diagram
    4d' -> r -> d
    Where r = d/2 (bottleneck_ratio = 2)
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, input_dim)
        Returns:
            output: (B, N, output_dim)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DepthwiseConv2d(nn.Module):
    """Depthwise Convolution for CCF-FFN"""
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim
        )
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
            H, W: spatial dimensions
        Returns:
            output: (B, N, C)
        """
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class CCFFFN(nn.Module):
    """
    CCF-FFN with DWConv from diagram
    - Compression Layer (Linear)
    - GELU
    - DWConv (3x3, groups=hidden)
    - Expansion Layer (Linear)
    - Dropout
    """
    def __init__(self, embed_dim: int, mlp_ratio: float = 0.5, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.dwconv = DepthwiseConv2d(hidden_dim, kernel_size=3)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim)
        Returns:
            output: (B, N, embed_dim)
        """
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.dwconv(x, H, W)
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x


class QuadAttentionBlock(nn.Module):
    """
    Quad-Attention Transformer Block
    Implements the complete pipeline from diagram:
    1. LayerNorm
    2. Four parallel branches: SWA, MSDA, CGA, Cross-Attention
    3. Per-branch LayerNorm + Compression
    4. Hybrid Fusion
    5. Bottleneck MLP
    6. Residual connection
    7. CCF-FFN with second residual
    """
    def __init__(self, config: QAViTConfig, global_bank: GlobalTokenBank, drop_path: float = 0.):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.compressed_dim = config.embed_dim // config.compress_ratio  # d' = d/4
        
        # Pre-norm for attention branches
        self.norm1 = nn.LayerNorm(config.embed_dim)
        
        # Four attention branches
        self.swa = EfficientSpatialWindowAttention(config, global_bank)
        self.msda = EfficientMultiScaleDilatedAttention(config, global_bank)
        self.cga = EfficientChannelGroupAttention(config, global_bank)
        self.cross_attn = CrossAttentionBranch(config, global_bank)
        
        # Per-branch post-processing
        self.norm_swa = nn.LayerNorm(config.embed_dim)
        self.norm_msda = nn.LayerNorm(config.embed_dim)
        self.norm_cga = nn.LayerNorm(config.embed_dim)
        self.norm_cross = nn.LayerNorm(config.embed_dim)
        
        # Per-branch compression: d -> d'
        self.compress_swa = nn.Linear(config.embed_dim, self.compressed_dim)
        self.compress_msda = nn.Linear(config.embed_dim, self.compressed_dim)
        self.compress_cga = nn.Linear(config.embed_dim, self.compressed_dim)
        self.compress_cross = nn.Linear(config.embed_dim, self.compressed_dim)
        
        # Hybrid Fusion
        self.fusion = HybridFusion(self.compressed_dim, num_branches=4)
        
        # Bottleneck MLP: 4d' -> r -> d, where r = d/2
        bottleneck_hidden = config.embed_dim // config.bottleneck_ratio
        self.bottleneck_mlp = BottleneckMLP(
            input_dim=4 * self.compressed_dim,
            hidden_dim=bottleneck_hidden,
            output_dim=config.embed_dim,
            dropout=config.dropout
        )
        
        # Drop path
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Second norm and FFN
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.ccf_ffn = CCFFFN(config.embed_dim, mlp_ratio=config.mlp_ratio, dropout=config.dropout)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim)
        Returns:
            output: (B, N, embed_dim)
        """
        # 1. Pre-norm
        x_norm = self.norm1(x)
        
        # 2. Four parallel branches on LN(X)
        swa_out = self.swa(x_norm)
        msda_out = self.msda(x_norm)
        cga_out = self.cga(x_norm)
        cross_out = self.cross_attn(x_norm)
        
        # 3. Per-branch LayerNorm
        swa_out = self.norm_swa(swa_out)
        msda_out = self.norm_msda(msda_out)
        cga_out = self.norm_cga(cga_out)
        cross_out = self.norm_cross(cross_out)
        
        # 4. Per-branch compression: d -> d'
        swa_compressed = self.compress_swa(swa_out)
        msda_compressed = self.compress_msda(msda_out)
        cga_compressed = self.compress_cga(cga_out)
        cross_compressed = self.compress_cross(cross_out)
        
        # 5. Hybrid Fusion (softmax-weighted + concatenate -> 4d')
        fused = self.fusion([swa_compressed, msda_compressed, cga_compressed, cross_compressed])
        
        # 6. Bottleneck MLP: 4d' -> r -> d
        mlp_out = self.bottleneck_mlp(fused)
        
        # 7. First residual connection
        x = x + self.drop_path1(mlp_out)
        
        # 8. Second norm and CCF-FFN
        x = x + self.drop_path2(self.ccf_ffn(self.norm2(x)))
        
        return x


class PatchEmbed(nn.Module):
    """
    Pure ViT Patch Embedding
    Image to Patch Embedding via linear projection
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            patches: (B, N, embed_dim)
        """
        B, C, H, W = x.shape
        
        # Patch projection
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        x = self.norm(x)
        
        return x


class QAViT(nn.Module):
    """
    Quad-Attention Vision Transformer (QAViT)
    Complete model matching all architectural diagrams
    """
    def __init__(self, config: QAViTConfig):
        super().__init__()
        self.config = config
        self.num_patches = (config.img_size // config.patch_size) ** 2
        
        # Patch embedding (pure ViT)
        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, config.embed_dim))
        self.pos_drop = nn.Dropout(p=config.dropout)
        
        # Global Token Bank (shared across all blocks)
        self.global_bank = GlobalTokenBank(config.global_bank_size, config.embed_dim)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path, config.depth)]
        
        # Quad-Attention Transformer Blocks
        self.blocks = nn.ModuleList([
            QuadAttentionBlock(
                config=config,
                global_bank=self.global_bank,
                drop_path=dpr[i]
            )
            for i in range(config.depth)
        ])
        
        # Final norm and head
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        
        # Initialize weights
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
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            features: (B, N, embed_dim)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply Quad-Attention blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm
        x = self.norm(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        x = self.forward_features(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification head
        x = self.head(x)
        
        return x


# ==============================================================================
# Model Configurations
# ==============================================================================

def qavit_tiny(**kwargs) -> QAViT:
    """QAViT-Tiny model"""
    # Default config
    default_config = {
        'img_size': 224,
        'patch_size': 16,
        'in_channels': 3,
        'num_classes': 1000,
        'embed_dim': 256,  # d
        'depth': 12,
        'num_heads': 4,
        'compress_ratio': 4,  # d' = d/4 = 64
        'bottleneck_ratio': 2,  # r = d/2 = 128
        'mlp_ratio': 0.5,
        'global_bank_size': 16,
        'dropout': 0.1,
        'drop_path': 0.1,
        'window_size': 7,
        'dilation_factors': (1, 2, 3),
        'landmark_pooling_stride': 2,
        'num_channel_groups': 8,
        'linformer_k': 64,
    }
    # Override with kwargs
    default_config.update(kwargs)
    config = QAViTConfig(**default_config)
    model = QAViT(config)
    return model


def qavit_small(**kwargs) -> QAViT:
    """QAViT-Small model"""
    # Default config
    default_config = {
        'img_size': 224,
        'patch_size': 16,
        'in_channels': 3,
        'num_classes': 1000,
        'embed_dim': 384,  # d
        'depth': 16,
        'num_heads': 6,
        'compress_ratio': 4,  # d' = d/4 = 96
        'bottleneck_ratio': 2,  # r = d/2 = 192
        'mlp_ratio': 0.5,
        'global_bank_size': 32,
        'dropout': 0.1,
        'drop_path': 0.15,
        'window_size': 7,
        'dilation_factors': (1, 2, 3),
        'landmark_pooling_stride': 2,
        'num_channel_groups': 8,  # 8 groups: 192/8=24, 24/6=4 per head
        'linformer_k': 64,
    }
    # Override with kwargs
    default_config.update(kwargs)
    config = QAViTConfig(**default_config)
    model = QAViT(config)
    return model


# ==============================================================================
# Modular Dataset Configuration
# ==============================================================================

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from typing import Callable, Optional


class DatasetConfig:
    """Modular dataset configuration"""
    def __init__(
        self,
        dataset_name: str,
        data_root: str,
        num_classes: int,
        img_size: int = 224,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
        batch_size: int = 128,
        num_workers: int = 4,
    ):
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.num_classes = num_classes
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.num_workers = num_workers


# Predefined dataset configurations
DATASET_CONFIGS = {
    'imagenet': DatasetConfig(
        dataset_name='imagenet',
        data_root='/path/to/imagenet',
        num_classes=1000,
        img_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        batch_size=128,
        num_workers=8,
    ),
    'cifar10': DatasetConfig(
        dataset_name='cifar10',
        data_root='./data',
        num_classes=10,
        img_size=224,  # Resize to match ViT input
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        batch_size=256,
        num_workers=4,
    ),
    'cifar100': DatasetConfig(
        dataset_name='cifar100',
        data_root='./data',
        num_classes=100,
        img_size=224,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
        batch_size=256,
        num_workers=4,
    ),
    'custom': DatasetConfig(
        dataset_name='custom',
        data_root='./data/custom',
        num_classes=10,  # Change this
        img_size=224,
        mean=(0.5, 0.5, 0.5),  # Change this
        std=(0.5, 0.5, 0.5),   # Change this
        batch_size=128,
        num_workers=4,
    ),
}


def get_transforms(config: DatasetConfig, is_train: bool = True):
    """Get data transforms based on config - optimized for speed"""
    if is_train:
        # Optimize for CIFAR datasets - resize first, then crop
        if 'cifar' in config.dataset_name.lower():
            transform = transforms.Compose([
                transforms.Resize((config.img_size, config.img_size), antialias=True),  # 32→224 directly
                transforms.RandomCrop(config.img_size, padding=int(config.img_size * 0.125)),  # Crop 224 with padding ~28
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.mean, std=config.std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((config.img_size, config.img_size), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(config.img_size, padding=4),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.mean, std=config.std),
            ])
    else:
        transform = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ])
    return transform


def build_dataset(config: DatasetConfig, is_train: bool = True):
    """Build dataset based on config"""
    transform = get_transforms(config, is_train=is_train)
    
    if config.dataset_name == 'imagenet':
        split = 'train' if is_train else 'val'
        dataset = datasets.ImageFolder(
            root=Path(config.data_root) / split,
            transform=transform
        )
    elif config.dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(
            root=config.data_root,
            train=is_train,
            transform=transform,
            download=True
        )
    elif config.dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(
            root=config.data_root,
            train=is_train,
            transform=transform,
            download=True
        )
    elif config.dataset_name == 'custom':
        # For custom datasets using ImageFolder structure
        split = 'train' if is_train else 'val'
        dataset = datasets.ImageFolder(
            root=Path(config.data_root) / split,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")
    
    return dataset


def build_dataloader(config: DatasetConfig, is_train: bool = True):
    """Build dataloader based on config - optimized for performance"""
    dataset = build_dataset(config, is_train=is_train)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_train,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2 if config.num_workers > 0 else None
    )
    
    return dataloader


# ==============================================================================
# Training Setup Example
# ==============================================================================

def create_model_and_dataloaders(
    model_size: str = 'tiny',
    dataset_name: str = 'cifar10',
    custom_config: Optional[DatasetConfig] = None
):
    """
    Create model and dataloaders with easy configuration
    
    Args:
        model_size: 'tiny' or 'small'
        dataset_name: dataset name from DATASET_CONFIGS or 'custom'
        custom_config: optional custom DatasetConfig
    
    Returns:
        model, train_loader, val_loader, dataset_config
    """
    # Get dataset config
    if custom_config is not None:
        dataset_config = custom_config
    else:
        dataset_config = DATASET_CONFIGS[dataset_name]
    
    # Create model
    if model_size == 'tiny':
        model = qavit_tiny(num_classes=dataset_config.num_classes)
    elif model_size == 'small':
        model = qavit_small(num_classes=dataset_config.num_classes)
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    # Create dataloaders
    train_loader = build_dataloader(dataset_config, is_train=True)
    val_loader = build_dataloader(dataset_config, is_train=False)
    
    print(f"Model: QAViT-{model_size.capitalize()}")
    print(f"Dataset: {dataset_config.dataset_name}")
    print(f"Num classes: {dataset_config.num_classes}")
    print(f"Image size: {dataset_config.img_size}")
    print(f"Batch size: {dataset_config.batch_size}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    return model, train_loader, val_loader, dataset_config


# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == "__main__":
    # Example 1: Use predefined dataset config
    print("="*80)
    print("Example 1: CIFAR-10 with QAViT-Tiny")
    print("="*80)
    model, train_loader, val_loader, config = create_model_and_dataloaders(
        model_size='tiny',
        dataset_name='cifar10'
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    print("\n" + "="*80)
    print("Example 2: Custom dataset config")
    print("="*80)
    
    # Example 2: Custom dataset
    custom_config = DatasetConfig(
        dataset_name='custom',
        data_root='./my_custom_dataset',
        num_classes=20,  # Your number of classes
        img_size=224,
        mean=(0.5, 0.5, 0.5),  # Your dataset mean
        std=(0.5, 0.5, 0.5),   # Your dataset std
        batch_size=64,
        num_workers=4,
    )
    
    print(f"\nCustom config created:")
    print(f"  Classes: {custom_config.num_classes}")
    print(f"  Image size: {custom_config.img_size}")
    print(f"  Mean: {custom_config.mean}")
    print(f"  Std: {custom_config.std}")
    
    # You can then use:
    # model, train_loader, val_loader, _ = create_model_and_dataloaders(
    #     model_size='small',
    #     custom_config=custom_config
    # )
    
    print("\n" + "="*80)
    print("Setup complete! Ready for training.")
    print("="*80)
