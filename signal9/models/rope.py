"""Rotary Position Embedding (RoPE) and Attention.
Fixes B3: RoPE is applied to per-head Q and K inside attention.
Freqs table is computed in fp32 outside autocast to prevent underflow.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryTable(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        # Always compute freqs in fp32 outside autocast
        with torch.autocast(device_type=device.type if device.type != "mps" else "cpu", enabled=False):
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.float())  # (L, D/2)
            return freqs.cos(), freqs.sin()

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary rotation to per-head query or key tensor (B, H, L, D)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    rot = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return rot

class RoPEAttention(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.h = n_heads
        self.d = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.rope = RotaryTable(self.d)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, L, d_model), pad_mask: (B, L) where True indicates padding."""
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is (B, H, L, D)

        cos, sin = self.rope(L, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_mask = None if pad_mask is None else ~pad_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(B, L, -1))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 8, ffn_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.att = RoPEAttention(d_model, n_heads, dropout)
        self.n2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.att(self.n1(x), pad_mask)
        x = x + self.ffn(self.n2(x))
        return x
