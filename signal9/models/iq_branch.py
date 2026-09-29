"""1D Time-Domain IQ Transformer Branch with Conv stem downsampling to 256 tokens."""
from __future__ import annotations
import torch
import torch.nn as nn
from .rope import TransformerBlock

class IQBranch(nn.Module):
    """Processes raw IQ signals.
    Input: (B, 2, 4096) or (B, 4, 4096) [I, Q, |x|, dphi]
    Output: (B, out_dim) (default out_dim=128 or 256)
    """
    def __init__(self, in_chans: int = 4, d_model: int = 128, n_layers: int = 4, n_heads: int = 8, out_dim: int = 256):
        super().__init__()
        # Conv stem: stride 2 x 4 = stride 16 -> 4096 samples becomes 256 tokens
        ch = [in_chans, 32, 64, 96, d_model]
        stem, k = [], [15, 11, 7, 5]
        for i in range(4):
            stem += [nn.Conv1d(ch[i], ch[i+1], k[i], stride=2, padding=k[i]//2),
                     nn.BatchNorm1d(ch[i+1]), nn.GELU()]
        self.stem = nn.Sequential(*stem)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, ffn_dim=d_model * 4)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # Attentive pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_chans, 4096)"""
        if x.ndim == 2:
            # (B, 4096) -> treat as real/imag interleaved or single channel
            x = x.unsqueeze(1)
        z = self.stem(x)  # (B, d_model, L=256)
        z = z.transpose(1, 2)  # (B, L=256, d_model)

        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)

        # Attentive pooling: softmax(z @ pool_query^T)
        scores = torch.softmax((z @ self.pool_query.transpose(1, 2)).squeeze(-1), dim=1)  # (B, L)
        pooled = (scores.unsqueeze(-1) * z).sum(dim=1)  # (B, d_model)
        return self.out_proj(pooled)
