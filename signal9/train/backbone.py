"""Multi-task neural backbone with complex-valued conv stem and expert feature fusion (SIGNAL INFINITY §2.1)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv1dStem(nn.Module):
    """4-stage 1D convolution stem downsampling 4096 complex samples to 256 tokens."""
    def __init__(self, in_channels: int = 2, stem_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, stem_dim // 2, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(stem_dim // 2)
        self.conv2 = nn.Conv1d(stem_dim // 2, stem_dim, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(stem_dim)
        self.conv3 = nn.Conv1d(stem_dim, stem_dim * 2, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(stem_dim * 2)
        self.conv4 = nn.Conv1d(stem_dim * 2, stem_dim * 2, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(stem_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 2, 4096) -> real and imag
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.gelu(self.bn4(self.conv4(x)))
        return x  # (B, 256, 256)


class ResNetBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.gelu(x + res)


class CrossCovarianceAttention1D(nn.Module):
    """XCiT-style channel attention with linear sequence-length cost."""
    def __init__(self, dim: int = 256, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(b, n, self.heads, self.head_dim).permute(0, 2, 3, 1)
        k = k.reshape(b, n, self.heads, self.head_dim).permute(0, 2, 3, 1)
        v = v.reshape(b, n, self.heads, self.head_dim).permute(0, 2, 3, 1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 3, 1, 2).reshape(b, n, c)
        return self.proj(out)


class XCiTBlock1D(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8, mlp_ratio: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.local = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CrossCovarianceAttention1D(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.local(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class SignalInfinityBackbone(nn.Module):
    """Unified multi-task encoder embedding IQ windows to 512-D space with optional pre-fusion SNR gating."""
    def __init__(self,
                 in_channels: int = 2,
                 stem_dim: int = 128,
                 expert_dim: int = 32,
                 emb_dim: int = 512):
        super().__init__()
        self.expert_dim = int(expert_dim)
        self.stem = ComplexConv1dStem(in_channels=in_channels, stem_dim=stem_dim)
        
        # ResNet body
        self.body = nn.Sequential(
            ResNetBlock1D(stem_dim * 2),
            ResNetBlock1D(stem_dim * 2),
            ResNetBlock1D(stem_dim * 2),
        )
        
        # Attentive / Adaptive pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Pre-fusion SNR estimator (operates purely on uncontaminated learned waveform features)
        self.pre_snr_head = nn.Sequential(
            nn.Linear(stem_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Learned SNR gate: maps estimated SNR (dB) to an attenuation scalar in [0, 1]
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Projection fusing learned embedding and classical expert cumulants (R21)
        self.fusion = nn.Sequential(
            nn.Linear(stem_dim * 2 + expert_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, iq: torch.Tensor,
                expert_feats: torch.Tensor | None = None,
                enable_gating: bool = False,
                return_snr: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # iq: (B, 2, 4096)
        h = self.stem(iq)
        h = self.body(h)
        z_learned = self.pool(h).squeeze(-1)  # (B, 256)

        snr_est = self.pre_snr_head(z_learned)  # (B, 1)

        if expert_feats is None:
            # Default zero vector if not supplied
            expert_feats = torch.zeros(iq.size(0), self.expert_dim, device=iq.device, dtype=iq.dtype)

        if enable_gating:
            gate = self.gate_mlp(snr_est)
            gated_expert = expert_feats * gate
        else:
            gated_expert = expert_feats
        
        fused = torch.cat([z_learned, gated_expert], dim=1)
        z = self.fusion(fused)  # (B, 512)

        if return_snr:
            return z, snr_est.squeeze(-1)
        return z


class ResearchHybridXCiTBackbone(nn.Module):
    """Residual RF front-end + XCiT channel attention + attentive pooling.

    The convolutional front-end captures local symbol transitions; XCA models
    long-range channel relationships at linear cost in window length. Expert
    statistics are fused only after learned attention pooling.
    """
    def __init__(self, in_channels: int = 2, stem_dim: int = 128,
                 expert_dim: int = 32, emb_dim: int = 512,
                 depth: int = 3, heads: int = 8):
        super().__init__()
        self.expert_dim = int(expert_dim)
        token_dim = stem_dim * 2
        self.stem = ComplexConv1dStem(in_channels, stem_dim)
        self.body = nn.Sequential(ResNetBlock1D(token_dim), ResNetBlock1D(token_dim))
        self.blocks = nn.ModuleList([XCiTBlock1D(token_dim, heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(token_dim)
        self.attn_pool = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2), nn.Tanh(),
            nn.Linear(token_dim // 2, 1, bias=False),
        )
        self.expert_norm = nn.LayerNorm(expert_dim)
        self.expert_gate = nn.Sequential(
            nn.Linear(token_dim + expert_dim, expert_dim), nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Linear(token_dim + expert_dim, emb_dim),
            nn.BatchNorm1d(emb_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, iq: torch.Tensor, expert_feats: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.body(self.stem(iq)).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        weights = torch.softmax(self.attn_pool(tokens), dim=1)
        learned = torch.sum(tokens * weights, dim=1)
        if expert_feats is None:
            expert_feats = torch.zeros(iq.size(0), self.expert_dim, device=iq.device, dtype=iq.dtype)
        expert = self.expert_norm(expert_feats)
        expert = expert * self.expert_gate(torch.cat([learned, expert], dim=1))
        return self.fusion(torch.cat([learned, expert], dim=1))


def build_backbone(architecture: str = "baseline_resnet", **kwargs) -> nn.Module:
    if architecture == "baseline_resnet":
        return SignalInfinityBackbone(**kwargs)
    if architecture == "hybrid_xcit":
        return ResearchHybridXCiTBackbone(**kwargs)
    raise ValueError(f"unsupported architecture {architecture!r}")
