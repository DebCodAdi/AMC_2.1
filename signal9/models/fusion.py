"""Modality fusion module with branch dropout."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityDropout(nn.Module):
    """Randomly zeros out an entire modality during training with probability p."""
    def __init__(self, p: float = 0.15):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        # Ensure at least one modality stays active
        mask = (torch.rand(x.shape[0], 1, device=x.device) > self.p).float()
        return x * mask

class ModalityFusion(nn.Module):
    """Fuses IQ, Spectrogram, and Expert feature branches."""
    def __init__(self, iq_dim: int = 128, spec_dim: int = 256, expert_dim: int = 32, meta_dim: int = 16, fused_dim: int = 256, drop_p: float = 0.15):
        super().__init__()
        if not 0.0 <= drop_p < 1.0:
            raise ValueError("drop_p must be in [0, 1)")
        self.drop_p = float(drop_p)

        self.expert_dim = expert_dim
        if expert_dim > 0:
            self.expert_encoder = nn.Sequential(
                nn.Linear(expert_dim, 64),
                nn.GELU(),
                nn.Linear(64, 64)
            )
            expert_out = 64
        else:
            self.expert_encoder = None
            expert_out = 0
            
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64)
        )

        in_total = iq_dim + spec_dim + expert_out + 64
        self.fusion_mlp = nn.Sequential(
            nn.Linear(in_total, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, fused_dim),
        )

    def forward(self, iq_feat: torch.Tensor,
                spec_feat: torch.Tensor,
                expert_feat: torch.Tensor | None,
                meta_feat: torch.Tensor) -> torch.Tensor:
        raw_modalities = [iq_feat, spec_feat]
        if self.expert_encoder is not None and expert_feat is not None:
            raw_modalities.append(expert_feat)

        # Draw masks jointly.  Independent branch-dropout modules could erase
        # every signal-bearing branch in the same example.  Keep at least one
        # branch active while retaining the requested marginal dropout rate.
        if self.training and self.drop_p > 0.0:
            b = iq_feat.shape[0]
            masks = torch.rand(b, len(raw_modalities), device=iq_feat.device) > self.drop_p
            all_off = ~masks.any(dim=1)
            if all_off.any():
                chosen = torch.randint(len(raw_modalities), (int(all_off.sum()),), device=iq_feat.device)
                masks[all_off] = False
                masks[all_off, chosen] = True
            raw_modalities = [x * masks[:, i:i + 1].to(x.dtype)
                              for i, x in enumerate(raw_modalities)]

        iq, spec = raw_modalities[:2]
        
        parts = [iq, spec]
        if self.expert_encoder is not None and expert_feat is not None:
            expert = self.expert_encoder(raw_modalities[2])
            parts.append(expert)
            
        meta = self.meta_encoder(meta_feat)
        parts.append(meta)

        cat = torch.cat(parts, dim=-1)
        fused = self.fusion_mlp(cat)
        return F.normalize(fused, dim=-1)
