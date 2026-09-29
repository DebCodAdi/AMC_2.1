"""Self-supervised pretraining (Masked Spectrogram Modeling and Contrastive IQ)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveIQHead(nn.Module):
    """SimCLR-style contrastive projection head for task-aware raw IQ views."""
    def __init__(self, in_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Normalized Temperature-scaled Cross Entropy Loss for contrastive representation learning."""
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)
    
    # Mask self-similarity
    diag_mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim.masked_fill_(diag_mask, -1e9)

    # Positive pairs are (i, i + B) and (i + B, i)
    pos_indices = torch.arange(batch_size, device=z.device)
    pos_targets_1 = pos_indices + batch_size
    pos_targets_2 = pos_indices
    targets = torch.cat([pos_targets_1, pos_targets_2], dim=0)

    loss = F.cross_entropy(sim, targets)
    return loss
