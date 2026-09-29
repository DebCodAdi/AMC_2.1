"""Classification and auxiliary estimation heads."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineHead(nn.Module):
    """Cosine similarity classifier head: logits = scale * (norm(x) @ norm(W)^T)."""
    def __init__(self, in_features: int, num_classes: int, scale: float = 16.0):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_norm = F.normalize(self.weight, dim=-1)
        x_norm = F.normalize(x, dim=-1)
        return self.scale * F.linear(x_norm, w_norm)

class HierarchicalHeads(nn.Module):
    """3-Level Hierarchical heads + auxiliary estimation heads."""
    def __init__(self, in_dim: int = 256, n_l1: int = 9, n_l2: int = 24, n_l3: int = 19):
        super().__init__()
        self.head_l1 = CosineHead(in_dim, n_l1)
        self.head_l2 = CosineHead(in_dim, n_l2)
        self.head_l3 = CosineHead(in_dim, n_l3)

        # Auxiliary regression heads
        self.head_snr = nn.Linear(in_dim, 1)
        self.head_lbaud = nn.Linear(in_dim, 1)
        self.head_burst = nn.Linear(in_dim, 1)
        
        # Outlier Exposure head
        self.head_outlier = nn.Linear(in_dim, 1)

    def forward(self, emb: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "l1": self.head_l1(emb),
            "l2": self.head_l2(emb),
            "l3": self.head_l3(emb),
            "snr": self.head_snr(emb).squeeze(-1),
            "log_baud": self.head_lbaud(emb).squeeze(-1),
            "burst": self.head_burst(emb).squeeze(-1),
            "outlier": self.head_outlier(emb).squeeze(-1),
        }
