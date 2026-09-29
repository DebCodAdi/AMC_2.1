"""Full Signal9 multi-modal neural network architecture."""
from __future__ import annotations
import torch
import torch.nn as nn
from .iq_branch import IQBranch
from .spec_branch import SpecBranch
from .fusion import ModalityFusion
from .heads import HierarchicalHeads

class Signal9Net(nn.Module):
    """End-to-end Signal 9 AMC Neural Architecture."""
    def __init__(self,
                 n_l1: int = 9,
                 n_l2: int = 24,
                 n_l3: int = 19,
                 emb_dim: int = 256,
                 expert_dim: int = 32,
                 meta_dim: int = 16):
        super().__init__()
        self.iq_branch = IQBranch(in_chans=4, d_model=128, n_layers=4, n_heads=8, out_dim=256)
        self.spec_branch = SpecBranch(in_chans=1, out_dim=256)
        # Banning expert vector as per Blind_Signal Appendix A13
        self.fusion = ModalityFusion(iq_dim=256, spec_dim=256, expert_dim=0, meta_dim=meta_dim, fused_dim=emb_dim)
        self.heads = HierarchicalHeads(in_dim=emb_dim, n_l1=n_l1, n_l2=n_l2, n_l3=n_l3)

    def forward(self, iq: torch.Tensor,
                spec: torch.Tensor,
                expert: torch.Tensor | None,
                meta: torch.Tensor) -> dict[str, torch.Tensor]:
        """iq: (B, 4, 4096), spec: (B, 1, 224, 224), meta: (B, 16)"""
        iq_feat = self.iq_branch(iq)
        spec_feat = self.spec_branch(spec)
        emb = self.fusion(iq_feat, spec_feat, expert, meta)
        outputs = self.heads(emb)
        outputs["embedding"] = emb
        return outputs
