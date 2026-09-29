"""Multi-task prediction heads for Signal Infinity (SIGNAL INFINITY §2.1)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SignalInfinityMultiTaskHeads(nn.Module):
    """8 specialized prediction heads off the shared 512-D embedding space."""
    def __init__(self,
                 emb_dim: int = 512,
                 n_mods: int = 32,
                 n_fecs: int = 5,
                 n_scramblers: int = 5):
        super().__init__()
        # 1. Modulation classification head
        self.mod_head = nn.Linear(emb_dim, n_mods)

        # 2. Symbol rate regression head (predicts log(symrate))
        self.symrate_head = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # 3. Roll-off factor regression head (predicts alpha in [0.1, 0.8])
        self.rolloff_head = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # scaled later to [0.0, 1.0]
        )

        # 4. CFO regression head (predicts cfo / fs)
        self.cfo_head = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh()  # in [-0.5, 0.5]
        )

        # 5. SNR regression head
        self.snr_head = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # 6. FEC classification head
        self.fec_head = nn.Linear(emb_dim, n_fecs)

        # 7. Scrambler classification head
        self.scrambler_head = nn.Linear(emb_dim, n_scramblers)

        # 8. Binary presence / active detection head (Bucket F)
        self.presence_head = nn.Linear(emb_dim, 1)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "mod_logits": self.mod_head(z),
            "log_symrate": self.symrate_head(z).squeeze(-1),
            "rolloff": 0.1 + 0.7 * self.rolloff_head(z).squeeze(-1),
            "cfo_norm": 0.5 * self.cfo_head(z).squeeze(-1),
            "snr_db": self.snr_head(z).squeeze(-1),
            "fec_logits": self.fec_head(z),
            "scrambler_logits": self.scrambler_head(z),
            "presence_logit": self.presence_head(z).squeeze(-1),
        }
