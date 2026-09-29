"""Parameter-Efficient Fine-Tuning (LoRA / Prompt Adapter) for hardware adaptation."""
from __future__ import annotations
import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) layer for linear projections."""
    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        in_f = original_linear.in_features
        out_f = original_linear.out_features
        self.rank = rank
        self.scale = alpha / rank

        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.original(x)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scale
        return orig_out + lora_out
