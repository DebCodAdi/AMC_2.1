"""Continual learning regularisation: EWC and LwF distillation (SIGNAL INFINITY §4.1)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinualRegularizer(nn.Module):
    """Elastic Weight Consolidation (EWC) and Learning without Forgetting (LwF)."""
    def __init__(self,
                 old_model: nn.Module | None = None,
                 fisher_diag: dict[str, torch.Tensor] | None = None,
                 ewc_lambda: float = 1000.0,
                 lwf_lambda: float = 1.0,
                 temperature: float = 2.0):
        super().__init__()
        self.old_params = {}
        if old_model:
            for name, param in old_model.named_parameters():
                self.old_params[name] = param.detach().clone()

        self.fisher_diag = fisher_diag or {}
        self.ewc_lambda = ewc_lambda
        self.lwf_lambda = lwf_lambda
        self.temperature = temperature

    def ewc_loss(self, current_model: nn.Module) -> torch.Tensor:
        if not self.old_params:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=next(current_model.parameters()).device)
        for name, param in current_model.named_parameters():
            if name in self.old_params:
                old_p = self.old_params[name].to(param.device)
                f_i = self.fisher_diag.get(name, torch.ones_like(param)).to(param.device)
                loss += torch.sum(f_i * (param - old_p) ** 2)

        return self.ewc_lambda * loss

    def lwf_loss(self, new_logits: torch.Tensor, old_logits: torch.Tensor) -> torch.Tensor:
        """Knowledge distillation loss matching old model predictions on replayed samples."""
        t = self.temperature
        p_new = F.log_softmax(new_logits / t, dim=-1)
        p_old = F.softmax(old_logits / t, dim=-1)
        kl = F.kl_div(p_new, p_old, reduction="batchmean") * (t ** 2)
        return self.lwf_lambda * kl
