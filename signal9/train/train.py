"""Supervised training loop with AMP, hierarchical loss, and EMA."""
from __future__ import annotations
import math
import copy
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from .losses import hier_loss
from ..models.backbone import Signal9Net

class EMAModel:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters() if p.requires_grad}

    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])

def train_epoch(model: Signal9Net,
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                scaler: torch.cuda.amp.GradScaler | None = None,
                ema: EMAModel | None = None,
                use_amp: bool = True) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        iq = batch["iq"].to(device)
        spec = batch["spec"].to(device)
        expert = batch["expert"].to(device)
        meta = batch["meta"].to(device)
        targets = {k: v.to(device) for k, v in batch["targets"].items()}
        sample_w = batch.get("sample_w", None)
        if sample_w is not None:
            sample_w = sample_w.to(device)

        optimizer.zero_grad()
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        with torch.autocast(device_type=device.type if device.type != "mps" else "cpu",
                            dtype=amp_dtype,
                            enabled=use_amp and device.type == "cuda"):
            outputs = model(iq, spec, expert, meta)
            loss = hier_loss(outputs, targets, sample_w=sample_w)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)
