"""Exponential Moving Average (EMA) weight tracker (SIGNAL INFINITY §2.5).

Follows the YOLOv5 / timm ModelEMA design.  Two defects in the previous version
made the EMA model -- the one validated and saved as "best" -- unusable:

* **No warm-up.**  With a fixed decay of 0.9998 the average after N updates
  still holds 0.9998**N of the random initialisation (96 % after 200 steps,
  13 % after 10 000), so any short or medium run deployed an essentially
  untrained network.  The decay now ramps as ``decay * (1 - exp(-t / tau))``.
* **BatchNorm statistics were never tracked.**  Only ``parameters()`` were
  averaged; the running mean/variance buffers stayed at their initial 0 / 1,
  so the EMA model normalised every activation with the wrong statistics in
  eval mode no matter how long training ran.  Every floating-point entry of
  the state dict is now averaged, buffers included.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


class ModelEMA:
    """Exponential Moving Average of a model's full state (parameters and buffers)."""

    def __init__(self, model: nn.Module, decay: float = 0.9998, tau: float = 2000.0):
        self.ema_model = copy.deepcopy(model).eval()
        self.decay = float(decay)
        self.tau = float(tau)
        self.updates = 0
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def current_decay(self) -> float:
        return self.decay * (1.0 - math.exp(-self.updates / self.tau))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.current_decay()
        msd = model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])        # e.g. BatchNorm num_batches_tracked

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)
