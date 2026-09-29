"""Auto-curriculum sampling weight computation from capability surface (SIGNAL INFINITY §3.6)."""
from __future__ import annotations
import numpy as np


def compute_curriculum_weights(measured_acc: dict[str, float],
                               sample_counts: dict[str, int],
                               target_acc: float = 0.95,
                               gamma: float = 1.5,
                               beta: float = 2.0,
                               floor_weight: float = 0.05,
                               max_ratio: float = 20.0) -> dict[str, float]:
    """Compute per-cell sampling weights reinforcing weak and under-explored regions."""
    raw_weights = {}
    for cell, acc in measured_acc.items():
        n = sample_counts.get(cell, 1)
        # 1. Error deficit term
        deficit = np.clip((target_acc - acc) / target_acc, 0.0, 1.0)
        w_deficit = (max(floor_weight, deficit)) ** gamma

        # 2. Exploration bonus for under-sampled cells
        bonus = 1.0 + beta / np.sqrt(1.0 + n)

        raw_weights[cell] = float(w_deficit * bonus)

    # Apply ceiling cap to prevent single pathological cell from consuming budget
    min_w = min(raw_weights.values()) if raw_weights else 1.0
    max_allowed = min_w * max_ratio

    capped_weights = {
        cell: float(np.clip(w, floor_weight, max_allowed))
        for cell, w in raw_weights.items()
    }

    # Normalize weights so mean is 1.0
    mean_w = np.mean(list(capped_weights.values())) if capped_weights else 1.0
    return {k: float(v / mean_w) for k, v in capped_weights.items()}
