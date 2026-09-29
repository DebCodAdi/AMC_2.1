"""Detector-style bounding-box jitter and crop augmentation."""
from __future__ import annotations
import numpy as np

def jitter_box(f_lo: float, f_hi: float, t_lo: float, t_hi: float,
               max_center_jitter_frac: float = 0.10,
               width_scale_range: tuple[float, float] = (0.7, 1.5),
               rng: np.random.Generator | None = None) -> tuple[float, float, float, float]:
    """Augment bounding box coordinates to simulate detector boundary errors.
    Ensures downstream classifiers are invariant to realistic imperfect localization.
    """
    rng = rng or np.random.default_rng()
    bw = f_hi - f_lo
    fc = 0.5 * (f_lo + f_hi)
    
    # 1. Center jitter
    shift = rng.uniform(-max_center_jitter_frac, max_center_jitter_frac) * bw
    new_fc = fc + shift
    
    # 2. Width scaling
    scale = rng.uniform(width_scale_range[0], width_scale_range[1])
    new_bw = max(100.0, bw * scale)
    
    new_f_lo = new_fc - 0.5 * new_bw
    new_f_hi = new_fc + 0.5 * new_bw

    # 3. Time jitter
    dt = t_hi - t_lo
    t_shift = rng.uniform(-0.05, 0.05) * dt
    new_t_lo = max(0.0, t_lo + t_shift)
    new_t_hi = max(new_t_lo + 0.01, t_hi + t_shift)

    return new_f_lo, new_f_hi, new_t_lo, new_t_hi
