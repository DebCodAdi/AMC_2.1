"""Spectral flatness and duty cycle metrics."""
from __future__ import annotations
import numpy as np

def spectral_flatness(p_linear: np.ndarray) -> float:
    """Compute spectral flatness (Wiener entropy): geometric mean / arithmetic mean."""
    p_safe = np.clip(p_linear, 1e-12, None)
    geom_mean = np.exp(np.mean(np.log(p_safe)))
    arith_mean = np.mean(p_safe)
    return float(geom_mean / arith_mean)

def on_off_duty_cycle(mag_sq: np.ndarray, threshold_ratio: float = 0.1) -> float:
    """Estimate the time-domain duty cycle (fraction of time signal is 'on')."""
    peak = np.max(mag_sq)
    on_mask = mag_sq > (peak * threshold_ratio)
    return float(np.mean(on_mask))
