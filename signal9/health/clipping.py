"""ADC clipping and rail-hit detection."""
from __future__ import annotations
import numpy as np

def clipping_fraction(raw_int: np.ndarray | None, fmt: str = "cs16") -> float:
    """Fraction of I or Q samples sitting on the ADC rails.
    MUST run on raw integers BEFORE any float normalization or DC offset removal.
    """
    if raw_int is None or len(raw_int) == 0:
        return 0.0
    if not np.issubdtype(raw_int.dtype, np.integer):
        # Float capture fallback: check if magnitude >= 0.999
        return float(np.mean(np.abs(raw_int) >= 0.999))

    info = np.iinfo(raw_int.dtype)
    rail_low = info.min + 1
    rail_high = info.max - 1
    hit_mask = (raw_int <= rail_low) | (raw_int >= rail_high)
    return float(np.mean(hit_mask))
