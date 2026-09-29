"""AGC step change-point detection."""
from __future__ import annotations
import numpy as np

def detect_agc_ruptures(x: np.ndarray, block_size: int = 1024, min_jump_db: float = 3.0) -> list[int]:
    """Detect sudden AGC gain steps using power block comparisons (CUSUM-like).
    Returns a list of indices where a gain step is suspected.
    """
    if len(x) < 2 * block_size:
        return []
        
    # Compute block-wise power
    n_blocks = len(x) // block_size
    x_view = x[:n_blocks * block_size].reshape(n_blocks, block_size)
    p = np.mean(np.abs(x_view)**2, axis=1)
    
    # Avoid log of zero
    p_db = 10.0 * np.log10(p + 1e-12)
    
    ruptures = []
    # Simple adjacent block jump detection (naive CUSUM alternative for speed)
    for i in range(1, n_blocks):
        if abs(p_db[i] - p_db[i-1]) > min_jump_db:
            ruptures.append(i * block_size)
            
    return ruptures
