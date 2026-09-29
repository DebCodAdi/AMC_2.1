"""Fixed-length crop slicing and multi-crop aggregation."""
from __future__ import annotations
import numpy as np

def make_crops(x: np.ndarray,
               crop_len: int = 4096,
               overlap: float = 0.5,
               max_crops: int = 8) -> list[np.ndarray]:
    """Slice a channelized signal into fixed-length crops (default 4096 samples).
    Applies strict unit-RMS normalization per-crop.
    If shorter than crop_len, zero-pads symmetrically.
    """
    if len(x) == 0:
        return [np.zeros(crop_len, dtype=np.complex64)]

    def _normalize(c: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(np.abs(c)**2))
        return (c / (rms + 1e-12)).astype(np.complex64)

    if len(x) < crop_len:
        pad_total = crop_len - len(x)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        padded = np.pad(x, (pad_left, pad_right), mode="constant")
        return [_normalize(padded)]

    step = max(1, int(crop_len * (1.0 - overlap)))
    crops: list[np.ndarray] = []
    for start in range(0, len(x) - crop_len + 1, step):
        crops.append(_normalize(x[start:start + crop_len]))
        if len(crops) >= max_crops:
            break

    # If step didn't reach the very end, include the final tail window
    if len(crops) < max_crops and (len(x) - crop_len) % step != 0:
        crops.append(_normalize(x[-crop_len:]))

    return crops

def aggregate_crops(logits_list: list[np.ndarray], snrs: list[float] | None = None) -> np.ndarray:
    """Combine logits across multiple crops of the same track.
    Uses SNR-weighted average if SNRs provided, otherwise simple mean.
    """
    if not logits_list:
        raise ValueError("Empty logits list")
    stacked = np.stack(logits_list, axis=0)  # (N, C)
    if snrs is None:
        return np.mean(stacked, axis=0)
    
    # Weight by linear SNR
    w = np.clip(10.0 ** (np.array(snrs) / 10.0), 0.1, 100.0)
    w = w / np.sum(w)
    return np.sum(stacked * w[:, None], axis=0)
