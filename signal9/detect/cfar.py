"""2D Spectrogram CFAR detector with morphological clustering."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import ndimage

@dataclass
class DetectionBox:
    f_lo: float
    f_hi: float
    t_lo: float
    t_hi: float
    peak_db: float
    center_freq: float
    bandwidth: float
    duration: float

def detect_cfar(Pw_db: np.ndarray,
                f: np.ndarray,
                t: np.ndarray,
                thr_db: float = 6.0,
                min_bins: int = 2,
                min_frames: int = 3) -> list[dict]:
    """Threshold on whitened spectrogram + morphological connected components.
    Returns list of detection bounding box dicts.
    """
    mask = Pw_db > thr_db
    # Morphological cleaning along frequency axis to preserve time boundaries
    mask = ndimage.binary_opening(mask, structure=np.ones((2, 1)))
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 1)))
    
    lab, _ = ndimage.label(mask)
    boxes: list[dict] = []
    df = float(f[1] - f[0]) if len(f) > 1 else 1.0
    min_fr = min(min_frames, max(1, len(t) // 2))

    for sl in ndimage.find_objects(lab):
        fsl, tsl = sl
        if (fsl.stop - fsl.start) < min_bins or (tsl.stop - tsl.start) < min_fr:
            continue
        f_lo = float(f[fsl.start] - df / 2.0)
        f_hi = float(f[fsl.stop - 1] + df / 2.0)
        t_lo = float(t[tsl.start]) if tsl.start > 1 else 0.0
        t_hi = float(t[tsl.stop - 1]) if tsl.stop < len(t) - 1 else float(t[-1])
        peak_val = float(Pw_db[sl].max())
        
        boxes.append({
            "f_lo": f_lo,
            "f_hi": f_hi,
            "t_lo": t_lo,
            "t_hi": t_hi,
            "peak_db": peak_val,
            "center_freq": 0.5 * (f_lo + f_hi),
            "bandwidth": abs(f_hi - f_lo),
            "duration": max(0.0, t_hi - t_lo),
        })

    return boxes
