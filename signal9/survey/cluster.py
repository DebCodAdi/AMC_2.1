"""Burst clustering and bounding-box extraction from CFAR detections (Phase 3).

Groups positive detection pixels on the time-frequency grid into discrete
signal emissions and bursts.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class SignalBurst:
    burst_id: int
    f_center_hz: float
    bandwidth_hz: float
    f_lo_hz: float
    f_hi_hz: float
    t_start_s: float
    t_end_s: float
    duration_s: float
    peak_power_db: float
    snr_db_est: float
    pixel_count: int


def cluster_detections(mask: np.ndarray,
                       f: np.ndarray,
                       t: np.ndarray,
                       P_db: np.ndarray,
                       min_pixels: int = 4,
                       noise_floor_db: float | None = None) -> list[SignalBurst]:
    """Cluster 2D detection mask into discrete SignalBurst structures.

    Parameters
    ----------
    mask : 2D boolean array (f x t)
    f : 1D array of frequency bins in Hz
    t : 1D array of time points in seconds
    P_db : 2D float array of power in dB
    min_pixels : int, default 4
        Minimum pixel count to reject isolated noise false alarms.
    noise_floor_db : float, optional
        Estimated noise floor in dB for SNR calculation.

    Returns
    -------
    bursts : list of SignalBurst
    """
    if not np.any(mask):
        return []

    # 8-connectivity structuring element
    struct = np.ones((3, 3), dtype=int)
    labeled_array, num_features = ndimage.label(mask, structure=struct)

    if num_features == 0:
        return []

    if noise_floor_db is None:
        noise_floor_db = float(np.median(P_db[~mask])) if np.any(~mask) else float(np.median(P_db))

    bursts: list[SignalBurst] = []
    df = abs(f[1] - f[0]) if len(f) > 1 else 1.0
    dt = abs(t[1] - t[0]) if len(t) > 1 else 1.0

    for b_id in range(1, num_features + 1):
        idx_f, idx_t = np.where(labeled_array == b_id)
        if len(idx_f) < min_pixels:
            continue

        f_lo = float(f[np.min(idx_f)]) - df / 2.0
        f_hi = float(f[np.max(idx_f)]) + df / 2.0
        t_lo = float(t[np.min(idx_t)]) - dt / 2.0
        t_hi = float(t[np.max(idx_t)]) + dt / 2.0

        f_center = 0.5 * (f_lo + f_hi)
        bw = max(df, f_hi - f_lo)
        duration = max(dt, t_hi - t_lo)

        burst_powers = P_db[idx_f, idx_t]
        peak_pwr = float(np.max(burst_powers))
        snr_est = max(0.0, peak_pwr - noise_floor_db)

        bursts.append(SignalBurst(
            burst_id=b_id,
            f_center_hz=f_center,
            bandwidth_hz=bw,
            f_lo_hz=f_lo,
            f_hi_hz=f_hi,
            t_start_s=t_lo,
            t_end_s=t_hi,
            duration_s=duration,
            peak_power_db=peak_pwr,
            snr_db_est=snr_est,
            pixel_count=len(idx_f),
        ))

    # Sort bursts chronologically by t_start
    bursts.sort(key=lambda b: (b.t_start_s, b.f_lo_hz))
    return bursts
