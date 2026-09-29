"""2D Ordered-Statistic Constant False Alarm Rate (OS-CFAR) detector (Phase 3).

Detects signals and bursts in non-homogeneous noise, interference, and dense
spectral environments without target masking.
"""
from __future__ import annotations
import numpy as np


def os_cfar_2d(P_db: np.ndarray,
               guard_cells: tuple[int, int] = (2, 2),
               ref_cells: tuple[int, int] = (6, 6),
               rank_pct: float = 0.75,
               threshold_db: float = 8.0) -> np.ndarray:
    """Execute 2D OS-CFAR detection over a power spectrogram.

    Parameters
    ----------
    P_db : 2D float array (frequencies x time)
    guard_cells : (guard_f, guard_t)
        Half-width of guard area around Cell Under Test (CUT).
    ref_cells : (ref_f, ref_t)
        Half-width of reference area around CUT.
    rank_pct : float, default 0.75
        Percentile of ordered reference cells used as noise baseline (e.g. 75th percentile).
    threshold_db : float, default 8.0
        Threshold offset in dB above the k-th order statistic.

    Returns
    -------
    mask : 2D boolean array (same shape as P_db)
        True where signal power exceeds OS-CFAR adaptive threshold.
    """
    P_db = np.asarray(P_db, dtype=np.float32)
    n_f, n_t = P_db.shape
    if n_f < 3 or n_t < 3:
        return np.zeros_like(P_db, dtype=bool)

    gf, gt = guard_cells
    rf, rt = ref_cells

    # Bound reference cells
    rf = min(rf, (n_f - 1) // 2)
    rt = min(rt, (n_t - 1) // 2)
    gf = min(gf, rf - 1)
    gt = min(gt, rt - 1)

    # Relative coordinates of reference window excluding guard window
    ref_coords = []
    for df in range(-rf, rf + 1):
        for dt in range(-rt, rt + 1):
            if abs(df) <= gf and abs(dt) <= gt:
                continue
            ref_coords.append((df, dt))

    n_ref = len(ref_coords)
    if n_ref == 0:
        return P_db > (np.median(P_db) + threshold_db)

    k_idx = int(np.clip(rank_pct * n_ref, 0, n_ref - 1))

    # Fast sliding extraction using strided or local neighborhood
    # For large spectrograms, sample reference cells
    detection_mask = np.zeros((n_f, n_t), dtype=bool)

    # Pad spectrogram with reflection or edge values
    padded = np.pad(P_db, ((rf, rf), (rt, rt)), mode="reflect")

    # Stack reference windows
    ref_slices = np.stack([
        padded[rf + df : rf + df + n_f, rt + dt : rt + dt + n_t]
        for df, dt in ref_coords
    ], axis=0)  # Shape: (n_ref, n_f, n_t)

    # Compute k-th order statistic across axis 0
    # partition is much faster than full sort
    kth_stat = np.partition(ref_slices, k_idx, axis=0)[k_idx]

    # CUT threshold
    detection_mask = P_db > (kth_stat + threshold_db)

    return detection_mask
