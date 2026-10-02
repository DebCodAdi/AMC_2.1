"""Welch-averaged multiscale detection and independent in-band SNR.

Thresholds are robust empirical standard deviations, not a claim of a
calibrated probability of false alarm. Noise-only sweeps measure that rate.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def spectral_floor(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average time in linear power; iteratively mask occupied frequencies."""
    mean = np.mean(power, axis=1)
    n = len(mean)
    valid = np.isfinite(mean) & (mean > 0)
    if not valid.any():
        return mean, np.ones(n)
    clean = valid.copy()
    floor = np.full(n, np.median(mean[clean]))
    for _ in range(4):
        ratio = mean / np.maximum(floor, 1e-30)
        center = np.median(ratio[clean])
        scatter = max(1.4826 * np.median(np.abs(ratio[clean] - center)), 1e-3)
        clean &= ratio < center + 3.5 * scatter
        if np.count_nonzero(clean) < max(8, n // 10):
            break
        interp = np.interp(np.arange(n), np.flatnonzero(clean), mean[clean])
        floor = ndimage.gaussian_filter1d(interp, max(8, n // 32))
    return mean, np.maximum(floor, 1e-30)


def detect_multiscale(P_db: np.ndarray, f: np.ndarray, t: np.ndarray,
                      sigma_threshold: float = 6.0) -> list[dict]:
    """Supplement burst CFAR with widths 2..64 on the averaged spectrum."""
    if len(f) < 16 or len(t) < 8:
        return []
    power = 10.0 ** (np.asarray(P_db, dtype=float) / 10.0)
    mean, floor = spectral_floor(power)
    ratio = mean / floor
    best = np.zeros(len(f))
    for width in (2, 4, 8, 16, 32, 64):
        if width > len(f) // 4:
            continue
        sm = ndimage.uniform_filter1d(ratio, width, mode="nearest")
        center = float(np.median(sm))
        scale = max(1.4826 * float(np.median(np.abs(sm - center))),
                    np.sqrt(2.0 / (len(t) * width)))
        score = (sm - center) / scale
        # Exclude smoothing windows crossing capture boundaries.
        score[:width // 2] = 0
        score[-width // 2:] = 0
        best = np.maximum(best, score)
    sm = ndimage.uniform_filter1d(ratio, 3, mode="nearest")
    edge_excess = min(10 ** 0.15 - 1, 2.5 * np.sqrt(2 / (3 * len(t))))
    support = (sm > 1 + edge_excess)
    support = ndimage.binary_closing(support, structure=np.ones(3))
    labels, _ = ndimage.label(support)
    df = float(f[1] - f[0])
    dt = float(np.median(np.diff(t)))
    boxes = []
    for (sl,) in ndimage.find_objects(labels):
        width = sl.stop - sl.start
        if width < 2 or np.max(best[sl]) < sigma_threshold:
            continue
        temporal = np.mean(power[sl] / floor[sl, None], axis=0)
        temporal = ndimage.uniform_filter1d(temporal, 5, mode="nearest")
        active = np.flatnonzero(temporal > 1 + 1.5 * np.sqrt(2 / (5 * width)))
        if len(active) < 3:
            continue
        lo, hi = float(f[sl.start] - df / 2), float(f[sl.stop - 1] + df / 2)
        tlo = max(0.0, float(t[active[0]] - dt))
        thi = float(t[active[-1]] + dt)
        boxes.append(dict(f_lo=lo, f_hi=hi, t_lo=tlo, t_hi=thi,
                          center_freq=(lo + hi) / 2, bandwidth=hi - lo,
                          duration=thi - tlo, peak_db=float(10 * np.log10(np.max(ratio[sl]))),
                          detection_sigma=float(np.max(best[sl])), source="multiscale"))
    return boxes


def inband_snr(P_db: np.ndarray, f: np.ndarray, t: np.ndarray,
               track: dict, tracks: list[dict]) -> float | None:
    """Signal excess / noise integrated over the same band; mask neighbours."""
    active = (t >= track["t_lo"]) & (t <= track["t_hi"])
    band = (f >= track["f_lo"]) & (f <= track["f_hi"])
    if not active.any() or not band.any():
        return None
    mean = np.mean(10.0 ** (P_db[:, active] / 10.0), axis=1)
    clean = np.ones(len(f), dtype=bool)
    df = float(f[1] - f[0])
    for other in tracks:
        clean &= ~((f >= other["f_lo"] - 2 * df) & (f <= other["f_hi"] + 2 * df))
    left = np.flatnonzero(clean & (f < track["f_lo"]))[-32:]
    right = np.flatnonzero(clean & (f > track["f_hi"]))[:32]
    if len(left) < 4 or len(right) < 4:
        return None
    # Welch-bin median bias is small after averaging, and is corrected here.
    k = max(1, int(np.count_nonzero(active)))
    correction = (1 - 1 / (9 * k)) ** 3
    noise = np.interp(f[band], [np.mean(f[left]), np.mean(f[right])],
                      [np.median(mean[left]) / correction, np.median(mean[right]) / correction])
    noise_power = float(noise.sum())
    excess = float(np.sum(mean[band]) - noise_power)
    if excess <= 0 or noise_power <= 0:
        return None
    return float(10 * np.log10(excess / noise_power))
