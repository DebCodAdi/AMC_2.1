"""Continuous-signal-resistant 2D noise-floor whitening."""
from __future__ import annotations
import numpy as np
from scipy import ndimage

def whiten_spectrogram(P_db: np.ndarray,
                       smooth: tuple[int, int] = (3, 8),
                       win_frac: float = 0.125) -> tuple[np.ndarray, np.ndarray]:
    """Whiten spectrogram noise floor to be invariant to receiver passband ripple
    and robust to continuous (always-on) signals.
    1. Smooth linear power
    2. Compute median across time per frequency bin
    3. Compute rolling 10th percentile lower envelope across frequency
    4. Subtract estimated floor
    Returns (whitened_P_db, estimated_floor_db).
    """
    # 1. Smooth linear power
    P_linear = 10.0 ** (P_db / 10.0)
    h, w = P_linear.shape
    sm_f = 15 if w < 4 else min(smooth[0], max(1, h))
    sm_t = min(smooth[1], max(1, w))
    P_smooth = ndimage.uniform_filter(P_linear, size=(sm_f, sm_t), mode="nearest")
    L = 10.0 * np.log10(P_smooth + 1e-20)

    # 2. Median over time per frequency bin
    M = np.median(L, axis=1)

    # 3. Robust noise floor estimation:
    # A rolling percentile window treats any signal wider than ~15% of the band as
    # noise floor, hollowing out the carrier center. We use masked noise-floor interpolation:
    q20 = float(np.percentile(M, 20))
    is_signal = M > (q20 + 4.0)
    noise_bins = np.copy(M)
    noise_bins[is_signal] = np.nan
    nans = np.isnan(noise_bins)

    if np.sum(~nans) >= 10:
        x_coords = np.arange(len(M))
        noise_bins[nans] = np.interp(x_coords[nans], x_coords[~nans], noise_bins[~nans])
        floor = ndimage.gaussian_filter1d(noise_bins, sigma=max(15, int(0.05 * len(M))))
    else:
        win = max(31, int(0.5 * M.size) | 1)
        floor = ndimage.percentile_filter(M, 10, size=win, mode="nearest")
        floor = ndimage.uniform_filter1d(floor, size=win, mode="nearest")

    # 4. Subtract floor
    Pw_db = L - floor[:, None]
    return Pw_db, floor
