"""DC offset, LO leakage detection, and carrier-safe masking."""
from __future__ import annotations
import numpy as np
from scipy import signal as sps

def detect_dc_spike(x: np.ndarray, fs: float, nfft: int = 1024, threshold_db: float = 12.0) -> bool:
    """Detect if 0 Hz bin carries a local oscillator leakage spike.
    Rather than blindly notch-filtering (which deletes real AM and CW carriers),
    we detect the DC spike and mark/mask it.
    IMPORTANT: Must set detrend=False in Welch PSD, otherwise scipy subtracts the DC mean!
    """
    if len(x) < nfft:
        return False
    # Use Welch PSD with detrend=False so DC component is preserved
    f, Pxx = sps.welch(x, fs=fs, nperseg=nfft, return_onesided=False, detrend=False)
    Pxx = np.fft.fftshift(Pxx)
    f = np.fft.fftshift(f)
    center_idx = len(f) // 2
    
    # Compare center bin to adjacent bins (excluding DC bin ±1)
    adj_left = max(0, center_idx - 10)
    adj_right = min(len(f), center_idx + 11)
    adj_bins = np.concatenate([Pxx[adj_left:center_idx - 1], Pxx[center_idx + 2:adj_right]])
    if len(adj_bins) == 0:
        return False
    
    local_median = np.median(adj_bins) + 1e-20
    center_power = Pxx[center_idx]
    ratio_db = 10.0 * np.log10(center_power / local_median)
    return bool(ratio_db > threshold_db)

def mask_dc_spike(P_db: np.ndarray, dc_bins: int = 3) -> np.ndarray:
    """Mask center bins by interpolating neighboring values in the spectrogram."""
    out = P_db.copy()
    center = out.shape[0] // 2
    lo = max(0, center - dc_bins)
    hi = min(out.shape[0], center + dc_bins + 1)
    for t in range(out.shape[1]):
        val_lo = out[lo - 1, t] if lo > 0 else out[hi, t]
        val_hi = out[hi, t] if hi < out.shape[0] else out[lo - 1, t]
        out[lo:hi, t] = np.linspace(val_lo, val_hi, hi - lo)
    return out
