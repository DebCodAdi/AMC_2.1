"""Cyclic features for symbol rate and framing detection."""
from __future__ import annotations
import numpy as np
from scipy.signal import welch

def cyclic_features(iq: np.ndarray, fs: float) -> dict[str, float]:
    """Compute cyclic spectrum features on magnitude squared.
    Captures bauded timing lines and symbol rates.
    """
    mag_sq = np.abs(iq)**2
    # Remove DC
    mag_sq -= np.mean(mag_sq)
    
    # Welch PSD of the magnitude squared
    f, p = welch(mag_sq, fs=fs, nperseg=min(1024, len(mag_sq)), return_onesided=True)
    
    # Ignore DC bin and very low frequencies
    valid_idx = f > 0.01 * fs
    if not np.any(valid_idx):
        return {"cyclic_peak_hz": 0.0, "cyclic_peak_db": -100.0, "cyclic_snr": 0.0}
        
    p_valid = p[valid_idx]
    f_valid = f[valid_idx]
    
    peak_idx = np.argmax(p_valid)
    peak_p = p_valid[peak_idx]
    peak_f = f_valid[peak_idx]
    
    median_p = np.median(p_valid)
    
    return {
        "cyclic_peak_hz": float(peak_f),
        "cyclic_peak_db": float(10 * np.log10(peak_p + 1e-12)),
        "cyclic_snr": float(10 * np.log10((peak_p + 1e-12) / (median_p + 1e-12)))
    }
