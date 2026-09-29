"""Occupied bandwidth, in-band SNR, and instantaneous frequency features.
Implements the exact noise-floor-referenced connected-component bandwidth estimator (Fixes A3, B5).
"""
from __future__ import annotations
import numpy as np
from scipy import signal as sps
from scipy import ndimage

def psd_db(iq: np.ndarray, fs: float, nfft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density in dB."""
    f, p = sps.welch(iq, fs=fs, nperseg=min(nfft, len(iq)), return_onesided=False,
                     scaling="density", detrend=False)
    order = np.argsort(f)
    return f[order], 10.0 * np.log10(p[order] + 1e-30)

def noise_floor_db(p_db: np.ndarray, q: float = 20.0) -> float:
    """Robust noise floor estimate: lower-percentile of PSD + 1 dB."""
    return float(np.percentile(p_db, q)) + 1.0

def occupied_bandwidth(iq: np.ndarray, fs: float, thresh_db: float = 4.0,
                       power_frac: float = 0.99, nfft: int = 4096) -> tuple[float, float, float, float]:
    """Noise-floor-referenced occupied bandwidth of the strongest emission.
    Returns (bw_hz, f_lo_hz, f_hi_hz, inband_snr_db).
    Unlike 'peak - 20 dB' edges, this does not collapse to the full sample rate when SNR < 20 dB.
    """
    if len(iq) < 32:
        return 0.0, 0.0, 0.0, -np.inf

    f, p = psd_db(iq, fs, nfft)
    nf = noise_floor_db(p)
    mask = p > nf + thresh_db

    # Close small gaps (FSK tones, OFDM pilots, SSB voice formants)
    k = max(3, int(round(0.01 * len(p))))
    mask = ndimage.binary_closing(mask, structure=np.ones(k))
    if not mask.any():
        return 0.0, 0.0, 0.0, -np.inf

    pk = int(np.argmax(np.where(mask, p, -np.inf)))
    lo = pk
    while lo > 0 and mask[lo - 1]:
        lo -= 1
    hi = pk
    while hi < len(mask) - 1 and mask[hi + 1]:
        hi += 1

    lin = np.maximum(10.0 ** (p[lo:hi + 1] / 10.0) - 10.0 ** (nf / 10.0), 0.0)
    total_lin = lin.sum()
    if total_lin <= 0:
        return 0.0, 0.0, 0.0, -np.inf

    c = np.cumsum(lin) / (total_lin + 1e-30)
    a = int(np.searchsorted(c, (1.0 - power_frac) / 2.0))
    b = int(np.searchsorted(c, 1.0 - (1.0 - power_frac) / 2.0))
    df = float(f[1] - f[0]) if len(f) > 1 else 1.0

    f_lo = float(f[lo + a])
    f_hi = float(f[lo + min(b, len(lin) - 1)])
    bw = float(f_hi - f_lo + df)

    sig_pow = total_lin * df
    noise_pow = (10.0 ** (nf / 10.0)) * bw
    inband_snr = float(10.0 * np.log10(sig_pow / (noise_pow + 1e-30)))

    return bw, f_lo, f_hi, inband_snr

def estimate_snr(x: np.ndarray, fs: float, bw_hz: float) -> float:
    """Convenience wrapper for estimating in-band SNR."""
    _, _, _, snr = occupied_bandwidth(x, fs)
    return snr

def instantaneous_freq_features(x: np.ndarray, fs: float) -> dict[str, float]:
    """Instantaneous amplitude and frequency distribution features."""
    if len(x) < 8:
        return {"papr": 0.0, "env_var": 0.0, "freq_var": 0.0}

    amp = np.abs(x)
    mean_p = np.mean(amp ** 2) + 1e-20
    papr = float(np.max(amp ** 2) / mean_p)
    norm_amp = amp / np.sqrt(mean_p)
    env_var = float(np.var(norm_amp))

    # Instantaneous frequency via phase difference
    phase = np.unwrap(np.angle(x))
    dphi = np.diff(phase) * (fs / (2.0 * np.pi))
    freq_var = float(np.var(dphi))

    return {
        "papr": papr,
        "env_var": env_var,
        "freq_var": freq_var,
    }

def psk_order_cues(iq: np.ndarray, fs: float) -> dict[str, float]:
    """2nd, 4th, 8th power spectral line strength (M-PSK cues)."""
    res = {}
    for p in (2, 4, 8):
        x_p = iq ** p
        f, psd = psd_db(x_p, fs, nfft=1024)
        peak = np.max(psd)
        med = np.median(psd)
        res[f"psk_{p}_cues"] = float(peak - med)
    return res

def usb_lsb_asymmetry(iq: np.ndarray, fs: float) -> float:
    """Ratio of USB energy to LSB energy."""
    f, p = psd_db(iq, fs, nfft=1024)
    usb = np.sum(10**(p[f > 0.01*fs]/10))
    lsb = np.sum(10**(p[f < -0.01*fs]/10))
    return float(np.log10(usb / (lsb + 1e-12) + 1e-12))

def ofdm_cp_lags(iq: np.ndarray) -> float:
    """Max autocorrelation outside the main lobe (OFDM CP)."""
    if len(iq) < 256:
        return 0.0
    corr = np.abs(np.correlate(iq, iq, mode='full'))
    corr = corr[len(corr)//2:] / (corr[len(corr)//2] + 1e-12)
    # Ignore main lobe
    main_lobe = 10
    if len(corr) > main_lobe:
        return float(np.max(corr[main_lobe:]))
    return 0.0
