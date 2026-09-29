"""Noise injection models (in-band AWGN, impulsive noise, and real noise bank)."""
from __future__ import annotations
import numpy as np

def add_awgn_snr(x: np.ndarray, target_snr_db: float, sig_bw_hz: float, fs: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add complex AWGN such that the in-band SNR across sig_bw_hz matches target_snr_db."""
    rng = rng or np.random.default_rng()
    sig_power = np.mean(np.abs(x) ** 2)
    # Target in-band noise power
    snr_lin = 10.0 ** (target_snr_db / 10.0)
    in_band_noise_power = sig_power / (snr_lin + 1e-12)
    # Total noise across full sampling band
    total_noise_power = in_band_noise_power * (fs / max(sig_bw_hz, 1.0))
    
    noise_sigma = np.sqrt(total_noise_power / 2.0)
    noise = (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))) * noise_sigma
    return (x + noise).astype(x.dtype)

def add_impulsive_noise(x: np.ndarray, p_impulse: float = 0.005, impulse_ratio_db: float = 20.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add Bernoulli-Gaussian impulsive noise (simulates lightning/powerline spikes on HF)."""
    rng = rng or np.random.default_rng()
    sig_rms = np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)
    impulse_amp = sig_rms * (10.0 ** (impulse_ratio_db / 20.0))
    
    mask = rng.random(len(x)) < p_impulse
    n_hits = np.sum(mask)
    if n_hits == 0:
        return x

    impulses = (rng.standard_normal(n_hits) + 1j * rng.standard_normal(n_hits)) * (impulse_amp / np.sqrt(2.0))
    out = x.copy()
    out[mask] += impulses
    return out

def inject_noise_bank(x: np.ndarray, noise_bank: list[np.ndarray], target_snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Inject a real recorded noise segment from the noise bank."""
    if not noise_bank:
        return x
    rng = rng or np.random.default_rng()
    donor = noise_bank[rng.integers(0, len(noise_bank))]
    if len(donor) < len(x):
        donor = np.tile(donor, int(np.ceil(len(x) / len(donor))))
    start = rng.integers(0, len(donor) - len(x) + 1)
    noise_crop = donor[start:start + len(x)]
    
    sig_power = np.mean(np.abs(x) ** 2)
    noise_power = np.mean(np.abs(noise_crop) ** 2) + 1e-12
    desired_noise_power = sig_power / (10.0 ** (target_snr_db / 10.0))
    scaled_noise = noise_crop * np.sqrt(desired_noise_power / noise_power)
    return (x + scaled_noise).astype(x.dtype)
