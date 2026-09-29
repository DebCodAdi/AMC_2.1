"""Augmentation and physical channel impairment stack for Signal 9."""
from __future__ import annotations
import numpy as np

from .noise import add_awgn_snr, add_impulsive_noise, inject_noise_bank
from .watterson import watterson_channel
from .fading import rician_channel
from .hardware import rapp_pa, adc_quantize, agc_gain_step, add_cfo_and_drift, add_phase_noise
from .interference import inject_donor_interferer, add_spur
from .boxjitter import jitter_box

def random_chain(x: np.ndarray,
                 fs: float,
                 sig_bw_hz: float,
                 band: str = "HF",
                 rng: np.random.Generator | None = None,
                 noise_bank: list[np.ndarray] | None = None,
                 donor: np.ndarray | None = None) -> np.ndarray:
    """Apply complete physical impairment chain in physical order:
    PA -> Channel (Watterson for HF, Rician for VHF/UHF) -> CFO -> Interference -> Noise -> Quantize/Clip.
    """
    rng = rng or np.random.default_rng()
    out = x.copy()

    # 0. Physical Invariances (Random phase rotation & spectral mirroring)
    phase_rot = np.exp(1j * rng.uniform(0, 2 * np.pi))
    out = out * phase_rot
    if rng.random() < 0.5:
        out = np.conj(out) # Spectral mirror for sideband ambiguity

    # 1. Transmitter PA Non-linearity
    if rng.random() < 0.4:
        out = rapp_pa(out, backoff_db=rng.uniform(1.0, 8.0), p=rng.uniform(1.5, 3.0))

    # 2. Channel Propagation
    if band.upper() == "HF":
        preset = rng.choice(["good", "moderate", "poor", "flutter"])
        out = watterson_channel(out, fs=fs, preset=preset, rng=rng)
    else:
        k_db = rng.uniform(-10.0, 15.0)
        doppler = rng.uniform(0.5, 20.0)
        out = rician_channel(out, fs=fs, k_factor_db=k_db, max_doppler_hz=doppler, rng=rng)

    # 3. Frequency & Phase Offsets
    cfo = rng.uniform(-0.04, 0.04) * fs
    out = add_cfo_and_drift(out, fs, cfo_hz=cfo)
    if rng.random() < 0.3:
        out = add_phase_noise(out, standard_dev=rng.uniform(0.005, 0.03))

    # 4. Donor Interference & Spurs
    if donor is not None and rng.random() < 0.35:
        sir = rng.uniform(5.0, 25.0)
        out = inject_donor_interferer(out, donor, sir_db=sir, fs=fs)

    # 5. Noise Injection
    snr = rng.uniform(-6.0, 30.0)
    if noise_bank and rng.random() < 0.3:
        out = inject_noise_bank(out, noise_bank, target_snr_db=snr, rng=rng)
    else:
        out = add_awgn_snr(out, target_snr_db=snr, sig_bw_hz=sig_bw_hz, fs=fs, rng=rng)

    # 6. Impulsive Spikes (HF)
    if band.upper() == "HF" and rng.random() < 0.2:
        out = add_impulsive_noise(out, p_impulse=0.002, rng=rng)

    # 7. Receiver Hardware / AGC / Quantization
    if rng.random() < 0.25:
        out = agc_gain_step(out, step_time_frac=rng.uniform(0.2, 0.8), gain_multiplier=rng.uniform(0.5, 2.0))

    bits = int(rng.choice([4, 6, 8, 10, 12, 16]))
    out = adc_quantize(out, effective_bits=bits)

    # Ensure unit-RMS normalization
    rms = np.sqrt(np.mean(np.abs(out) ** 2) + 1e-12)
    return (out / rms).astype(np.complex64)

__all__ = [
    "add_awgn_snr",
    "add_impulsive_noise",
    "inject_noise_bank",
    "watterson_channel",
    "rician_channel",
    "rapp_pa",
    "adc_quantize",
    "agc_gain_step",
    "add_cfo_and_drift",
    "add_phase_noise",
    "inject_donor_interferer",
    "add_spur",
    "jitter_box",
    "random_chain",
]
