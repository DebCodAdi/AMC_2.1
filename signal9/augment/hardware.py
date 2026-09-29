"""RF front-end and converter hardware impairments."""
from __future__ import annotations
import numpy as np

def rapp_pa(x: np.ndarray, backoff_db: float = 3.0, p: float = 2.0) -> np.ndarray:
    """Rapp solid-state power amplifier model (AM/AM conversion).
    Fixes B7: Saturation amplitude is set RELATIVE to signal RMS power (back-off),
    so that compression actually occurs on unit-RMS signals.
    """
    rms = np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)
    # A_sat based on input back-off from RMS
    a_sat = rms * (10.0 ** (backoff_db / 20.0))
    r = np.abs(x)
    gain = 1.0 / (1.0 + (r / a_sat) ** (2.0 * p)) ** (1.0 / (2.0 * p))
    return (x * gain).astype(x.dtype)

def adc_quantize(x: np.ndarray, effective_bits: int = 8, clip_ratio: float = 1.0) -> np.ndarray:
    """Per-rail ADC quantizer with clipping and effective bit depths (3 to 16 bits).
    Simulates SDR ADC resolution limits and weak signals occupying few bits next to strong interferers.
    """
    effective_bits = max(3, min(16, effective_bits))
    levels = 2 ** (effective_bits - 1)
    
    # Scale based on clip_ratio
    max_val = max(np.max(np.abs(x.real)), np.max(np.abs(x.imag)), 1e-6) * clip_ratio
    scale = levels / max_val
    
    # Quantize I and Q rails independently
    i_q = np.clip(np.round(x.real * scale), -levels, levels - 1) / scale
    q_q = np.clip(np.round(x.imag * scale), -levels, levels - 1) / scale
    return (i_q + 1j * q_q).astype(x.dtype)

def agc_gain_step(x: np.ndarray, step_time_frac: float = 0.5, gain_multiplier: float = 2.0) -> np.ndarray:
    """Simulate sudden receiver AGC gain step."""
    out = x.copy()
    step_idx = int(len(x) * step_time_frac)
    out[step_idx:] = out[step_idx:] * gain_multiplier
    return out

def add_cfo_and_drift(x: np.ndarray, fs: float, cfo_hz: float, drift_hz_per_sec: float = 0.0) -> np.ndarray:
    """Simulate carrier frequency offset and thermal drift."""
    n = np.arange(len(x))
    t = n / fs
    phase = 2.0 * np.pi * (cfo_hz * t + 0.5 * drift_hz_per_sec * (t ** 2))
    return (x * np.exp(1j * phase)).astype(x.dtype)

def add_phase_noise(x: np.ndarray, standard_dev: float = 0.02) -> np.ndarray:
    """Simulate oscillator Wiener process phase noise."""
    rng = np.random.default_rng()
    dphi = rng.standard_normal(len(x)) * standard_dev
    phi = np.cumsum(dphi)
    return (x * np.exp(1j * phi)).astype(x.dtype)
