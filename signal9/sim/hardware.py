"""Realistic SDR receiver front-end impairments (Phase 1).

Models real-world SDR receiver non-idealities:
  - Frequency-dependent and frequency-independent IQ imbalance
  - Local oscillator (LO) leakage / DC spike
  - ADC quantization (8-bit uint8 with 127.5 offset, 16-bit int16, clipping)
  - AGC step changes / gain ruptures
  - Local oscillator phase noise (Lorentzian 1/f^2)
  - Discrete CW spurs / mixer intermodulation products
  - DEF-054 sample dropouts (dropped USB/network FIFO buffers)
"""
from __future__ import annotations
import numpy as np


def apply_iq_imbalance(iq: np.ndarray,
                       gain_imb_db: float = 1.0,
                       phase_imb_deg: float = 3.0) -> np.ndarray:
    """Apply frequency-independent IQ imbalance.

    I_out = I_in * cos(phi/2) * 10^(g/40) - Q_in * sin(phi/2) * 10^(g/40)
    Q_out = -I_in * sin(phi/2) * 10^(-g/40) + Q_in * cos(phi/2) * 10^(-g/40)
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) == 0 or (gain_imb_db == 0.0 and phase_imb_deg == 0.0):
        return iq

    phi = np.deg2rad(phase_imb_deg)
    g_i = 10.0 ** (gain_imb_db / 40.0)
    g_q = 10.0 ** (-gain_imb_db / 40.0)

    i_in = iq.real
    q_in = iq.imag

    i_out = g_i * (i_in * np.cos(phi / 2.0) - q_in * np.sin(phi / 2.0))
    q_out = g_q * (-i_in * np.sin(phi / 2.0) + q_in * np.cos(phi / 2.0))

    return (i_out + 1j * q_out).astype(np.complex64)


def apply_lo_leakage(iq: np.ndarray,
                     dc_level_db: float = -20.0) -> np.ndarray:
    """Inject direct-conversion receiver LO leakage (DC spike).

    Parameters
    ----------
    iq : np.ndarray
    dc_level_db : float
        DC spike power relative to signal power in dB.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) == 0:
        return iq

    p_sig = np.mean(np.abs(iq) ** 2)
    dc_amplitude = np.sqrt(p_sig * (10.0 ** (dc_level_db / 10.0)))
    dc_offset = dc_amplitude * (1.0 + 1j) / np.sqrt(2.0)
    return (iq + dc_offset).astype(np.complex64)


def apply_phase_noise(iq: np.ndarray,
                      phase_noise_std_deg: float = 1.0,
                      seed: int | None = None) -> np.ndarray:
    """Apply random walk oscillator phase noise (Lorentzian linewidth)."""
    iq = np.asarray(iq, dtype=np.complex64)
    N = len(iq)
    if N == 0 or phase_noise_std_deg <= 0.0:
        return iq

    rng = np.random.default_rng(seed)
    sigma_rad = np.deg2rad(phase_noise_std_deg)
    # Random walk phase: cumulative sum of white phase increments
    dphi = rng.standard_normal(N) * sigma_rad
    phi = np.cumsum(dphi)
    return (iq * np.exp(1j * phi)).astype(np.complex64)


def apply_spurs(iq: np.ndarray,
                fs: float = 48000.0,
                spur_offsets_hz: list[float] | None = None,
                spur_level_db: float = -25.0) -> np.ndarray:
    """Inject discrete CW spurs from mixer harmonics and clock bleed."""
    iq = np.asarray(iq, dtype=np.complex64)
    N = len(iq)
    if N == 0 or not spur_offsets_hz:
        return iq

    p_sig = np.mean(np.abs(iq) ** 2)
    spur_amp = np.sqrt(p_sig * (10.0 ** (spur_level_db / 10.0)))

    t = np.arange(N, dtype=np.float64) / fs
    spur_signal = np.zeros(N, dtype=np.complex64)
    for f_spur in spur_offsets_hz:
        spur_signal += spur_amp * np.exp(1j * 2.0 * np.pi * f_spur * t).astype(np.complex64)

    return (iq + spur_signal).astype(np.complex64)


def apply_agc_steps(iq: np.ndarray,
                    step_indices: list[int] | None = None,
                    step_dbs: list[float] | None = None) -> np.ndarray:
    """Inject sudden receiver gain jumps (AGC state changes)."""
    iq = np.asarray(iq, dtype=np.complex64).copy()
    N = len(iq)
    if N == 0 or not step_indices:
        return iq

    dbs = step_dbs or [6.0] * len(step_indices)
    gain_envelope = np.ones(N, dtype=np.float32)

    for idx, db in zip(step_indices, dbs):
        if 0 <= idx < N:
            gain_factor = 10.0 ** (db / 20.0)
            gain_envelope[idx:] *= gain_factor

    return (iq * gain_envelope).astype(np.complex64)


def apply_sample_dropouts(iq: np.ndarray,
                          dropout_len: int = 128,
                          period: int = 4096) -> np.ndarray:
    """Simulate dropped USB/network FIFO buffers (DEF-054)."""
    iq = np.asarray(iq, dtype=np.complex64).copy()
    N = len(iq)
    for start in range(period, N, period):
        end = min(N, start + dropout_len)
        iq[start:end] = 0.0 + 0j
    return iq


def quantize_adc(iq: np.ndarray,
                 bits: int = 8,
                 mode: str = "uint8_rtl",
                 clip_headroom_db: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Simulate hardware ADC quantization and return both raw integer bytes and reconstructed float IQ.

    Modes:
      - 'uint8_rtl': RTL-SDR format (8-bit unsigned uint8 with 127.5 offset, range [0, 255])
      - 'int8': Signed 8-bit integers (range [-128, 127])
      - 'int16': Signed 16-bit integers (range [-32768, 32767])
    """
    iq = np.asarray(iq, dtype=np.complex64)
    N = len(iq)
    if N == 0:
        return np.zeros(0, dtype=np.uint8), iq

    # Peak amplitude based on headroom
    rms = np.sqrt(np.mean(np.abs(iq) ** 2)) + 1e-12
    v_peak = rms * (10.0 ** (clip_headroom_db / 20.0))

    # Scale to [-1, 1] full scale
    norm_i = np.clip(iq.real / v_peak, -1.0, 1.0)
    norm_q = np.clip(iq.imag / v_peak, -1.0, 1.0)

    if mode == "uint8_rtl":
        # Scale to [0, 255] with 127.5 center
        raw_i = np.round(norm_i * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        raw_q = np.round(norm_q * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        # Interleave I and Q
        raw_bytes = np.empty(2 * N, dtype=np.uint8)
        raw_bytes[0::2] = raw_i
        raw_bytes[1::2] = raw_q
        # Reconstructed complex float
        recon_i = (raw_i.astype(np.float32) - 127.5) / 127.5 * v_peak
        recon_q = (raw_q.astype(np.float32) - 127.5) / 127.5 * v_peak
        return raw_bytes, (recon_i + 1j * recon_q).astype(np.complex64)

    elif mode == "int16":
        max_val = 32767.0
        raw_i = np.round(norm_i * max_val).clip(-32768, 32767).astype(np.int16)
        raw_q = np.round(norm_q * max_val).clip(-32768, 32767).astype(np.int16)
        raw_bytes = np.empty(2 * N, dtype=np.int16)
        raw_bytes[0::2] = raw_i
        raw_bytes[1::2] = raw_q
        recon_i = raw_i.astype(np.float32) / max_val * v_peak
        recon_q = raw_q.astype(np.float32) / max_val * v_peak
        return raw_bytes, (recon_i + 1j * recon_q).astype(np.complex64)

    else:
        # Standard signed int8
        max_val = 127.0
        raw_i = np.round(norm_i * max_val).clip(-128, 127).astype(np.int8)
        raw_q = np.round(norm_q * max_val).clip(-128, 127).astype(np.int8)
        raw_bytes = np.empty(2 * N, dtype=np.int8)
        raw_bytes[0::2] = raw_i
        raw_bytes[1::2] = raw_q
        recon_i = raw_i.astype(np.float32) / max_val * v_peak
        recon_q = raw_q.astype(np.float32) / max_val * v_peak
        return raw_bytes, (recon_i + 1j * recon_q).astype(np.complex64)


def apply_hardware_impairments(iq: np.ndarray,
                               fs: float = 48000.0,
                               gain_imb_db: float = 0.5,
                               phase_imb_deg: float = 2.0,
                               dc_level_db: float = -25.0,
                               phase_noise_deg: float = 0.5,
                               spur_offsets_hz: list[float] | None = None,
                               agc_steps: list[int] | None = None,
                               adc_mode: str | None = None,
                               seed: int | None = None) -> np.ndarray:
    """Composite hardware front-end simulation applying all impairments."""
    y = apply_iq_imbalance(iq, gain_imb_db, phase_imb_deg)
    y = apply_lo_leakage(y, dc_level_db)
    y = apply_phase_noise(y, phase_noise_deg, seed=seed)
    if spur_offsets_hz:
        y = apply_spurs(y, fs, spur_offsets_hz)
    if agc_steps:
        y = apply_agc_steps(y, agc_steps)
    if adc_mode:
        _, y = quantize_adc(y, mode=adc_mode)
    return y.astype(np.complex64)
