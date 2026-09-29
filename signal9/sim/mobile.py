"""Mobile multi-path fading channels and LEO satellite dynamic Doppler ramp.

Implements:
  - Rayleigh fading via sum-of-sinusoids (SOS) / Jakes' spectrum.
  - Rician fading with line-of-sight (LOS) component and K-factor.
  - LEO satellite Doppler trajectory: continuous frequency ramps (e.g. ±10 kHz at 350 Hz/s).
"""
from __future__ import annotations
import numpy as np


def rayleigh_channel(iq: np.ndarray,
                     fs: float = 48000.0,
                     fd_hz: float = 50.0,
                     n_sinusoids: int = 16,
                     seed: int | None = None) -> np.ndarray:
    """Apply Rayleigh flat fading using Jakes' sum-of-sinusoids model.

    Parameters
    ----------
    iq : np.ndarray
        Complex baseband input.
    fs : float
        Sampling frequency in Hz.
    fd_hz : float
        Maximum Doppler frequency shift in Hz.
    n_sinusoids : int, default 16
        Number of multipath arrival angles.
    seed : int, optional
        RNG seed.

    Returns
    -------
    y : np.ndarray
        Rayleigh faded output.
    """
    iq = np.asarray(iq, dtype=np.complex64).ravel()
    N = len(iq)
    if N == 0:
        return iq
    if fd_hz <= 0:
        return iq

    rng = np.random.default_rng(seed)
    t = np.arange(N, dtype=np.float64) / fs

    # Jakes' model angles of arrival
    n = np.arange(1, n_sinusoids + 1)
    alpha = (2.0 * np.pi * n - np.pi) / (4.0 * n_sinusoids)
    
    # Random initial phases
    phi = rng.uniform(-np.pi, np.pi, size=n_sinusoids)
    theta = rng.uniform(-np.pi, np.pi, size=n_sinusoids)

    # In-phase and quadrature fading processes
    # xc(t) = 2/sqrt(M) * sum(cos(beta_n) * cos(w_d * t * cos(alpha_n) + theta_n))
    # xs(t) = 2/sqrt(M) * sum(sin(beta_n) * cos(w_d * t * cos(alpha_n) + theta_n))
    omega_d = 2.0 * np.pi * fd_hz
    
    h_i = np.zeros(N, dtype=np.float64)
    h_q = np.zeros(N, dtype=np.float64)

    for i in range(n_sinusoids):
        freq_term = omega_d * np.cos(alpha[i]) * t + phi[i]
        h_i += np.cos(freq_term)
        h_q += np.sin(freq_term + theta[i])

    norm_factor = np.sqrt(2.0 / n_sinusoids)
    h = (h_i + 1j * h_q) * (norm_factor / np.sqrt(2.0))

    y = iq * h.astype(np.complex64)
    
    # Power preservation
    p_in = np.mean(np.abs(iq) ** 2) + 1e-12
    p_out = np.mean(np.abs(y) ** 2) + 1e-12
    y *= np.sqrt(p_in / p_out)

    return y.astype(np.complex64)


def rician_channel(iq: np.ndarray,
                   fs: float = 48000.0,
                   fd_hz: float = 50.0,
                   k_factor_db: float = 6.0,
                   seed: int | None = None) -> np.ndarray:
    """Apply Rician fading with line-of-sight (LOS) component and K-factor.

    Parameters
    ----------
    iq : np.ndarray
        Complex baseband input.
    fs : float
        Sampling frequency in Hz.
    fd_hz : float
        Maximum Doppler spread in Hz.
    k_factor_db : float
        Ratio of specular power to diffuse power in dB.
    seed : int, optional
        RNG seed.

    Returns
    -------
    y : np.ndarray
        Rician faded output.
    """
    iq = np.asarray(iq, dtype=np.complex64).ravel()
    N = len(iq)
    if N == 0:
        return iq

    k_linear = 10.0 ** (k_factor_db / 10.0)
    specular_weight = np.sqrt(k_linear / (k_linear + 1.0))
    diffuse_weight = np.sqrt(1.0 / (k_linear + 1.0))

    # Diffuse Rayleigh component
    ray_iq = rayleigh_channel(iq, fs=fs, fd_hz=fd_hz, seed=seed)
    
    # Specular LOS component
    y = specular_weight * iq + diffuse_weight * ray_iq

    # Energy conservation
    p_in = np.mean(np.abs(iq) ** 2) + 1e-12
    p_out = np.mean(np.abs(y) ** 2) + 1e-12
    y *= np.sqrt(p_in / p_out)

    return y.astype(np.complex64)


def apply_doppler_ramp(iq: np.ndarray,
                       fs: float = 48000.0,
                       f_start_hz: float = -10000.0,
                       ramp_rate_hz_per_s: float = 350.0) -> np.ndarray:
    """Apply continuous Doppler ramp simulating a LEO satellite overpass.

    The instantaneous frequency offset is f(t) = f_start + ramp_rate * t.
    The phase is phi(t) = 2*pi * (f_start * t + 0.5 * ramp_rate * t^2).

    Parameters
    ----------
    iq : np.ndarray
        Complex baseband input.
    fs : float
        Sample rate in Hz.
    f_start_hz : float, default -10000.0 Hz (-10 kHz)
        Initial Doppler shift at t=0.
    ramp_rate_hz_per_s : float, default 350.0 Hz/s
        Doppler chirp rate (linear frequency acceleration).

    Returns
    -------
    y : np.ndarray
        Doppler-ramped signal.
    """
    iq = np.asarray(iq, dtype=np.complex64).ravel()
    N = len(iq)
    if N == 0:
        return iq

    t = np.arange(N, dtype=np.float64) / fs
    phase = 2.0 * np.pi * (f_start_hz * t + 0.5 * ramp_rate_hz_per_s * (t ** 2))
    carrier = np.exp(1j * phase).astype(np.complex64)
    return (iq * carrier).astype(np.complex64)
