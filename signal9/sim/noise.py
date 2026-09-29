"""Middleton Class-A impulsive noise simulation (Track F-3 / Phase 1).

Models non-Gaussian atmospheric, urban, and ignition impulsive noise
as a Poisson mixture of Gaussian processes.

Reference:
  D. Middleton, 'Statistical-physical models of electromagnetic noise',
  IEEE Trans. Electromagn. Compat., vol. EMC-19, no. 3, 1977.
"""
from __future__ import annotations
import numpy as np


def middleton_class_a_noise(n_samples: int,
                            total_power: float = 1.0,
                            impulse_index_A: float = 0.1,
                            gamma_ratio: float = 0.01,
                            seed: int | None = None) -> np.ndarray:
    """Generate complex Middleton Class-A impulsive noise.

    Parameters
    ----------
    n_samples : int
        Number of complex samples to generate.
    total_power : float, default 1.0
        Total average noise power (sigma_G^2 + sigma_I^2).
    impulse_index_A : float, default 0.1
        Impulse index (overlap index). Smaller values (e.g. 0.01-0.1) produce
        sharper, rarer, more intense impulses. Large values (e.g. A > 10) approach
        pure Gaussian noise.
    gamma_ratio : float, default 0.01
        Ratio of Gaussian background power to impulsive power (sigma_G^2 / sigma_I^2).
    seed : int, optional
        RNG seed.

    Returns
    -------
    noise : np.ndarray
        Complex64 array of impulsive noise samples.
    """
    if n_samples <= 0:
        return np.zeros(0, dtype=np.complex64)

    rng = np.random.default_rng(seed)
    A = max(1e-5, impulse_index_A)
    gamma = max(1e-6, gamma_ratio)

    # Number of impulsive events per sample drawn from Poisson distribution
    m = rng.poisson(lam=A, size=n_samples)

    # Conditional variance for each sample: sigma_m^2
    var_m = total_power * (m / A + gamma) / (1.0 + gamma)
    sigma_m = np.sqrt(np.maximum(1e-12, var_m / 2.0))

    # Generate zero-mean complex Gaussian with local standard deviation sigma_m
    w_i = rng.standard_normal(n_samples) * sigma_m
    w_q = rng.standard_normal(n_samples) * sigma_m

    noise = (w_i + 1j * w_q).astype(np.complex64)
    return noise


def add_middleton_noise(iq: np.ndarray,
                        snr_db: float = 20.0,
                        impulse_index_A: float = 0.1,
                        gamma_ratio: float = 0.01,
                        seed: int | None = None) -> np.ndarray:
    """Add Middleton Class-A noise to signal at specified SNR.

    Parameters
    ----------
    iq : np.ndarray
        Clean complex input.
    snr_db : float
        Signal-to-noise ratio in dB.
    impulse_index_A : float
        Middleton A parameter.
    gamma_ratio : float
        Middleton Gamma parameter.
    seed : int, optional
        RNG seed.

    Returns
    -------
    noisy_iq : np.ndarray
        Signal corrupted by impulsive Middleton noise.
    """
    iq = np.asarray(iq, dtype=np.complex64).ravel()
    N = len(iq)
    if N == 0:
        return iq

    signal_pwr = np.mean(np.abs(iq) ** 2)
    noise_pwr = signal_pwr / (10.0 ** (snr_db / 10.0))

    noise = middleton_class_a_noise(
        n_samples=N,
        total_power=noise_pwr,
        impulse_index_A=impulse_index_A,
        gamma_ratio=gamma_ratio,
        seed=seed,
    )
    return (iq + noise).astype(np.complex64)
