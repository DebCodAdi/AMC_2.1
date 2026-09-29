"""Impulsive noise detection and robust front-end mitigation (Track F-3).

Implements:
  - Kurtosis / Middleton impulsive noise detection
  - Huber M-estimator winsorization (phase-preserving outlier clipping)
  - Maximum Correntropy Criterion Multi-Modulus Algorithm (MCC-MMA) equalizer

Spec reference: Track F-3 (PS26147 / SIGNAL10).
"""
from __future__ import annotations
import numpy as np


def detect_impulsive_noise(iq: np.ndarray,
                           kurtosis_threshold: float = 3.2) -> tuple[bool, float]:
    """Detect presence of impulsive noise via fourth-order complex kurtosis.

    Parameters
    ----------
    iq : np.ndarray
        Complex baseband samples.
    kurtosis_threshold : float, default 3.2
        Threshold above which noise is classified as impulsive.
        (Circular complex Gaussian noise has theoretical kurtosis = 2.0).

    Returns
    -------
    (is_impulsive, kurtosis)
    """
    iq = np.asarray(iq, dtype=np.complex64).ravel()
    if len(iq) < 32:
        return False, 2.0

    mag2 = np.abs(iq) ** 2
    m2 = np.mean(mag2)
    if m2 < 1e-15:
        return False, 2.0

    m4 = np.mean(mag2 ** 2)
    kurt = float(m4 / (m2 ** 2))

    is_impulsive = kurt > kurtosis_threshold
    return is_impulsive, kurt


def huber_winsorize(iq: np.ndarray,
                    k_sigma: float = 3.0) -> np.ndarray:
    """Robust Huber M-estimator winsorization preserving sample phase.

    Smoothly shrinks impulsive spikes exceeding k_sigma * sigma_MAD without
    clipping phase or introducing sharp discontinuities.

    Parameters
    ----------
    iq : np.ndarray
        Complex baseband signal.
    k_sigma : float, default 3.0
        Outlier rejection threshold in robust standard deviations.

    Returns
    -------
    y : np.ndarray
        Winsorized complex signal.
    """
    iq = np.asarray(iq, dtype=np.complex64).ravel()
    if len(iq) == 0:
        return iq

    mag = np.abs(iq)
    med_mag = float(np.median(mag))
    # Robust standard deviation via Median Absolute Deviation (MAD)
    mad = float(np.median(np.abs(mag - med_mag)))
    sigma_mad = 1.4826 * mad

    # Fallback to standard deviation or median floor if MAD collapses
    scale_est = max(sigma_mad, float(np.std(mag)), 0.1 * max(1e-6, med_mag))
    thresh = k_sigma * scale_est

    # Scale factor: min(1.0, thresh / mag)
    scale = np.ones_like(mag, dtype=np.float32)
    outliers = mag > thresh
    scale[outliers] = thresh / (mag[outliers] + 1e-12)

    return (iq * scale).astype(np.complex64)


def mcc_mma_equalizer(iq: np.ndarray,
                      sps: int = 2,
                      n_taps: int = 21,
                      mu: float = 0.005,
                      kernel_sigma: float = 1.0,
                      mod_type: str = "QPSK") -> tuple[np.ndarray, np.ndarray]:
    """Maximum Correntropy Criterion Multi-Modulus Algorithm (MCC-MMA) equalizer.

    Robust to impulsive noise by weighting the MMA adaptation gradient by the
    Gaussian correntropy kernel exp(-|e|^2 / (2 * kernel_sigma^2)).
    """
    iq = np.asarray(iq, dtype=np.complex64).ravel()
    N = len(iq)
    if N < n_taps:
        return iq, np.ones(1, dtype=np.complex64)

    # Reference radii for QPSK / 16-QAM
    R_i = 1.0 / np.sqrt(2.0)
    R_q = 1.0 / np.sqrt(2.0)
    R2_i = R_i ** 2
    R2_q = R_q ** 2

    # Center-spike initialization
    w = np.zeros(n_taps, dtype=np.complex64)
    w[n_taps // 2] = 1.0 + 0j

    half = n_taps // 2
    y = np.zeros(N, dtype=np.complex64)
    inv_2sig2 = 1.0 / (2.0 * max(1e-3, kernel_sigma) ** 2)

    for n in range(half, N - half):
        x_vec = iq[n - half : n + half + 1][::-1]
        y_n = np.dot(w, x_vec)
        y[n] = y_n

        # MMA error per branch
        e_i = (y_n.real ** 2 - R2_i) * y_n.real
        e_q = (y_n.imag ** 2 - R2_q) * y_n.imag
        err = e_i + 1j * e_q

        # MCC Gaussian correntropy weighting: rejects large impulsive errors
        err_mag2 = float(e_i ** 2 + e_q ** 2)
        weight = float(np.exp(-err_mag2 * inv_2sig2))

        # Weight gradient
        w -= (mu * weight * err) * np.conj(x_vec)

    return y.astype(np.complex64), w
