"""Non-Data-Aided (NDA) SNR estimation via three-way fusion (IV-F).

Implements:
  1. M2M4 estimator: Classical second- and fourth-order moment estimator.
  2. Split-Symbol Moment Estimator (SSME): Splits symbol into two halves to separate
     correlated signal from uncorrelated noise (robust at low/negative SNR).
  3. Signal-to-Variation Ratio (SVR): Moments of instantaneous power.
  4. Robust fusion: Median combination with confidence bounds.

Spec reference: IV-F, DEF-017 (PS26147).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SNREstimate:
    """Estimated SNR with breakdown across estimators."""
    snr_db: float
    m2m4_db: float
    ssme_db: float
    svr_db: float
    confidence: float  # [0, 1], based on estimator agreement


def estimate_snr_m2m4(y: np.ndarray, mod_type: str = "QPSK") -> float:
    """M2M4 NDA SNR estimator.

    Parameters
    ----------
    y : complex array of symbol-rate samples
    mod_type : str ('BPSK', 'QPSK', '8PSK', '16QAM', '64QAM')

    Returns
    -------
    float : estimated SNR in dB (clamped to [-20.0, 50.0])
    """
    y = np.asarray(y, dtype=np.complex128).ravel()
    if len(y) < 16:
        return 10.0

    mag2 = np.abs(y) ** 2
    M2 = float(np.mean(mag2))
    M4 = float(np.mean(mag2 ** 2))

    # Modulation kurtosis ka = E[|s|^4] / (E[|s|^2])^2
    mod = mod_type.upper().replace("-", "")
    ka_map = {
        "BPSK": 1.0,
        "QPSK": 1.0,
        "8PSK": 1.0,
        "16QAM": 1.32,
        "64QAM": 1.381,
    }
    ka = ka_map.get(mod, 1.0)

    # Radicand term for Pauluzzi & Beaulieu M2M4: 2*M2^2 - M4 >= 0
    discrim = 2.0 * (M2 ** 2) - M4
    if discrim <= 0:
        # Very low SNR or noise-dominated
        return -10.0

    S = np.sqrt(max(0.0, (2.0 * M2 ** 2 - M4) / (2.0 - ka)))
    N = M2 - S
    if N <= 1e-12:
        return 40.0
    snr_lin = max(1e-4, S / N)
    return float(np.clip(10.0 * np.log10(snr_lin), -20.0, 50.0))


def estimate_snr_ssme(y_2sps: np.ndarray) -> float:
    """Split-Symbol Moment Estimator (SSME).

    Requires 2 SPS input. Splits each symbol into first half and second half.
    Signal components are correlated, noise components are uncorrelated.
    """
    y = np.asarray(y_2sps, dtype=np.complex128).ravel()
    N = len(y) // 2
    if N < 16:
        return 10.0

    y1 = y[0 : 2 * N : 2]
    y2 = y[1 : 2 * N : 2]

    # Correlated power: Re{E[y1 * conj(y2)]} = P_sig
    cross = np.real(np.mean(y1 * np.conj(y2)))
    # Difference power: E[|y1 - y2|^2] = 2 * P_noise
    diff_pwr = np.mean(np.abs(y1 - y2) ** 2)

    if diff_pwr <= 1e-12 or cross <= 0:
        return -5.0 if cross <= 0 else 40.0

    p_sig = cross
    p_noise = diff_pwr / 2.0
    snr_lin = max(1e-4, p_sig / p_noise)
    return float(np.clip(10.0 * np.log10(snr_lin), -20.0, 50.0))


def estimate_snr_svr(y: np.ndarray) -> float:
    """Signal-to-Variation Ratio (SVR) estimator.

    Uses the ratio of the square of the mean of |y|^2 to its variance.
    """
    y = np.asarray(y, dtype=np.complex128).ravel()
    if len(y) < 16:
        return 10.0

    r = np.abs(y) ** 2
    mu = float(np.mean(r))
    var = float(np.var(r))

    if var <= 1e-12:
        return 40.0

    beta = (mu ** 2) / var
    # For constant modulus signals in AWGN, beta increases monotonically with SNR
    if beta <= 1.0:
        return -10.0

    snr_lin = np.sqrt(beta - 1.0)
    return float(np.clip(10.0 * np.log10(max(1e-4, snr_lin)), -20.0, 50.0))


def estimate_snr(symbols: np.ndarray,
                 symbols_2sps: np.ndarray | None = None,
                 mod_type: str = "QPSK") -> SNREstimate:
    """Three-way fused Non-Data-Aided (NDA) SNR estimation (IV-F).

    Parameters
    ----------
    symbols : 1D complex array at 1 SPS
    symbols_2sps : 1D complex array at 2 SPS (optional, for SSME)
    mod_type : str ('BPSK', 'QPSK', '16QAM', etc.)

    Returns
    -------
    SNREstimate dataclass with fused snr_db and component estimates.
    """
    m2m4 = estimate_snr_m2m4(symbols, mod_type)
    svr = estimate_snr_svr(symbols)

    if symbols_2sps is not None and len(symbols_2sps) >= 32:
        ssme = estimate_snr_ssme(symbols_2sps)
        estimates = [m2m4, ssme, svr]
    else:
        ssme = m2m4  # fallback
        estimates = [m2m4, svr]

    # Robust median fusion
    fused = float(np.median(estimates))

    # Agreement confidence: 1.0 if all agree within 2 dB, drops with spread
    spread = float(np.max(estimates) - np.min(estimates))
    confidence = float(np.clip(1.0 - (spread / 15.0), 0.1, 1.0))

    return SNREstimate(
        snr_db=fused,
        m2m4_db=m2m4,
        ssme_db=ssme,
        svr_db=svr,
        confidence=confidence
    )


# ─────────────────────────────────────────────────────────────────────────────
# EVM (Error Vector Magnitude) estimation (Plan §5.2, blindbits.py §3)
# ─────────────────────────────────────────────────────────────────────────────

def evm_db(symbols: np.ndarray, constellation_points: np.ndarray) -> float:
    """Compute Error Vector Magnitude in dB against the nearest constellation point.

    EVM_dB = 10 * log10( mean(|sym - nearest|^2) )

    The symbol array is normalised to unit average power before computing EVM,
    matching the convention in blindbits.py §3 and the plan §5.2.

    Parameters
    ----------
    symbols : 1D complex array of received symbols.
    constellation_points : 1D complex array of ideal constellation symbols.

    Returns
    -------
    float : EVM in dB (negative; e.g. -22 dB is a good QPSK link).
    """
    sym = np.asarray(symbols, dtype=np.complex128).ravel()
    pts = np.asarray(constellation_points, dtype=np.complex128).ravel()
    sym = sym[np.isfinite(sym.real) & np.isfinite(sym.imag)]
    pts = pts[np.isfinite(pts.real) & np.isfinite(pts.imag)]
    if sym.size == 0 or pts.size == 0:
        return float("nan")
    power = float(np.mean(np.abs(sym) ** 2))
    if not np.isfinite(power):
        return float("nan")
    sym = sym / np.sqrt(max(power, 1e-20))
    d2 = np.min(np.abs(sym[:, None] - pts[None, :]) ** 2, axis=1)
    return float(10.0 * np.log10(np.mean(d2) + 1e-20))


def snr_from_evm(evm_db_val: float) -> float:
    """Approximate SNR (dB) from EVM (dB).  Valid for Gaussian noise-limited links.

    SNR_dB ≈ -EVM_dB  (for constant-modulus constellations).
    """
    return -evm_db_val
