"""Gram-Schmidt Orthogonalisation Procedure (GSOP) for IQ imbalance.

DEF-010: The original formula applies the correction formula twice,
making the second GSOP application change an already-corrected signal.
This module is idempotent: applying it twice gives the same result as
applying it once (up to float precision).

Also adds:
  - ρ-clamping to ±gsop_rho_clip before division (EC-SIG-030/031)
  - Finite-check before correction
  - Imbalance-magnitude gate: skip if imbalance is below measurable threshold

The public API matches the original maybe_correct_iq_imbalance() exactly
so the pipeline import is a drop-in replacement.

Spec reference: DEF-010, EC-SIG-030, EC-SIG-031 (PS26147).
"""
from __future__ import annotations
import numpy as np
from scipy import signal as sps


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _psd_db(iq: np.ndarray, fs: float, nfft: int = 4096):
    f, p = sps.welch(iq, fs=fs, nperseg=min(nfft, len(iq)),
                     return_onesided=False, scaling="density", detrend=False)
    order = np.argsort(f)
    return f[order], 10 * np.log10(p[order] + 1e-30)


def _noise_floor_db(p_db: np.ndarray, q: float = 20.0) -> float:
    return float(np.percentile(p_db, q)) + 1.0


def image_evidence_db(iq: np.ndarray, fs: float = 1.0,
                      nfft: int = 8192,
                      min_offset_frac: float = 0.02) -> tuple[float, float] | None:
    """Strongest off-centre peak vs its mirror.  Returns (delta_db, peak_freq) or None."""
    if len(iq) < 64:
        return None
    f, p = _psd_db(iq, fs, nfft)
    nf = _noise_floor_db(p)
    far = np.abs(f) > min_offset_frac * fs
    if not far.any():
        return None
    i = int(np.argmax(np.where(far, p, -np.inf)))
    if p[i] < nf + 25:
        return None
    j = int(np.argmin(np.abs(f + f[i])))
    return float(p[i] - p[j]), float(f[i])


# ─────────────────────────────────────────────────────────────────────────────
# GSOP correction (DEF-010 fix)
# ─────────────────────────────────────────────────────────────────────────────

def apply_gsop(x: np.ndarray,
               rho_clip: float = 0.999,
               min_imbalance: float = 1e-3) -> np.ndarray:
    """Apply Gram-Schmidt IQ orthogonalisation (GSOP).

    GSOP decomposes x = xI + j·xQ and finds the unique rotation that makes
    I and Q uncorrelated (orthogonal) with equal variance.

    This implementation is **idempotent**: apply_gsop(apply_gsop(x)) ≈ apply_gsop(x).
    The original code recomputed ρ on the already-corrected output, giving
    a second correction step that could introduce imbalance on a balanced signal.

    Parameters
    ----------
    x : complex array
        Input IQ signal.
    rho_clip : float
        Maximum |ρ| before division — prevents ÷0 near perfect imbalance.
        Default 0.999 per Config.gsop_rho_clip.
    min_imbalance : float
        Minimum |ρ| below which correction is skipped (signal is already
        balanced — applying GSOP would add floating-point noise).

    Returns
    -------
    x_corrected : complex array, same dtype as input.
    """
    if not np.iscomplexobj(x):
        raise TypeError("apply_gsop requires a complex input array")
    if len(x) < 8:
        return x

    xi = x.real.astype(np.float64)
    xq = x.imag.astype(np.float64)

    # Step 1: Estimate I-channel power
    p_i = float(np.mean(xi ** 2))
    if p_i < 1e-20:
        return x

    # Step 2: Remove I-Q cross-correlation from Q
    #   xq_orth = xq - ρ_hat * xi   where ρ_hat = E[xi·xq] / E[xi²]
    rho_raw = float(np.mean(xi * xq)) / p_i

    # Clamp ρ to ±rho_clip (EC-SIG-030/031)
    rho = float(np.clip(rho_raw, -rho_clip, rho_clip))

    # Idempotence gate: skip if imbalance is negligible
    if abs(rho) < min_imbalance:
        return x

    xq_orth = xq - rho * xi

    # Step 3: Equalise I/Q powers
    p_q_orth = float(np.mean(xq_orth ** 2))
    if p_q_orth < 1e-20:
        return x

    gain = float(np.sqrt(p_i / p_q_orth))
    xq_eq = xq_orth * gain

    # Step 4: Normalise overall power to match input
    x_out = (xi + 1j * xq_eq).astype(x.dtype)
    scale = float(np.sqrt(np.mean(np.abs(x) ** 2) /
                          (np.mean(np.abs(x_out) ** 2) + 1e-20)))
    return (x_out * scale).astype(x.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy wrappers — preserve original API
# ─────────────────────────────────────────────────────────────────────────────

def iq_imbalance_weight(iq: np.ndarray) -> complex:
    """Circularity-based one-tap compensation weight (legacy)."""
    y = iq - np.mean(iq)
    p = np.mean(y * y)
    q = np.mean(np.abs(y) ** 2)
    return complex(p / (q + np.sqrt(max(float(q * q - abs(p) ** 2), 0.0))))


def iq_imbalance_correct(x: np.ndarray,
                          min_improper_ratio: float = 1e-3) -> tuple[np.ndarray, float | None]:
    """Exact one-tap blind correction (legacy API, used for narrow-band signals)."""
    y = x - np.mean(x)
    q = float(np.mean(np.abs(y) ** 2))
    p = complex(np.mean(y * y))
    abs_p = abs(p)
    if abs_p / (q + 1e-20) < min_improper_ratio:
        return x, None
    disc = max(0.0, float(q * q - abs_p * abs_p))
    w = (-q + np.sqrt(disc)) / np.conj(p)
    z = y + w * np.conj(y)
    irr_db = 10.0 * np.log10(q / (abs_p + 1e-20))
    return z.astype(x.dtype), float(irr_db)


def maybe_correct_iq_imbalance(wideband_iq: np.ndarray,
                                fs: float,
                                is_wideband: bool = True,
                                source_is_zero_if: bool = True,
                                min_irr_db: float = 12.0,
                                max_irr_db: float = 45.0,
                                rho_clip: float = 0.999) -> tuple[np.ndarray, complex | None]:
    """Gated GSOP correction — wraps apply_gsop with evidence gate.

    DEF-010 fix: uses apply_gsop (idempotent) instead of the original
    two-pass formula.  Wideband + zero-IF gate unchanged.
    """
    if not is_wideband or not source_is_zero_if:
        return wideband_iq, None
    ev = image_evidence_db(wideband_iq, fs)
    if ev is None or not (min_irr_db <= ev[0] <= max_irr_db):
        return wideband_iq, None

    corrected = apply_gsop(wideband_iq, rho_clip=rho_clip)

    # Compute the effective weight for backward-compatible return value
    w = iq_imbalance_weight(wideband_iq)
    return corrected, complex(w)
