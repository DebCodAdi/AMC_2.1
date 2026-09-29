"""Adaptive blind equaliser with CMA, MMA, and Decision-Directed modes.

Fixes:
  - DEF-036: Unbounded CMA divergence guard — input power normalization to 1.0,
    tap-norm monitoring (resets with halved step size if ||w|| > 10*||w0||),
    and NaN/Inf protection.
  - DEF-037 / IV-D: Multi-Modulus Algorithm (MMA) for 16-QAM / 64-QAM — decouples
    real and imaginary dispersion, reducing residual MSE by 4-6 dB compared to CMA.
  - Dual-mode switching: Transitions from blind CMA/MMA to Decision-Directed (DD)
    LMS once the constellation eye opens (EVM < 0.25).
  - Taps lengthened to 21-31 taps with center-spike initialization (DEF-004).

Spec reference: IV-D, DEF-004, DEF-036, DEF-037 (PS26147).
"""
from __future__ import annotations
import numpy as np


def _slice_qam(y: complex, mod_type: str) -> complex:
    """Slice symbol to nearest ideal constellation point."""
    mod = mod_type.upper().replace("-", "")
    if mod == "BPSK":
        return complex(1.0 if y.real >= 0 else -1.0, 0.0)
    elif mod == "QPSK":
        r = 1.0 if y.real >= 0 else -1.0
        i = 1.0 if y.imag >= 0 else -1.0
        return complex(r / np.sqrt(2.0), i / np.sqrt(2.0))
    elif mod in ("16QAM", "QAM16"):
        # Levels {-3, -1, +1, +3} / sqrt(10)
        norm = np.sqrt(10.0)
        levels = np.array([-3.0, -1.0, 1.0, 3.0]) / norm
        r = levels[np.argmin(np.abs(y.real - levels))]
        i = levels[np.argmin(np.abs(y.imag - levels))]
        return complex(r, i)
    elif mod in ("64QAM", "QAM64"):
        # Levels {-7, -5, -3, -1, +1, +3, +5, +7} / sqrt(42)
        norm = np.sqrt(42.0)
        levels = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]) / norm
        r = levels[np.argmin(np.abs(y.real - levels))]
        i = levels[np.argmin(np.abs(y.imag - levels))]
        return complex(r, i)
    elif mod in ("8PSK", "PSK8"):
        theta = np.angle(y)
        k = int(np.round(8.0 * theta / (2.0 * np.pi))) % 8
        target_theta = k * 2.0 * np.pi / 8.0
        return complex(np.cos(target_theta), np.sin(target_theta))
    else:
        # Default unit circle
        mag = abs(y) + 1e-12
        return y / mag


def adaptive_equalize(x: np.ndarray,
                      mod_type: str = "QPSK",
                      n_taps: int = 21,
                      mu: float = 1e-3,
                      enable_dd: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """Adaptive blind equaliser (CMA / MMA / DD-LMS).

    Parameters
    ----------
    x : complex array
        Input signal at 2 SPS (or 1 SPS).
    mod_type : str
        Modulation type ('BPSK', 'QPSK', '8PSK', '16QAM', '64QAM').
    n_taps : int, default 21
        Filter tap count (odd integer).
    mu : float, default 1e-3
        Adaptation step size.
    enable_dd : bool, default True
        Allow transition to Decision-Directed mode once converged.

    Returns
    -------
    (equalized_symbols, final_taps, evm)
        equalized_symbols : 1D complex64 array
        final_taps : 1D complex64 array of filter coefficients
        evm : final estimated Error Vector Magnitude
    """
    x = np.asarray(x, dtype=np.complex64).ravel()
    N = len(x)
    if N < n_taps:
        return x, np.array([1.0 + 0j], dtype=np.complex64), 1.0

    # DEF-036: Normalize input power to 1.0 before equalisation
    pwr = float(np.mean(np.abs(x) ** 2))
    if pwr <= 1e-12 or not np.isfinite(pwr):
        return x, np.array([1.0 + 0j], dtype=np.complex64), 1.0
    x_norm = x / np.sqrt(pwr)

    mod = mod_type.upper().replace("-", "")
    is_qam = mod in ("16QAM", "QAM16", "64QAM", "QAM64")

    # Target dispersion constants
    # CMA target R2 = E[|s|^4] / E[|s|^2]
    # MMA target R_mma = E[s_R^4] / E[s_R^2]
    if mod == "BPSK":
        R2 = 1.0
        R_mma = 1.0
    elif mod in ("QPSK", "8PSK"):
        R2 = 1.0
        R_mma = 0.5  # E[s_R^4]/E[s_R^2] for QPSK is 0.25 / 0.5 = 0.5
    elif mod in ("16QAM", "QAM16"):
        R2 = 1.32
        R_mma = 0.82  # E[s_R^4]/E[s_R^2] for unit-power 16QAM: 0.41 / 0.5 = 0.82
    elif mod in ("64QAM", "QAM64"):
        R2 = 1.381
        R_mma = 0.881 # E[s_R^4]/E[s_R^2] for unit-power 64QAM: 0.4405 / 0.5 = 0.881
    else:
        R2 = 1.0
        R_mma = 0.5

    # Center-spike initialization (DEF-004)
    w0 = np.zeros(n_taps, dtype=np.complex64)
    center = n_taps // 2
    w0[center] = 1.0 + 0j
    w = w0.copy()
    w0_norm = float(np.linalg.norm(w0))

    step_mu = mu
    y_out = np.zeros(N - n_taps + 1, dtype=np.complex64)
    dd_mode = False

    i = 0
    while i < N - n_taps + 1:
        # Extract filter input vector (time-reversed for convolution)
        reg = x_norm[i : i + n_taps][::-1]
        y = complex(np.dot(w, reg))

        # Check for NaN / Inf
        if not np.isfinite(y.real) or not np.isfinite(y.imag):
            # DEF-036: Tap explosion safeguard — halve step size and restart
            step_mu *= 0.5
            w = w0.copy()
            i = 0
            continue

        if dd_mode:
            # Decision-Directed LMS: e = y - slice(y)
            s_hat = _slice_qam(y, mod_type)
            e = y - s_hat
            w -= step_mu * 0.5 * e * np.conj(reg)
        else:
            if is_qam:
                # Multi-Modulus Algorithm (MMA / DEF-037):
                # e_R = y_R * (y_R^2 - R_mma), e_I = y_I * (y_I^2 - R_mma)
                e_R = y.real * (y.real ** 2 - R_mma)
                e_I = y.imag * (y.imag ** 2 - R_mma)
                e = complex(e_R, e_I)
            else:
                # Constant Modulus Algorithm (CMA):
                # e = y * (|y|^2 - R2)
                e = y * (abs(y) ** 2 - R2)

            w -= step_mu * e * np.conj(reg)

        # DEF-036: Tap-norm monitor (halve mu and restart if ||w|| > 10 * ||w0||)
        w_norm = float(np.linalg.norm(w))
        if w_norm > 10.0 * w0_norm:
            step_mu *= 0.5
            w = w0.copy()
            i = 0
            continue

        y_out[i] = y

        # Check for eye opening and switch to DD mode
        if enable_dd and not dd_mode and i > 500 and i % 200 == 0:
            recent = y_out[max(0, i - 200) : i]
            # Sliced EVM
            slices = np.array([_slice_qam(s, mod_type) for s in recent])
            evm_sample = float(np.sqrt(np.mean(np.abs(recent - slices) ** 2)))
            if evm_sample < 0.25:
                dd_mode = True

        i += 1

    # Final EVM calculation over the last 20% of symbols
    tail_len = max(32, len(y_out) // 5)
    tail = y_out[-tail_len:]
    tail_slices = np.array([_slice_qam(s, mod_type) for s in tail])
    evm_final = float(np.sqrt(np.mean(np.abs(tail - tail_slices) ** 2)))

    return y_out, w, evm_final
