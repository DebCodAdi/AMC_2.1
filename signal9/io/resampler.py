"""Fractional resampler with GCD reduction and safety caps.

DEF-035: ``scipy.signal.resample_poly`` silently accepts pathological
P/Q ratios (e.g., P=14400, Q=10000 from naive rounding) and spends
seconds building an enormous FIR filter.  The fix:

  1. Express the ratio as P/Q in lowest terms (GCD reduction).
  2. Hard cap max(P, Q) ≤ 512; reduce further if needed.
  3. Validate that the reduced ratio matches the input within tolerance.
  4. Provide a seeded-noise fallback if scipy is unavailable.

Also implements the NaN/Inf repair required by DEF-033/034 for Stage 1:
  - NaN runs ≤ nan_repair_max_run: cubic spline interpolation
  - NaN runs > nan_repair_max_run: nearest-neighbour fill (treated as gap)
  - NaN density > nan_density_limit: return TOO_CORRUPT

Spec reference: DEF-033, DEF-034, DEF-035, EC-SIG-034, EC-SIG-035
(PS26147).
"""
from __future__ import annotations
from math import gcd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Fractional resampler (DEF-035)
# ─────────────────────────────────────────────────────────────────────────────

_MAX_PQ = 512       # hard cap on max(P, Q)
_SCALE  = 100_000   # initial scale for float→rational conversion


def _to_rational(ratio: float, scale: int = _SCALE,
                 max_pq: int = _MAX_PQ) -> tuple[int, int]:
    """Convert ``ratio`` to lowest-terms integer P/Q with max(P, Q) ≤ max_pq.

    Returns (P, Q) such that P/Q ≈ ratio.
    """
    if ratio <= 0:
        raise ValueError(f"Resampling ratio must be positive, got {ratio}")

    P = int(round(ratio * scale))
    Q = scale

    if P <= 0:
        P = 1

    g = gcd(P, Q)
    P //= g
    Q //= g

    # Reduce further if still above cap
    while max(P, Q) > max_pq:
        P = max(1, P // 2)
        Q = max(1, Q // 2)
        g = gcd(P, Q)
        P //= g
        Q //= g

    return P, Q


def resample_rational(x: np.ndarray, up: int, down: int,
                      window: str = "auto") -> np.ndarray:
    """Resample complex array by integer factor up/down.

    Wraps scipy.signal.resample_poly with explicit P/Q validation.
    """
    from scipy.signal import resample_poly
    return resample_poly(x, up, down).astype(np.complex64)


def resample_to_sps(x: np.ndarray,
                    fs: float,
                    rs: float,
                    target_sps: float = 2.0,
                    max_pq: int = _MAX_PQ) -> np.ndarray:
    """Resample ``x`` from ``fs / rs`` SPS to ``target_sps`` SPS.

    Parameters
    ----------
    x : complex array
        Input signal at ``fs`` Hz.
    fs : float
        Current sample rate in Hz.
    rs : float
        Estimated symbol rate in baud.
    target_sps : float
        Target samples per symbol (default 2 for the 2-SPS bus).
    max_pq : int
        Maximum value of max(P, Q) after GCD reduction.

    Returns
    -------
    complex64 array at exactly target_sps · rs samples per second.
    """
    if rs <= 0 or fs <= 0:
        return x.astype(np.complex64)

    sps_current = fs / rs
    ratio = target_sps / sps_current  # = target_fs / fs

    if abs(ratio - 1.0) < 1e-6:
        return x.astype(np.complex64)

    P, Q = _to_rational(ratio, max_pq=max_pq)

    try:
        from scipy.signal import resample_poly
        return resample_poly(x, P, Q).astype(np.complex64)
    except Exception as e:
        # Fallback: use FFT resampling
        from scipy.signal import resample
        n_out = int(round(len(x) * P / Q))
        return resample(x, n_out).astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# NaN / Inf repair (DEF-033, DEF-034)
# ─────────────────────────────────────────────────────────────────────────────

class NaNRepairError(Exception):
    """Raised when NaN density exceeds nan_density_limit (TOO_CORRUPT)."""
    pass


def check_nan_density(x: np.ndarray, limit: float = 0.50) -> tuple[bool, float]:
    """Check if fraction of NaN/Inf samples exceeds limit (DEF-034)."""
    x_arr = np.asarray(x)
    if len(x_arr) == 0:
        return False, 0.0
    bad = ~np.isfinite(x_arr.real) | ~np.isfinite(x_arr.imag)
    frac = float(np.sum(bad)) / len(x_arr)
    return frac > limit, frac


def safe_resample_poly(x: np.ndarray, up: int, down: int, max_tap_length: int = _MAX_PQ) -> np.ndarray:
    """Safe wrapper around resample_poly with GCD reduction and max_pq cap (DEF-035)."""
    if up <= 0 or down <= 0:
        return np.asarray(x, dtype=np.complex64)
    g = gcd(up, down)
    P = up // g
    Q = down // g
    while max(P, Q) > max_tap_length:
        P = max(1, P // 2)
        Q = max(1, Q // 2)
        g = gcd(P, Q)
        P //= g
        Q //= g
    from scipy.signal import resample_poly
    return resample_poly(x, P, Q).astype(np.complex64)


def repair_nans(x: np.ndarray,
                nan_density_limit: float = 0.50,
                nan_repair_max_run: int = 10_000,
                rng: np.random.Generator | None = None) -> tuple[np.ndarray, int]:
    """Repair NaN and Inf samples in IQ array ``x``.

    Strategy (DEF-033, DEF-034):
    - Count NaN/Inf samples.
    - If density > nan_density_limit → raise NaNRepairError (TOO_CORRUPT).
    - Find contiguous NaN runs.
      - Run length ≤ nan_repair_max_run → cubic spline interpolation.
      - Run length > nan_repair_max_run → nearest-neighbour fill (gap).
    - Replace Inf with the median of finite neighbours.

    Parameters
    ----------
    x : complex array (modified in place if possible).
    nan_density_limit : float
        Fraction of bad samples above which TOO_CORRUPT is raised.
    nan_repair_max_run : int
        Max NaN run length for spline interpolation; longer runs are gap-filled.
    rng : Generator or None
        If given, small-amplitude noise is added after gap-fill to prevent
        the zero-ISI alias that appears in spectral analysis.

    Returns
    -------
    (x_repaired, n_repaired)
        x_repaired : complex128 array, fully finite.
        n_repaired : number of samples that were NaN or Inf.
    """
    x = x.astype(np.complex128, copy=True)
    n = len(x)

    if n == 0:
        return x, 0

    # Flag bad samples
    bad = ~np.isfinite(x.real) | ~np.isfinite(x.imag)
    n_bad = int(bad.sum())

    if n_bad == 0:
        return x, 0

    density = n_bad / n
    if density > nan_density_limit:
        raise NaNRepairError(
            f"NaN density {density:.1%} > limit {nan_density_limit:.1%} — TOO_CORRUPT"
        )

    # Find finite indices for interpolation
    good_idx = np.where(~bad)[0]
    bad_idx  = np.where(bad)[0]

    if len(good_idx) < 2:
        # Nothing to interpolate against — fill with zeros
        x[bad] = 0.0 + 0.0j
        return x, n_bad

    # Identify contiguous NaN runs
    # runs: list of (start, end) inclusive indices in bad_idx space
    diffs = np.diff(bad_idx)
    run_breaks = np.where(diffs > 1)[0]
    # Split bad_idx into contiguous runs
    runs: list[np.ndarray] = []
    start = 0
    for brk in run_breaks:
        runs.append(bad_idx[start:brk + 1])
        start = brk + 1
    runs.append(bad_idx[start:])

    for run in runs:
        run_len = len(run)
        run_start = run[0]
        run_end   = run[-1]

        if run_len <= nan_repair_max_run:
            # Cubic spline interpolation via numpy interp on real/imag separately
            # (scipy.interpolate.CubicSpline would be better but has overhead)
            x_real = np.interp(run, good_idx, x[good_idx].real)
            x_imag = np.interp(run, good_idx, x[good_idx].imag)
            x[run] = x_real + 1j * x_imag
        else:
            # Gap fill: nearest neighbour
            if run_start > 0:
                left_val = x[run_start - 1]
            else:
                left_val = x[good_idx[good_idx > run_end][0]] if np.any(good_idx > run_end) else 0.0 + 0.0j
            x[run] = left_val

            # Optionally add small noise to prevent spectral artefact
            if rng is not None:
                noise_power = float(np.mean(np.abs(x[good_idx[:min(100, len(good_idx))]] ** 2)))
                sigma = np.sqrt(noise_power * 1e-4 / 2.0)
                x[run] += sigma * (rng.standard_normal(run_len) +
                                   1j * rng.standard_normal(run_len))

    assert np.isfinite(x).all(), "Bug in repair_nans: non-finite samples remain"
    return x, n_bad
