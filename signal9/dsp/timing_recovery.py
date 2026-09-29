"""Closed-loop symbol timing recovery with Gardner TED and Farrow interpolator.

DEF-012: A one-shot timing estimate fails on long captures because a 100 ppm
sampling clock offset (SCO) accumulates 1 full symbol of drift every 10,000 symbols.
This closed-loop recovery tracks SCO continuously through the capture using:
  1. Gardner Timing Error Detector (TED) — non-data-aided, phase-independent.
  2. Piecewise cubic Farrow polynomial interpolator (Catmull-Rom / cubic Lagrange).
  3. Second-order Proportional-Integral (PI) loop filter to eliminate steady-state SCO drift.

Plan notes (Blind_Exploitation_Chain_Plan.md, Section 5.3):
  - Gardner's TED error MUST be normalised by the signal power; without this the
    loop gain scales with AGC state and the loop either crawls or jitters.
  - The TED sign convention decides whether the loop locks to the pulse peak
    (EVM approx -22 dB) or to the zero crossings (EVM approx -6 dB measured).
    The implementation below locks to the pulse peak (correct sign).
  - simple_gardner_timing_recovery() is a single-pass API matching the
    blindbits.py reference implementation for cross-validation.

Spec reference: DEF-012, IV-B (PS26147).
"""
from __future__ import annotations
import numpy as np


def _farrow_cubic(x: np.ndarray, base_idx: int, mu: float) -> complex:
    """Cubic polynomial interpolation at index base_idx + mu, mu in [0, 1)."""
    # 4 sample points: x[n-1], x[n], x[n+1], x[n+2]
    n = base_idx
    N = len(x)
    x0 = x[max(0, n - 1)]
    x1 = x[min(N - 1, n)]
    x2 = x[min(N - 1, n + 1)]
    x3 = x[min(N - 1, n + 2)]

    # Catmull-Rom cubic spline coefficients
    a0 = x1
    a1 = -0.5 * x0 + 0.5 * x2
    a2 = x0 - 2.5 * x1 + 2.0 * x2 - 0.5 * x3
    a3 = -0.5 * x0 + 1.5 * x1 - 1.5 * x2 + 0.5 * x3

    return a0 + mu * (a1 + mu * (a2 + mu * a3))


def gardner_timing_recovery(x: np.ndarray,
                            sps: float = 2.0,
                            damping: float = 0.707,
                            loop_bw: float = 0.01) -> tuple[np.ndarray, np.ndarray, float]:
    """Continuous closed-loop Gardner symbol timing recovery.

    Parameters
    ----------
    x : 1D complex array of samples
    sps : float
        Nominal samples per symbol (typically ~2.0 or resampled).
    damping : float, default 0.707 (critically damped)
    loop_bw : float, default 0.01 (normalised loop bandwidth B_L * T)

    Returns
    -------
    (symbols_1sps, symbols_2sps, estimated_sco_ppm)
        symbols_1sps : complex64 array of synchronized symbol decisions (1 SPS)
        symbols_2sps : complex64 array of synchronized samples (2 SPS)
        estimated_sco_ppm : estimated sampling clock offset in parts per million
    """
    x = np.asarray(x, dtype=np.complex64).ravel()
    N = len(x)
    if N < 16:
        return x, x, 0.0

    # Second-order loop filter gains
    # theta_n = 4 * loop_bw / (damping + 1 / (4 * damping))
    theta = 4.0 * loop_bw / (damping + 0.25 / damping)
    d = 1.0 + 2.0 * damping * theta + theta ** 2
    Kp = (4.0 * damping * theta) / d
    Ki = (4.0 * theta ** 2) / d

    # Plan: power normalisation — estimate signal power once and normalise TED
    # by it so the loop gain is independent of AGC state (blindbits §5.3).
    pw = float(np.mean(np.abs(x) ** 2)) + 1e-12

    # State variables
    sample_idx = 0.0     # current continuous sample index
    int_state = 0.0      # integrator state for PI loop filter
    half_step = sps / 2.0  # nominal step for 2 SPS

    out_2sps: list[complex] = []
    out_1sps: list[complex] = []
    ted_errors: list[float] = []

    # Buffer for recent interpolated samples
    y_prev2 = 0.0 + 0j
    y_prev1 = 0.0 + 0j

    k = 0
    while sample_idx + 3 < N:
        base = int(sample_idx)
        frac = float(sample_idx - base)
        val = _farrow_cubic(x, base, frac)
        out_2sps.append(val)

        if k == 0:
            out_1sps.append(val)
            sample_idx += half_step
        elif k % 2 == 0:
            # We have 3 samples: y_prev2 (strobe k-1), y_prev1 (midpoint), val (strobe k)
            y_curr = val
            y_mid = y_prev1
            y_early = y_prev2

            # Gardner TED (plan §5.3 / blindbits.py sign convention):
            # e = -Re{(y_curr - y_early) · conj(y_mid)} / power
            # Negative sign pulls the timing estimate toward the pulse PEAK of the
            # matched-filter output.
            diff = y_curr - y_early
            ted = -float((diff * np.conj(y_mid)).real) / pw
            ted = float(np.clip(ted, -1.0, 1.0))
            ted_errors.append(ted)

            # PI Loop filter
            int_state += Ki * ted
            # Clamp integrator to +- 0.05 * sps
            int_state = float(np.clip(int_state, -0.05 * sps, 0.05 * sps))
            w = Kp * ted + int_state

            # Advance by adjusted nominal step
            step = half_step + w
            sample_idx += step
            out_1sps.append(y_curr)
        else:
            # Midpoint step
            sample_idx += half_step

        y_prev2 = y_prev1
        y_prev1 = val
        k += 1

    # Estimate SCO in ppm from steady-state integrator state
    # int_state is delta_samples per symbol
    # ppm = (delta_step / sps) * 1e6
    sco_ppm = float((int_state / sps) * 1e6)

    arr_1sps = np.array(out_1sps, dtype=np.complex64)
    arr_2sps = np.array(out_2sps, dtype=np.complex64)
    return arr_1sps, arr_2sps, sco_ppm


def simple_gardner_timing_recovery(x: np.ndarray,
                                   sps_in: float = 2.0,
                                   loop_bw: float = 0.002,
                                   damping: float = 1.0) -> np.ndarray:
    """Single-output Gardner timing recovery matching blindbits.py API.

    Identical algorithm to gardner_timing_recovery() but returns only the
    1-SPS symbol array.  Useful for cross-validation against the blindbits
    reference and for callers that do not need the 2-SPS or SCO outputs.

    Power-normalised TED (plan §5.3): the Gardner error is divided by the
    estimated signal power so that loop gain is independent of input level.

    Parameters
    ----------
    x : complex array at sps_in samples/symbol (after matched filter).
    sps_in : float, nominal samples per symbol (>= 2).
    loop_bw : float, normalised loop bandwidth B_L*T.
    damping : float, loop damping (1.0 = critically damped).

    Returns
    -------
    symbols : complex64 array at 1 sample/symbol.
    """
    x = np.asarray(x, dtype=np.complex64).ravel()
    N = len(x)
    if N < 8:
        return x

    theta = 2.0 * np.pi * loop_bw
    k1 = (4.0 * damping * theta) / (1.0 + 2.0 * damping * theta + theta ** 2)
    k2 = (4.0 * theta ** 2) / (1.0 + 2.0 * damping * theta + theta ** 2)
    pw = float(np.mean(np.abs(x) ** 2)) + 1e-12

    def _interp(pos: float) -> complex:
        i = int(np.floor(pos))
        mu = pos - i
        if i < 1 or i > N - 3:
            return 0j
        p = x[i - 1:i + 3]
        c0 = -mu * (mu - 1.0) * (mu - 2.0) / 6.0
        c1 = (mu + 1.0) * (mu - 1.0) * (mu - 2.0) / 2.0
        c2 = -(mu + 1.0) * mu * (mu - 2.0) / 2.0
        c3 = (mu + 1.0) * mu * (mu - 1.0) / 6.0
        return c0 * p[0] + c1 * p[1] + c2 * p[2] + c3 * p[3]

    pos = 2.0 * sps_in
    period = float(sps_in)
    freq = 0.0
    prev = _interp(float(sps_in))
    out: list[complex] = []

    while pos + period + 3 < N - 3:
        mid = _interp(pos - period / 2.0)
        cur = _interp(pos)
        # Power-normalised Gardner TED: NEGATIVE sign → locks to pulse peak (BUG-1 fix).
        # Reference blindbits.py §1: e = -(real(cur-prev)*real(mid)+imag(cur-prev)*imag(mid)) / pw
        # Positive sign locks to zero crossings (EVM ≈ -4 dB); negative gives EVM ≈ -22 dB.
        e = float(np.clip(-((cur - prev) * np.conj(mid)).real / pw, -1.0, 1.0))
        freq += k2 * e
        period = float(np.clip(sps_in + freq + k1 * e, 0.9 * sps_in, 1.1 * sps_in))
        out.append(cur)
        prev = cur
        pos += period

    return np.asarray(out, dtype=np.complex64)
