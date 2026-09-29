"""Narrowband channelizer and zoom-rate rational resampler."""
from __future__ import annotations
from fractions import Fraction
import numpy as np
from scipy import signal as sps

ZOOM_RATES = [1_000, 4_000, 16_000, 64_000, 256_000, 1_024_000]

def pick_rate(bw_hz: float, fill: float = 0.5) -> int:
    """Select standard zoom rate where signal bandwidth occupies ~fill of passband with strict Nyquist margin."""
    min_rate = int(2.0 * bw_hz)
    for r in ZOOM_RATES:
        if r >= min_rate and bw_hz <= fill * r:
            return r
    return ZOOM_RATES[-1]

def channelize(x: np.ndarray,
               fs: float,
               f_center: float,
               bw_hz: float,
               t0: float = 0.0,
               t1: float | None = None) -> tuple[np.ndarray, int]:
    """Cut one detection out of a wideband capture:
    1. Slice time extent [t0, t1]
    2. Frequency shift center frequency to 0 Hz
    3. Resample to standard zoom tier using rational polyphase FIR (never upsampling beyond capture rate)
    """
    s0 = int(max(t0, 0.0) * fs)
    s1 = len(x) if t1 is None else min(len(x), int(t1 * fs))
    if s1 <= s0:
        return np.empty(0, dtype=np.complex64), int(fs)

    seg = x[s0:s1]
    n = np.arange(seg.size)
    # 1. Frequency shift
    seg_shifted = seg * np.exp(-2j * np.pi * f_center / fs * n)

    # 2. Zoom rate selection
    out_fs = pick_rate(bw_hz)
    if out_fs >= fs or (len(seg) * out_fs / fs < 256):
        # Keep capture rate if out_fs >= fs or if downsampling leaves < 256 samples
        return seg_shifted.astype(np.complex64), int(fs)

    # 3. Rational resample with bounded denominator
    ratio = Fraction(out_fs, int(round(fs))).limit_denominator(4096)
    y = sps.resample_poly(seg_shifted, ratio.numerator, ratio.denominator)
    return y.astype(np.complex64), out_fs
