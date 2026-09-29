"""Differential testing of resampler vs soxr reference library (SIGNAL10 §G.1 / Rule R9)."""
from __future__ import annotations
import pytest
import numpy as np
import soxr
from signal9.io.resampler import resample_rational


@pytest.mark.parametrize("up,down", [(3, 2), (4, 3), (2, 1), (1, 2)])
@pytest.mark.parametrize("seed", range(3))
def test_resample_agreement_with_soxr(up, down, seed):
    """Signal9 rational resampler must cross-correlate > 0.98 with independent soxr engine."""
    rng = np.random.default_rng(seed)
    n = 2000
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)

    # Reference resample via soxr (resample real & imag independently)
    yr = soxr.resample(x.real.astype(np.float32), in_rate=down, out_rate=up)
    yi = soxr.resample(x.imag.astype(np.float32), in_rate=down, out_rate=up)
    y_ref = yr + 1j * yi

    # Signal9 rational resampler
    y_ours = resample_rational(x, up=up, down=down)

    min_len = min(len(y_ref), len(y_ours))
    margin = 50  # ignore filter transition edges
    s_ref = y_ref[margin : min_len - margin]
    s_ours = y_ours[margin : min_len - margin]

    # Normalized cross-correlation
    corr = np.abs(np.vdot(s_ref, s_ours)) / (np.linalg.norm(s_ref) * np.linalg.norm(s_ours))
    assert corr > 0.98, f"Resampler correlation with soxr ({corr:.4f}) < 0.98 for ratio {up}/{down}"
