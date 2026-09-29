"""Differential testing of RRC matched filter vs commpy (SIGNAL10 §G.1 / Rule R9)."""
from __future__ import annotations
import pytest
import numpy as np
from commpy.filters import rrcosfilter
from signal9.dsp.matched_filter import rrc_taps


@pytest.mark.parametrize("alpha", [0.1, 0.2, 0.35, 0.5, 0.8])
@pytest.mark.parametrize("sps", [2, 4, 8])
def test_rrc_taps_match_commpy(alpha, sps):
    """Signal9 RRC filter taps must match commpy reference to within 1e-12."""
    span = 10
    h_ours = rrc_taps(alpha=alpha, sps=sps, span=span)

    # commpy uses even N = 2 * span * sps
    _, h_ref = rrcosfilter(N=2 * span * sps, alpha=alpha, Ts=1.0, Fs=float(sps))

    n = min(len(h_ours), len(h_ref))
    h_ours_norm = h_ours[:n] / np.sqrt(np.sum(h_ours[:n] ** 2))
    h_ref_norm = h_ref / np.sqrt(np.sum(h_ref ** 2))

    max_diff = np.max(np.abs(h_ours_norm - h_ref_norm))
    assert max_diff < 1e-12, f"RRC tap mismatch vs commpy for alpha={alpha}, sps={sps}: diff={max_diff:.2e}"
