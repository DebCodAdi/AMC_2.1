"""Differential testing of Convolutional Codec vs commpy (SIGNAL10 §G.1 / Rule R9)."""
from __future__ import annotations
import pytest
import numpy as np
from commpy.channelcoding.convcode import Trellis, conv_encode, viterbi_decode
from signal9.fec.conv import conv_encode as s9_encode, viterbi_decode as s9_decode

TRELLIS = Trellis(memory=np.array([6]), g_matrix=np.array([[0o171, 0o133]]))


@pytest.mark.parametrize("n", [64, 128, 256, 512])
@pytest.mark.parametrize("seed", range(5))
def test_conv_encode_bit_exact(n, seed):
    """Signal9 convolutional encoding must be bit-exact with commpy reference."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n).astype(np.uint8)

    ref = conv_encode(bits, TRELLIS)
    ours = s9_encode(bits, K=7, gens=[0o171, 0o133])

    # commpy terminates with 6 zeros (12 bits) at end; unpadded prefix must match exactly
    assert np.array_equal(ours, ref[:len(ours)]), f"Divergence in conv encoding at n={n}, seed={seed}"


@pytest.mark.parametrize("ber", [0.0, 0.01, 0.03])
def test_viterbi_decode_agreement(ber):
    """Signal9 soft-decision Viterbi decoder must agree with commpy reference (> 99.5%)."""
    agreements = []
    for seed in range(10):
        rng = np.random.default_rng(seed + 100)
        bits = rng.integers(0, 2, 256).astype(np.uint8)
        cw = s9_encode(bits, K=7, gens=[0o171, 0o133])

        # Add binary symmetric channel noise
        noise_flips = rng.random(len(cw)) < ber
        noisy_cw = cw ^ noise_flips.astype(np.uint8)

        # Soft LLR input
        llr = (2.0 * noisy_cw - 1.0).astype(float)

        dec_ours = s9_decode(llr, K=7, gens=[0o171, 0o133], tb_depth=96)[:256]
        dec_ref = viterbi_decode(llr, TRELLIS, tb_depth=96, decoding_type="unquantized")[:256]

        agreements.append(float(np.mean(dec_ours == dec_ref)))

    mean_agreement = float(np.mean(agreements))
    assert mean_agreement >= 0.995, f"Viterbi decoder agreement with commpy was {mean_agreement:.4f} < 0.995 at BER={ber}"
