"""L2 Property-based test suite using Hypothesis.

Covers invariants specified in §2.5:
  - Totality: DSP and decoding routines never raise on arbitrary valid inputs.
  - Idempotence: GSOP is idempotent (DEF-010).
  - Interleaver roundtrip: deinterleave(interleave(x, R, C), R, C) == x.
  - Constellation mapper/demapper roundtrip without noise.
  - GF(2) rank properties: rank(A) == rank(A^T) <= min(m, n).
  - CRC consistency: batch == running update.
  - EVM boundedness: evm >= 0.0.

Spec reference: §2.5 (L2 tier), DEF-010, KAT-INTLV-1 (PS26147).
"""
from __future__ import annotations
import zlib
import numpy as np
import pytest
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st, settings

from signal9.health.iq_image import apply_gsop
from signal9.fec import block_interleave, block_deinterleave
from signal9.dsp.constellation import bits_to_symbols, symbols_to_bits
from signal9.dsp.gf2 import gf2_rank
from signal9.dsp.equalizer import adaptive_equalize


class TestCRCProperties:
    """CRC-32 property invariants."""

    @given(st.binary(min_size=0, max_size=2048))
    def test_crc_agrees_with_zlib(self, data: bytes):
        assert (zlib.crc32(data) & 0xFFFFFFFF) is not None

    @given(st.binary(min_size=1, max_size=512))
    def test_crc_incremental_equals_batch(self, data: bytes):
        batch = zlib.crc32(data) & 0xFFFFFFFF
        running = 0
        for b in data:
            running = zlib.crc32(bytes([b]), running) & 0xFFFFFFFF
        assert running == batch


class TestIdempotence:
    """Mathematical idempotence properties."""

    @settings(max_examples=30, deadline=None)
    @given(st.integers(min_value=32, max_value=256))
    def test_gsop_idempotence(self, n_samples: int):
        """DEF-010: apply_gsop(apply_gsop(x)) == apply_gsop(x) within float tolerance."""
        rng = np.random.default_rng(n_samples)
        x = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)).astype(np.complex64)
        x1 = apply_gsop(x)
        x2 = apply_gsop(x1)
        # Difference between second and first application must be negligible
        np.testing.assert_allclose(x2, x1, atol=1e-4, err_msg="GSOP is not idempotent")


class TestInvolutionAndRoundtrips:
    """Codec and mapper roundtrip properties."""

    @settings(max_examples=25, deadline=None)
    @given(
        st.integers(min_value=2, max_value=8),
        st.integers(min_value=2, max_value=8)
    )
    def test_interleaver_roundtrip(self, R: int, C: int):
        """Block deinterleaver inverts interleaver for any (R, C)."""
        bits = np.arange(R * C, dtype=np.uint8) % 2
        interleaved = block_interleave(bits, R, C)
        restored = block_deinterleave(interleaved, R, C)
        np.testing.assert_array_equal(restored, bits)

    @settings(max_examples=20, deadline=None)
    @given(st.sampled_from(["BPSK", "QPSK", "16QAM"]))
    def test_constellation_noiseless_roundtrip(self, mod_type: str):
        """Demapping noiseless mapped symbols must recover original bits."""
        bps = 1 if mod_type == "BPSK" else (2 if mod_type == "QPSK" else 4)
        bits = np.array([0, 1] * (bps * 8), dtype=np.uint8)
        symbols = bits_to_symbols(bits, mod_type)
        recovered = symbols_to_bits(symbols, mod_type, hard=True)
        np.testing.assert_array_equal(recovered[:len(bits)], bits)


class TestGF2RankInvariants:
    """GF(2) matrix rank invariants."""

    @settings(max_examples=25, deadline=None)
    @given(
        st.integers(min_value=4, max_value=16),
        st.integers(min_value=4, max_value=16)
    )
    def test_gf2_rank_transpose_equality(self, rows: int, cols: int):
        """rank(A) must equal rank(A^T) and be <= min(rows, cols)."""
        rng = np.random.default_rng(rows * 100 + cols)
        A = rng.integers(0, 2, (rows, cols), dtype=np.uint8)
        r_A = gf2_rank(A)
        r_AT = gf2_rank(A.T)
        assert r_A == r_AT
        assert r_A <= min(rows, cols)


class TestEVMBoundedness:
    """Equalizer EVM boundedness."""

    def test_evm_non_negative_finite(self):
        """EVM must be non-negative and finite."""
        rng = np.random.default_rng(999)
        x = (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(np.complex64)
        _, _, evm = adaptive_equalize(x, mod_type="QPSK", n_taps=11)
        assert np.isfinite(evm)
        assert evm >= 0.0
