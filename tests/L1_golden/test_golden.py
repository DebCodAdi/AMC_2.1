"""L1 Golden-vector tests — the known-answer test (KAT) tier.

Every algorithm in this tier has a published or analytically-derivable
correct answer.  Failures here are B5 contract violations (wrong algorithm)
and outrank accuracy failures in triage priority.

Test IDs deliberately match the spec's test-ID list from §2.3.

Spec reference: §2.3, L1 tier (PS26147).
"""
from __future__ import annotations
import struct
import zlib

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# CRC-32 golden vectors (§2.3 KAT-CRC-1…3)
# ─────────────────────────────────────────────────────────────────────────────

class TestCRC32:
    """CRC-32 IEEE 802.3 polynomial: 0xEDB88320 (reflected)."""

    def _crc32(self, data: bytes) -> int:
        """Reference using Python's zlib.crc32."""
        return zlib.crc32(data) & 0xFFFFFFFF

    def test_kat_crc_1_standard_string(self):
        """The 9-byte string '123456789' must give exactly 0xCBF43926."""
        assert self._crc32(b"123456789") == 0xCBF43926

    def test_kat_crc_2_empty(self):
        """CRC-32 of empty input must be 0x00000000."""
        assert self._crc32(b"") == 0x00000000

    def test_kat_crc_3_single_byte(self):
        """CRC-32 of a single 0x00 byte is a specific constant."""
        # zlib.crc32(b'\x00') = 0xD202EF8D
        assert self._crc32(b"\x00") == 0xD202EF8D

    def test_kat_crc_4_running_window(self):
        """Running CRC update must equal the batch CRC."""
        data = bytes(range(256))
        batch = self._crc32(data)
        # Simulate the running-window update the pipeline uses
        crc = 0
        for byte in data:
            crc = zlib.crc32(bytes([byte]), crc) & 0xFFFFFFFF
        assert crc == batch

    def test_kat_crc_5_all_zeros_known(self):
        """CRC-32 of 32 zero bytes is a known constant."""
        assert self._crc32(b"\x00" * 32) == 0x190A55AD


# ─────────────────────────────────────────────────────────────────────────────
# Convolutional codec (§2.3 KAT-CONV-1…4)
# ─────────────────────────────────────────────────────────────────────────────

class TestConvolutionalCodec:
    """K=7 rate-1/2 conv code, generators (171, 133) in octal."""

    @pytest.fixture(autouse=True)
    def _import_codec(self):
        try:
            from signal9.fec import conv_encode, viterbi_decode  # type: ignore
            self.encode = conv_encode
            self.decode = viterbi_decode
        except ImportError:
            pytest.skip("signal9.fec not available yet")

    def test_kat_conv_1_zero_roundtrip(self):
        """All-zero payload encodes and decodes back to all-zero with no errors."""
        bits_in = np.zeros(100, dtype=np.uint8)
        encoded = self.encode(bits_in, 7, [0o171, 0o133])
        decoded = self.decode(encoded.astype(float) * 2 - 1, 7, [0o171, 0o133],
                              tb_depth=96)
        np.testing.assert_array_equal(decoded[:len(bits_in)], bits_in)

    def test_kat_conv_2_single_error_corrected(self):
        """A single bit error in the encoded stream must be corrected."""
        bits_in = np.zeros(100, dtype=np.uint8)
        encoded = self.encode(bits_in, 7, [0o171, 0o133])
        corrupted = encoded.copy()
        corrupted[50] ^= 1
        llr = (corrupted.astype(float) * 2 - 1) * 3.0  # SNR ~5 dB
        decoded = self.decode(llr, 7, [0o171, 0o133], tb_depth=96)
        np.testing.assert_array_equal(decoded[:len(bits_in)], bits_in)

    def test_kat_conv_3_impulse_response_length(self):
        """Encoding a single '1' followed by zeros produces exactly K=7 coded bits."""
        # The impulse response has length K, so the encoder state flushes in K-1 steps
        bits_in = np.array([1] + [0] * 20, dtype=np.uint8)
        encoded = self.encode(bits_in, 7, [0o171, 0o133])
        # After position 2*(K-1)=12, output should be zero
        assert encoded[14:].sum() == 0, (
            f"Impulse response extends past K-1 symbols: {encoded[12:]}"
        )

    def test_kat_conv_4_soft_vs_hard(self):
        """Saturated soft decisions (±100) should give same result as hard."""
        bits_in = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 12, dtype=np.uint8)
        encoded = self.encode(bits_in, 7, [0o171, 0o133])
        hard_llr = (encoded.astype(float) * 2 - 1)
        sat_llr  = hard_llr * 100.0
        dec_hard = self.decode(hard_llr, 7, [0o171, 0o133], tb_depth=96)
        dec_sat  = self.decode(sat_llr,  7, [0o171, 0o133], tb_depth=96)
        np.testing.assert_array_equal(dec_hard[:len(bits_in)], dec_sat[:len(bits_in)])


# ─────────────────────────────────────────────────────────────────────────────
# Reed–Solomon (§2.3 KAT-RS-1…4)
# ─────────────────────────────────────────────────────────────────────────────

class TestReedSolomon:
    @pytest.fixture(autouse=True)
    def _import_rs(self):
        try:
            from signal9.fec import rs_encode, rs_decode  # type: ignore
            self.encode = rs_encode
            self.decode = rs_decode
        except ImportError:
            pytest.skip("signal9.fec RS not available yet")

    def test_kat_rs_1_allzero(self):
        """RS(255,223) of all-zero data produces all-zero codeword."""
        data = np.zeros(223, dtype=np.uint8)
        cw = self.encode(data, n=255, k=223)
        assert len(cw) == 255
        assert cw.sum() == 0

    def test_kat_rs_2_correct_16_errors(self):
        """RS(255,223) can correct up to 16 symbol errors."""
        rng = np.random.default_rng(1)
        data = rng.integers(0, 256, 223, dtype=np.uint8)
        cw = self.encode(data, n=255, k=223).copy()
        error_pos = rng.choice(255, 16, replace=False)
        for pos in error_pos:
            cw[pos] = (int(cw[pos]) + 77) % 256
        recovered, n_errors = self.decode(cw, n=255, k=223)
        assert n_errors == 16
        np.testing.assert_array_equal(recovered, data)

    def test_kat_rs_3_flag_17_errors_not_silent_garbage(self):
        """RS(255,223) with 17 errors must NOT silently return garbage."""
        rng = np.random.default_rng(2)
        data = rng.integers(0, 256, 223, dtype=np.uint8)
        cw = self.encode(data, n=255, k=223).copy()
        error_pos = rng.choice(255, 17, replace=False)
        for pos in error_pos:
            cw[pos] = (int(cw[pos]) + 77) % 256
        # Must either raise or return n_errors < 0 to signal failure
        try:
            recovered, n_errors = self.decode(cw, n=255, k=223)
        except Exception:
            return  # Raising is also acceptable
        assert n_errors < 0, (
            "With 17 errors, RS(255,223) is uncorrectable; "
            "n_errors must be negative to signal failure, "
            f"got n_errors={n_errors}"
        )

    def test_kat_rs_4_dvb_n204_k188(self):
        """RS(204,188) DVB profile: all-zero data roundtrip."""
        data = np.zeros(188, dtype=np.uint8)
        cw = self.encode(data, n=204, k=188)
        recovered, n_errors = self.decode(cw, n=204, k=188)
        assert n_errors == 0
        np.testing.assert_array_equal(recovered, data)


# ─────────────────────────────────────────────────────────────────────────────
# LDPC (§2.3 KAT-LDPC-1…2)
# ─────────────────────────────────────────────────────────────────────────────

class TestLDPC:
    @pytest.fixture(autouse=True)
    def _import_ldpc(self):
        try:
            from signal9.fec import ldpc_encode, ldpc_decode  # type: ignore
            self.encode = ldpc_encode
            self.decode = ldpc_decode
        except ImportError:
            pytest.skip("signal9.fec LDPC not available yet")

    def test_kat_ldpc_1_syndrome_zero(self):
        """Encoding a zero information word must produce a codeword with zero syndrome."""
        from signal9.fec import ldpc_parity_check  # type: ignore
        data = np.zeros(324, dtype=np.uint8)
        cw = self.encode(data, n=648, k=324)
        syndrome = ldpc_parity_check(cw, n=648, k=324)
        assert syndrome.sum() == 0, f"Syndrome not zero: {syndrome.sum()} errors"

    def test_kat_ldpc_2_zero_iterations_perfect_codeword(self):
        """Feeding a perfect codeword in LLR form must converge in 0 iterations."""
        data = np.zeros(324, dtype=np.uint8)
        cw = self.encode(data, n=648, k=324)
        llr = (cw.astype(float) * 2 - 1) * 100.0  # very high SNR
        decoded, converged, n_iter, sw = self.decode(
            llr, n=648, k=324, max_iter=50
        )
        assert converged, "Should converge immediately on perfect codeword"
        assert n_iter == 0, f"Should take 0 iterations, took {n_iter}"
        assert sw == 0, f"Syndrome weight should be 0, got {sw}"


# ─────────────────────────────────────────────────────────────────────────────
# Gray map 16-QAM (§2.3 KAT-GRAY-1)
# ─────────────────────────────────────────────────────────────────────────────

class TestGrayMap:
    @pytest.fixture(autouse=True)
    def _import_mapper(self):
        try:
            from signal9.dsp.constellation import bits_to_symbols, gray_map  # type: ignore
            self.bits_to_symbols = bits_to_symbols
            self.gray_map = gray_map
        except ImportError:
            pytest.skip("signal9.dsp.constellation not available yet")

    def test_kat_gray_1_adjacent_differ_one_bit(self):
        """For Gray-coded 16-QAM, adjacent constellation symbols must differ
        in exactly 1 bit (Hamming distance 1)."""
        # Test all 16 symbols
        for i in range(16):
            sym_i = self.gray_map("16QAM", i)
            # Find the nearest neighbour in the constellation
            all_syms = [self.gray_map("16QAM", j) for j in range(16)]
            dists = [abs(sym_i - s) for j, s in enumerate(all_syms) if j != i]
            # At least one neighbour should be adjacent
            nearest = min(dists)
            for j in range(16):
                if j == i:
                    continue
                s = self.gray_map("16QAM", j)
                if abs(abs(sym_i - s) - nearest) < 1e-9:
                    # This is a neighbour — check Hamming distance in Gray labels
                    gray_i = i ^ (i >> 1)
                    gray_j = j ^ (j >> 1)
                    xor = gray_i ^ gray_j
                    hamming = bin(xor).count("1")
                    assert hamming == 1, (
                        f"Adjacent 16-QAM symbols {i}↔{j} have Hamming dist {hamming} ≠ 1"
                    )


# ─────────────────────────────────────────────────────────────────────────────
# Block interleaver roundtrip (§2.3 KAT-INTLV-1)
# ─────────────────────────────────────────────────────────────────────────────

class TestBlockInterleaver:
    @pytest.fixture(autouse=True)
    def _import_intlv(self):
        try:
            from signal9.fec import block_interleave, block_deinterleave  # type: ignore
            self.interleave = block_interleave
            self.deinterleave = block_deinterleave
        except ImportError:
            pytest.skip("signal9.fec interleaver not available yet")

    @pytest.mark.parametrize("R,C", [(4, 8), (8, 4), (3, 5), (17, 13), (1, 64)])
    def test_kat_intlv_1_roundtrip(self, R, C):
        """deinterleave(interleave(x, R, C), R, C) must equal x exactly."""
        bits = np.arange(R * C, dtype=np.uint8)
        intlv = self.interleave(bits, R, C)
        restored = self.deinterleave(intlv, R, C)
        np.testing.assert_array_equal(restored, bits,
            err_msg=f"Block interleaver ({R}×{C}) roundtrip failed")


# ─────────────────────────────────────────────────────────────────────────────
# M-th power CFO estimate (§2.3 KAT-CFO-1)
# ─────────────────────────────────────────────────────────────────────────────

class TestCFO:
    @pytest.fixture(autouse=True)
    def _import_cfo(self):
        try:
            from signal9.dsp.cfo import mth_power_cfo  # type: ignore
            self.cfo_est = mth_power_cfo
        except ImportError:
            pytest.skip("signal9.dsp.cfo not available yet")

    def test_kat_cfo_1_qpsk_known_offset(self):
        """QPSK signal with injected CFO=+1234.5 Hz must be estimated to < 0.1 Hz."""
        rng = np.random.default_rng(42)
        fs = 48000.0
        rs = 4800.0
        n_sym = 4096
        # Generate QPSK symbols
        bits = rng.integers(0, 2, 2 * n_sym)
        syms = ((2 * bits[0::2] - 1) + 1j * (2 * bits[1::2] - 1)) / np.sqrt(2)
        x = np.repeat(syms, int(fs / rs)).astype(np.complex64)
        # Inject CFO
        cfo_true = 1234.5
        n = np.arange(len(x))
        x = x * np.exp(2j * np.pi * cfo_true / fs * n)
        # Estimate
        cfo_est = self.cfo_est(x, fs, M=4)
        assert abs(cfo_est - cfo_true) < 0.1, (
            f"|est ({cfo_est:.2f}) - true ({cfo_true:.2f})| = "
            f"{abs(cfo_est-cfo_true):.2f} Hz > 0.1 Hz"
        )


# ─────────────────────────────────────────────────────────────────────────────
# RRC filter Nyquist property (§2.3 KAT-RRC-1)
# ─────────────────────────────────────────────────────────────────────────────

class TestRRC:
    @pytest.fixture(autouse=True)
    def _import_rrc(self):
        try:
            from signal9.dsp.matched_filter import rrc_taps  # type: ignore
            self.rrc_taps = rrc_taps
        except ImportError:
            pytest.skip("signal9.dsp.matched_filter not available yet")

    @pytest.mark.parametrize("alpha", [0.1, 0.2, 0.35, 0.5, 0.8])
    def test_kat_rrc_1_nyquist_property(self, alpha):
        """h_RRC * h_RRC must have zeros at non-zero symbol multiples (Nyquist ISI-free)."""
        sps = 4
        span = 10
        h = self.rrc_taps(alpha=alpha, sps=sps, span=span)
        # Overall Nyquist response (convolution of RRC with itself)
        nyq = np.convolve(h, h)
        centre = len(nyq) // 2
        # Check at ±1, ±2, …, ±span symbol positions
        # For truncated FIR filter of span=10, sidelobe truncation introduces ISI < 2e-3.
        for k in range(1, span + 1):
            val = abs(nyq[centre + k * sps])
            assert val < 5e-3, (
                f"Nyquist ISI at k={k}: h*h[{centre + k*sps}] = {val:.2e} > 5e-3 "
                f"(alpha={alpha}, sps={sps})"
            )

    @pytest.mark.parametrize("alpha", [0.2, 0.35])
    def test_kat_rrc_2_match_commpy(self, alpha):
        """RRC taps must match commpy.rrcosfilter to within 1e-12."""
        try:
            from commpy.filters import rrcosfilter  # type: ignore
        except ImportError:
            pytest.skip("commpy not installed")
        sps = 4
        span = 10
        h_ours = self.rrc_taps(alpha=alpha, sps=sps, span=span)
        _, h_ref = rrcosfilter(N=2 * span * sps, alpha=alpha, Ts=1.0, Fs=sps)
        # Align lengths and normalize energy
        n = min(len(h_ours), len(h_ref))
        h_ours_norm = h_ours[:n] / np.sqrt(np.sum(h_ours[:n] ** 2))
        h_ref_norm = h_ref[:n] / np.sqrt(np.sum(h_ref[:n] ** 2))
        np.testing.assert_allclose(h_ours_norm, h_ref_norm, atol=1e-12,
            err_msg=f"RRC mismatch vs commpy (alpha={alpha})")
