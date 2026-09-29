"""L16 Open-set recognition and unknown waveform rejection test suite (Phase 4).

Spec reference: Track F-5 (PS26147 / SIGNAL10).
"""
from __future__ import annotations
import pytest
import numpy as np

from signal9.decide.openset import OpenSetClassifier
from signal9.status import Status
from signal9.dsp.constellation import bits_to_symbols


@pytest.fixture
def classifier():
    return OpenSetClassifier(evm_reject_threshold=0.38, unknown_prob_threshold=0.5)


class TestKnownModulations:
    @pytest.mark.parametrize("mod", ["BPSK", "QPSK", "8PSK", "16QAM"])
    def test_known_modulations_accepted(self, classifier, mod):
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, 2048, dtype=np.uint8)
        syms = bits_to_symbols(bits, mod_type=mod)

        # Add modest AWGN (SNR = 22 dB)
        sigma = np.sqrt(1.0 / (2.0 * (10.0 ** 2.2)))
        noisy = syms + sigma * (rng.standard_normal(len(syms)) + 1j * rng.standard_normal(len(syms)))

        dec = classifier.classify(noisy)
        assert dec.is_known, f"Known modulation {mod} was rejected as unknown"
        assert dec.predicted_class == mod
        assert dec.unknown_probability < 0.5
        assert dec.min_evm < 0.35


class TestUnknownRejection:
    def test_lfm_chirp_rejected(self, classifier):
        # Linear Frequency Modulation (radar chirp)
        fs = 48000.0
        t = np.arange(2048) / fs
        chirp = np.exp(1j * np.pi * (20000.0 / (2048 / fs)) * (t ** 2)).astype(np.complex64)

        dec = classifier.classify(chirp)
        assert not dec.is_known, "LFM chirp was falsely accepted as known digital modulation"
        assert dec.predicted_class == "UNKNOWN"
        assert dec.status == Status.UNKNOWN_MODULATION
        assert dec.unknown_probability >= 0.5

    def test_pure_noise_rejected(self, classifier):
        rng = np.random.default_rng(99)
        noise = (rng.standard_normal(2048) + 1j * rng.standard_normal(2048)).astype(np.complex64)

        dec = classifier.classify(noise)
        assert not dec.is_known, "Pure noise was falsely accepted as known modulation"
        assert dec.predicted_class == "UNKNOWN"
        assert dec.unknown_probability >= 0.5
