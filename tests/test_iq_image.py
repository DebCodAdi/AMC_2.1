"""IQ imbalance and BPSK protection guard tests.
Fixes A2, A3: Ensures closed-form tap works and BPSK is never destroyed.
"""
import unittest
import numpy as np
from scipy import signal as sps
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal9.health.iq_image import image_evidence_db, maybe_correct_iq_imbalance, iq_imbalance_correct

class TestIQImage(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_exact_iq_correction_wideband(self):
        # Generate an off-center tone with synthetic amplitude & phase imbalance
        n = np.arange(65536)
        tone = np.exp(2j * np.pi * 0.1 * n)
        noise = (self.rng.standard_normal(len(n)) + 1j * self.rng.standard_normal(len(n))) * 1e-3
        
        # Apply 1.5 dB amplitude imbalance, 8 deg phase imbalance
        g = 10.0 ** (1.5 / 20.0)
        phi = np.deg2rad(8.0)
        k1 = (1.0 + g * np.exp(-1j * phi)) / 2.0
        k2 = (1.0 - g * np.exp(1j * phi)) / 2.0
        distorted = (k1 * tone + k2 * np.conj(tone) + noise).astype(np.complex64)

        ev_before = image_evidence_db(distorted)
        self.assertIsNotNone(ev_before)
        improp_before, _ = ev_before
        self.assertLess(improp_before, 20.0)  # Significant image present

        # Correct using closed-form one-tap wideband corrector
        corrected, w = maybe_correct_iq_imbalance(distorted, fs=1.0, is_wideband=True)
        self.assertIsNotNone(w)

        ev_after = image_evidence_db(corrected)
        self.assertIsNotNone(ev_after)
        improp_after, _ = ev_after
        self.assertGreater(improp_after, improp_before + 25.0)

    def test_never_correct_channelized_bpsk(self):
        # Centered BPSK is naturally improper: E[s^2] != 0
        n = 65536
        bits = self.rng.integers(0, 2, n) * 2 - 1
        bpsk = (bits * np.exp(1j * 0.4)).astype(np.complex64)
        
        # Low-pass filter (pulse shaping)
        b = sps.firwin(65, 0.1)
        bpsk_shaped = sps.lfilter(b, 1, bpsk).astype(np.complex64)
        power_before = np.mean(np.abs(bpsk_shaped) ** 2)

        # Ensure gate REFUSES correction when is_wideband=False
        out, w = maybe_correct_iq_imbalance(bpsk_shaped, fs=1.0, is_wideband=False)
        self.assertIsNone(w)
        power_after = np.mean(np.abs(out) ** 2)
        self.assertAlmostEqual(power_before, power_after, places=5)

if __name__ == "__main__":
    unittest.main()
