"""Pathology and hardware flaw robustness tests (Fixes G8)."""
import unittest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal9.health.clipping import clipping_fraction
from signal9.health.dc_lo import detect_dc_spike, mask_dc_spike
from signal9.features.cumulants import cumulant_features
from signal9.features.spectral import occupied_bandwidth

class TestPathologies(unittest.TestCase):
    def test_all_zero_signal(self):
        zeros = np.zeros(8192, dtype=np.complex64)
        # Should not throw DivisionByZero or NaN
        c = cumulant_features(zeros)
        self.assertEqual(len(c), 5)
        self.assertTrue(np.all(np.isfinite(c)))

        bw, lo, hi, snr = occupied_bandwidth(zeros, 100000.0)
        self.assertEqual(bw, 0.0)

    def test_heavy_clipping(self):
        # Heavy ADC saturation on 16-bit integers
        raw_saturated = np.full(4096, 32767, dtype=np.int16)
        clip_frac = clipping_fraction(raw_saturated, "cs16")
        self.assertAlmostEqual(clip_frac, 1.0)

    def test_massive_dc_spike(self):
        fs = 1_000_000.0
        n = np.arange(65536)
        noise = (np.random.randn(len(n)) + 1j * np.random.randn(len(n))).astype(np.complex64) * 0.1
        spike = 10.0 + 0j  # Massive DC offset
        sig_with_dc = noise + spike

        has_spike = detect_dc_spike(sig_with_dc, fs, threshold_db=12.0)
        self.assertTrue(has_spike)

if __name__ == "__main__":
    unittest.main()
