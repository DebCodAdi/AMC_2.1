"""Tests occupied bandwidth estimation accuracy at low and high SNR."""
import unittest
import numpy as np
from scipy import signal as sps
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal9.features.spectral import occupied_bandwidth

class TestBandwidth(unittest.TestCase):
    def test_qpsk_bandwidth_low_snr(self):
        fs = 100000.0
        n = 200000
        rng = np.random.default_rng(42)
        # 10 kHz QPSK signal
        syms = rng.choice([-1, 1], n // 10) + 1j * rng.choice([-1, 1], n // 10)
        filt = sps.firwin(129, 5000.0, fs=fs)
        q = sps.lfilter(filt, 1, np.repeat(syms, 10))
        q = q / np.sqrt(np.mean(np.abs(q) ** 2))

        for snr in [30, 15, 6]:
            noise = (rng.standard_normal(len(q)) + 1j * rng.standard_normal(len(q))) * (10.0 ** (-snr / 20.0))
            bw, lo, hi, snr_est = occupied_bandwidth(q + noise, fs)
            # QPSK 10 kHz should measure between 7 kHz and 13 kHz even at 6 dB SNR
            self.assertGreater(bw, 7000.0)
            self.assertLess(bw, 13000.0)

    def test_narrow_cw(self):
        fs = 12000.0
        n = 120000
        t = np.arange(n)
        cw = np.exp(2j * np.pi * 700.0 / fs * t) * (t % 6000 < 3000)
        bw, lo, hi, _ = occupied_bandwidth(cw, fs)
        self.assertLess(bw, 300.0)
        self.assertGreater(lo, 500.0)
        self.assertLess(hi, 900.0)

if __name__ == "__main__":
    unittest.main()
