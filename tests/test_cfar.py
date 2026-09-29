"""Tests 2D spectrogram CFAR detection and noise floor suppression."""
import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal9.detect.spectrogram import log_spectrogram
from signal9.detect.whiten import whiten_spectrogram
from signal9.detect.cfar import detect_cfar

class TestCFAR(unittest.TestCase):
    def test_two_emissions_detected(self):
        np.random.seed(42)
        fs = 2_000_000.0
        n = 200_000
        t = np.arange(n)
        
        # Emission 1: tone at +300 kHz
        sig1 = 0.5 * np.exp(2j * np.pi * 300_000.0 / fs * t)
        # Emission 2: burst at -400 kHz
        sig2 = np.zeros(n, dtype=np.complex64)
        sig2[n // 2:] = 0.5 * np.exp(2j * np.pi * -400_000.0 / fs * t[n // 2:])
        noise = (np.random.randn(n) + 1j * np.random.randn(n)) * 0.02

        composite = (sig1 + sig2 + noise).astype(np.complex64)
        f, time_axis, P = log_spectrogram(composite, fs=fs, nfft=1024)
        Pw, _ = whiten_spectrogram(P)
        boxes = detect_cfar(Pw, f, time_axis, thr_db=8.0)

        self.assertEqual(len(boxes), 2)

    def test_pure_noise_no_boxes(self):
        np.random.seed(42)
        fs = 1_000_000.0
        n = 100_000
        noise = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64) * 0.1
        f, time_axis, P = log_spectrogram(noise, fs=fs, nfft=1024)
        Pw, _ = whiten_spectrogram(P)
        boxes = detect_cfar(Pw, f, time_axis, thr_db=8.0)
        self.assertEqual(len(boxes), 0)

if __name__ == "__main__":
    unittest.main()
