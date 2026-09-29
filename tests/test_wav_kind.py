"""Tests WAV kind identification (mono audio vs dual-mono vs true stereo IQ)."""
import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal9.io.wav_io import wav_kind

class TestWavKind(unittest.TestCase):
    def test_mono_audio(self):
        mono = np.random.randn(1000)
        self.assertEqual(wav_kind(mono), "mono_audio")

    def test_dual_mono(self):
        mono = np.random.randn(1000)
        dual = np.stack([mono, mono], axis=1)
        self.assertEqual(wav_kind(dual), "mono_duplicated")

    def test_true_iq(self):
        i = np.random.randn(2000)
        q = np.random.randn(2000)
        iq = np.stack([i, q], axis=1)
        self.assertEqual(wav_kind(iq), "iq")

if __name__ == "__main__":
    unittest.main()
