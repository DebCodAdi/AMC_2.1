"""Tests random impairment augmentation chain."""
import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal9.augment import random_chain

class TestAugment(unittest.TestCase):
    def test_random_chain_invariants(self):
        rng = np.random.default_rng(42)
        fs = 12000.0
        n = 4096
        x = np.exp(2j * np.pi * 500.0 / fs * np.arange(n)).astype(np.complex64)

        for _ in range(50):
            band = rng.choice(["HF", "VHF", "UHF"])
            aug = random_chain(x, fs=fs, sig_bw_hz=1000.0, band=band, rng=rng)
            self.assertEqual(aug.dtype, np.complex64)
            self.assertEqual(len(aug), n)
            self.assertTrue(np.all(np.isfinite(aug)))
            p = float(np.mean(np.abs(aug) ** 2))
            self.assertAlmostEqual(p, 1.0, places=4)

if __name__ == "__main__":
    unittest.main()
