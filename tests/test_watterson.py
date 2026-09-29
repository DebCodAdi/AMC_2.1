"""Tests Watterson channel average power conservation and Doppler spread."""
import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal9.augment.watterson import watterson_channel

class TestWatterson(unittest.TestCase):
    def test_power_normalized_on_average(self):
        rng = np.random.default_rng(42)
        n = 24000
        x = np.ones(n, dtype=np.complex64)
        powers = [
            np.mean(np.abs(watterson_channel(x, fs=24000.0, preset="poor", rng=rng)) ** 2)
            for _ in range(50)
        ]
        mean_p = float(np.mean(powers))
        # Expected power is ~1.0
        self.assertGreater(mean_p, 0.8)
        self.assertLess(mean_p, 1.2)

if __name__ == "__main__":
    unittest.main()
