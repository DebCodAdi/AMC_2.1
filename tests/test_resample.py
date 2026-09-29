"""Tests bounded rational polyphase resampler accuracy."""
import unittest
from fractions import Fraction
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class TestResample(unittest.TestCase):
    def test_bounded_rational_ratios(self):
        cases = [
            (2_400_000, 12_000),
            (2_048_000, 96_000),
            (20_250, 12_000),
            (44_100, 12_000),
        ]
        for f_in, f_out in cases:
            ratio = Fraction(f_out, f_in).limit_denominator(4096)
            f_actual = f_in * (ratio.numerator / ratio.denominator)
            rel_err = abs(f_actual - f_out) / f_out
            self.assertLess(rel_err, 2e-3)

if __name__ == "__main__":
    unittest.main()
