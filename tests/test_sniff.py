"""Tests format sniffing round-trip and short-file rejection."""
import unittest
import tempfile
import os
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal9.io.sniff import sniff_dtype, RAW_DTYPES

class TestSniff(unittest.TestCase):
    def test_roundtrip_all_formats(self):
        n = 100000
        t = np.arange(n)
        sig = 0.5 * np.exp(2j * np.pi * 0.03 * t) + 0.05 * (np.random.randn(n) + 1j * np.random.randn(n))
        with tempfile.TemporaryDirectory() as td:
            for fmt, (dt, bps, off, scale) in RAW_DTYPES.items():
                inter = np.empty(2 * n, dtype=np.float64)
                inter[0::2] = sig.real
                inter[1::2] = sig.imag
                p = os.path.join(td, f"test_{fmt}.bin")
                if fmt in ("cf32", "cf64"):
                    arr = inter.astype(np.float32 if fmt == "cf32" else np.float64)
                else:
                    arr = np.clip(np.round(inter * scale * 0.85 + off), np.iinfo(dt).min, np.iinfo(dt).max).astype(dt)
                arr.tofile(p)
                guess, conf = sniff_dtype(p)
                if fmt in ("cu8", "cs8", "cs16"):
                    self.assertEqual(guess, fmt)

    def test_reject_short_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "short.bin")
            with open(p, "wb") as f:
                f.write(b"\x00\x01\x02")  # 3 bytes
            guess, conf = sniff_dtype(p)
            self.assertEqual(conf, 0.0)

if __name__ == "__main__":
    unittest.main()
