"""Format sniffing and memory-mapped bounded read tests."""
import unittest
import tempfile
import os
from pathlib import Path
import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal9.io.sniff import sniff_dtype, RAW_DTYPES
from signal9.io.readers import read_iq_block, iter_raw_iq

class TestReaders(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_sniff_and_read_all_raw_formats(self):
        n_samples = 100000
        n = np.arange(n_samples)
        # Continuous tone + noise has high temporal correlation
        sig = 0.5 * np.exp(2j * np.pi * 0.02 * n) + 0.05 * (self.rng.standard_normal(n_samples) + 1j * self.rng.standard_normal(n_samples))

        with tempfile.TemporaryDirectory() as td:
            for fmt_name, (dt, bps, offset, scale) in RAW_DTYPES.items():
                inter = np.empty(2 * n_samples, dtype=np.float64)
                inter[0::2] = sig.real
                inter[1::2] = sig.imag

                file_path = os.path.join(td, f"capture_{fmt_name}.bin")
                if fmt_name in ("cf32", "cf64"):
                    arr = inter.astype(np.float32 if fmt_name == "cf32" else np.float64)
                else:
                    arr = np.clip(np.round(inter * scale * 0.85 + offset), np.iinfo(dt).min, np.iinfo(dt).max).astype(dt)
                arr.tofile(file_path)

                # Test blind sniffing
                guessed_fmt, conf = sniff_dtype(file_path)
                if fmt_name in ("cu8", "cs8", "cs16"):
                    self.assertEqual(guessed_fmt, fmt_name, f"Failed sniffing {fmt_name}, got {guessed_fmt}")

                # Test bounded read
                block, raw_int = read_iq_block(file_path, fmt=fmt_name, start=100, count=4096)
                self.assertEqual(len(block), 4096)
                self.assertEqual(block.dtype, np.complex64)
                self.assertTrue(np.isfinite(block).all())

                # Test generator
                gen = iter_raw_iq(file_path, fmt=fmt_name, block=16384)
                first_chunk = next(gen)
                self.assertEqual(len(first_chunk), 16384)

if __name__ == "__main__":
    unittest.main()
