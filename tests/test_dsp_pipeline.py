"""End-to-end signal processing and model integration test."""
import unittest
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal9.pipeline import Signal9Engine
from signal9.io.sigmf_io import write_sigmf

class TestDSPPipeline(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_full_pipeline_multi_emitter(self):
        fs = 2_000_000.0  # 2 MS/s
        n_samples = 200_000
        n = np.arange(n_samples)
        
        # Emitter 1: Continuous tone at +300 kHz
        sig1 = 0.5 * np.exp(2j * np.pi * 300_000 / fs * n)
        # Emitter 2: Bursty tone at -400 kHz in second half
        sig2 = np.zeros(n_samples, dtype=np.complex64)
        sig2[n_samples // 2:] = 0.4 * np.exp(2j * np.pi * -400_000 / fs * n[n_samples // 2:])
        # AWGN noise
        noise = (self.rng.standard_normal(n_samples) + 1j * self.rng.standard_normal(n_samples)) * 0.02
        
        composite = (sig1 + sig2 + noise).astype(np.complex64)

        with tempfile.TemporaryDirectory() as td:
            sigmf_stem = os.path.join(td, "test_capture")
            write_sigmf(sigmf_stem, composite, fs=fs, fc=144_000_000.0)

            engine = Signal9Engine()
            report = engine.process_file(f"{sigmf_stem}.sigmf-meta")

            self.assertIn("signals", report)
            self.assertGreaterEqual(report["total_signals_detected"], 1)
            
            first_sig = report["signals"][0]
            self.assertIn("classification", first_sig)
            cls = first_sig["classification"]
            self.assertIn("l1", cls)
            self.assertIn("l2", cls)
            self.assertIn("l3", cls)
            self.assertIn("decision_state", cls["l1"])
            self.assertIn("prediction", cls["l1"])

if __name__ == "__main__":
    unittest.main()
