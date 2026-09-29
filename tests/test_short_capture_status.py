"""Short captures must report an explicit failure, never a silent no-signal."""
from pathlib import Path
import numpy as np

from signal9.pipeline import Signal9Engine


def test_file_below_detection_minimum_is_explicit(tmp_path):
    capture = tmp_path / "tiny.iq"
    np.zeros(26, dtype=np.complex64).tofile(capture)
    report = Signal9Engine(checkpoint=None).process_file(capture, fs=4800, fmt="cf32", exploit=False)
    assert report["status"] == "INSUFFICIENT_SAMPLES"
    assert report["total_signals_detected"] == 0
