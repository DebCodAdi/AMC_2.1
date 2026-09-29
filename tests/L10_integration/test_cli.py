"""L10 Integration tier — CLI and End-to-End Pipeline Smoke Tests.

Verifies:
  - CLI parser flags (--fs, --fc, --format, --json, --save-json).
  - Clean error handling on missing / invalid files without unhandled traceback.
  - End-to-end execution producing valid JSON output.

Spec reference: §2.4 (L10 tier), §6.1 (PS26147).
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import pytest
import numpy as np


class TestL10CLIIntegration:
    """CLI end-to-end integration tests."""

    def test_cli_help(self):
        """CLI --help must return 0."""
        cmd = [sys.executable, "main.py", "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "Signal 9" in res.stdout

    def test_cli_missing_file_exit_clean(self):
        """CLI on non-existent file must exit with non-zero and clean error message."""
        cmd = [sys.executable, "main.py", "non_existent_file_12345.iq"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode != 0
        assert "Traceback" not in res.stderr
        assert "non_existent_file_12345.iq" in res.stderr
        assert "Error" in res.stderr or "not found" in res.stderr.lower()

    def test_cli_synthetic_raw_iq(self, tmp_path: Path):
        """CLI processing a synthetic raw IQ file with --json outputs valid JSON."""
        test_file = tmp_path / "test_synth_cf32.iq"
        # Generate 4096 complex64 samples
        rng = np.random.default_rng(42)
        samples = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(np.complex64)
        samples.tofile(str(test_file))

        cmd = [
            sys.executable,
            "main.py",
            str(test_file),
            "--fs", "1000000",
            "--format", "cf32",
            "--json"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        data = json.loads(res.stdout)
        assert "sample_rate_sps" in data
        assert "health" in data
        assert data["sample_rate_sps"] == 1000000.0
