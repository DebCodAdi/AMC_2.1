"""Generic decoder runners for VHF/UHF and HF protocols."""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
import numpy as np

def run_external_decoder(cmd_args: list[str], timeout_sec: float = 10.0) -> bool:
    """Execute external CLI decoder tool and verify if valid frames were produced."""
    bin_name = cmd_args[0]
    if not shutil.which(bin_name):
        return False
    try:
        res = subprocess.run(cmd_args, capture_output=True, text=True, timeout=timeout_sec)
        return res.returncode == 0 and len(res.stdout.strip()) > 0
    except Exception:
        return False
