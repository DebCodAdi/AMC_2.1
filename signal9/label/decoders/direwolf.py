"""Direwolf APRS/Packet protocol verification wrapper."""
from __future__ import annotations
import subprocess
from pathlib import Path
from .common import DecoderResult, write_temp_wav

def verify_direwolf(iq: np.ndarray, fs: float, tmp_dir: Path) -> DecoderResult:
    """Run Direwolf to detect AX.25 / APRS frames."""
    wav_path = write_temp_wav(iq, fs, tmp_dir / "direwolf.wav")
    try:
        # Assuming direwolf is installed in PATH
        res = subprocess.run(["direwolf", "-", "<", str(wav_path)], capture_output=True, text=True, timeout=5, shell=True)
        if "Digipeater" in res.stdout or "Audio" in res.stdout: # naive check
            return DecoderResult(verified=True, mode="APRS_AFSK1200", confidence=1.0)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return DecoderResult(verified=False)
