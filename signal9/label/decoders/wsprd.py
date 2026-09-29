"""WSPRd verification wrapper."""
from __future__ import annotations
import subprocess
from pathlib import Path
from .common import DecoderResult, write_temp_wav

def verify_wsprd(iq: np.ndarray, fs: float, tmp_dir: Path) -> DecoderResult:
    wav_path = write_temp_wav(iq, fs, tmp_dir / "wspr.wav")
    try:
        res = subprocess.run(["wsprd", str(wav_path)], capture_output=True, text=True, timeout=5)
        if len(res.stdout.strip().splitlines()) > 0: # wsprd outputs decoded lines
            return DecoderResult(verified=True, mode="WSPR", confidence=1.0)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return DecoderResult(verified=False)
