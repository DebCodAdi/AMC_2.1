"""rtl_433 ISM verification wrapper."""
from __future__ import annotations
import subprocess
from pathlib import Path
from .common import DecoderResult, write_temp_wav

def verify_rtl433(iq: np.ndarray, fs: float, tmp_dir: Path) -> DecoderResult:
    """Run rtl_433 for ISM band signal verification."""
    wav_path = write_temp_wav(iq, fs, tmp_dir / "rtl433.wav")
    try:
        res = subprocess.run(["rtl_433", "-r", str(wav_path)], capture_output=True, text=True, timeout=5)
        if "model" in res.stdout:
            return DecoderResult(verified=True, mode="ISM_OOK_REMOTE", confidence=1.0)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return DecoderResult(verified=False)
