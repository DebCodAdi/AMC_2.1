"""AIS-catcher verification wrapper."""
from __future__ import annotations
import subprocess
from pathlib import Path
from .common import DecoderResult, write_temp_wav

def verify_aiscatcher(iq: np.ndarray, fs: float, tmp_dir: Path) -> DecoderResult:
    wav_path = write_temp_wav(iq, fs, tmp_dir / "ais.wav")
    try:
        res = subprocess.run(["AIS-catcher", "-r", str(wav_path)], capture_output=True, text=True, timeout=5)
        if "AIVDM" in res.stdout or "!AIVDO" in res.stdout:
            return DecoderResult(verified=True, mode="AIS", confidence=1.0)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return DecoderResult(verified=False)
