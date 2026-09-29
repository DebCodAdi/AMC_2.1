"""multimon-ng POCSAG/FLEX verification wrapper."""
from __future__ import annotations
import subprocess
from pathlib import Path
from .common import DecoderResult, write_temp_wav

def verify_multimon(iq: np.ndarray, fs: float, tmp_dir: Path) -> DecoderResult:
    wav_path = write_temp_wav(iq, fs, tmp_dir / "multimon.wav")
    try:
        res = subprocess.run(["multimon-ng", "-a", "POCSAG512", "-a", "POCSAG1200", "-a", "POCSAG2400", "-a", "FLEX", str(wav_path)], capture_output=True, text=True, timeout=5)
        if "POCSAG" in res.stdout:
            return DecoderResult(verified=True, mode="POCSAG", confidence=1.0)
        elif "FLEX" in res.stdout:
            return DecoderResult(verified=True, mode="FLEX", confidence=1.0)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return DecoderResult(verified=False)
