"""DSD-FME DMR/NXDN/P25 verification wrapper."""
from __future__ import annotations
import subprocess
from pathlib import Path
from .common import DecoderResult, write_temp_wav

def verify_dsdfme(iq: np.ndarray, fs: float, tmp_dir: Path) -> DecoderResult:
    wav_path = write_temp_wav(iq, fs, tmp_dir / "dsd.wav")
    try:
        res = subprocess.run(["dsd-fme", "-i", str(wav_path)], capture_output=True, text=True, timeout=5)
        out = res.stdout + res.stderr
        if "DMR" in out:
            return DecoderResult(verified=True, mode="DMR", confidence=1.0)
        elif "NXDN" in out:
            return DecoderResult(verified=True, mode="NXDN", confidence=1.0)
        elif "P25" in out:
            return DecoderResult(verified=True, mode="P25_P1", confidence=1.0)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return DecoderResult(verified=False)
