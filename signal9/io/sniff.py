"""Blind format sniffing for headerless IQ captures."""
from __future__ import annotations
from pathlib import Path
import numpy as np

RAW_DTYPES = {  # name -> (numpy dtype, bytes per complex sample, offset, scale)
    "cu8": (np.uint8, 2, 127.5, 127.5),
    "cs8": (np.int8, 2, 0.0, 128.0),
    "cs16": (np.int16, 4, 0.0, 32768.0),
    "cf32": (np.float32, 8, 0.0, 1.0),
    "cf64": (np.float64, 16, 0.0, 1.0),
}

def _smoothness(x: np.ndarray) -> float:
    """Computes temporal correlation smoothness metric.
    Under correct framing, oversampled SDR captures exhibit temporal correlation (smoothness > 0).
    Under wrong byte alignment or interpretation, samples look like white noise (smoothness near 0).
    """
    if len(x) < 4:
        return -1.0
    xf = x.astype(np.float64)
    xf = xf - np.mean(xf)
    s = np.std(xf)
    if s < 1e-12:
        return 0.0
    diff_s = np.std(np.diff(xf))
    return float(1.0 - diff_s / (np.sqrt(2.0) * s))

def sniff_dtype(path: str | Path, nbytes: int = 1 << 20) -> tuple[str, float]:
    """Blind guess of raw IQ sample format -> (name, confidence 0..1).
    Heuristic: under the correct interpretation, adjacent samples are correlated;
    wrong interpretation scrambles bytes and looks white.
    """
    path = Path(path)
    ext = path.suffix.lower().lstrip(".")
    if ext in RAW_DTYPES:
        return ext, 1.0
    if ext == "cfile":
        return "cf32", 1.0

    raw = np.fromfile(path, dtype=np.uint8, count=nbytes)
    if len(raw) < 64:
        return "cf32", 0.0

    scores: dict[str, float] = {}
    
    # cu8: unsigned 8-bit, mean around 127.5
    cu8_sm = _smoothness(raw[0::2])
    cu8_mean = float(raw.mean())
    cu8_score = cu8_sm + (0.3 if (abs(cu8_mean - 127.5) < 12 and cu8_sm > 0.05) else -0.6)
    scores["cu8"] = cu8_score

    # cs8: signed 8-bit, mean around 0
    cs8_view = raw.view(np.int8)
    cs8_sm = _smoothness(cs8_view[0::2])
    cs8_mean = float(cs8_view.mean())
    cs8_score = cs8_sm + (0.3 if (abs(cs8_mean) < 12 and cs8_sm > 0.05) else -0.6)
    scores["cs8"] = cs8_score

    # cs16: signed 16-bit
    n16 = (raw.size // 4) * 4
    if n16 >= 16:
        cs16_view = raw[:n16].view(np.int16)
        scores["cs16"] = _smoothness(cs16_view[0::2])
    else:
        scores["cs16"] = -1.0

    # cf32: 32-bit float
    if n16 >= 16:
        f32_view = raw[:n16].view(np.float32)
        ok = np.all(np.isfinite(f32_view)) and (np.max(np.abs(f32_view)) < 1e4) and (np.median(np.abs(f32_view)) > 1e-7)
        scores["cf32"] = _smoothness(f32_view[0::2]) if ok else -2.0
    else:
        scores["cf32"] = -2.0

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, s1 = ranked[0]
    _, s2 = ranked[1]
    conf = float(np.clip((s1 - s2) * 2.0, 0.0, 1.0))
    return best, conf

def sniff_raw_dtype(path: str | Path, probe_bytes: int = 1 << 20) -> str:
    """Convenience wrapper returning the best guessed dtype string."""
    best, _ = sniff_dtype(path, probe_bytes)
    return best
