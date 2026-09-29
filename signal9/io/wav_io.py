"""WAV audio and IQ ingestion."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert

def wav_kind(channels: np.ndarray) -> str:
    """Classify 2-channel WAV as either 'iq' or 'stereo_audio'/'mono_duplicated'."""
    if channels.ndim == 1 or channels.shape[1] == 1:
        return "mono_audio"
    c0 = channels[:, 0].astype(np.float64)
    c1 = channels[:, 1].astype(np.float64)
    diff = np.mean(np.abs(c0 - c1))
    std0 = np.std(c0)
    if std0 < 1e-9 or (diff / (std0 + 1e-9)) < 1e-4:
        return "mono_duplicated"
    # In genuine IQ, cross-correlation at lag 0 is typically small, variance similar
    p0, p1 = np.var(c0) + 1e-12, np.var(c1) + 1e-12
    balance = abs(np.log10(p0 / p1))
    if balance > 1.0:  # power differs by >10x
        return "stereo_audio"
    return "iq"

def read_wav(path: str | Path) -> tuple[np.ndarray, int, str]:
    """Read a WAV file and return (memory mapped array, sample_rate, kind).
    Note: For large mono audio, Hilbert is expensive on the full array.
    """
    path = Path(path)
    fs, data = wavfile.read(path, mmap=True)
    
    # Kind detection needs a small slice to avoid loading whole array
    probe = data[: min(len(data), 1 << 18)]
    if probe.ndim == 1:
        return data, fs, "mono_audio"

    # Scale probe for kind detection
    if np.issubdtype(probe.dtype, np.integer):
        info = np.iinfo(probe.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        probe_f = probe.astype(np.float32) / scale
    else:
        probe_f = probe.astype(np.float32)

    kind = wav_kind(probe_f)
    return data, fs, kind
