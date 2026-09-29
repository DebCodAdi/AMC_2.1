"""Memory-bounded streaming iterators for large captures."""
from __future__ import annotations
from pathlib import Path
import numpy as np

from .sniff import RAW_DTYPES

def iter_raw_iq(path: str | Path, dtype_name: str, block: int = 1 << 20, offset_samples: int = 0, overlap: int = 0):
    """Yield complex64 blocks from a raw header-less file (memory-bounded)."""
    dt, bps, off, scale = RAW_DTYPES[dtype_name]
    mm = np.memmap(path, dtype=dt, mode="r")
    n_samp = len(mm) // 2
    step = max(1, block - overlap)
    for s in range(offset_samples, n_samp, step):
        e = min(s + block, n_samp)
        v = np.asarray(mm[2 * s:2 * e], dtype=np.float32)
        yield ((v[0::2] - off) + 1j * (v[1::2] - off)).astype(np.complex64) / np.float32(scale)
        if e == n_samp:
            break

def iter_wav_iq(data_mmap: np.ndarray,
                block: int = 1 << 20,
                offset_samples: int = 0,
                overlap: int = 0,
                kind: str = "iq"):
    """Yield complex64 blocks from a memory-mapped WAV array.

    ``kind`` comes from :func:`signal9.io.wav_io.read_wav` and decides how the
    channels are interpreted:

      - ``"iq"``              left = I, right = Q (already complex baseband)
      - ``"mono_audio"``      real audio -> Hilbert -> analytic signal
      - ``"mono_duplicated"`` identical channels -> take one -> Hilbert
      - ``"stereo_audio"``    genuine two-channel audio -> channel mean -> Hilbert

    The analytic conversion matters: a real audio recording handed downstream as
    ``real + 0j`` has a two-sided spectrum, so every detector sees a mirror image
    of each emission at -f and every carrier/CFO estimator is pulled toward DC.
    Earlier revisions dropped ``kind`` here and the pipeline's
    ``not np.iscomplexobj(chunk)`` guard could never fire (this generator always
    yields complex64), so the analytic path was dead and .wav audio inputs
    produced duplicated mirrored detections.
    """
    n_samp = len(data_mmap)
    if np.issubdtype(data_mmap.dtype, np.integer):
        info = np.iinfo(data_mmap.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
    else:
        scale = 1.0

    from .audio_path import audio_to_analytic

    step = max(1, block - overlap)
    for s in range(offset_samples, n_samp, step):
        e = min(s + block, n_samp)
        v = data_mmap[s:e]

        if v.ndim == 1:
            real = v.astype(np.float32) / scale
        elif kind == "iq":
            c0 = v[:, 0].astype(np.float32) / scale
            c1 = v[:, 1].astype(np.float32) / scale
            yield (c0 + 1j * c1).astype(np.complex64)
            if e == n_samp:
                break
            continue
        elif kind == "stereo_audio":
            real = (v[:, 0].astype(np.float32) + v[:, 1].astype(np.float32)) / (2.0 * scale)
        else:  # mono_duplicated or unknown multi-channel audio
            real = v[:, 0].astype(np.float32) / scale

        yield audio_to_analytic(real)
        if e == n_samp:
            break

def iter_sigmf_iq(data_mmap: np.ndarray, block: int = 1 << 20, offset_samples: int = 0, overlap: int = 0):
    """Yield complex64 blocks from SigMF mapped array."""
    n_samp = len(data_mmap)
    step = max(1, block - overlap)
    for s in range(offset_samples, n_samp, step):
        e = min(s + block, n_samp)
        yield data_mmap[s:e].astype(np.complex64)
        if e == n_samp:
            break
