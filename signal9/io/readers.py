"""Memory-mapped, bounded readers for raw IQ data."""
from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np
from .sniff import RAW_DTYPES

def read_iq_block(path: str | Path, fmt: str, start: int = 0, count: int = 4096) -> tuple[np.ndarray, np.ndarray | None]:
    """Memory-mapped bounded read of complex IQ samples.
    Returns (complex64 block, raw integer block or None).
    """
    path = Path(path)
    if fmt not in RAW_DTYPES:
        raise ValueError(f"Unsupported format '{fmt}'. Supported: {list(RAW_DTYPES.keys())}")

    dt, bps, offset, scale = RAW_DTYPES[fmt]
    file_bytes = path.stat().st_size
    total_samples = file_bytes // bps
    if start >= total_samples:
        return np.empty(0, dtype=np.complex64), None

    actual_count = min(count, total_samples - start)
    if actual_count <= 0:
        return np.empty(0, dtype=np.complex64), None

    if fmt in ("cf32", "cf64"):
        # 32-bit or 64-bit complex float interleaved or direct
        # RAW_DTYPES cf32 uses np.float32, 8 bytes per complex sample (I, Q as float32)
        mm = np.memmap(path, dtype=np.float32 if fmt == "cf32" else np.float64, mode="r")
        raw = np.asarray(mm[2 * start: 2 * (start + actual_count)])
        x = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
        return x, None
    else:
        # Integer formats (cu8, cs8, cs16)
        mm = np.memmap(path, dtype=dt, mode="r")
        raw_ints = np.asarray(mm[2 * start: 2 * (start + actual_count)])
        raw_ints = raw_ints[: raw_ints.size // 2 * 2]
        x_f = (raw_ints.astype(np.float32) - offset) / scale
        x_c = (x_f[0::2] + 1j * x_f[1::2]).astype(np.complex64)
        return x_c, raw_ints

def iter_raw_iq(path: str | Path, fmt: str, block: int = 65536) -> Iterator[np.ndarray]:
    """Generator yielding complex64 blocks from a raw IQ file using memmap."""
    path = Path(path)
    if fmt not in RAW_DTYPES:
        raise ValueError(f"Unknown format {fmt}")
    dt, bps, offset, scale = RAW_DTYPES[fmt]
    total_samples = path.stat().st_size // bps
    for start in range(0, total_samples, block):
        chunk, _ = read_iq_block(path, fmt, start, block)
        if len(chunk) == 0:
            break
        yield chunk
