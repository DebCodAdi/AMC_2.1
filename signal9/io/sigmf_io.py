"""SigMF reader and writer according to SigMF specifications."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from .readers import read_iq_block

SIGMF_DTYPE_MAP = {
    "cf32_le": "cf32",
    "cf64_le": "cf64",
    "ci16_le": "cs16",
    "ci8": "cs8",
    "cu8": "cu8",
}

REV_SIGMF_DTYPE_MAP = {
    "cf32": "cf32_le",
    "cf64": "cf64_le",
    "cs16": "ci16_le",
    "cs8": "ci8",
    "cu8": "cu8",
}

def read_sigmf(path_stem: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a SigMF dataset (.sigmf-meta + .sigmf-data)."""
    p = Path(path_stem)
    if p.name.endswith(".sigmf-meta") or p.name.endswith(".sigmf-data"):
        stem = p.parent / p.stem
    else:
        stem = p

    meta_file = stem.with_suffix(".sigmf-meta")
    data_file = stem.with_suffix(".sigmf-data")

    if not meta_file.exists():
        raise FileNotFoundError(f"SigMF metadata file not found: {meta_file}")
    if not data_file.exists():
        raise FileNotFoundError(f"SigMF data file not found: {data_file}")

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    global_meta = meta.get("global", {})
    datatype = global_meta.get("core:datatype", "cf32_le")
    raw_fmt = SIGMF_DTYPE_MAP.get(datatype, "cf32")

    iq, _ = read_iq_block(data_file, raw_fmt, start=0, count=10000000)
    return iq, meta

def write_sigmf(path_stem: str | Path, iq: np.ndarray, fs: float, fc: float = 0.0,
                 annotations: list[dict] | None = None) -> None:
    """Write complex IQ array to .sigmf-meta and .sigmf-data."""
    stem = Path(path_stem)
    data_file = stem.with_suffix(".sigmf-data")
    meta_file = stem.with_suffix(".sigmf-meta")

    iq_c64 = iq.astype(np.complex64)
    # Write interleaved float32 I, Q
    raw = np.empty(2 * len(iq_c64), dtype=np.float32)
    raw[0::2] = iq_c64.real
    raw[1::2] = iq_c64.imag
    raw.tofile(data_file)

    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": float(fs),
            "core:version": "1.0.0",
            "core:recorder": "Signal9-AMC",
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": float(fc),
            }
        ],
        "annotations": annotations or []
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
