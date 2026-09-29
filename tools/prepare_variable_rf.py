#!/usr/bin/env python3
"""Make the variable-length RF corpus self-describing and auditable.

The source file contains valid IQ/label/SNR arrays but no class-name table.
This tool copies it without changing row order or numeric labels, adds the
authoritative 25-class table, and validates every group.  The source is never
modified; the destination must be a new path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

CLASS_NAMES = [
    "BPSK", "QPSK", "8PSK", "16PSK", "32PSK", "16QAM", "32QAM",
    "64QAM", "128QAM", "256QAM", "OOK", "4ASK", "8ASK", "PAM4",
    "2FSK", "4FSK", "8FSK", "CPFSK", "GFSK", "MSK", "AM-DSB",
    "AM-SSB", "FM", "WBFM", "NOISE",
]
EXPECTED = {"len_128": 128, "len_256": 256, "len_512": 512}


def copy_group(src: h5py.Group, dst: h5py.Group, length: int, chunk_rows: int) -> None:
    iq = src["iq"]
    mod = src["mod"]
    snr = src["snr"]
    if iq.shape != (5_000_000, 2, length):
        raise ValueError(f"{src.name}/iq has unexpected shape {iq.shape}")
    if mod.shape != (iq.shape[0],) or snr.shape != (iq.shape[0],):
        raise ValueError(f"{src.name}: IQ/mod/SNR row counts do not match")
    if int(mod[:].min()) < 0 or int(mod[:].max()) >= len(CLASS_NAMES):
        raise ValueError(f"{src.name}/mod contains labels outside 0..24")
    out_iq = dst.create_dataset("iq", shape=iq.shape, dtype=np.float16,
                                chunks=(min(chunk_rows, iq.shape[0]), 2, length),
                                compression="lzf")
    out_mod = dst.create_dataset("mod", shape=mod.shape, dtype=np.int16,
                                 chunks=(min(chunk_rows, mod.shape[0]),), compression="lzf")
    out_snr = dst.create_dataset("snr", shape=snr.shape, dtype=np.float16,
                                 chunks=(min(chunk_rows, snr.shape[0]),), compression="lzf")
    for start in range(0, iq.shape[0], chunk_rows):
        end = min(start + chunk_rows, iq.shape[0])
        x = np.asarray(iq[start:end])
        y = np.asarray(mod[start:end])
        z = np.asarray(snr[start:end])
        if not np.isfinite(x).all() or not np.isfinite(z).all():
            raise ValueError(f"{src.name}: non-finite values in rows {start}:{end}")
        out_iq[start:end] = x
        out_mod[start:end] = y
        out_snr[start:end] = z
    dst.attrs["class_names"] = np.asarray(CLASS_NAMES, dtype=h5py.string_dtype())
    dst.attrs["label_order"] = "variable-rf-25-v1"
    dst.attrs["source_rows_preserved"] = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("destination", type=Path)
    ap.add_argument("--chunk-rows", type=int, default=8192)
    args = ap.parse_args()
    if args.source.resolve() == args.destination.resolve():
        raise ValueError("Destination must be different from source")
    if args.destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {args.destination}")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.source, "r") as src, h5py.File(args.destination, "w") as dst:
        dst.attrs["format"] = "variable-rf-corpus-v1"
        dst.attrs["class_names"] = np.asarray(CLASS_NAMES, dtype=h5py.string_dtype())
        dst.attrs["source_file"] = str(args.source)
        for group, length in EXPECTED.items():
            if group not in src:
                raise KeyError(f"Missing group {group}")
            copy_group(src[group], dst.create_group(group), length, args.chunk_rows)
    print(f"Wrote validated ordered corpus: {args.destination}")
    print(f"Rows preserved: 5,000,000 per group; classes: {len(CLASS_NAMES)}")


if __name__ == "__main__":
    main()
