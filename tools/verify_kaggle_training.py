#!/usr/bin/env python3
"""Fail-fast verification for the repaired Kaggle AMC trainer and datasets."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import h5py
import numpy as np


EXPECTED_VARIABLE_CLASSES = [
    "BPSK", "QPSK", "8PSK", "16PSK", "32PSK", "16QAM", "32QAM",
    "64QAM", "128QAM", "256QAM", "OOK", "4ASK", "8ASK", "PAM4",
    "2FSK", "4FSK", "8FSK", "CPFSK", "GFSK", "MSK", "AM-DSB",
    "AM-SSB", "FM", "WBFM", "NOISE",
]

EXPECTED_SIG53_CLASSES = [
    "BPSK", "QPSK", "8PSK", "16PSK", "32PSK", "QAM16", "QAM32",
    "QAM64", "QAM128", "QAM256", "2FSK", "4FSK", "8FSK", "16FSK",
    "GFSK", "MSK", "GMSK", "OQPSK", "DQPSK", "D8PSK", "PI4QPSK",
    "AM-DSB", "AM-SSB", "AM-SC", "FM", "WFM", "OFDM", "DSSS",
    "LORA", "CHIRP", "NOISE", "UNKNOWN",
]


def text_values(value) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in value]


def verify_script(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name == "train_satvik_grand_corpus_dual_gpu"), None)
    if fn is None:
        raise AssertionError("training function not found")
    epoch_loop = next((n for n in ast.walk(fn) if isinstance(n, ast.For)
                       and isinstance(n.target, ast.Name) and n.target.id == "epoch"), None)
    if epoch_loop is None:
        raise AssertionError("epoch loop not found")
    loop_calls = [n.func.id for n in ast.walk(epoch_loop)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    if "evaluate_holdout" not in loop_calls:
        raise AssertionError("validation is not inside the epoch loop")
    required = [
        "blocked-group-session-80-train-10-cal-10-val-v4",
        "finite-component-median+huber-kurtosis3.2+rms-v1",
        "variable-rf-excludes-parent-sig53-v1",
        "_balanced_hardware_group_partitions",
        "condition_classifier_batch(iq_batch)",
        "ema_heads",
        "ema_arcface",
        "expert11-native-paprdb-v3",
        "H5_SPLIT_GROUP_ROWS = 64",
        "--synthetic-fraction",
        "--input-length",
        "resolve_kiwi_sigmf",
        "--arcface-weight",
        "--architecture",
        "ResearchHybridXCiTBackbone",
        "temperature-scaled-energy-v1",
        'cal_auroc >= 0.65',
        "arcface(emb[valid_mod], t_mod[valid_mod])",
        't_mod != MOD_TO_IDX["NOISE"]',
        '"8FSK", "16FSK"' not in source,
        "TORCHSIG_NARROWBAND_CLASSES" not in source,
    ]
    for marker in required[:-2]:
        if marker not in source:
            raise AssertionError(f"missing repaired trainer marker: {marker}")
    if required[-2] is not True:
        raise AssertionError("8FSK is still present in the Sig53 skip set")
    if required[-1] is not True:
        raise AssertionError("trainer still contains an assumed Sig53 fallback class table")
    if '("classes-fixed.json", "classes-fixed.txt")' not in source:
        raise AssertionError("RadioML does not require a corrected class-order sidecar")
    print(f"[OK] trainer: {path}")


def verify_label_vector(dataset: h5py.Dataset, class_count: int, name: str) -> np.ndarray:
    counts = np.zeros(class_count, dtype=np.int64)
    chunk = 1_000_000
    for start in range(0, len(dataset), chunk):
        labels = np.asarray(dataset[start:start + chunk], dtype=np.int64)
        if labels.ndim != 1:
            raise AssertionError(f"{name} labels are not a vector: {labels.shape}")
        if labels.size and (labels.min() < 0 or labels.max() >= class_count):
            raise AssertionError(f"{name} label outside 0..{class_count - 1}")
        counts += np.bincount(labels, minlength=class_count)
    if (counts == 0).any():
        missing = np.flatnonzero(counts == 0).tolist()
        raise AssertionError(f"{name} has empty classes: {missing}")
    return counts


def finite_stratified_probe(dataset: h5py.Dataset, rows: int,
                            blocks: int = 16, block_rows: int = 256) -> bool:
    """Read small contiguous blocks across a large HDF5 dataset.

    h5py point-selection across millions of rows can be pathologically slow on
    compressed/external-drive files.  Contiguous blocks provide the intended
    coverage without accidentally reading most of the IQ corpus.
    """
    for start in np.linspace(0, max(0, rows - block_rows), min(blocks, rows), dtype=np.int64):
        if not np.isfinite(np.asarray(dataset[int(start):int(start) + block_rows])).all():
            return False
    return True


def verify_variable(path: Path) -> None:
    with h5py.File(path, "r") as f:
        names = text_values(f.attrs.get("class_names", f.attrs.get("classes", [])))
        if names != EXPECTED_VARIABLE_CLASSES:
            raise AssertionError(f"wrong variable-RF class table: {names}")
        for length in (128, 256, 512):
            group = f[f"len_{length}"]
            if group["iq"].shape[1:] != (2, length):
                raise AssertionError(f"len_{length}/iq shape is {group['iq'].shape}")
            rows = group["iq"].shape[0]
            if group["mod"].shape != (rows,) or group["snr"].shape != (rows,):
                raise AssertionError(f"len_{length} row counts disagree")
            counts = verify_label_vector(group["mod"], len(names), f"len_{length}")
            if counts.max() / counts.min() > 1.05:
                raise AssertionError(f"len_{length} is unexpectedly imbalanced: {counts.tolist()}")
            if (not finite_stratified_probe(group["iq"], rows)
                    or not finite_stratified_probe(group["snr"], rows)):
                raise AssertionError(f"len_{length} contains non-finite values in stratified probe")
    print(f"[OK] variable-length corpus: {path}")


def verify_sig53(path: Path) -> None:
    with h5py.File(path, "r") as f:
        required = {"X", "y_mod", "y_snr", "class_names"}
        missing = required.difference(f.keys())
        if missing:
            raise AssertionError(f"Sig53 missing datasets: {sorted(missing)}")
        names = text_values(f["class_names"][:])
        if names != EXPECTED_SIG53_CLASSES:
            raise AssertionError(f"wrong Sig53 embedded class table: {names}")
        rows = f["X"].shape[0]
        if f["X"].shape[1:] != (2, 1024):
            raise AssertionError(f"Sig53 X shape is {f['X'].shape}; expected (N,2,1024)")
        if f["y_mod"].shape != (rows,) or f["y_snr"].shape != (rows,):
            raise AssertionError("Sig53 row counts disagree")
        counts = verify_label_vector(f["y_mod"], len(names), "Sig53")
        if counts.max() / counts.min() > 1.01:
            raise AssertionError(f"Sig53 is unexpectedly imbalanced: {counts.tolist()}")
        if not finite_stratified_probe(f["X"], rows):
            raise AssertionError("Sig53 contains non-finite IQ values in stratified probe")
        if not finite_stratified_probe(f["y_snr"], rows):
            raise AssertionError("Sig53 contains non-finite SNR values in stratified probe")
    print(f"[OK] Sig53 corpus: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--variable-rf", type=Path)
    parser.add_argument("--sig53", type=Path)
    args = parser.parse_args()
    verify_script(args.script)
    if args.variable_rf:
        verify_variable(args.variable_rf)
    if args.sig53:
        verify_sig53(args.sig53)
    print("RESULT: REPAIRED TRAINER AND DATA CONTRACT VERIFIED")


if __name__ == "__main__":
    main()
