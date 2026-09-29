"""Test suite for direct Kaggle dataset loading without manual downloads."""
from __future__ import annotations
import pickle
import numpy as np
import torch
import pytest
from pathlib import Path

from signal9.train.trainer import KaggleRadioMLDataset, find_kaggle_dataset


def test_kaggle_radioml_dataset_mock(tmp_path: Path):
    """Verify KaggleRadioMLDataset handles standard RadioML dict format cleanly."""
    mock_dict = {
        ("BPSK", 10): np.random.randn(15, 2, 128).astype(np.float32),
        ("QPSK", 18): np.random.randn(15, 2, 128).astype(np.float32),
        (b"8PSK", 6): np.random.randn(10, 2, 128).astype(np.float32),
    }
    pkl_file = tmp_path / "RML2016.10a_dict.pkl"
    with open(pkl_file, "wb") as f:
        pickle.dump(mock_dict, f)

    ds = KaggleRadioMLDataset(pkl_file, max_samples=100)
    assert len(ds) == 40

    sample = ds[0]
    assert "iq" in sample
    assert "expert" in sample
    assert "mod" in sample
    # Native 128-sample windows: tiling to 4096 made every example periodic.
    assert sample["iq"].shape == (2, 128)
    assert sample["expert"].shape == (32,)
    assert sample["is_anchor"] is True
    assert isinstance(sample["mod"], torch.Tensor)

    # Labels are canonical vocabulary indices decoded by name, not dict order.
    from signal9.train.labels import MOD_VOCAB
    names = {MOD_VOCAB[int(ds[i]["mod"])] for i in range(len(ds))}
    assert names == {"BPSK", "QPSK", "8PSK"}


def test_kaggle_radioml_pam4_is_not_bpsk(tmp_path: Path):
    """PAM4 was missing from the old map and silently labelled BPSK."""
    from signal9.train.labels import MOD_VOCAB
    pkl_file = tmp_path / "RML2016.10a_dict.pkl"
    with open(pkl_file, "wb") as f:
        pickle.dump({("PAM4", 0): np.random.randn(4, 2, 128).astype(np.float32)}, f)
    ds = KaggleRadioMLDataset(pkl_file, max_samples=10)
    assert MOD_VOCAB[int(ds[0]["mod"])] == "PAM4"


def test_radioml2018_lazy_dataset_mock(tmp_path: Path):
    """Verify RadioML2018LazyDataset loads from HDF5 lazily without loading full array."""
    import h5py
    from signal9.train.trainer import RadioML2018LazyDataset

    h5_file = tmp_path / "GOLD_XYZ_OSC.0001_1024.hdf5"
    n_samples = 60
    with h5py.File(h5_file, "w") as f:
        f.create_dataset("X", data=np.random.randn(n_samples, 1024, 2).astype(np.float32))
        # 24 classes one-hot
        y = np.zeros((n_samples, 24), dtype=np.float32)
        for i in range(n_samples):
            y[i, i % 24] = 1.0
        f.create_dataset("Y", data=y)
        f.create_dataset("Z", data=np.linspace(-20, 30, n_samples)[:, None].astype(np.float32))

    ds = RadioML2018LazyDataset(h5_file, max_samples=40)
    assert len(ds) == 40

    sample = ds[0]
    assert "iq" in sample
    assert "mod" in sample
    assert "snr_db" in sample
    assert sample["iq"].shape == (2, 1024)
    # One-hot position 0 is OOK in DeepSig's corrected (classes-fixed) order.
    from signal9.train.labels import MOD_VOCAB
    assert MOD_VOCAB[sample["mod"].item()] == "OOK"
    assert sample["is_anchor"] is True
