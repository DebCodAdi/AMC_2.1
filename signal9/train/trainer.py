"""Multi-task training: datasets, train/validation loop, checkpoints (SIGNAL INFINITY §2.6).

Datasets (see DATASETS.md for how to obtain each one)
-----------------------------------------------------
* :class:`RadioML2018LazyDataset`  -- DeepSig RadioML 2018.01A HDF5, 24 classes,
  1024-sample windows, read lazily so the 20 GB file is never loaded into RAM.
* :class:`KaggleRadioMLDataset`   -- DeepSig RadioML 2016.10a/b pickle, 11/10
  classes, 128-sample windows.
* :class:`SyntheticRFDataset`     -- generated on the fly with exact labels
  (:mod:`signal9.train.synth`); needs no download.
* :class:`SyntheticBatchDataset`  -- the synthetic set plus Rule-R15 "anchor"
  real captures from a local directory.

Every dataset emits the same sample schema, with labels in the canonical
vocabulary of :mod:`signal9.train.labels` and ``-1`` / NaN for "not labelled"
(the loss masks those out).

Two data-handling defects fixed here, both of which would have trained the
model on artefacts rather than on signals:

* **Tiling.**  Both RadioML loaders repeated each window (``np.tile``) until it
  reached 4096 samples -- the 128-sample 2016 windows 32 times over.  A tiled
  window is exactly periodic, so its spectrum is a comb of lines at multiples of
  fs/128: an artificial cyclostationary feature present in every training
  example and in no real signal.  The backbone ends in adaptive average
  pooling, so windows are now used at their native length.
* **Placeholder labels.**  Every RadioML example carried symbol-rate 2400,
  roll-off 0.35, FEC 0 and scrambler 0 whatever the waveform, and the synthetic
  set used ``idx % 4`` for FEC and scrambler.  Unknown labels are now masked.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, Sampler

from ..preprocess import condition_for_classifier
from .backbone import SignalInfinityBackbone, build_backbone, ResearchHybridXCiTBackbone
from .augment import OnlineRFAugmenter
from .ema import ModelEMA
from .features import EXPERT_DIM, expert_features
from .heads import SignalInfinityMultiTaskHeads
from .labels import (
    MOD_VOCAB,
    RML2016_CLASSES,
    RML2018_CLASSES_FIXED,
    canonical_mod,
    mod_index,
)
from .losses import SignalInfinityLoss
from .synth import DEFAULT_MODS, draw_params, generate, labels_for

CHECKPOINT_FORMAT = "signal9-multitask-v1"
CONDITIONING_CONTRACT = "finite-component-median+huber-kurtosis3.2+rms-v1"
_NAN = float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Sample construction shared by every dataset
# ─────────────────────────────────────────────────────────────────────────────

def make_sample(iq: np.ndarray, labels: dict, is_anchor: bool = False) -> dict[str, Any]:
    """Build one training sample in the common schema from complex IQ + labels."""
    x = np.asarray(iq, dtype=np.complex64).ravel()
    x, _ = condition_for_classifier(x)
    return {
        "iq": torch.from_numpy(np.stack([x.real, x.imag]).astype(np.float32)),
        "expert": torch.from_numpy(expert_features(x, full=True)),
        "mod": torch.tensor(int(labels.get("mod", -1)), dtype=torch.long),
        "symrate": torch.tensor(float(labels.get("symrate", _NAN)), dtype=torch.float32),
        "rolloff": torch.tensor(float(labels.get("rolloff", _NAN)), dtype=torch.float32),
        "cfo_norm": torch.tensor(float(labels.get("cfo_norm", _NAN)), dtype=torch.float32),
        "snr_db": torch.tensor(float(labels.get("snr_db", _NAN)), dtype=torch.float32),
        "fec": torch.tensor(int(labels.get("fec", -1)), dtype=torch.long),
        "scrambler": torch.tensor(int(labels.get("scrambler", -1)), dtype=torch.long),
        "presence": torch.tensor(int(labels.get("presence", 1)), dtype=torch.long),
        "is_anchor": bool(is_anchor),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticRFDataset(Dataset):
    """On-the-fly synthetic examples; example ``i`` is deterministic in (seed, i)."""

    def __init__(self, n_samples: int = 20000, length: int = 1024, seed: int = 0,
                 mods: tuple[str, ...] = DEFAULT_MODS):
        self.n_samples = int(n_samples)
        self.length = int(length)
        self.seed = int(seed)
        self.mods = tuple(mods)

    def __len__(self) -> int:
        return self.n_samples

    def _draw(self, idx: int):
        rng = np.random.default_rng((self.seed, int(idx)))
        p = draw_params(rng, self.mods)
        return p, rng

    def snr_of(self, idx: int) -> float:
        p, _ = self._draw(idx)
        return _NAN if p.mod == "NOISE" else p.snr_db

    def label_of(self, idx: int) -> int:
        p, _ = self._draw(idx)
        return mod_index(p.mod)

    def group_of(self, idx: int) -> tuple:
        return ("synthetic", int(idx))

    def __getitem__(self, idx: int):
        p, rng = self._draw(idx)
        return make_sample(generate(p, self.length, rng), labels_for(p))


def _mod_from_filename(name: str) -> str | None:
    """Modulation named in a capture's filename, or None when it names none.

    The previous rule was ``0 if "bpsk" in name else (1 if "qpsk" in name else 0)``,
    which labelled every file that was not QPSK -- including 16QAM, FSK and
    noise captures -- as BPSK.  A file that names no known modulation is now
    skipped rather than given a guessed label.
    """
    tokens = name.upper().replace("-", "_").replace(".", "_").split("_")
    for tok in tokens:
        try:
            return canonical_mod(tok)
        except ValueError:
            continue
    return None


class SyntheticBatchDataset(Dataset):
    """Synthetic examples mixed with Rule-R15 "anchor" real captures.

    Anchors are complex64 ``*.cf32`` / ``*.raw`` files in ``corpus_dir`` whose
    filename names their modulation (e.g. ``qpsk_hf_2024-03-01.cf32``).  At
    least ``anchor_fraction`` of the examples come from them when any exist.
    """

    def __init__(self, n_samples: int = 256, anchor_fraction: float = 0.15,
                 corpus_dir: str | Path = "dataset", length: int = 1024, seed: int = 0):
        if anchor_fraction < 0.10:
            raise ValueError(f"Rule R15 VIOLATION: anchor fraction must be >= 0.10, got {anchor_fraction}")
        self.n_samples = int(n_samples)
        self.anchor_fraction = float(anchor_fraction)
        self.length = int(length)
        self.synth = SyntheticRFDataset(n_samples, length=length, seed=seed)
        cdir = Path(corpus_dir)
        files = sorted(list(cdir.glob("*.cf32")) + list(cdir.glob("*.raw"))) if cdir.exists() else []
        self.anchors = [(f, m) for f in files if (m := _mod_from_filename(f.stem)) is not None]
        self.mod_names = list(MOD_VOCAB)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        n_anchor = int(math.ceil(self.n_samples * self.anchor_fraction)) if self.anchors else 0
        if idx < n_anchor:
            f, mod = self.anchors[idx % len(self.anchors)]
            raw = np.fromfile(f, dtype=np.complex64)
            rng = np.random.default_rng(idx)
            if len(raw) > self.length:
                s0 = int(rng.integers(0, len(raw) - self.length))
                raw = raw[s0:s0 + self.length]
            elif len(raw) < self.length:
                raw = np.pad(raw, (0, self.length - len(raw)))
            return make_sample(raw, {"mod": mod_index(mod), "presence": 1}, is_anchor=True)
        return self.synth[idx]

    def group_of(self, idx: int):
        n_anchor = int(math.ceil(self.n_samples * self.anchor_fraction)) if self.anchors else 0
        if idx < n_anchor:
            return ("anchor_file", str(self.anchors[idx % len(self.anchors)][0].resolve()))
        return self.synth.group_of(idx)

    def label_of(self, idx: int) -> int:
        n_anchor = int(math.ceil(self.n_samples * self.anchor_fraction)) if self.anchors else 0
        if idx < n_anchor:
            return mod_index(self.anchors[idx % len(self.anchors)][1])
        return self.synth.label_of(idx)


# ─────────────────────────────────────────────────────────────────────────────
# RadioML
# ─────────────────────────────────────────────────────────────────────────────

class KaggleRadioMLDataset(Dataset):
    """RadioML 2016.10a / 2016.10b pickle: ``{(mod_name, snr): array(N, 2, 128)}``."""

    def __init__(self, pkl_path: str | Path, max_samples: int = 50000):
        import pickle
        self.pkl_path = Path(pkl_path)
        with open(self.pkl_path, "rb") as f:
            raw_dict = pickle.load(f, encoding="latin1")

        # Take an equal share of every (mod, snr) cell so max_samples cannot
        # truncate the dataset to its first few classes, which dictionary order
        # would otherwise decide.
        keys = sorted(raw_dict.keys(), key=lambda k: (str(k[0]), float(k[1])))
        per_cell = max(1, int(max_samples) // max(1, len(keys)))
        self.samples: list[tuple[np.ndarray, int, float]] = []
        for mod_name, snr in keys:
            idx = mod_index(mod_name)          # raises on an unknown name
            arr = raw_dict[(mod_name, snr)]
            for i in range(min(len(arr), per_cell)):
                self.samples.append((arr[i], idx, float(snr)))

    def __len__(self) -> int:
        return len(self.samples)

    def snr_of(self, idx: int) -> float:
        return self.samples[idx][2]

    def label_of(self, idx: int) -> int:
        return self.samples[idx][1]

    def group_of(self, idx: int) -> tuple:
        _, label, snr = self.samples[idx]
        return ("radioml2016_block", int(label), float(snr), int(idx) // 64)

    def __getitem__(self, idx: int):
        iq_2xN, mod_idx, snr = self.samples[idx]
        x = iq_2xN[0].astype(np.float32) + 1j * iq_2xN[1].astype(np.float32)
        return make_sample(x, {"mod": mod_idx, "snr_db": snr, "presence": 1}, is_anchor=True)


def _rml2018_classes(h5_path: str | Path) -> tuple[str, ...]:
    """Class order for a RadioML 2018 file: a classes-fixed file beside it, else the fixed order.

    ``classes.txt`` from the original release is deliberately *not* read: its
    order does not match the labels in the HDF5 (see labels.RML2018_CLASSES_FIXED).
    """
    base = Path(h5_path).parent
    for name in ("classes-fixed.json", "classes-fixed.txt"):
        f = base / name
        if f.exists():
            text = f.read_text(encoding="utf-8")
            try:
                names = json.loads(text)
            except json.JSONDecodeError:
                names = [t.strip().strip("'\"") for t in text.replace("\n", ",").split(",") if t.strip()]
            if len(names) == 24:
                return tuple(canonical_mod(n) for n in names)
    return RML2018_CLASSES_FIXED


class RadioML2018LazyDataset(Dataset):
    """Memory-mapped, lazy HDF5 loader for RadioML 2018.01A (2.56 M examples, 24 classes)."""

    def __init__(self, h5_path: str | Path, indices: list[int] | np.ndarray | None = None,
                 max_samples: int | None = None):
        import h5py
        self.h5_path = str(h5_path)
        with h5py.File(self.h5_path, "r") as f:
            self.total_dataset_len = f["X"].shape[0]
            self.n_classes = f["Y"].shape[1]
            # Z (SNR) is small; read it once for stratified splitting and reporting.
            self._snr = np.asarray(f["Z"][:]).reshape(-1).astype(np.float32)
            y_idx = np.argmax(np.asarray(f["Y"][:], dtype=np.float32), axis=1) \
                if self.total_dataset_len <= 3_000_000 else None
        names = _rml2018_classes(h5_path)
        if self.n_classes != len(names):
            raise ValueError(f"{h5_path}: expected {len(names)} classes, file has {self.n_classes}")
        self.class_map = np.array([mod_index(n) for n in names], dtype=np.int64)
        self._y_idx = y_idx

        if indices is not None:
            self.indices = np.array(indices, dtype=np.int64)
        elif max_samples is not None and max_samples < self.total_dataset_len:
            # An even stride preserves the class/SNR balance: the file is
            # stored in contiguous (class, SNR) blocks.
            step = max(1, self.total_dataset_len // int(max_samples))
            self.indices = np.arange(0, self.total_dataset_len, step, dtype=np.int64)[:max_samples]
        else:
            self.indices = np.arange(self.total_dataset_len, dtype=np.int64)
        self._h5 = None

    def _get_h5(self):
        if self._h5 is None:
            import h5py
            # Opened lazily so each DataLoader worker gets its own handle.
            self._h5 = h5py.File(self.h5_path, "r")
            self._X = self._h5["X"]
            self._Y = self._h5["Y"]

    def __len__(self) -> int:
        return len(self.indices)

    def snr_of(self, idx: int) -> float:
        return float(self._snr[int(self.indices[idx])])

    def label_of(self, idx: int) -> int:
        real = int(self.indices[idx])
        if self._y_idx is not None:
            return int(self.class_map[self._y_idx[real]])
        self._get_h5()
        return int(self.class_map[int(np.argmax(self._Y[real]))])

    def group_of(self, idx: int) -> tuple[str, int]:
        # The public export has no session IDs.  Keep neighboring rows
        # together as a conservative proxy instead of randomly interleaving
        # them across train and validation.
        return ("radioml2018_block", int(self.indices[idx]) // 64)

    def __getitem__(self, idx: int):
        self._get_h5()
        real = int(self.indices[idx])
        x_raw = self._X[real]                       # (1024, 2) float32
        x = x_raw[:, 0] + 1j * x_raw[:, 1]
        mod = int(self.class_map[int(np.argmax(self._Y[real]))])
        return make_sample(x, {"mod": mod, "snr_db": float(self._snr[real]), "presence": 1},
                           is_anchor=True)


def find_kaggle_dataset(search_dirs: list[str | Path] | None = None) -> tuple[Path | None, str | None]:
    """Find a mounted RadioML dataset (HDF5 2018 first, then 2016 pickle).

    Only dataset locations are searched -- not the working directory, whose
    ``rglob`` used to pick up any stray ``.pkl`` in the repository.
    """
    dirs = [Path(d) for d in (search_dirs or ["/kaggle/input", "../input", "dataset", "data"])]
    for base in dirs:
        if base.exists():
            for ext in ("*.hdf5", "*.h5"):
                for p in sorted(base.rglob(ext)):
                    return p, "hdf5"
    for base in dirs:
        if base.exists():
            for ext in ("*.pkl", "*.dat", "*.pickle"):
                for p in sorted(base.rglob(ext)):
                    if any(k in p.name.lower() for k in ("rml", "radioml", "deepsig", "2016")):
                        return p, "pickle"
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Splitting and evaluation
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split(dataset: Dataset, val_frac: float = 0.1, seed: int = 0) -> tuple[Subset, Subset]:
    """Class-stratified, group-safe split when provenance groups are available."""
    n = len(dataset)
    labels = np.array([dataset.label_of(i) for i in range(n)]) if hasattr(dataset, "label_of") \
        else np.zeros(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    tr, va = [], []
    groups: dict[Any, list[int]] = {}
    for i in range(n):
        group = dataset.group_of(i) if hasattr(dataset, "group_of") else ("row", i)
        groups.setdefault(group, []).append(i)
    groups_by_class: dict[int, list[Any]] = {}
    for group, members in groups.items():
        member_labels = labels[np.asarray(members)]
        values, counts = np.unique(member_labels, return_counts=True)
        dominant = int(values[int(np.argmax(counts))])
        groups_by_class.setdefault(dominant, []).append(group)
    for c in sorted(groups_by_class):
        class_groups = groups_by_class[c]
        rng.shuffle(class_groups)
        k = int(round(len(class_groups) * val_frac))
        if len(class_groups) > 1:
            k = min(max(k, 1), len(class_groups) - 1)
        val_groups = set(class_groups[:k])
        for group in class_groups:
            (va if group in val_groups else tr).extend(groups[group])
    return Subset(dataset, sorted(tr)), Subset(dataset, sorted(va))


class LengthGroupedBatchSampler(Sampler):
    """Batches samples of the same sequence length together so tensors stack without dimension errors."""
    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.groups: dict[int, list[int]] = {}
        if isinstance(dataset, Subset) and hasattr(dataset.dataset, "sample_lengths"):
            for sub_idx, orig_idx in enumerate(dataset.indices):
                L = dataset.dataset.sample_lengths[orig_idx]
                self.groups.setdefault(L, []).append(sub_idx)
        elif hasattr(dataset, "sample_lengths"):
            for idx, L in enumerate(dataset.sample_lengths):
                self.groups.setdefault(L, []).append(idx)
        else:
            self.groups[0] = list(range(len(dataset)))

    def __iter__(self):
        import random
        batches = []
        for L, idxs in self.groups.items():
            c = list(idxs)
            if self.shuffle:
                random.shuffle(c)
            for i in range(0, len(c), self.batch_size):
                chunk = c[i:i + self.batch_size]
                if chunk:
                    batches.append(chunk)
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return sum(int(math.ceil(len(idxs) / self.batch_size)) for idxs in self.groups.values())


@torch.no_grad()
def evaluate(backbone: nn.Module, heads: nn.Module, loader: DataLoader, device: str) -> dict:
    """Overall and per-SNR modulation accuracy on a labelled loader."""
    backbone.eval()
    heads.eval()
    correct = total = 0
    by_snr: dict[int, list[int]] = {}
    for batch in loader:
        iq = batch["iq"].to(device)
        ex = batch["expert"].to(device)
        pred = heads(backbone(iq, ex))["mod_logits"].argmax(dim=1).cpu()
        y = batch["mod"]
        snr = batch["snr_db"]
        ok = (pred == y) & (y >= 0)
        valid = y >= 0
        correct += int(ok.sum())
        total += int(valid.sum())
        for s, o, v in zip(snr.tolist(), ok.tolist(), valid.tolist()):
            if not v or not math.isfinite(s):
                continue
            b = int(round(s / 2.0) * 2)           # 2 dB buckets
            by_snr.setdefault(b, [0, 0])
            by_snr[b][0] += int(o)
            by_snr[b][1] += 1
    backbone.train()
    heads.train()
    return {
        "val_acc": correct / total if total else 0.0,
        "val_count": total,
        "val_acc_by_snr": {k: round(c / t, 4) for k, (c, t) in sorted(by_snr.items()) if t},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_step(backbone: nn.Module,
               heads: nn.Module,
               criterion: nn.Module,
               batch: dict[str, torch.Tensor],
               optimizer: torch.optim.Optimizer,
               ema: ModelEMA | None = None,
               scaler: torch.amp.GradScaler | None = None,
               use_amp: bool = True) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    device_type = batch["iq"].device.type
    amp_enabled = bool(use_amp and device_type == "cuda")
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
        z = backbone(batch["iq"], batch["expert"])
        preds = heads(z)
        loss_dict = criterion(preds, batch, embeddings=z)
        total_loss = loss_dict["loss"]
    parameters = list(backbone.parameters()) + list(heads.parameters())
    if scaler is not None and amp_enabled:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
    if ema:
        ema.update(backbone)
    return {k: float(v.item()) for k, v in loss_dict.items()}


def _augment_training_batch(batch: dict[str, torch.Tensor],
                            augmenter: OnlineRFAugmenter) -> dict[str, torch.Tensor]:
    """Apply RF augmentation and recompute the coupled expert features.

    Keeping expert statistics from the unaugmented waveform would create a
    second train-serve mismatch, so both inputs are updated together.
    """
    iq = batch["iq"].detach().cpu().numpy()
    augmented_iq = []
    augmented_expert = []
    for row in iq:
        z = row[0].astype(np.float32) + 1j * row[1].astype(np.float32)
        # Phase/gain is always safe; stronger channel/noise augmentation is
        # used on a minority of rows to avoid erasing already-low-SNR data.
        z = augmenter.weak_augment(z)
        if augmenter.rng.random() < 0.25:
            z = augmenter.strong_augment(z)
        z, _ = condition_for_classifier(z)
        augmented_iq.append(np.stack([z.real, z.imag]).astype(np.float32))
        augmented_expert.append(expert_features(z, full=True))
    out = dict(batch)
    out["iq"] = torch.from_numpy(np.stack(augmented_iq))
    out["expert"] = torch.from_numpy(np.stack(augmented_expert))
    return out


def build_dataset(source: str | Path | None = None, max_samples: int = 50000,
                  anchor_fraction: float = 0.15, length: int = 1024, seed: int = 0) -> tuple[Dataset, str]:
    """Pick the training data: an explicit path, a mounted RadioML set, or synthetic."""
    if source not in (None, "auto"):
        if str(source) == "synthetic":
            return SyntheticRFDataset(max_samples, length=length, seed=seed), "synthetic"
        path = Path(source)
        if path.suffix.lower() in (".hdf5", ".h5"):
            return RadioML2018LazyDataset(path, max_samples=max_samples), f"radioml2018:{path.name}"
        return KaggleRadioMLDataset(path, max_samples=max_samples), f"radioml2016:{path.name}"
    ds_path, ds_type = find_kaggle_dataset()
    if ds_type == "hdf5" and ds_path:
        return RadioML2018LazyDataset(ds_path, max_samples=max_samples), f"radioml2018:{ds_path.name}"
    if ds_type == "pickle" and ds_path:
        return KaggleRadioMLDataset(ds_path, max_samples=max_samples), f"radioml2016:{ds_path.name}"
    return (SyntheticBatchDataset(n_samples=max_samples, anchor_fraction=anchor_fraction,
                                  length=length, seed=seed), "synthetic+anchors")


def run_training(epochs: int = 1,
                 anchor_fraction: float = 0.15,
                 device: str = "cpu",
                 batch_size: int = 32,
                 max_samples: int = 50000,
                 save_path: str | Path | None = None,
                 dataset: str | Path | Dataset | None = None,
                 val_frac: float = 0.1,
                 lr: float = 3e-4,
                 seed: int = 0,
                 num_workers: int = 0,
                 log_every: int = 50,
                 architecture: str = "hybrid_xcit",
                 expert_dim: int = EXPERT_DIM) -> dict:
    """Train the multi-task model; returns metrics and writes the best checkpoint.

    The checkpoint holds the weights together with everything needed to use
    them: the class vocabulary, the input length, the model dimensions and the
    validation metrics.  Validation runs on a held-out stratified split every
    epoch and the best epoch (by modulation accuracy) is the one saved.
    """
    if anchor_fraction < 0.10:
        raise RuntimeError("Rule R15 Violation: Cannot train without anchor set >= 10%")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    if isinstance(dataset, Dataset):
        ds, ds_name = dataset, type(dataset).__name__
    else:
        ds, ds_name = build_dataset(dataset, max_samples=max_samples,
                                    anchor_fraction=anchor_fraction, seed=seed)
    print(f"[Dataset] {ds_name}: {len(ds)} examples")

    train_ds, val_ds = stratified_split(ds, val_frac=val_frac, seed=seed)
    input_len = int(ds[0]["iq"].shape[-1])
    pin = device.startswith("cuda")
    has_dynamic = hasattr(ds, "sample_lengths") or (isinstance(ds, Subset) and hasattr(ds.dataset, "sample_lengths"))
    if has_dynamic:
        train_sampler = LengthGroupedBatchSampler(train_ds, batch_size=batch_size, shuffle=True)
        val_sampler = LengthGroupedBatchSampler(val_ds, batch_size=max(batch_size, 64), shuffle=False)
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=pin)
        val_loader = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=num_workers, pin_memory=pin)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size,
                                  num_workers=num_workers, pin_memory=pin)
        val_loader = DataLoader(val_ds, batch_size=max(batch_size, 64), shuffle=False,
                                num_workers=num_workers, pin_memory=pin)

    backbone = build_backbone(architecture, expert_dim=expert_dim).to(device)
    heads = SignalInfinityMultiTaskHeads(n_mods=len(MOD_VOCAB)).to(device)
    criterion = SignalInfinityLoss().to(device)
    ema = ModelEMA(backbone)
    params = list(backbone.parameters()) + list(heads.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
    steps = max(1, epochs * max(1, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps,
                                                    pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    augmenter = OnlineRFAugmenter(seed=seed)

    history: list[dict] = []
    best = {"val_acc": -1.0}
    last: dict[str, float] = {}
    t0 = time.time()
    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            batch = _augment_training_batch(batch, augmenter)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            last = train_step(backbone, heads, criterion, batch, optimizer, ema,
                              scaler=scaler, use_amp=True)
            scheduler.step()
            if log_every and step % log_every == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {last['loss']:.4f} "
                      f"mod {last['l_mod']:.4f}")
        # Validate the EMA weights -- they are what gets deployed.
        live = {k: v.detach().clone() for k, v in backbone.state_dict().items()}
        backbone.load_state_dict(ema.state_dict())
        metrics = evaluate(backbone, heads, val_loader, device)
        backbone.load_state_dict(live)
        metrics.update({"epoch": epoch, "train_loss": last.get("loss", float("nan")),
                        "elapsed_s": round(time.time() - t0, 1)})
        history.append(metrics)
        print(f"[Epoch {epoch}] train loss {metrics['train_loss']:.4f}  "
              f"val acc {metrics['val_acc']:.4f} ({metrics['val_count']} examples)")
        if metrics["val_acc"] > best["val_acc"]:
            best = dict(metrics)
            if save_path:
                save_checkpoint(save_path, backbone, heads, ema, input_len=input_len,
                                dataset=ds_name, metrics=metrics,
                                architecture=architecture, expert_dim=expert_dim)

    out = dict(last)
    out.update({
        "val_acc": best.get("val_acc", 0.0),
        "val_acc_by_snr": best.get("val_acc_by_snr", {}),
        "best_epoch": best.get("epoch", -1),
        "history": history,
        "dataset": ds_name,
        "input_len": input_len,
        "class_names": list(MOD_VOCAB),
    })
    return out


def save_checkpoint(path: str | Path, backbone: nn.Module, heads: nn.Module, ema: ModelEMA | None,
                    input_len: int, dataset: str, metrics: dict,
                    architecture: str = "hybrid_xcit", expert_dim: int = EXPERT_DIM) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    stem_d = 128
    if hasattr(backbone, "stem") and hasattr(backbone.stem, "conv1"):
        stem_d = int(backbone.stem.conv1.out_channels * 2)
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "architecture": architecture,
        "backbone": backbone.state_dict(),
        "heads": heads.state_dict(),
        "ema_backbone": ema.state_dict() if ema is not None else None,
        "class_names": list(MOD_VOCAB),
        "input_len": int(input_len),
        "expert_dim": int(expert_dim),
        "emb_dim": int(heads.mod_head.in_features),
        "stem_dim": stem_d,
        "n_fecs": int(heads.fec_head.out_features),
        "n_scramblers": int(heads.scrambler_head.out_features),
        "alignment_preprocess": False,
        "conditioning_contract": CONDITIONING_CONTRACT,
        "dataset": dataset,
        "metrics": metrics,
    }, p)
