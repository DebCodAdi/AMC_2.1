"""Train a robust, hardened AMC model accustomed to real-world hard channel impairments.

Training Options:
-----------------
1. Synthetic Hardened Mode: Trains across all 15 modulations (PSK, QAM, PAM, FSK, GFSK,
   CPFSK, AM-DSB, AM-SSB, WBFM) with multipath fading, IQ imbalance, carrier leakage/DC,
   phase noise, and PA non-linear saturation.
2. Dual-Domain Hybrid Mode: Ingests RadioML 2016.10a anchors mixed with synthetic hard
   cases, eliminating domain gaps across both simulated and off-the-air signals.
3. Multi-Length Invariance: Supports arbitrary sequence lengths (L=128, 256, 512, 1024)
   or dynamic length variation across batches.
"""
from __future__ import annotations

import os
import sys
import glob
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signal9.train.trainer import (
    SyntheticRFDataset,
    KaggleRadioMLDataset,
    run_training,
)
from signal9.train.backbone import build_backbone
from signal9.train.heads import SignalInfinityMultiTaskHeads
from signal9.train.labels import MOD_VOCAB, mod_index


def find_rml_dataset(user_path: str | None = None) -> str | None:
    """Auto-discover RadioML 2016 pickle file across local, parent, and Kaggle directories."""
    candidates = []
    if user_path:
        candidates.append(user_path)

    # Check environment variable
    if "RML_PATH" in os.environ:
        candidates.append(os.environ["RML_PATH"])

    # Current and parent directories
    root = os.path.dirname(os.path.abspath(__file__))
    candidates.extend([
        os.path.join(root, "RML2016.10a_dict_optimized.pkl"),
        os.path.join(root, "RML2016.10a.pkl"),
        os.path.join(os.path.dirname(root), "RML2016.10a_dict_optimized.pkl"),
        os.path.join(os.path.dirname(root), "RML2016.10a.pkl"),
        r"C:\Users\adity\Downloads\AMC_1-satvik\AMC_1-satvik\RML2016.10a_dict_optimized.pkl",
    ])

    # Kaggle input paths
    kaggle_patterns = [
        "/kaggle/input/**/RML2016*.pkl",
        "/kaggle/input/**/rml2016*.pkl",
    ]
    for pat in kaggle_patterns:
        candidates.extend(glob.glob(pat, recursive=True))

    for p in candidates:
        if p and os.path.exists(p) and os.path.isfile(p):
            return os.path.abspath(p)
    return None


class DynamicMultiLengthDataset(Dataset):
    """Mixes hard-channel synthetic signals with real RadioML 2016 anchors with multi-length support."""
    def __init__(self,
                 pkl_path: str | None = None,
                 n_synthetic: int = 35000,
                 length: int = 128,
                 dynamic_lengths: tuple[int, ...] | None = None,
                 seed: int = 42):
        self.length = length
        self.dynamic_lengths = dynamic_lengths
        self.synth = SyntheticRFDataset(
            n_samples=n_synthetic,
            length=max(dynamic_lengths) if dynamic_lengths else length,
            seed=seed
        )
        self.rml = None
        if pkl_path and os.path.exists(pkl_path):
            print(f"[*] Ingesting RadioML anchors from: {pkl_path}")
            self.rml = KaggleRadioMLDataset(pkl_path, max_samples=min(25000, n_synthetic))
            print(f"[+] Loaded {len(self.rml)} RadioML anchors.")
        else:
            print("[!] RadioML dataset not found / not used; using 100% synthetic hardened mode.")

        self.total_len = len(self.synth) + (len(self.rml) if self.rml else 0)
        if self.dynamic_lengths:
            self.sample_lengths = [self.dynamic_lengths[i % len(self.dynamic_lengths)] for i in range(self.total_len)]
        else:
            self.sample_lengths = [self.length] * self.total_len

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int):
        target_len = self.sample_lengths[idx]

        if self.rml is not None and idx % 2 == 1:
            rml_idx = (idx // 2) % len(self.rml)
            sample = self.rml[rml_idx]
            iq = sample["iq"]  # (2, 128)
            cur_len = iq.size(-1)
            if cur_len < target_len:
                reps = int(np.ceil(target_len / cur_len))
                sample["iq"] = iq.repeat(1, reps)[:, :target_len]
            elif cur_len > target_len:
                sample["iq"] = iq[:, :target_len]
            return sample

        synth_idx = idx % len(self.synth)
        sample = self.synth[synth_idx]
        cur_len = sample["iq"].size(-1)
        if cur_len > target_len:
            sample["iq"] = sample["iq"][:, :target_len]
        return sample


def train_hardened_pipeline(epochs: int = 5,
                            batch_size: int = 128,
                            n_samples: int = 35000,
                            length: int = 128,
                            dynamic_lengths: tuple[int, ...] | None = None,
                            lr: float = 4e-4,
                            use_rml_anchors: bool = True,
                            rml_path: str | None = None,
                            save_name: str | None = None,
                            output_dir: str | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if save_name is None:
        if dynamic_lengths:
            save_name = "s9_hybrid_xcit_multilength_hardened.ckpt"
        else:
            save_name = f"s9_hybrid_xcit_L{length}_hardened.ckpt"

    print("=" * 80)
    print("  TRAINING HARDENED HIGH-ROBUSTNESS AMC MODEL (NTRO SIH26147)")
    print(f"  Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Sequence Len  : {dynamic_lengths if dynamic_lengths else length} samples")
    print(f"  Epochs        : {epochs}")
    print(f"  Batch Size    : {batch_size}")
    print(f"  Total Samples : {n_samples}")
    print(f"  Target Checkpt: {save_name}")
    print("=" * 80)

    found_rml = find_rml_dataset(rml_path) if use_rml_anchors else None
    dataset = DynamicMultiLengthDataset(
        pkl_path=found_rml,
        n_synthetic=n_samples,
        length=length,
        dynamic_lengths=dynamic_lengths
    )

    if output_dir:
        save_path = os.path.join(output_dir, save_name)
    elif os.path.exists("/kaggle/working"):
        save_path = os.path.join("/kaggle/working/models", save_name)
    else:
        save_path = os.path.join(os.path.dirname(__file__), "models", save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"[*] Starting training loop across {len(dataset)} examples...")
    metrics = run_training(
        epochs=epochs,
        device=device,
        batch_size=batch_size,
        dataset=dataset,
        save_path=save_path,
        lr=lr,
        log_every=25,
        num_workers=0,
        architecture="hybrid_xcit",
        expert_dim=32
    )

    print("\n" + "=" * 80)
    print(f"  TRAINING COMPLETED! BEST CHECKPOINT SAVED TO: {save_path}")
    print("=" * 80)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hardened AMC model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--samples", type=int, default=35000, help="Number of synthetic samples")
    parser.add_argument("--length", type=int, default=128, help="IQ sequence length (e.g. 128, 256, 512, 1024)")
    parser.add_argument("--dynamic-lengths", action="store_true", help="Train with mixed lengths [128, 256, 512, 1024]")
    parser.add_argument("--rml-path", type=str, default=None, help="Path to RML2016.10a pickle dataset")
    parser.add_argument("--no-rml", action="store_true", help="Disable RadioML anchors")
    parser.add_argument("--save-name", type=str, default=None, help="Output checkpoint filename")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save checkpoints")
    args = parser.parse_args()

    dyn_lens = (128, 256, 512, 1024) if args.dynamic_lengths else None

    train_hardened_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_samples=args.samples,
        length=args.length,
        dynamic_lengths=dyn_lens,
        use_rml_anchors=not args.no_rml,
        rml_path=args.rml_path,
        save_name=args.save_name,
        output_dir=args.output_dir
    )
