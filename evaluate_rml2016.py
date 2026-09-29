"""Comprehensive Evaluation of NTRO Model on RadioML 2016.10a Dataset."""
from __future__ import annotations

import os
import sys
import glob
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

# Ensure local package path is root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signal9.train.trainer import KaggleRadioMLDataset
from signal9.train.backbone import build_backbone
from signal9.train.heads import SignalInfinityMultiTaskHeads
from signal9.train.labels import MOD_VOCAB


def find_file(user_path: str | None, search_names: list[str], kaggle_pats: list[str]) -> str | None:
    if user_path and os.path.exists(user_path):
        return os.path.abspath(user_path)
    root = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    for name in search_names:
        candidates.append(os.path.join(root, name))
        candidates.append(os.path.join(root, "models", name))
        candidates.append(os.path.join(os.path.dirname(root), name))
        candidates.append(os.path.join(os.path.dirname(root), "models", name))
        candidates.append(os.path.join(str(Path.home()), "Downloads", name))
    for pat in kaggle_pats:
        candidates.extend(glob.glob(pat, recursive=True))
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            return os.path.abspath(c)
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate NTRO Model on RadioML 2016.10a")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Path to .ckpt model file")
    parser.add_argument("--pkl-path", type=str, default=None, help="Path to RML2016.10a dataset pickle")
    parser.add_argument("--batch-size", type=int, default=512, help="Inference batch size")
    parser.add_argument("--max-samples", type=int, default=50000, help="Max test samples to evaluate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("  EVALUATING NTRO SIH26147 MODEL ON RadioML 2016.10a")
    print(f"  Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    # 1. Load Checkpoint
    ckpt_names = [
        "s9_hybrid_xcit_L128_hardened.ckpt",
        "s9_hybrid_xcit_L128_best.ckpt",
        "s9_hybrid_xcit_best.ckpt",
        "s9_satvik_grand_corpus_best.ckpt",
    ]
    ckpt_path = find_file(args.ckpt_path, ckpt_names, ["/kaggle/input/**/*.ckpt", "models/*.ckpt"])
    if not ckpt_path:
        print("[!] No valid checkpoint file found! Please specify --ckpt-path.")
        return

    print(f"[*] Loading model checkpoint: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)

    expert_dim = int(ck.get("expert_dim", 11))
    stem_dim = ck.get("stem_dim", 128)
    emb_dim = ck.get("emb_dim", 512)
    n_mods = len(ck.get("class_names") or MOD_VOCAB)
    arch = ck.get("architecture", "hybrid_xcit")

    backbone = build_backbone(arch, expert_dim=expert_dim, stem_dim=stem_dim, emb_dim=emb_dim)
    heads = SignalInfinityMultiTaskHeads(emb_dim=emb_dim, n_mods=n_mods, n_fecs=ck.get("n_fecs", 5), n_scramblers=ck.get("n_scramblers", 5))

    selected_backbone = ck.get("ema_backbone") or ck.get("backbone")
    backbone.load_state_dict(selected_backbone, strict=True)
    heads.load_state_dict(ck.get("ema_heads") or ck["heads"])

    backbone.to(device).eval()
    heads.to(device).eval()
    print(f"[+] Model loaded successfully. Architecture: {arch} (expert_dim={expert_dim}, {n_mods} classes)")

    # 2. Load Dataset
    pkl_names = [
        "RML2016.10a_dict_optimized.pkl",
        "RML2016.10a.pkl",
    ]
    pkl_path = find_file(args.pkl_path, pkl_names, ["/kaggle/input/**/RML2016*.pkl"])
    if not pkl_path:
        print("[!] RML2016 dataset not found! Please specify --pkl-path.")
        return

    print(f"[*] Ingesting RadioML 2016.10a dataset: {pkl_path}")
    ds = KaggleRadioMLDataset(pkl_path, max_samples=args.max_samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[+] Loaded {len(ds)} test samples across 11 modulation schemes.")

    # 3. Evaluation Loop
    total = 0
    correct = 0
    class_stats = {}
    snr_stats = {}

    print("[*] Running inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            iq = batch["iq"].to(device)
            labels = batch["mod"].to(device)
            expert = batch["expert"].to(device)
            if expert.size(1) > expert_dim:
                expert = expert[:, :expert_dim]
            elif expert.size(1) < expert_dim:
                expert = F.pad(expert, (0, expert_dim - expert.size(1)))
            snrs = batch["snr_db"].cpu().numpy()

            z = backbone(iq, expert)
            preds = heads(z)
            mod_preds = preds["mod_logits"].argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            total += len(labels_np)
            correct += int(np.sum(mod_preds == labels_np))

            for p, l, s in zip(mod_preds, labels_np, snrs):
                s_int = int(round(float(s)))
                snr_stats.setdefault(s_int, [0, 0])
                snr_stats[s_int][0] += 1
                if p == l:
                    snr_stats[s_int][1] += 1

                class_stats.setdefault(l, [0, 0])
                class_stats[l][0] += 1
                if p == l:
                    class_stats[l][1] += 1

    # 4. Results Reporting
    overall_acc = 100.0 * correct / total
    print("\n" + "=" * 80)
    print("  RadioML 2016.10a — OVERALL EVALUATION RESULTS")
    print("=" * 80)
    print(f"  Total Samples Evaluated: {total}")
    print(f"  Overall Top-1 Accuracy : {correct} / {total} ({overall_acc:.2f}%)")
    print("-" * 80)

    # Class-by-class breakdown
    print("  1. Class-by-Class Classification Breakdown:")
    print(f"    {'Modulation':12s} {'Correct / Total':18s} {'Accuracy':10s}")
    sorted_classes = sorted(class_stats.keys(), key=lambda k: class_stats[k][1]/class_stats[k][0], reverse=True)
    for cls_idx in sorted_classes:
        c, t = class_stats[cls_idx][1], class_stats[cls_idx][0]
        name = MOD_VOCAB[cls_idx] if cls_idx < len(MOD_VOCAB) else f"Class_{cls_idx}"
        pct = 100.0 * c / t
        print(f"    {name:12s} {c:5d} / {t:5d}          {pct:6.2f}%")

    # High-SNR vs Low-SNR performance
    print("-" * 80)
    print("  2. Accuracy Across Signal-to-Noise Ratio (SNR):")
    sorted_snrs = sorted(snr_stats.keys())
    high_snr_corr = sum(snr_stats[s][1] for s in sorted_snrs if s >= 0)
    high_snr_tot = sum(snr_stats[s][0] for s in sorted_snrs if s >= 0)
    print(f"    High-SNR (>= 0 dB) Accuracy: {high_snr_corr}/{high_snr_tot} ({100.0*high_snr_corr/high_snr_tot:.2f}%)")
    print()
    print(f"    {'SNR (dB)':10s} {'Correct / Total':18s} {'Accuracy':10s}")
    for s in sorted_snrs[::2]: # Every 4 dB
        c, t = snr_stats[s][1], snr_stats[s][0]
        pct = 100.0 * c / t
        bar = "#" * int(pct // 5)
        print(f"    {s:+3d} dB     {c:5d} / {t:5d}          {pct:6.2f}%  |{bar}")
    print("=" * 80)


if __name__ == "__main__":
    main()
