"""Seeded detector and order-selection benchmark (not end-to-end BER).

Run: python tools/benchmark_receiver_review.py --trials 20 --output review.json
The holdout seed range is disjoint; keep it unused while tuning development.
Order metrics use known symbol timing and independent ground-truth Es/N0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signal9.detect.cfar import detect_cfar
from signal9.detect.multiscale import detect_multiscale
from signal9.detect.spectrogram import log_spectrogram
from signal9.detect.tracker import track_detections
from signal9.detect.whiten import whiten_spectrogram
from signal9.dsp.constellation import get_constellation
from signal9.dsp.matched_filter import rrc_taps
from signal9.dsp.mod_order import select_modulation


def interval(successes, total):
    if not total:
        return {"rate": None, "count": 0, "ci95": None}
    p, z = successes / total, 1.96
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return {"rate": p, "count": total, "ci95": [max(0, center - half), min(1, center + half)]}


def detect(x, fs, enhanced):
    f, t, p = log_spectrogram(x, fs)
    w, _ = whiten_spectrogram(p)
    boxes = detect_cfar(w, f, t)
    if enhanced:
        boxes += detect_multiscale(p, f, t)
    return track_detections(boxes, freq_tol_hz=0)


def run(trials=10, split="development"):
    mods = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]
    snrs = [-6, 0, 4, 8, 16, 25]
    spec = {"version": 1, "split": split, "seed_base": 31000 if split == "development" else 1000000,
            "trials": trials, "mods": mods, "inband_snr_db": snrs,
            "fs": 256000, "sps": 16, "alpha": 0.35, "symbols": 2048}
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    results = {"spec": spec, "spec_sha256": digest,
               "scope": "AWGN detector; known-timing symbol order. No end-to-end BER or real-SDR claim.", "bins": []}
    for si, snr in enumerate(snrs):
        counts = dict(old_pd=0, new_pd=0, old_fa=0, new_fa=0, correct=0, decided=0, old_correct=0, old_decided=0)
        confusion = {}
        center_errors = []
        for mi, mod in enumerate(mods):
            for trial in range(trials):
                rng = np.random.default_rng(spec["seed_base"] + si * 10000 + mi * trials + trial)
                fs, sps, n = spec["fs"], spec["sps"], spec["symbols"]
                symbols = rng.choice(get_constellation(mod), n)
                up = np.zeros(n * sps, dtype=complex)
                up[::sps] = symbols
                x = fftconvolve(up, rrc_taps(spec["alpha"], sps), mode="same")
                center = rng.uniform(35000, 65000)
                x *= np.exp(2j * np.pi * center * np.arange(len(x)) / fs)
                bandwidth = fs / sps * (1 + spec["alpha"])
                noise_power = np.mean(abs(x) ** 2) / (10 ** (snr / 10) * bandwidth / fs)
                noise = (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x))) * np.sqrt(noise_power / 2)
                for enhanced, key in ((False, "old"), (True, "new")):
                    boxes = detect(x + noise, fs, enhanced)
                    matches = [b for b in boxes if b["f_lo"] < center < b["f_hi"] and b["bandwidth"] >= bandwidth / 2]
                    counts[key + "_pd"] += bool(matches)
                    counts[key + "_fa"] += bool(detect(noise, fs, enhanced))
                    if enhanced and matches:
                        center_errors.append(abs(min(matches, key=lambda b: abs(b["center_freq"] - center))["center_freq"] - center))
                esn0 = snr + 10 * np.log10(1 + spec["alpha"])
                noisy_symbols = symbols + (rng.normal(size=n) + 1j * rng.normal(size=n)) * np.sqrt(10 ** (-esn0 / 10) / 2)
                old = select_modulation(noisy_symbols)
                new = select_modulation(noisy_symbols, snr_db=esn0)
                label = new["modulation"] if not new["ambiguous"] else "UNRESOLVED"
                old_label = old["modulation"] if not old["ambiguous"] else "UNRESOLVED"
                counts["old_decided"] += old_label != "UNRESOLVED"
                counts["old_correct"] += old_label == mod
                counts["decided"] += label != "UNRESOLVED"
                counts["correct"] += label == mod
                row = confusion.setdefault(mod, {})
                row[label] = row.get(label, 0) + 1
        total = len(mods) * trials
        results["bins"].append({"snr_db": snr, "captures": total,
                                "detector_baseline_pd": interval(counts["old_pd"], total),
                                "detector_pd": interval(counts["new_pd"], total),
                                "noise_capture_pfa": interval(counts["new_fa"], total),
                                "baseline_noise_capture_pfa": interval(counts["old_fa"], total),
                                "order_accuracy": interval(counts["correct"], total),
                                "decided_accuracy": interval(counts["correct"], counts["decided"]),
                                "baseline_decided_accuracy": interval(counts["old_correct"], counts["old_decided"]),
                                "abstention_rate": 1 - counts["decided"] / total,
                                "coarse_center_mae_hz": float(np.mean(center_errors)) if center_errors else None,
                                "confusion": confusion})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10, help="Captures per modulation per SNR")
    parser.add_argument("--split", choices=["development", "holdout"], default="development")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1 or args.trials > 1000:
        parser.error("trials must be between 1 and 1000")
    result = run(args.trials, args.split)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}; spec SHA256 {result['spec_sha256']}")
