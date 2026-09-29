"""Corpus diversity auditor and non-decreasing diversity floor verifier (Rule R22)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


METRICS = [
    "entropy_mod",
    "entropy_channel",
    "entropy_snr_bin",
    "entropy_device",
    "entropy_fec",
    "entropy_scrambler",
    "n_unique_groups",
    "coverage_cells_filled"
]


def shannon_entropy(counts: list[int] | np.ndarray) -> float:
    arr = np.array(counts, dtype=np.float64)
    total = np.sum(arr)
    if total <= 0:
        return 0.0
    p = arr / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_diversity_metrics(samples: list[dict]) -> dict[str, float]:
    if not samples:
        return {m: 0.0 for m in METRICS}

    def get_counts(key: str) -> list[int]:
        freqs: dict[str, int] = {}
        for s in samples:
            val = str(s.get(key, "unknown"))
            freqs[val] = freqs.get(val, 0) + 1
        return list(freqs.values())

    # SNR bins: -10..30 in 5 dB increments
    snr_bins: dict[int, int] = {}
    for s in samples:
        snr = float(s.get("snr_db", 0.0))
        b = int(np.floor(snr / 5.0) * 5)
        snr_bins[b] = snr_bins.get(b, 0) + 1

    # Unique groups
    groups = set()
    cells = set()
    for s in samples:
        grp = (s.get("prov_session_id", ""), s.get("prov_device_id", ""))
        groups.add(grp)
        # Capability cell: (mod, snr_bin, channel, fec)
        cell = (s.get("mod", ""), int(s.get("snr_db", 0) // 5), s.get("fec", ""))
        cells.add(cell)

    return {
        "entropy_mod": shannon_entropy(get_counts("mod")),
        "entropy_channel": shannon_entropy(get_counts("channel")),
        "entropy_snr_bin": shannon_entropy(list(snr_bins.values())),
        "entropy_device": shannon_entropy(get_counts("prov_device_id")),
        "entropy_fec": shannon_entropy(get_counts("fec")),
        "entropy_scrambler": shannon_entropy(get_counts("scrambler")),
        "n_unique_groups": float(len(groups)),
        "coverage_cells_filled": float(len(cells)),
    }


def audit_diversity_floor(current_metrics: dict[str, float], baseline_metrics: dict[str, float]) -> tuple[bool, list[str]]:
    """Enforce Rule R22: Corpus diversity metrics may never decrease between generations."""
    violations = []
    for m in METRICS:
        cur = current_metrics.get(m, 0.0)
        base = baseline_metrics.get(m, 0.0)
        # Allow small numerical epsilon
        if cur < (base - 1e-4):
            violations.append(f"Diversity metric '{m}' decreased from {base:.4f} to {cur:.4f} (Rule R22 violation)")
    
    return len(violations) == 0, violations


def main():
    parser = argparse.ArgumentParser(description="Corpus diversity floor auditor (Rule R22)")
    parser.add_argument("--current", type=str, required=True, help="Current diversity JSON")
    parser.add_argument("--baseline", type=str, required=False, help="Baseline diversity JSON")
    args = parser.parse_args()

    cur_p = Path(args.current)
    cur_data = json.loads(cur_p.read_text(encoding="utf-8")) if cur_p.exists() else {}

    if args.baseline and Path(args.baseline).exists():
        base_data = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        passed, viols = audit_diversity_floor(cur_data, base_data)
        if not passed:
            print("FAILED Rule R22 diversity floor audit:")
            for v in viols:
                print("  *", v)
            exit(1)
        print("PASSED: Diversity floor holds across all metrics.")
    else:
        print("Diversity metrics computed and recorded.")


if __name__ == "__main__":
    main()
