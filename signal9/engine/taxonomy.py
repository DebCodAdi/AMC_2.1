"""Failure taxonomy reporter (SIGNAL INFINITY §3.1)."""
from __future__ import annotations
from pathlib import Path
from .mine import FailureRecord


def generate_taxonomy_report(failures: list[FailureRecord], out_path: str | Path) -> str:
    counts: dict[str, int] = {
        "F-A": 0, "F-B": 0, "F-C": 0, "F-D": 0, "F-E": 0, "F-F": 0, "F-G": 0, "F-H": 0, "F-I": 0
    }
    for f in failures:
        counts[f.failure_mode] = counts.get(f.failure_mode, 0) + 1

    total = len(failures)
    md = [
        "# Signal Infinity — Failure Mode Taxonomy Report",
        f"\n**Total Failures Mined:** {total}\n",
        "| Mode | Name | Count | Share | Severity | Action |",
        "|---|---|---|---|---|---|",
        f"| `F-A` | Confidently wrong | {counts['F-A']} | {counts['F-A']/max(1,total):.1%} | P0 | 8x weight & synthesize neighbourhood |",
        f"| `F-B` | Correct but uncalibrated | {counts['F-B']} | {counts['F-B']/max(1,total):.1%} | P2 | Calibration data & temperature scaling |",
        f"| `F-C` | Near-miss (high plausibility) | {counts['F-C']} | {counts['F-C']/max(1,total):.1%} | P1 | Parameter neighbourhood blanket synthesis |",
        f"| `F-D` | Misclassification / Disagreement | {counts['F-D']} | {counts['F-D']/max(1,total):.1%} | P2 | Active learning review |",
        f"| `F-E` | Unknown-not-rejected | {counts['F-E']} | {counts['F-E']/max(1,total):.1%} | P1 | Expand Bucket F negatives |",
        f"| `F-F` | Known-wrongly-rejected | {counts['F-F']} | {counts['F-F']/max(1,total):.1%} | P2 | Rejection threshold coverage tuning |",
        f"| `F-G` | Unstable / Non-deterministic | {counts['F-G']} | {counts['F-G']/max(1,total):.1%} | P0 | Determinism bug fix |",
        f"| `F-H` | Latency budget violation | {counts['F-H']} | {counts['F-H']/max(1,total):.1%} | P2 | Performance profiling |",
        f"| `F-I` | Oracle divergence | {counts['F-I']} | {counts['F-I']/max(1,total):.1%} | P0 | External decoder triage |",
    ]
    report = "\n".join(md) + "\n"
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report, encoding="utf-8")
    return report
