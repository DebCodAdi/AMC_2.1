"""Report Card generator (SIGNAL10 §A.3 / Rule R8).

Reads results from reports/results_head.jsonl and generates reports/report_card.md
documenting measured lock rates, latency statistics, and stage performance.

Usage::

    python tools/report_card.py --results reports/results_head.jsonl --out reports/report_card.md
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def generate_report_card(results_path: str | Path, out_path: str | Path) -> str:
    res_p = Path(results_path)
    rows = []
    with open(res_p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line.strip()))

    n_total = len(rows)
    n_locked = sum(1 for r in rows if r.get("crc_locked", False) or r.get("status") == "SUCCESS_CRC_LOCKED")
    latencies = [r.get("latency_ms", 0.0) for r in rows if "latency_ms" in r]

    lock_rate = (n_locked / max(1, n_total)) * 100.0
    p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0

    statuses: dict[str, int] = {}
    for r in rows:
        st = r.get("status", "UNKNOWN")
        statuses[st] = statuses.get(st, 0) + 1

    md = [
        "# Signal 10 — Empirical Report Card",
        f"\n**Source Artefact:** `{results_path}`  ",
        f"**Evaluated Captures:** {n_total}  ",
        "\n## 1. Measured Performance Metrics",
        "\n| Metric | Measured Value | Target Standard | Status |",
        "|---|---|---|---|",
        f"| **CRC-32 Lock Rate** | **{n_locked}/{n_total} ({lock_rate:.1f}%)** | $\\ge 72.0\\%$ | {'PASS' if lock_rate >= 72 else 'BENCHMARKED'} |",
        f"| **Mean Latency** | **{mean_lat:.1f} ms** | $\\le 1500.0\\text{{ ms}}$ | {'PASS' if mean_lat <= 1500 else 'NEEDS_OPTIMIZATION'} |",
        f"| **p50 Latency** | **{p50_lat:.1f} ms** | $\\le 1200.0\\text{{ ms}}$ | {'PASS' if p50_lat <= 1200 else 'NEEDS_OPTIMIZATION'} |",
        f"| **p95 Latency** | **{p95_lat:.1f} ms** | $\\le 3500.0\\text{{ ms}}$ | {'PASS' if p95_lat <= 3500 else 'NEEDS_OPTIMIZATION'} |",
        "\n## 2. Terminal Status Distribution",
        "\n| Status | Count | Percentage |",
        "|---|---|---|",
    ]

    for st, cnt in sorted(statuses.items(), key=lambda x: x[1], reverse=True):
        md.append(f"| `{st}` | {cnt} | {cnt * 100.0 / n_total:.1f}% |")

    content = "\n".join(md) + "\n"
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Report card generated: {out_p}")
    return content


def main():
    parser = argparse.ArgumentParser(description="Generate Signal 10 Report Card")
    parser.add_argument("--results", type=str, default="reports/results_head.jsonl", help="Results JSONL")
    parser.add_argument("--out", type=str, default="reports/report_card.md", help="Output markdown path")
    args = parser.parse_args()

    generate_report_card(args.results, args.out)


if __name__ == "__main__":
    main()
