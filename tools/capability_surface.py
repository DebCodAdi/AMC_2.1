"""Capability Surface — P(crc_lock) heatmaps vs pipeline axes.

Reads a JSONL file of results from the L5 Monte-Carlo sweep (each line
must include all Truth fields from the synthetic generator plus the
pipeline result).  Generates an HTML report with interactive Plotly
heatmaps, one per pair of axes.

Usage::

    python tools/capability_surface.py --results sweep.jsonl --out reports/

Spec reference: §2.7, §5.1 (PS26147).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


_AXES = {
    "snr_db":     ([-6, -3, 0, 3, 6, 10, 15, 20, 30], "SNR (dB)"),
    "mod":        (["BPSK","QPSK","8PSK","16QAM","64QAM","2FSK","4FSK","GMSK"], "Modulation"),
    "sps":        ([1.0, 2.0, 2.5, 4.0, 8.0, 32.0], "SPS"),
    "alpha":      ([0.1, 0.2, 0.35, 0.5, 0.8], "Roll-off α"),
    "cfo_frac":   ([0, 1e-4, 1e-3, 1e-2, 0.05, 0.2], "CFO frac"),
    "sco_ppm":    ([0, 5, 20, 100, 500], "SCO (ppm)"),
    "fec":        (["none","conv_k7_r12","conv_p23","rs255_223","ldpc648_324"], "FEC"),
    "n_slips":    ([0, 1, 3], "Cycle slips"),
    "payload_len":([32, 100, 256, 1024, 4096], "Payload length (bytes)"),
}

_PRIMARY_PAIRS = [
    ("snr_db", "mod"),
    ("snr_db", "sps"),
    ("snr_db", "alpha"),
    ("snr_db", "cfo_frac"),
    ("snr_db", "sco_ppm"),
    ("snr_db", "n_slips"),
    ("snr_db", "payload_len"),
    ("mod",    "fec"),
]


def _load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return []
        if text.startswith("["):
            return json.loads(text)
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows


def capability_surface(results_jsonl: str | Path,
                       out_dir: str | Path,
                       pairs: list[tuple[str, str]] | None = None) -> None:
    """Generate the capability-surface HTML report.

    Parameters
    ----------
    results_jsonl : path
        JSONL from the L5 Monte-Carlo sweep.
    out_dir : path
        Directory where ``capability.html`` is written.
    pairs : list or None
        Axis pairs to plot.  Defaults to ``_PRIMARY_PAIRS``.
    """
    try:
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("ERROR: pandas and plotly are required.\n"
              "  pip install pandas plotly", file=sys.stderr)
        sys.exit(1)

    rows = _load_jsonl(results_jsonl)
    df = pd.DataFrame(rows)
    df["locked"] = (df["status"] == "SUCCESS_CRC_LOCKED").astype(float)

    if pairs is None:
        pairs = _PRIMARY_PAIRS

    # Filter to only pairs whose columns exist
    valid_pairs = [(a, b) for a, b in pairs
                   if a in df.columns and b in df.columns]

    n_plots = len(valid_pairs)
    if n_plots == 0:
        print("No axis pairs found in results — check column names.")
        return

    fig = make_subplots(
        rows=(n_plots + 1) // 2,
        cols=2,
        subplot_titles=[f"{a} × {b}" for a, b in valid_pairs],
    )

    for idx, (ax, bx) in enumerate(valid_pairs):
        row = idx // 2 + 1
        col = idx % 2 + 1

        pivot = (
            df.groupby([ax, bx])["locked"]
              .mean()
              .unstack(bx)
        )

        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=[str(c) for c in pivot.columns],
                y=[str(r) for r in pivot.index],
                colorscale="RdYlGn",
                zmin=0, zmax=1,
                showscale=(idx == 0),
                colorbar={"title": "P(CRC lock)"},
                text=[[f"{v:.2f}" if v == v else "" for v in row_]
                      for row_ in pivot.values],
                texttemplate="%{text}",
            ),
            row=row, col=col,
        )
        fig.update_xaxes(title_text=_AXES.get(bx, (None, bx))[1], row=row, col=col)
        fig.update_yaxes(title_text=_AXES.get(ax, (None, ax))[1], row=row, col=col)

    n_files = len(df)
    n_locked = int(df["locked"].sum())
    fig.update_layout(
        title_text=(
            f"Signal9 Capability Surface — {n_locked}/{n_files} = "
            f"{n_locked/n_files:.1%} CRC lock<br>"
            f"<sup>P(CRC lock) by parameter axis pair (L5 Monte-Carlo corpus)</sup>"
        ),
        height=max(500, 400 * ((n_plots + 1) // 2)),
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / "capability.html"
    fig.write_html(str(out_html))
    print(f"Capability surface written to {out_html}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Capability Surface HTML report — PS26147 §2.7"
    )
    p.add_argument("--results", required=True,
                   help="JSONL from the L5 sweep")
    p.add_argument("--out", default="reports",
                   help="Output directory (default: reports/)")
    args = p.parse_args(argv)
    capability_surface(args.results, args.out)


if __name__ == "__main__":
    main()
