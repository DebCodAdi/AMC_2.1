"""Failure Attribution Waterfall — the primary diagnostic instrument.

Reads a JSONL file of pipeline results and a ground-truth JSON, then
produces:

  1. The conditional waterfall table (cumulative count at each stage).
  2. Six diagnostic cross-tabs per spec §2.2.
  3. A CSV dump of the waterfall sorted by leak size (largest first).

Usage::

    python tools/waterfall.py --results results.jsonl \\
                              --ground-truth ground_truth.json \\
                              --out waterfall.csv

Spec reference: §2.2, §6.5 (PS26147).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def waterfall(results_jsonl: str | Path,
              ground_truth_json: str | Path,
              out_csv: str | Path | None = None) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Build the Failure Attribution Waterfall.

    Parameters
    ----------
    results_jsonl : path
        JSONL file where each line is a result dict from the pipeline.
    ground_truth_json : path
        JSON file mapping file name → ground-truth fields.
    out_csv : path or None
        If given, write the waterfall CSV to this path.

    Returns
    -------
    pandas.DataFrame
        Columns: stage, files, cum_pct, leak.
        Sorted by *leak* descending — the biggest leak is your next task.
    """
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is required.  pip install pandas", file=sys.stderr)
        sys.exit(1)

    rows = _load_jsonl(results_jsonl)
    gt_raw = _load_json(ground_truth_json)

    df = pd.DataFrame(rows)
    gt = pd.DataFrame.from_dict(gt_raw, orient="index")
    gt.index.name = "file"
    gt = gt.reset_index()

    # Merge on the 'file' column
    df = df.merge(gt, on="file", suffixes=("", "_true"), how="left")

    # ── Waterfall steps (spec §2.2) ───────────────────────────────────────
    def _pct_diff(col, true_col, tol):
        """Relative difference within tolerance, handling None/NaN."""
        num = pd.to_numeric(df[col], errors="coerce")
        tru = pd.to_numeric(df[true_col], errors="coerce")
        return ((num / tru - 1).abs() < tol)

    steps = [
        ("ingested",
         df["status"] != "EMPTY_FILE"),
        ("fs ok (±0.1%)",
         _pct_diff("fs_est", "fs_true", 1e-3)
         if "fs_est" in df.columns and "fs_true" in df.columns
         else pd.Series(True, index=df.index)),
        ("rs ok (±0.1%)",
         _pct_diff("rs_est", "rs_true", 1e-3)
         if "rs_est" in df.columns and "rs_true" in df.columns
         else pd.Series(True, index=df.index)),
        ("amc correct",
         df["mod_est"] == df["mod_true"]
         if "mod_est" in df.columns and "mod_true" in df.columns
         else pd.Series(True, index=df.index)),
        ("evm < 0.35",
         pd.to_numeric(df["evm"], errors="coerce") < 0.35
         if "evm" in df.columns
         else pd.Series(True, index=df.index)),
        ("n_slips == 0",
         pd.to_numeric(df["n_slips"], errors="coerce") == 0
         if "n_slips" in df.columns
         else pd.Series(True, index=df.index)),
        ("fec identified",
         df["fec_est"] == df["fec_true"]
         if "fec_est" in df.columns and "fec_true" in df.columns
         else pd.Series(True, index=df.index)),
        ("interleaver ok",
         df["intlv_est"] == df["intlv_true"]
         if "intlv_est" in df.columns and "intlv_true" in df.columns
         else pd.Series(True, index=df.index)),
        ("crc locked",
         df["status"] == "SUCCESS_CRC_LOCKED"),
    ]

    mask = pd.Series(True, index=df.index)
    wf_rows = []
    for name, cond in steps:
        cond_filled = cond.fillna(False)
        mask = mask & cond_filled
        wf_rows.append((name, int(mask.sum()), 100.0 * mask.mean()))

    wf = pd.DataFrame(wf_rows, columns=["stage", "files", "cum_pct"])
    total = len(df)
    wf["leak"] = pd.Series(
        [total] + wf["files"].tolist()[:-1]
    ).values - wf["files"].values
    wf = wf.sort_values("leak", ascending=False).reset_index(drop=True)

    # ── Six cross-tabs (spec §2.2) ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FAILURE ATTRIBUTION WATERFALL")
    print("=" * 60)
    print(wf.to_string(index=False))

    _print_crosstabs(df)

    if out_csv:
        wf.to_csv(out_csv, index=False)
        print(f"\nWaterfall written to {out_csv}")

    return wf


def _print_crosstabs(df: "pd.DataFrame") -> None:
    """Print the six diagnostic cross-tabs from spec §2.2."""
    import pandas as pd

    print("\n" + "-" * 60)
    print("CROSS-TABS")
    print("-" * 60)

    locked = df["status"] == "SUCCESS_CRC_LOCKED"

    def _xtab(label, col, defect):
        if col not in df.columns:
            print(f"\n[{label}] — column '{col}' missing from results; skip.")
            return
        print(f"\n[{label}]  (tests {defect})")
        ct = pd.crosstab(
            df[col].fillna("unknown"),
            locked.map({True: "crc_locked", False: "crc_fail"}),
            margins=True
        )
        print(ct.to_string())

    # 1. crc × n_slips (DEF-003)
    _xtab("crc × n_slips", "n_slips", "DEF-003")

    # 2. crc × payload_length (DEF-012 SCO)
    if "payload_len" in df.columns:
        bins = [0, 100, 500, 2000, 10000, int(1e9)]
        labels = ["≤100", "101-500", "501-2000", "2001-10k", ">10k"]
        df = df.copy()
        df["payload_bin"] = pd.cut(
            pd.to_numeric(df["payload_len"], errors="coerce"),
            bins=bins, labels=labels, right=True
        )
        _xtab("crc × payload_length", "payload_bin", "DEF-012 (SCO)")

    # 3. crc × mod_true (DEF-037 16/64QAM)
    _xtab("crc × mod_true", "mod_true", "DEF-037")

    # 4. crc × wav channel count (DEF-008)
    _xtab("crc × wav_nchannels", "wav_nchannels", "DEF-008")

    # 5. amc_correct × snr_est (DEF-039/040)
    if "mod_est" in df.columns and "mod_true" in df.columns and "snr_est" in df.columns:
        import pandas as pd
        df2 = df.copy()
        df2["amc_ok"] = df2["mod_est"] == df2["mod_true"]
        snr_bins = [-30, -6, 0, 6, 12, 20, 50]
        snr_labels = ["<-6", "-6..0", "0..6", "6..12", "12..20", ">20"]
        df2["snr_bin"] = pd.cut(
            pd.to_numeric(df2["snr_est"], errors="coerce"),
            bins=snr_bins, labels=snr_labels
        )
        print("\n[amc_correct × snr_est]  (tests DEF-039/040)")
        ct = pd.crosstab(df2["snr_bin"], df2["amc_ok"].map({True: "amc_ok", False: "amc_fail"}),
                         margins=True)
        print(ct.to_string())

    # 6. entropy × crc (DEF-006)
    if "plausibility_score" in df.columns:
        import pandas as pd
        df3 = df.copy()
        df3["plaus_bin"] = pd.cut(
            pd.to_numeric(df3["plausibility_score"], errors="coerce"),
            bins=[-1, 0, 0.25, 0.5, 0.75, 1.01],
            labels=["0", "0–0.25", "0.25–0.5", "0.5–0.75", "0.75–1"]
        )
        print("\n[plausibility_score × crc]  (tests DEF-006 entropy gate)")
        ct = pd.crosstab(
            df3["plaus_bin"],
            locked.map({True: "crc_locked", False: "crc_fail"}),
            margins=True
        )
        print(ct.to_string())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Failure Attribution Waterfall — PS26147 §6.5"
    )
    p.add_argument("--results", required=True,
                   help="JSONL file of pipeline results")
    p.add_argument("--ground-truth", required=True,
                   help="JSON file of ground-truth labels")
    p.add_argument("--out", default=None,
                   help="Output CSV path (optional)")
    args = p.parse_args(argv)
    waterfall(args.results, args.ground_truth, args.out)


if __name__ == "__main__":
    main()
