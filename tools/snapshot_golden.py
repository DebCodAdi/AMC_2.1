"""Phase 0 golden payload snapshot tool.

Runs the pipeline on every file in a dataset directory (or JSONL list),
collects all SUCCESS_CRC_LOCKED results, and writes them to
tests/golden/MANIFEST.json + tests/golden/payloads/.

Usage::

    python tools/snapshot_golden.py --dataset /path/to/iq_files/ \\
                                    --out tests/golden/

This must be run exactly ONCE to freeze the baseline.  Re-running it
overwrites existing entries — use ab_compare.py to validate before doing so.

Spec reference: §2.4, Phase 0, L8 tier (PS26147).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


SUPPORTED_EXTENSIONS = {".iq", ".wav", ".sigmf-meta", ".sigmf-data",
                        ".cs8", ".cs16", ".cu8", ".cf32", ".cf64", ".bin"}


def snapshot(dataset_dir: str | Path,
             out_dir: str | Path,
             file_list: str | Path | None = None,
             overwrite: bool = False) -> None:
    """Run the pipeline over all files in ``dataset_dir`` and freeze golden payloads.

    Parameters
    ----------
    dataset_dir : path
        Directory containing the real IQ captures.
    out_dir : path
        Destination for MANIFEST.json, payloads/, and telemetry/.
    file_list : path or None
        Optional text file listing specific paths, one per line.
        Overrides dataset_dir discovery.
    overwrite : bool
        If False, skip files already in the manifest.
    """
    try:
        from signal9.pipeline import Signal9Engine
    except ImportError as e:
        print(f"ERROR: Could not import Signal9Engine: {e}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(out_dir)
    payload_dir = out_dir / "payloads"
    telemetry_dir = out_dir / "telemetry"
    manifest_path = out_dir / "MANIFEST.json"
    payload_dir.mkdir(parents=True, exist_ok=True)
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    # Load existing manifest
    existing: dict[str, dict] = {}
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            m = json.load(f)
            existing = {e["file"]: e for e in m.get("entries", [])}

    # Collect files
    if file_list:
        files = [Path(line.strip()) for line in Path(file_list).read_text().splitlines()
                 if line.strip()]
    else:
        files = [
            p for p in Path(dataset_dir).rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

    engine = Signal9Engine()

    entries: list[dict] = list(existing.values())
    locked_count = 0
    total = len(files)

    print(f"Running snapshot over {total} files …")
    for i, fpath in enumerate(sorted(files), 1):
        key = str(fpath)
        if not overwrite and key in existing:
            print(f"  [{i}/{total}] SKIP (already in manifest): {fpath.name}")
            continue

        print(f"  [{i}/{total}] Processing {fpath.name} … ", end="", flush=True)
        t0 = time.perf_counter()
        try:
            result = engine.process_file(str(fpath))
        except Exception as e:
            print(f"ERROR: {e!r}")
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        status = result.get("status", "")
        if status == "SUCCESS_CRC_LOCKED":
            payload_hex = result.get("payload_hex", "")
            if not payload_hex:
                print("SKIP (no payload_hex in result)")
                continue

            payload_bytes = bytes.fromhex(payload_hex)
            sha256 = hashlib.sha256(payload_bytes).hexdigest()

            # Write frozen payload bytes
            pfile = payload_dir / f"{sha256[:16]}.bin"
            pfile.write_bytes(payload_bytes)

            # Write telemetry snapshot
            tfile = telemetry_dir / f"{sha256[:16]}.json"
            tele = {k: result.get(k) for k in [
                "fs_est", "rs_est", "alpha_est", "cfo_est",
                "snr_est", "evm", "n_slips", "branch_name",
                "t_total_ms",
            ]}
            tfile.write_text(json.dumps(tele, indent=2))

            entry = {
                "file": key,
                "status": status,
                "payload_hex": payload_hex,
                "payload_sha256": sha256,
                "payload_len_bytes": len(payload_bytes),
                "t_total_ms": elapsed_ms,
            }
            existing[key] = entry
            locked_count += 1
            print(f"LOCKED  {len(payload_bytes)} bytes  sha256={sha256[:12]}…")
        else:
            print(f"no-lock ({status})")

    # Write manifest
    manifest = {
        "version": 1,
        "description": "Phase 0 golden payload snapshot — PS26147",
        "total_locked": locked_count,
        "entries": list(existing.values()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written: {manifest_path}  ({locked_count} golden entries)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Phase 0 golden payload snapshot — PS26147"
    )
    p.add_argument("--dataset", default=".",
                   help="Directory of IQ captures (default: current dir)")
    p.add_argument("--out", default="tests/golden",
                   help="Output directory (default: tests/golden/)")
    p.add_argument("--file-list", default=None,
                   help="Text file listing specific paths (overrides --dataset)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-snapshot files already in the manifest")
    args = p.parse_args(argv)
    snapshot(args.dataset, args.out, args.file_list, args.overwrite)


if __name__ == "__main__":
    main()
