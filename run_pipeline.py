"""Run the trained Signal 9 model on an IQ, WAV, or SigMF capture.

Example:
    python run_pipeline.py demo_capture.cf32 --fs 48000 --format cf32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from signal9.pipeline import Signal9Engine


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "models" / "s9_hybrid_xcit_multilength_hardened.ckpt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Signal 9 analysis pipeline")
    parser.add_argument("capture", type=Path, help="Input IQ, WAV, or SigMF capture")
    parser.add_argument("--fs", type=float, default=None, help="Sample rate in Hz")
    parser.add_argument("--fc", type=float, default=None, help="Centre frequency in Hz")
    parser.add_argument(
        "--format",
        choices=("cu8", "cs8", "cs16", "cf32", "cf64"),
        default=None,
        help="Raw IQ sample format",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pipeline_report.json"),
        help="Destination JSON report",
    )
    parser.add_argument("--max-tracks", type=int, default=8)
    args = parser.parse_args()

    if not args.capture.exists():
        parser.error(f"capture not found: {args.capture}")
    if not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    print(f"Loading model: {args.checkpoint}")
    engine = Signal9Engine(checkpoint=args.checkpoint)

    print(f"Analyzing: {args.capture}")
    report = engine.process_file(
        filepath=args.capture,
        fs=args.fs,
        fc=args.fc,
        fmt=args.format,
        max_exploit_tracks=args.max_tracks,
    )

    args.output.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    print(f"Detected signals: {report.get('total_signals_detected', 0)}")
    print(f"Report saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
