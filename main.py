"""Signal 9 — command-line entry point for blind signal analysis.

    python main.py capture.wav
    python main.py capture.cf32 --fs 48000 --json > report.json
    python main.py capture.iq  --fs 2.4e6 --format cu8 --save-json report.json

For the graphical interface run ``python -m signal9.gui`` (or ``python gui.py``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal9.pipeline import Signal9Engine


def _fmt_hz(v: float) -> str:
    v = float(v or 0.0)
    if abs(v) >= 1e6:
        return f"{v / 1e6:.3f} MHz"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.3f} kHz"
    return f"{v:.1f} Hz"


def _print_exploitation(ex: dict) -> None:
    d = ex.get("demodulation", {})
    alpha = d.get("rolloff")
    alpha_text = f"{alpha:.2f}" if alpha is not None else "n/a"
    print(f"  Demodulation : {d.get('modulation')} ({d.get('family')})  "
          f"Rs {d.get('symbol_rate_hz', 0):.1f} Bd  "
          + (f"roll-off {alpha_text}  EVM {d.get('evm_db', 0):.1f} dB" if d.get("family") == "linear" else
             f"tones {d.get('fsk', {}).get('n_tones')}  h {d.get('fsk', {}).get('modulation_index', 0):.2f}"))
    if "symbol_rate_per_sample" in d:
        print(f"                 (sample rate unknown: Rs = {d['symbol_rate_per_sample']:.5f} x fs)")
    amb = ex.get("bit_ambiguity", {})
    if amb:
        pol = amb.get("tone_polarity")
        rot = f"rotation {amb.get('rotation_deg')} deg, mirror {amb.get('mirror')}" if pol is None \
            else f"tone polarity {pol}"
        print(f"  Bit mapping  : {rot}, line code {amb.get('line_code')}, "
              f"differential {amb.get('differential')}")
    sc = ex.get("scrambler", {})
    print(f"  Scrambler    : {sc.get('type') or 'not run'}")
    il = ex.get("interleaver", {})
    geo = {k: il[k] for k in ("rows", "cols", "slope", "branches", "delay", "size", "phase") if k in il}
    print(f"  Interleaver  : {il.get('type') or 'not run'} {geo if geo else ''}")
    fec = ex.get("fec", {})
    if fec.get("class") == "concatenated":
        inner, outer = fec.get("inner", {}), fec.get("outer", {})
        print(f"  FEC          : concatenated  inner {inner.get('name')}  "
              f"outer RS({outer.get('n')},{outer.get('k')}) {outer.get('words_ok')}/{outer.get('words_checked')} words")
    else:
        extra = {k: fec[k] for k in ("name", "K", "rate", "n", "k", "reencode_agreement",
                                     "words_ok", "words_checked", "blocks_converged", "blocks_checked")
                 if k in fec}
        print(f"  FEC          : {fec.get('class') or 'not run'} {extra if extra else ''}")
    fr = ex.get("frames", {})
    crc = fr.get("crc") or {}
    sync = fr.get("sync") or {}
    if fr.get("period_bits"):
        print(f"  Frames       : period {fr.get('period_bits')} bits, sync {sync.get('length', 0)} bits "
              f"@ {sync.get('offset', '-')}, payload bits {fr.get('payload', {}).get('bits')}")
        print(f"  Frame CRC    : {crc.get('model')} {fr.get('crc_frames_ok')}/{fr.get('crc_frames_checked')} frames"
              if crc else "  Frame CRC    : none identified")
    dec = ex.get("decoded") or {}
    if dec.get("frames"):
        lines = [l for l in dec.get("message_text", "").split("\n") if l]
        print(f"  Decoded msg  : {len(lines)} CRC-valid frame(s); first: {lines[0] if lines else '—'}")
        for row in dec["frames"][:3]:
            mark = {True: "CRC ok ", False: "CRC BAD", None: "no CRC "}[row["crc_ok"]]
            print(f"    frame {row['index']:3d} [{mark}] {row['payload_text'][:60]}")
        if len(dec["frames"]) > 3:
            print(f"    ... {len(dec['frames'])} frames in total (full list in --json / GUI)")
    elif dec.get("preview_text"):
        print(f"  Decoded bytes: {dec['preview_text'][:80]}  (no frame model: raw preview)")
    if dec.get("byte_alignment"):
        print(f"  Byte alignment: {dec['byte_alignment']}")
    if "reference_ber" in ex:
        print(f"  Reference BER: {ex['reference_ber']}")
    print(f"  Status       : {ex.get('status')}")
    for note in ex.get("notes", []):
        print(f"    note: {note}")


def main():
    parser = argparse.ArgumentParser(
        description="Signal 9 — blind signal analysis: detection, classification, "
                    "demodulation, de-interleaving, FEC decoding and frame correlation"
    )
    parser.add_argument("file", help="Input capture (.wav, .iq, .raw, .cf32, .cu8, .cs16, .sigmf-meta/-data)")
    parser.add_argument("--fs", type=float, default=None, help="Sample rate in Hz (optional if in metadata/filename)")
    parser.add_argument("--fc", type=float, default=None, help="Centre frequency in Hz")
    parser.add_argument("--format", type=str, default=None, choices=["cu8", "cs8", "cs16", "cf32", "cf64"],
                        help="Raw sample format if headerless")
    parser.add_argument("--no-exploit", action="store_true",
                        help="Detection and classification only; skip demod/FEC/frame analysis")
    parser.add_argument("--max-tracks", type=int, default=8,
                        help="Maximum number of tracks to run the exploitation chain on (default 8)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Trained model checkpoint (from signal9.train.trainer) for neural classification")
    parser.add_argument("--json", action="store_true", help="Output full JSON report to stdout")
    parser.add_argument("--save-json", type=str, default=None, help="Save report to JSON file path")
    parser.add_argument("--reference-bits", type=Path, help="Test mode: text file of 0/1 reference bits")
    parser.add_argument("--reference-offset", type=int, default=0, help="Explicit decoded-bit offset for BER comparison")

    args = parser.parse_args()
    reference = None
    if args.reference_bits:
        import numpy as np
        text = "".join(args.reference_bits.read_text().split())
        if not text or set(text) - {"0", "1"} or args.reference_offset < 0:
            parser.error("Reference must contain 0/1 bits and offset must be nonnegative")
        reference = np.fromiter(map(int, text), dtype=np.uint8)
    artifacts = {} if reference is not None else None

    engine = Signal9Engine(checkpoint=args.checkpoint)
    if not args.json:
        print(f"[*] Signal 9: Analyzing '{args.file}'...")
    try:
        results = engine.process_file(
            filepath=args.file,
            fs=args.fs,
            fc=args.fc,
            fmt=args.format,
            exploit=not args.no_exploit,
            max_exploit_tracks=args.max_tracks,
            artifacts=artifacts,
        )
    except Exception as e:
        print(f"[!] Error processing file: {e}", file=sys.stderr)
        sys.exit(1)

    if reference is not None:
        from signal9.dsp.bit_alignment import reference_ber
        for track in artifacts.get("tracks", []):
            bits = track.get("decoded_bits")
            if bits is not None:
                sig = results["signals"][track["track_id"] - 1]
                sig["exploitation"]["reference_ber"] = reference_ber(bits, reference, args.reference_offset)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
        if args.save_json:
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
        return

    inp = results.get("input", {})
    print("=" * 74)
    print(f"SIGNAL 9 ANALYSIS REPORT: {Path(args.file).name}")
    print("=" * 74)
    fs_note = "  (ASSUMED — not in the file; pass --fs)" if inp.get("sample_rate_assumed") else ""
    print(f"Sample Rate : {_fmt_hz(results.get('sample_rate_sps', 0))}  "
          f"[source: {inp.get('sample_rate_source', 'unknown')}]{fs_note}")
    print(f"Center Freq : {_fmt_hz(results.get('center_freq_hz', 0))}  "
          f"[source: {inp.get('center_freq_source', 'unknown')}]")
    print(f"Input       : {inp.get('input_kind')}")
    health = results.get("health", {})
    print("Front-End Health:")
    print(f"  - ADC Clipping Fraction: {health.get('clipping_fraction', 0.0) * 100:.2f}%")
    improp = health.get("iq_impropriety_db")
    print(f"  - IQ Impropriety       : {f'{improp:.1f} dB' if improp is not None else 'N/A'}")
    print(f"  - LO Leakage / DC Spike: {'YES (masked)' if health.get('dc_spike_detected') else 'NO'}")
    print(f"\nDetections Found: {results.get('total_signals_detected', 0)}")
    print("-" * 74)

    for sig in results.get("signals", []):
        tid = sig.get("track_id")
        fc_rel = sig.get("freq_center_relative_hz", 0.0) / 1e3
        bw_khz = sig.get("bandwidth_hz", 0.0) / 1e3
        snr = sig.get("snr_db", 0.0)
        cls = sig.get("classification", {})
        l1, l2, l3 = cls.get("l1", {}), cls.get("l2", {}), cls.get("l3", {})

        print(f"Signal #{tid}: Offset {fc_rel:+.2f} kHz | BW {bw_khz:.1f} kHz | SNR {snr:+.1f} dB")
        if sig.get("carrier_offset_hz") is not None:
            print(f"  Total carrier: {sig['carrier_offset_hz']:+.3f} Hz relative to capture centre")
        print(f"  Decision     : {sig.get('modulation_decision', {})}")
        print(f"  L1 Family    : {l1.get('prediction')} (conf: {l1.get('confidence', 0) * 100:.1f}%) [{l1.get('decision_state')}]")
        print(f"  L2 Modulation: {l2.get('prediction')} (conf: {l2.get('confidence', 0) * 100:.1f}%) [{l2.get('decision_state')}]")
        print(f"  L3 Protocol  : {l3.get('prediction')} (conf: {l3.get('confidence', 0) * 100:.1f}%) [{l3.get('decision_state')}]")
        if sig.get("exploitation"):
            _print_exploitation(sig["exploitation"])
        print("-" * 74)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[+] Report saved to {args.save_json}")


if __name__ == "__main__":
    main()
