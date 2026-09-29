"""NTRO SIH 26147 — diagnostic end-to-end pipeline smoke test.

This script generates a known QPSK/AX.25 capture and reports what the real
pipeline recovered.  It is deliberately not a compliance certificate: it says
``PASS`` only when an end-to-end CRC lock is actually observed.

The pipeline implements paths for these requirements:
  1. Parameter Extraction (Fs, Fc, SNR, Baud, Roll-off)
  2. Blind Modulation Recognition & Demodulation (QPSK / FSK)
  3. De-interleaving (Matrix / Helical / Convolutional)
  4. FEC Decoding (Viterbi Convolutional K=7, Reed-Solomon, LDPC)
  5. Bit Stream Correlation (Sync word, Frame payload, CRC-16)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal9.pipeline import Signal9Engine
from signal9.sim.framing import frame_ax25_bits
from signal9.dsp.constellation import bits_to_symbols
from signal9.dsp.matched_filter import rrc_taps


def generate_demo_signal(filename: str = "demo_capture.cf32", fs: float = 48000.0) -> Path:
    """Generate a realistic test transmission containing framed, CRC-checked payload."""
    filepath = PROJECT_ROOT / filename
    print(f"[*] Generating realistic RF test capture: {filepath.name} (Fs = {fs/1000:.1f} kHz)...")

    # 1. Generate an on-air AX.25/HDLC frame: LSB-first octets, bit stuffing,
    # and CRC-16/X-25.  Do not convert the legacy byte-container with
    # np.unpackbits here: it is not an AX.25 radio wire stream.
    payload_text = b"NTRO-SIH-26147: SATELLITE_TELEMETRY_PAYLOAD_VALIDATED"
    bits = frame_ax25_bits(dest_call="DEST01", src_call="SRC001", payload=payload_text)

    # 2. Map the actual wire bitstream to the QPSK constellation.
    symbols = bits_to_symbols(bits, "QPSK")

    # 3. Transmit RRC pulse shaping at 4 samples/symbol.  Repeating each
    # symbol produces a rectangular waveform, not an RRC-shaped QPSK signal;
    # it also makes phase-transition impulses resemble FSK to a blind
    # discriminator.  This is a real transmitter waveform instead.
    samples_per_symbol = 4
    up = np.zeros(len(symbols) * samples_per_symbol, dtype=np.complex64)
    up[::samples_per_symbol] = symbols
    iq_burst = np.convolve(
        up, rrc_taps(alpha=0.35, sps=samples_per_symbol, span=10), mode="full"
    ).astype(np.complex64)

    # Repeat frames back-to-back.  HDLC flags already provide frame boundaries;
    # inserting a hard zero-amplitude gap creates an artificial carrier
    # discontinuity that is not part of the QPSK waveform and biases blind CFO
    # estimation.
    iq = np.tile(iq_burst, 12)

    # 4. Add channel noise (AWGN at +22 dB SNR) and slight frequency offset (+120 Hz)
    t = np.arange(len(iq)) / fs
    cfo_hz = 120.0
    iq = iq * np.exp(1j * 2 * np.pi * cfo_hz * t)
    noise_sigma = 0.05
    rng = np.random.default_rng(26147)
    noise = noise_sigma * (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq)))
    iq = (iq + noise).astype(np.complex64)

    # 5. Save to raw binary complex64 format (.cf32)
    iq.tofile(str(filepath))
    print(f"[+] Saved {len(iq)} complex samples to {filepath.name}")
    return filepath


def main():
    print("=" * 75)
    print("  NTRO SIH 26147: AUTOMATED SIGNAL ANALYSIS DEMONSTRATION")
    print("=" * 75)

    # Step 1: Create synthetic capture
    test_file = generate_demo_signal("demo_capture.cf32", fs=48000.0)

    # Step 2: Initialize Signal 9 Pipeline Engine
    print("\n[*] Initializing Signal 9 Automated Pipeline Engine...")
    checkpoint_path = PROJECT_ROOT / "models" / "s9_hybrid_xcit_L128_best.ckpt"
    ckpt_arg = str(checkpoint_path) if checkpoint_path.exists() else None
    
    engine = Signal9Engine(checkpoint=ckpt_arg)
    print(f"[+] Engine loaded successfully. (Classifier: {checkpoint_path.name if ckpt_arg else 'NOT TRAINED'})")

    # Step 3: Run Blind End-to-End Analysis
    print(f"[*] Processing file: {test_file.name}...")
    artifacts = {}
    report = engine.process_file(
        filepath=test_file,
        fs=48000.0,
        fmt="cf32",
        max_exploit_tracks=4,
        artifacts=artifacts
    )

    # Step 4: Display Output Summary
    print("\n" + "=" * 75)
    print("  ANALYSIS RESULTS (PROBLEM STATEMENT COMPLIANCE REPORT)")
    print("=" * 75)
    print(f"Sampling Frequency (Fs) : {report.get('sample_rate_sps', 0)/1000:.2f} kHz")
    print(f"Center Frequency (Fc)   : {report.get('center_freq_hz', 0)} Hz")
    print(f"Signal Format Detected  : {report.get('input', {}).get('input_kind', 'raw_iq')}")
    print(f"ADC Clipping Level      : {report.get('health', {}).get('clipping_fraction', 0)*100:.2f}%")
    print(f"DC / LO Leakage Status  : {report.get('health', {}).get('dc_spike_detected', False)}")

    tracks = report.get("signals", [])
    print(f"\nTracks / Emitters Detected: {len(tracks)}")
    for i, track in enumerate(tracks, 1):
        print(f"\n--- Signal Track #{i} ---")
        print(f"  Center Offset : {track.get('center_offset_hz', 0)/1000:+.2f} kHz")
        print(f"  Bandwidth     : {track.get('bandwidth_hz', 0)/1000:.2f} kHz")
        print(f"  Estimated SNR : {track.get('snr_db', 0):+.1f} dB")
        
        # Classification
        pred = track.get("classification", {})
        l1, l2 = pred.get("l1", {}), pred.get("l2", {})
        print(f"  Modulation L1 : {l1.get('prediction')} ({l1.get('confidence', 0)*100:.1f}%)")
        print(f"  Modulation L2 : {l2.get('prediction')} ({l2.get('confidence', 0)*100:.1f}%)")
        
        # Exploitation & Demodulation
        ex = track.get("exploitation", {})
        demod = ex.get("demodulation", {})
        print(f"  Demodulation  : {demod.get('modulation')} (Family: {demod.get('family')})")
        print(f"  Symbol Rate   : {demod.get('symbol_rate_hz', 0):.1f} Baud")
        print(f"  EVM / Quality : {demod.get('evm_db', 0):.1f} dB (CFO: {demod.get('cfo_hz', 0):+.1f} Hz)")
        
        # Interleaver & FEC
        il = ex.get("interleaver", {})
        print(f"  De-Interleaver: {il.get('type', 'none')}")
        fec = ex.get("fec", {})
        print(f"  FEC Decoder   : {fec.get('class', 'none')} ({fec.get('name', 'N/A')})")
        
        # Frame & Bit Correlation
        fr = ex.get("frames", {})
        if fr.get("crc"):
            print(f"  Frame CRC     : {fr['crc'].get('model')} (Valid frames: {fr.get('crc_frames_ok')}/{fr.get('crc_frames_checked')})")
        
        dec = ex.get("decoded", {})
        if dec.get("message_text"):
            print(f"  Decoded Payload: {dec['message_text'][:160]}")
        elif dec.get("preview_text"):
            print(f"  Decoded Stream: {dec['preview_text'][:80]}")

    # Step 5: Save JSON report
    out_json = PROJECT_ROOT / "demo_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        # Filter non-serializable objects
        json.dump(report, f, indent=2, default=str)
    print(f"\n[+] Full structured JSON report saved to: {out_json.name}")
    crc_locks = sum(
        int((track.get("exploitation", {}).get("frames", {}) or {}).get("crc_frames_ok", 0) or 0)
        for track in tracks
    )
    if crc_locks:
        print(f"\n[PASS] End-to-end CRC validation succeeded ({crc_locks} valid frame(s)).")
    else:
        print("\n[NOT VALIDATED] No CRC-valid frame was recovered. The JSON report is the "
              "authoritative diagnostic output; do not treat this run as a successful decode.")
    print("Run `python gui.py` to inspect this signal visually in the interactive GUI.")


if __name__ == "__main__":
    main()
