"""Local SDR capture utility (RTL-SDR / HackRF / Airspy) with automatic JSON sidecar."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

def capture_rtl(freq_hz: float,
                sample_rate: float = 2400000.0,
                gain_db: float = 30.0,
                num_samples: int = 24000000,
                out_dir: str | Path = "data/raw/local",
                hw_name: str = "RTL-SDR v4",
                antenna: str = "dipole") -> Path:
    """Execute rtl_sdr capture and write mandatory JSON sidecar."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"sess_{timestamp}_{int(freq_hz/1e6)}M_g{int(gain_db)}"
    bin_file = out_p / f"{stem}.cu8"
    sidecar_file = out_p / f"{stem}.json"

    rtl_bin = shutil.which("rtl_sdr")
    if rtl_bin:
        cmd = [
            rtl_bin,
            "-f", str(int(freq_hz)),
            "-s", str(int(sample_rate)),
            "-g", str(gain_db),
            "-n", str(num_samples),
            str(bin_file)
        ]
        print(f"[*] Executing: {' '.join(cmd)}")
        subprocess.run(cmd)
    else:
        print(f"[!] rtl_sdr binary not found in PATH. Simulating capture metadata.")
        # Create dummy file if rtl_sdr is not connected
        bin_file.touch()

    sidecar_data = {
        "sample_rate": float(sample_rate),
        "center_freq": float(freq_hz),
        "datatype": "cu8",
        "hw": hw_name,
        "zero_if": True,
        "gain_db": float(gain_db),
        "antenna": antenna,
        "session": stem,
        "timestamp_utc": timestamp,
    }

    with open(sidecar_file, "w", encoding="utf-8") as f:
        json.dump(sidecar_data, f, indent=2)

    print(f"[+] Capture sidecar written to {sidecar_file}")
    return bin_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local RTL-SDR capture with sidecar")
    parser.add_argument("--fc", type=float, required=True, help="Center freq in Hz")
    parser.add_argument("--fs", type=float, default=2400000.0, help="Sample rate in Hz")
    parser.add_argument("--gain", type=float, default=30.0, help="Gain in dB")
    parser.add_argument("--samples", type=int, default=24000000, help="Number of samples")
    parser.add_argument("--out", default="data/raw/local", help="Output directory")
    args = parser.parse_args()
    capture_rtl(args.fc, args.fs, args.gain, args.samples, args.out)
