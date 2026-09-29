"""KiwiSDR Automated Collection Bot (fixes A8, B13).
Uses kiwirecorder with --tlimit, -g, --kiwi-wav to preserve GNSS timestamps.
"""
from __future__ import annotations
import argparse
import subprocess
import shutil
from pathlib import Path

DEFAULT_KIWI_HOSTS = [
    ("kiwisdr.linkfanel.net", 8073, "kiwi_eu_1"),
    ("kiwisdr.linkfanel.net", 8074, "kiwi_eu_2"),
    ("rx.linkfanel.net", 8073, "kiwi_asia_1"),
]

def run_kiwi_capture(host: str,
                     port: int,
                     freq_khz: float = 14074.0,
                     duration_sec: int = 300,
                     out_dir: str | Path = "data/raw/kiwi",
                     station_id: str = "rx01") -> None:
    """Invokes kiwirecorder.py with required flags."""
    kiwi_bin = shutil.which("kiwirecorder.py") or shutil.which("kiwirecorder")
    if not kiwi_bin:
        print("[!] kiwirecorder not found in PATH. Clone from https://github.com/jks-prv/kiwiclient.git")
        return

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    cmd = [
        kiwi_bin,
        "-s", host,
        "-p", str(port),
        "-u", "sigsense-research",
        "-f", str(freq_khz),
        "-m", "iq",
        "-L", "-6000",
        "-H", "6000",
        "-g", "50",
        "--tlimit", str(duration_sec),
        "--kiwi-wav",
        f"--station={station_id}",
        "-d", str(out_p)
    ]
    print(f"[*] Starting Kiwi capture: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KiwiSDR HF collection bot")
    parser.add_argument("--host", default="kiwisdr.linkfanel.net")
    parser.add_argument("--port", type=int, default=8073)
    parser.add_argument("--freq", type=float, default=14074.0, help="Dial frequency in kHz")
    parser.add_argument("--tlimit", type=int, default=300, help="Capture limit in seconds")
    parser.add_argument("--out", default="data/raw/kiwi")
    args = parser.parse_args()
    run_kiwi_capture(args.host, args.port, args.freq, args.tlimit, args.out)
