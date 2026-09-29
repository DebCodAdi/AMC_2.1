"""Automated KiwiSDR IQ recording script."""
import argparse
import subprocess
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8073)
    parser.add_argument("--freq", type=float, required=True, help="kHz")
    parser.add_argument("--outdir", default="data/raw")
    parser.add_argument("--tlimit", type=int, default=60, help="seconds")
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    
    # Corrected kiwirecorder command from v5 review B13
    cmd = [
        "kiwirecorder",
        "-s", args.host,
        "-p", str(args.port),
        "-f", str(args.freq),
        "-m", "iq",
        "-g", "50", # Pin AGC
        "--kiwi-wav", # Keep GNSS timestamps
        "--tlimit", str(args.tlimit),
        "-d", args.outdir
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
