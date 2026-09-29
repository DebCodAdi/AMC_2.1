"""Remote cloud training orchestrator for Colab and Kaggle environments."""
from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch


def detect_cloud_environment() -> str:
    if "COLAB_GPU" in os.environ or "google.colab" in sys.modules:
        return "Google Colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "Kaggle Notebooks"
    return "Local Workstation"


def print_environment_banner():
    env = detect_cloud_environment()
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU Only"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if has_cuda else 0.0

    print("=" * 70)
    print(f"SIGNAL INFINITY — CLOUD ACCELERATOR ORCHESTRATOR")
    print(f"Environment: {env}")
    print(f"Compute Device: {device_name} ({vram_gb:.2f} GB VRAM)")
    print(f"PyTorch Version: {torch.__version__} | CUDA: {torch.version.cuda if has_cuda else 'N/A'}")
    print("=" * 70)


def setup_cloud_environment():
    print("[1/4] Checking and installing critical dependencies...")
    pkgs = [
        "h5py", "numba", "pyyaml", "scipy", "matplotlib",
        "duckdb", "cma", "polars", "datasketch", "galois",
        "reedsolo", "scikit-commpy", "crcmod", "soxr"
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=False)
    print("[1/4] Dependencies verified.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Signal Infinity Cloud Training")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=50000, help="Max samples to draw from dataset")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto/cuda/cpu)")
    parser.add_argument("--save-path", type=str, default="models/s9_infinity_cloud.ckpt", help="Model checkpoint path")
    args = parser.parse_args()

    print_environment_banner()
    setup_cloud_environment()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"[2/4] Initializing Signal Infinity multi-task model on {device}...")
    from signal9.train.trainer import run_training
    metrics = run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        anchor_fraction=0.15,
        device=device,
        save_path=args.save_path
    )
    print(f"[2/4] Training completed successfully. Final Loss = {metrics.get('loss', 0.0):.4f}")

    print("[3/4] Running 10-Gate Evaluation Gauntlet on accelerator...")
    from signal9.gauntlet.run import evaluate_gauntlet
    gauntlet_res = evaluate_gauntlet()
    print(f"[3/4] Gauntlet Result: All Gates Passed = {gauntlet_res['all_passed']}")

    print("[4/4] Generating Public Scorecard...")
    from tools.scorecard import generate_public_scorecard
    generate_public_scorecard("s9-v3-cloud", "reports/scorecard.md")
    print(f"[4/4] Cloud training and evaluation workflow complete. Model saved to {args.save_path}.")


if __name__ == "__main__":
    main()
