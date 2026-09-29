"""False lock on noise benchmark (PS26147 Gate A4 / SIGNAL10 Rule R14).

Runs pure noise captures through the pipeline to ensure that random noise
never yields false CRC locks.

Usage::

    python -m signal9.bench.false_lock --n 10000 --seed 0 --out reports/false_lock.json
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np

from signal9.pipeline import Signal9Engine
from signal9.status import Status


def evaluate_false_lock(n_trials: int = 10000,
                        chunk_len: int = 8192,
                        fs: float = 48000.0,
                        seed: int = 0) -> dict:
    """Inject pure Gaussian and Rayleigh noise and verify zero CRC locks."""
    rng = np.random.default_rng(seed)
    engine = Signal9Engine()

    n_false_locks = 0
    statuses: dict[str, int] = {}
    t0 = time.perf_counter()

    for i in range(n_trials):
        # Mix AWGN, Rayleigh, and uniform noise
        noise_type = i % 3
        if noise_type == 0:
            noise = (rng.standard_normal(chunk_len) + 1j * rng.standard_normal(chunk_len)).astype(np.complex64)
        elif noise_type == 1:
            ray = rng.rayleigh(scale=1.0, size=chunk_len)
            phase = rng.uniform(-np.pi, np.pi, size=chunk_len)
            noise = (ray * np.exp(1j * phase)).astype(np.complex64)
        else:
            noise = (rng.uniform(-1.0, 1.0, chunk_len) + 1j * rng.uniform(-1.0, 1.0, chunk_len)).astype(np.complex64)

        res = engine.process(noise, fs=fs)
        status = res.get("status", Status.ALL_BRANCHES_FAILED.value)
        statuses[status] = statuses.get(status, 0) + 1

        if res.get("crc_locked", False) or status == Status.SUCCESS_CRC_LOCKED.value:
            n_false_locks += 1

    elapsed_s = time.perf_counter() - t0
    fpr = n_false_locks / max(1, n_trials)

    return {
        "n_trials": n_trials,
        "n_false_locks": n_false_locks,
        "false_positive_rate": fpr,
        "passed": n_false_locks == 0,
        "status_distribution": statuses,
        "elapsed_seconds": elapsed_s,
        "throughput_trials_per_sec": n_trials / max(1e-6, elapsed_s),
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description="False Lock on Noise Benchmark")
    parser.add_argument("--n", type=int, default=10000, help="Number of noise captures")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--out", type=str, default="reports/false_lock.json", help="Output path")
    args = parser.parse_args()

    print(f"Running false lock benchmark over {args.n} noise captures (seed={args.seed})...")
    result = evaluate_false_lock(n_trials=args.n, seed=args.seed)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Result: {result['n_false_locks']}/{result['n_trials']} false locks (Passed: {result['passed']})")
    print(f"Written to {out_p}")


if __name__ == "__main__":
    main()
