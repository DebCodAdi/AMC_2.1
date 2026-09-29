"""L5 Monte-Carlo sweep runner (§6.2, §2.7).

Executes automated parameter sweeps over the synthetic capture factory using
the strength-2 covering array sampler. Computes accuracy metrics across
operating conditions (SNR, CFO, SCO, roll-off, multipath, FEC).

Spec reference: §6.2, §2.7 (PS26147).
"""
from __future__ import annotations
import argparse
import json
import time
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.L5_montecarlo.generator import make_capture, Truth
from tests.L5_montecarlo.covering_array import generate_pairwise_samples
from signal9.config import Config
from signal9.status import Status
from signal9.decide.blackboard import BlackboardArbiter
from signal9.dsp.matched_filter import apply_rrc
from signal9.dsp.timing_recovery import gardner_timing_recovery
from signal9.dsp.cfo import mth_power_cfo, correct_cfo


def run_single_capture(truth: Truth, config: Config) -> dict:
    """Run one synthetic capture through the hardened front-end & blackboard."""
    t0 = time.perf_counter()
    iq, _ = make_capture(truth)

    fs = 48000.0
    # 1. CFO coarse correction
    cfo_est = mth_power_cfo(iq, fs=fs, M=4)
    iq_cfo = correct_cfo(iq, fs=fs, cfo_hz=cfo_est)

    # 2. Matched filter if enabled
    if config.enable_matched_filter:
        iq_mf = apply_rrc(iq_cfo, alpha=truth.alpha, sps=int(round(truth.sps)))
    else:
        iq_mf = iq_cfo

    # 3. Timing recovery
    symbols_1sps, symbols_2sps, sco_est = gardner_timing_recovery(iq_mf, sps=truth.sps)

    # 4. Blackboard arbitration
    arbiter = BlackboardArbiter(config)
    winner, _ = arbiter.arbitrate(symbols_1sps, mod_type=truth.mod, fec_hint=truth.fec)

    t1 = time.perf_counter()
    crc_locked = bool(winner and winner.crc_locked)
    status_str = winner.status.value if winner else Status.ALL_BRANCHES_FAILED.value

    return {
        "mod_true": truth.mod,
        "snr_db": truth.snr_db,
        "sps": truth.sps,
        "alpha": truth.alpha,
        "cfo_frac": truth.cfo_frac,
        "sco_ppm": truth.sco_ppm,
        "fec_true": truth.fec,
        "multipath": truth.multipath,
        "crc_locked": crc_locked,
        "status": status_str,
        "latency_ms": (t1 - t0) * 1000.0,
    }


def run_monte_carlo_sweep(n_samples: int = 50,
                           seed: int = 42,
                           config: Config | None = None) -> list[dict]:
    """Execute L5 Monte-Carlo sweep."""
    if config is None:
        config = Config(
            enable_matched_filter=True,
            early_exit_on_crc=True,
            max_branches=64
        )

    samples = generate_pairwise_samples(seed=seed, max_cases=n_samples)
    results: list[dict] = []

    print(f"Starting L5 Monte-Carlo sweep over {len(samples)} parameter configurations...")
    for idx, truth in enumerate(samples):
        res = run_single_capture(truth, config)
        results.append(res)
        if (idx + 1) % 10 == 0 or idx == len(samples) - 1:
            n_locked = sum(1 for r in results if r["crc_locked"])
            print(f"  [{idx + 1}/{len(samples)}] Progress: {n_locked}/{len(results)} CRC locked ({100.0 * n_locked / len(results):.1f}%)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L5 Monte-Carlo Sweep Runner")
    parser.add_argument("--samples", type=int, default=30, help="Number of test captures")
    parser.add_argument("--out", type=str, default="reports/l5_sweep_results.json", help="Output JSON path")
    args = parser.parse_args()

    results = run_monte_carlo_sweep(n_samples=args.samples)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}")
