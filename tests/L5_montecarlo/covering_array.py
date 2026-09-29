"""Pairwise (strength-2 covering array) parameter sampler for L5 Monte-Carlo.

Generates a compact set of Truth configurations such that every pair of
parameter values is tested together at least once (all-pairs / strength-2).
Reduces combinatorial explosion from >10^7 grid points to ~60-120 test cases.

Spec reference: §6.1, §6.2, §2.7 (PS26147).
"""
from __future__ import annotations
import itertools
from typing import Any
import numpy as np
from tests.L5_montecarlo.generator import Truth

# Parameter domain definitions
PARAM_DOMAINS = {
    "mod": ["BPSK", "QPSK", "8PSK", "16QAM"],
    "snr_db": [-5.0, 0.0, 5.0, 15.0, 25.0],
    "sps": [2.0, 2.4, 4.0, 8.0],
    "alpha": [0.2, 0.35, 0.5],
    "cfo_frac": [-0.05, 0.0, 0.02, 0.1],
    "sco_ppm": [0.0, 50.0, 100.0],
    "phase_off": [0.0, float(np.pi / 4.0)],
    "iq_gain_db": [0.0, 1.0],
    "iq_phase_deg": [0.0, 5.0],
    "dc": [0.0 + 0j, 0.05 + 0.05j],
    "multipath": ["none", "flat", "2tap"],
    "fec": ["none", "conv_k7_r12", "rs255_223", "ldpc648_324"],
    "interleave": ["none", "block_4x8", "block_8x4"],
}


def generate_pairwise_samples(domains: dict[str, list[Any]] | None = None,
                              seed: int = 42,
                              max_cases: int = 80) -> list[Truth]:
    """Generate strength-2 covering array of Truth records using greedy IPOG-like heuristic.

    Parameters
    ----------
    domains : dict mapping parameter names to candidate lists
    seed : int master RNG seed
    max_cases : int target sample count

    Returns
    -------
    list of Truth records
    """
    if domains is None:
        domains = PARAM_DOMAINS

    keys = list(domains.keys())
    rng = np.random.default_rng(seed)

    # Collect pairs that must be covered
    all_pairs: set[tuple[tuple[str, Any], tuple[str, Any]]] = set()
    for i in range(len(keys)):
        k1 = keys[i]
        for j in range(i + 1, len(keys)):
            k2 = keys[j]
            for v1 in domains[k1]:
                for v2 in domains[k2]:
                    all_pairs.add(((k1, v1), (k2, v2)))

    uncovered = set(all_pairs)
    cases: list[dict[str, Any]] = []

    # Greedy generation
    while uncovered and len(cases) < max_cases:
        candidate: dict[str, Any] = {}
        # Pick random starting values
        for k in keys:
            candidate[k] = rng.choice(domains[k])

        # Optimize candidate to maximize uncovered pair coverage
        for k in keys:
            best_val = candidate[k]
            best_cov = -1
            for v in domains[k]:
                candidate[k] = v
                # Count pairs this candidate covers
                cov = 0
                for idx1 in range(len(keys)):
                    for idx2 in range(idx1 + 1, len(keys)):
                        pair = ((keys[idx1], candidate[keys[idx1]]), (keys[idx2], candidate[keys[idx2]]))
                        if pair in uncovered:
                            cov += 1
                if cov > best_cov:
                    best_cov = cov
                    best_val = v
            candidate[k] = best_val

        # Remove covered pairs
        for idx1 in range(len(keys)):
            for idx2 in range(idx1 + 1, len(keys)):
                pair = ((keys[idx1], candidate[keys[idx1]]), (keys[idx2], candidate[keys[idx2]]))
                uncovered.discard(pair)

        cases.append(candidate)

    # Convert to Truth records
    records: list[Truth] = []
    for idx, c in enumerate(cases):
        record = Truth(
            mod=c["mod"],
            snr_db=float(c["snr_db"]),
            sps=float(c["sps"]),
            alpha=float(c["alpha"]),
            cfo_frac=float(c["cfo_frac"]),
            sco_ppm=float(c["sco_ppm"]),
            phase_off=float(c["phase_off"]),
            iq_gain_db=float(c["iq_gain_db"]),
            iq_phase_deg=float(c["iq_phase_deg"]),
            dc=complex(c["dc"]),
            multipath=c["multipath"],
            fec=c["fec"],
            interleave=c["interleave"],
            payload=b"ANTIGRAVITY_TEST_PAYLOAD_" + bytes([idx % 256]),
            n_slips=0,
            seed=seed + idx * 17
        )
        records.append(record)

    return records
