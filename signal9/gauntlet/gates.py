"""The 10 Evaluation Gauntlet Gates G0 through G9 (SIGNAL INFINITY Part 5)."""
from __future__ import annotations
import numpy as np


def gate_g0_reproducibility(runs: list[dict]) -> dict:
    """G0: Byte-identical outputs across repeated runs with identical seed (U6)."""
    if len(runs) < 2:
        return {"pass": True, "detail": "Single run baseline"}
    first = runs[0]
    for i, r in enumerate(runs[1:], 1):
        if r != first:
            return {"pass": False, "detail": f"Run {i} diverged from run 0"}
    return {"pass": True, "detail": f"{len(runs)} runs byte-identical"}


def gate_g1_containment(new_correct_ids: set[str], old_correct_ids: set[str]) -> dict:
    """G1: Strict set containment: correct(new) ⊇ correct(old), lost == 0 (U3, Rule R20)."""
    lost = old_correct_ids - new_correct_ids
    gained = new_correct_ids - old_correct_ids
    is_pass = (len(lost) == 0)
    return {
        "pass": is_pass,
        "lost_count": len(lost),
        "gained_count": len(gained),
        "lost_ids": sorted(list(lost))[:10],
    }


def gate_g2_golden_payloads(golden_matches: int, total_golden: int) -> dict:
    """G2: Every frozen byte-exact golden payload reproduces without regression (Rule R4)."""
    is_pass = (golden_matches == total_golden and total_golden > 0)
    return {
        "pass": is_pass,
        "matches": golden_matches,
        "total": total_golden,
    }


def gate_g3_real_oracle_conformance(unvalidated_frames_emitted: int) -> dict:
    """G3: Real-oracle conformance: ZERO emitted frames failing their own protocol CRC (U2)."""
    is_pass = (unvalidated_frames_emitted == 0)
    return {
        "pass": is_pass,
        "unvalidated_frames": unvalidated_frames_emitted,
    }


def gate_g4_forgetting(forgetting_pts: float, bwt: float) -> dict:
    """G4: Continual learning forgetting <= 2.0 pts, BWT >= -1.0 (§4.3)."""
    is_pass = (forgetting_pts <= 2.0 and bwt >= -1.0)
    return {
        "pass": is_pass,
        "forgetting_pts": forgetting_pts,
        "bwt": bwt,
    }


def gate_g5_capability_surface(cell_deltas: list[float], coverage_non_decreasing: bool) -> dict:
    """G5: Capability surface: no cell regresses > 1.0 pt, coverage non-decreasing (U4, Rule R22)."""
    worst_drop = min(cell_deltas) if cell_deltas else 0.0
    is_pass = (worst_drop >= -1.0 and coverage_non_decreasing)
    return {
        "pass": is_pass,
        "worst_drop_pts": worst_drop,
        "coverage_non_decreasing": coverage_non_decreasing,
    }


def gate_g6_statistics(mcnemar_p_value: float, gained: int) -> dict:
    """G6: McNemar exact p < 0.01 with gained > 0."""
    is_pass = (mcnemar_p_value < 0.01 and gained > 0)
    return {
        "pass": is_pass,
        "p_value": mcnemar_p_value,
        "gained": gained,
    }


def gate_g7_calibration(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> dict:
    """G7: Expected Calibration Error (ECE) <= 0.03 on Tier-R (U1)."""
    if len(confidences) == 0:
        return {"pass": True, "ece": 0.0}

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if np.any(mask):
            bin_conf = np.mean(confidences[mask])
            bin_acc = np.mean(accuracies[mask])
            ece += (np.sum(mask) / len(confidences)) * np.abs(bin_acc - bin_conf)

    is_pass = (ece <= 0.03)
    return {
        "pass": is_pass,
        "ece": float(ece),
    }


def gate_g8_rejection(false_locks_on_noise: int, open_set_auroc: float, fpr_at_95_tpr: float) -> dict:
    """G8: 0 false CRC locks on noise, open-set AUROC >= 0.92, FPR@95TPR <= 0.08 (U1, U5, Rule R14)."""
    is_pass = (false_locks_on_noise == 0 and open_set_auroc >= 0.92 and fpr_at_95_tpr <= 0.08)
    return {
        "pass": is_pass,
        "false_locks": false_locks_on_noise,
        "auroc": open_set_auroc,
        "fpr_at_95_tpr": fpr_at_95_tpr,
    }


def gate_g9_budget_and_fallback(p50_ms: float, p95_ms: float, classical_passes_g1_g3: bool) -> dict:
    """G9: Latency within budget (p50 <= 1200ms, p95 <= 3500ms) and classical-only path passes G1-G3 (Rule R21)."""
    is_pass = (p50_ms <= 1200.0 and p95_ms <= 3500.0 and classical_passes_g1_g3)
    return {
        "pass": is_pass,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "classical_fallback_verified": classical_passes_g1_g3,
    }
