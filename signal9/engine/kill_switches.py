"""Automated kill switches with pre-declared trigger conditions (SIGNAL INFINITY Rule R24, §9.4)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class KillSwitchTrip:
    switch_id: str
    triggered: bool
    trigger_message: str
    action_required: str


def evaluate_kill_switches(status_dict: dict) -> list[KillSwitchTrip]:
    """Audit all 8 pre-declared kill switches (Rule R24)."""
    trips = []

    # 1. diversity_collapse: any corpus diversity metric decreases by > 2%
    div_drop = float(status_dict.get("diversity_drop_pct", 0.0))
    if div_drop > 2.0:
        trips.append(KillSwitchTrip(
            "diversity_collapse", True,
            f"Corpus diversity metric dropped by {div_drop:.1f}% (> 2% threshold)",
            "Halt engine; quarantine generation data; trigger alert (Rule R22)"
        ))

    # 2. tail_regression: any capability-surface cell drops > 2 pts
    worst_drop = float(status_dict.get("worst_cell_drop_pts", 0.0))
    if worst_drop < -2.0:
        trips.append(KillSwitchTrip(
            "tail_regression", True,
            f"Capability-surface tail cell regressed by {abs(worst_drop):.1f} pts (> 2 pts)",
            "Reject candidate; roll back; alert"
        ))

    # 3. hunter_yield_rise: confident error yield increases vs previous generation
    hunter_delta = float(status_dict.get("hunter_yield_delta", 0.0))
    if hunter_delta > 0.0:
        trips.append(KillSwitchTrip(
            "hunter_yield_rise", True,
            f"Hunter yield increased by {hunter_delta:.1f} errors/100k",
            "HALT IMMEDIATELY: model got worse where it matters most. Roll back."
        ))

    # 4. calibration_drift: ECE > 0.05 on the frozen holdout
    ece = float(status_dict.get("ece", 0.02))
    if ece > 0.05:
        trips.append(KillSwitchTrip(
            "calibration_drift", True,
            f"Expected Calibration Error {ece:.3f} exceeded 0.05",
            "Halt pseudo-labelling (miscalibrated model poisons corpus); recalibrate."
        ))

    # 5. pseudolabel_purity: audited pseudo-label error rate > 1%
    pseudo_error_rate = float(status_dict.get("pseudo_error_rate", 0.005))
    if pseudo_error_rate > 0.01:
        trips.append(KillSwitchTrip(
            "pseudolabel_purity", True,
            f"Pseudo-label audited error rate {pseudo_error_rate:.1%} exceeded 1.0%",
            "Halt pseudo-labelling; enforce Tier 1 only."
        ))

    # 6. holdout_touched: frozen holdout hash altered (Rule R19)
    holdout_tampered = bool(status_dict.get("holdout_tampered", False))
    if holdout_tampered:
        trips.append(KillSwitchTrip(
            "holdout_touched", True,
            "Frozen holdout hash diverged from recorded hash",
            "HARD STOP: Invalidate all metrics produced since modification (Rule R19)."
        ))

    # 7. anchor_missing: training batch anchor fraction < 10% (Rule R15)
    anchor_frac = float(status_dict.get("anchor_fraction", 0.15))
    if anchor_frac < 0.10:
        trips.append(KillSwitchTrip(
            "anchor_missing", True,
            f"Anchor fraction {anchor_frac:.1%} fell below 10%",
            "Abort the training run immediately (Rule R15)."
        ))

    # 8. provenance_gap: sample lacks provenance or generation distance > 1 (Rule R17, R18)
    provenance_gap = bool(status_dict.get("provenance_gap", False))
    if provenance_gap:
        trips.append(KillSwitchTrip(
            "provenance_gap", True,
            "Detected samples lacking full provenance or exceeding generation distance 1",
            "Quarantine offending samples; fail audit (Rule R17, R18)."
        ))

    return trips
