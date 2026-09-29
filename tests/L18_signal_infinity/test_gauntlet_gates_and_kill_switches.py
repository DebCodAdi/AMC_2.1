"""Tests for 10-Gate Gauntlet, Rule R20 Containment, and Rule R24 Kill Switches."""
import pytest
from signal9.gauntlet.gates import (
    gate_g0_reproducibility,
    gate_g1_containment,
    gate_g3_real_oracle_conformance,
    gate_g7_calibration,
    gate_g8_rejection
)
from signal9.gauntlet.run import evaluate_gauntlet
from signal9.engine.kill_switches import evaluate_kill_switches


def test_gate_g1_strict_containment():
    # If no cases lost -> pass
    res_pass = gate_g1_containment(new_correct_ids={"c1", "c2", "c3"}, old_correct_ids={"c1", "c2"})
    assert res_pass["pass"] is True
    assert res_pass["lost_count"] == 0

    # If any case lost (Rule R20) -> fail
    res_fail = gate_g1_containment(new_correct_ids={"c1", "c3"}, old_correct_ids={"c1", "c2"})
    assert res_fail["pass"] is False
    assert res_fail["lost_count"] == 1
    assert "c2" in res_fail["lost_ids"]


def test_gate_g8_rejection():
    # 0 false locks -> pass
    res_pass = gate_g8_rejection(false_locks_on_noise=0, open_set_auroc=0.95, fpr_at_95_tpr=0.04)
    assert res_pass["pass"] is True

    # Any false lock on noise -> fail (Rule R14)
    res_fail = gate_g8_rejection(false_locks_on_noise=1, open_set_auroc=0.95, fpr_at_95_tpr=0.04)
    assert res_fail["pass"] is False


def test_kill_switches_audit():
    # Clean status -> no active trips
    trips_clean = evaluate_kill_switches({
        "diversity_drop_pct": 0.0,
        "worst_cell_drop_pts": -0.5,
        "hunter_yield_delta": -1.0,
        "ece": 0.02,
        "anchor_fraction": 0.15,
    })
    assert not any(t.triggered for t in trips_clean)

    # Trigger diversity collapse (> 2%)
    trips_div = evaluate_kill_switches({"diversity_drop_pct": 3.5})
    assert any(t.switch_id == "diversity_collapse" for t in trips_div)

    # Trigger anchor missing (< 10%, Rule R15)
    trips_anc = evaluate_kill_switches({"anchor_fraction": 0.05})
    assert any(t.switch_id == "anchor_missing" for t in trips_anc)

    # Trigger frozen holdout tampered (Rule R19)
    trips_holdout = evaluate_kill_switches({"holdout_tampered": True})
    assert any(t.switch_id == "holdout_touched" for t in trips_holdout)
