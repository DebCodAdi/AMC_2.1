"""Tests for failure mining taxonomy, Rule R16 pseudo-label admission, and adversarial hunting."""
import pytest
import numpy as np
from signal9.engine.mine import classify_outcome, mine_failures
from signal9.engine.pseudolabel import admit_pseudo_label
from signal9.engine.synthesize import expand_failure_neighbourhood
from signal9.engine.adversary import hunt_adversarial_errors


def test_failure_taxonomy_classification():
    # Confidently wrong (F-A)
    f_a = classify_outcome({"pred_mod": "BPSK", "true_mod": "QPSK", "confidence": 0.95})
    assert f_a is not None
    assert f_a.failure_mode == "F-A"

    # Correct but uncalibrated (F-B)
    f_b = classify_outcome({"pred_mod": "QPSK", "true_mod": "QPSK", "confidence": 0.40})
    assert f_b is not None
    assert f_b.failure_mode == "F-B"

    # Near-miss with high plausibility (F-C)
    f_c = classify_outcome({"pred_mod": "BPSK", "true_mod": "QPSK", "confidence": 0.60, "plausibility": 0.85})
    assert f_c is not None
    assert f_c.failure_mode == "F-C"

    # Unknown-not-rejected (F-E)
    f_e = classify_outcome({"pred_mod": "QPSK", "true_mod": "NOISE", "confidence": 0.85, "is_negative": True})
    assert f_e is not None
    assert f_e.failure_mode == "F-E"


def test_rule_r16_pseudolabel_admission():
    # Tier 1: CRC valid -> admitted with weight 1.0
    adm_crc = admit_pseudo_label({"crc_valid": True, "label": "BPSK"})
    assert adm_crc.label == "BPSK"
    assert adm_crc.weight == 1.0
    assert not adm_crc.unlabelled

    # Tier 2: Multi-branch agreement -> admitted with weight 0.5
    adm_branch = admit_pseudo_label({
        "crc_valid": False, "n_branches_agreeing": 3, "syncword_found": True, "evm": 0.10, "label": "QPSK"
    })
    assert adm_branch.label == "QPSK"
    assert adm_branch.weight == 0.5

    # Pure model confidence alone MUST be rejected (Rule R16)
    adm_conf = admit_pseudo_label({"confidence": 0.999, "crc_valid": False, "label": "QAM16"})
    assert adm_conf.label is None
    assert adm_conf.weight == 0.0
    assert adm_conf.unlabelled is True
    assert adm_conf.verified_by == "rejected_unverified"


def test_failure_neighbourhood_synthesis():
    synth = expand_failure_neighbourhood({"true_mod": "QPSK", "snr_db": 12.0}, n_samples=10)
    assert len(synth) == 10
    assert synth[0]["mod"] == "QPSK"
    assert synth[0]["verified_by"] == "construction"
    assert len(synth[0]["iq"]) > 0


def test_adversarial_hunt():
    def mock_model(iq):
        # Biased dummy model
        return ("BPSK", 0.95) if np.real(iq[0]) > 0 else ("QPSK", 0.95)

    errors, yield_rate = hunt_adversarial_errors(mock_model, budget=20)
    assert isinstance(errors, list)
    assert yield_rate >= 0.0
