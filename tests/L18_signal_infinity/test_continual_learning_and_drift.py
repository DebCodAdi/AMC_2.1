"""Tests for continual learning buffer, regularisation, and deployment drift alarms."""
import pytest
import numpy as np
import torch
import torch.nn as nn
from signal9.continual.buffer import StratifiedReplayBuffer, ReplaySample
from signal9.continual.loss import ContinualRegularizer
from signal9.continual.drift import evaluate_drift_alarms, compute_mmd


def test_stratified_replay_buffer():
    buf = StratifiedReplayBuffer(budget=100)
    
    # Add anchor sample (Rule R15)
    s_anchor = ReplaySample("anc_01", np.zeros(64, dtype=np.complex64), "BPSK", 15, "awgn", "none", is_anchor=True)
    buf.add_sample(s_anchor)

    # Add regular stratum sample
    s_reg = ReplaySample("reg_01", np.zeros(64, dtype=np.complex64), "QPSK", 10, "rayleigh", "conv", is_anchor=False)
    buf.add_sample(s_reg)

    batch = buf.sample_batch(batch_size=4, anchor_fraction=0.25)
    assert len(batch) > 0
    # Must include anchor
    assert any(s.is_anchor for s in batch)


def test_ewc_and_lwf_losses():
    model_old = nn.Linear(10, 4)
    model_new = nn.Linear(10, 4)

    reg = ContinualRegularizer(old_model=model_old, ewc_lambda=100.0, lwf_lambda=1.0)
    ewc_l = reg.ewc_loss(model_new)
    assert ewc_l.item() >= 0.0

    z_new = torch.randn(4, 4)
    z_old = torch.randn(4, 4)
    lwf_l = reg.lwf_loss(z_new, z_old)
    assert lwf_l.item() >= 0.0


def test_deployment_drift_alarms():
    x = np.random.randn(20, 8)
    y = np.random.randn(20, 8) + 5.0  # shifted distribution
    mmd = compute_mmd(x, y)
    assert mmd > 0.0

    # High disagreement triggers alarms and fallback
    alarms_dict = evaluate_drift_alarms(
        live_embeddings=x,
        ref_embeddings=y,
        classical_preds=["BPSK"] * 20,
        learned_preds=["QPSK"] * 20,  # 100% disagreement
        rejected_count=5,
        total_count=20
    )
    assert "HIGH_MODEL_DISAGREEMENT" in alarms_dict["alarms"]
    assert alarms_dict["trigger_classical_fallback"] is True
