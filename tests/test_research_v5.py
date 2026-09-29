"""Contracts for the research-v5 architecture and open-set deployment path."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from signal9.train.backbone import build_backbone


def _trainer_module():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_grand_corpus_training.py"
    spec = importlib.util.spec_from_file_location("research_v5_trainer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hybrid_checkpoint_contract_matches_deployment():
    trainer = _trainer_module()
    trained = trainer.build_backbone("hybrid_xcit", stem_dim=128, expert_dim=32, emb_dim=512)
    deployed = build_backbone("hybrid_xcit", stem_dim=128, expert_dim=32, emb_dim=512)
    assert trained.state_dict().keys() == deployed.state_dict().keys()
    deployed.load_state_dict(trained.state_dict(), strict=True)
    trained.eval()
    deployed.eval()
    iq = torch.randn(3, 2, 128)
    expert = torch.randn(3, 32)
    with torch.no_grad():
        torch.testing.assert_close(trained(iq, expert), deployed(iq, expert), rtol=0, atol=0)


def test_both_backbones_accept_all_supported_lengths():
    for architecture in ("baseline_resnet", "hybrid_xcit"):
        model = build_backbone(architecture, stem_dim=128, expert_dim=32, emb_dim=512).eval()
        for length in (128, 256, 512, 1024):
            with torch.no_grad():
                out = model(torch.randn(2, 2, length), torch.randn(2, 32))
            assert out.shape == (2, 512)
            assert torch.isfinite(out).all()


def test_open_set_has_no_magic_deployment_threshold():
    source = (Path(__file__).resolve().parents[1] / "signal9" / "train" / "inference.py").read_text()
    assert 'open_set.get("enabled", False)' in source
    assert 'open_set.get("energy_threshold", float("inf"))' in source
    assert "energy_threshold=-5.0" not in source
