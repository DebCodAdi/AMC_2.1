"""Tests for multi-task neural backbone, 8 heads, loss functions, and EMA."""
import pytest
import torch
from signal9.train.backbone import SignalInfinityBackbone
from signal9.train.heads import SignalInfinityMultiTaskHeads
from signal9.train.losses import SignalInfinityLoss
from signal9.train.ema import ModelEMA
from signal9.train.trainer import train_step


def test_backbone_and_heads_forward_pass():
    backbone = SignalInfinityBackbone(in_channels=2, stem_dim=64, expert_dim=32, emb_dim=256)
    heads = SignalInfinityMultiTaskHeads(emb_dim=256, n_mods=16, n_fecs=4, n_scramblers=4)

    B = 4
    iq = torch.randn(B, 2, 4096)
    expert = torch.randn(B, 32)

    z = backbone(iq, expert)
    assert z.shape == (B, 256)

    out = heads(z)
    assert out["mod_logits"].shape == (B, 16)
    assert out["log_symrate"].shape == (B,)
    assert out["rolloff"].shape == (B,)
    assert out["cfo_norm"].shape == (B,)
    assert out["snr_db"].shape == (B,)
    assert out["fec_logits"].shape == (B, 4)
    assert out["scrambler_logits"].shape == (B, 4)
    assert out["presence_logit"].shape == (B,)


def test_multitask_loss_backward():
    backbone = SignalInfinityBackbone(stem_dim=32, emb_dim=128)
    heads = SignalInfinityMultiTaskHeads(emb_dim=128, n_mods=8, n_fecs=3, n_scramblers=3)
    criterion = SignalInfinityLoss()
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(heads.parameters()), lr=1e-3)
    ema = ModelEMA(backbone)

    B = 4
    batch = {
        "iq": torch.randn(B, 2, 4096),
        "expert": torch.randn(B, 32),
        "mod": torch.tensor([0, 1, 2, 3]),
        "symrate": torch.tensor([1200.0, 2400.0, 4800.0, 9600.0]),
        "rolloff": torch.tensor([0.25, 0.35, 0.5, 0.2]),
        "cfo_norm": torch.tensor([0.0, 0.05, -0.05, 0.01]),
        "snr_db": torch.tensor([5.0, 10.0, 15.0, 20.0]),
        "fec": torch.tensor([0, 1, 0, 2]),
        "scrambler": torch.tensor([0, 1, 2, 0]),
        "presence": torch.tensor([1, 1, 1, 0]),
    }

    loss_dict = train_step(backbone, heads, criterion, batch, optimizer, ema)
    assert "loss" in loss_dict
    assert loss_dict["loss"] > 0.0
    assert not torch.isnan(torch.tensor(loss_dict["loss"]))
