"""Regression tests for claims independently confirmed from the V5 review."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import threading
import time
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from signal9.models.fusion import ModalityFusion
from signal9.models.losses import Signal9Loss
from signal9.preprocess import condition_for_classifier
from signal9.protocol.ax25 import deframe_ax25
from signal9.protocol.ccsds import deframe_ccsds_tm
from signal9.protocol.ism import deframe_ism
from signal9.serve.onnx_export import export_onnx
from signal9.sim.framing import (
    CCSDS_ASM_BYTES,
    crc16_ccitt,
    frame_ax25,
    frame_ccsds,
    hdlc_bit_stuff,
)
from signal9.train.selftrain import filter_pseudo_labels
from signal9.train.augment import quantize_signed_8bit
from signal9.train.trainer import CONDITIONING_CONTRACT, make_sample
from signal9.stream.queue import OverflowPolicy, StreamingRingBuffer


ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "notebooks" / "kaggle_grand_corpus_training.py"


def _load_trainer():
    spec = importlib.util.spec_from_file_location("v5_review_trainer", TRAINER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_joint_modality_dropout_never_erases_every_signal_branch():
    fusion = ModalityFusion(iq_dim=2, spec_dim=2, expert_dim=0, meta_dim=1,
                            fused_dim=4, drop_p=0.95).train()
    fusion.meta_encoder = nn.Identity()
    fusion.fusion_mlp = nn.Identity()
    iq = torch.ones(512, 2)
    spec = torch.ones(512, 2)
    meta = torch.zeros(512, 1)
    out = fusion(iq, spec, None, meta)
    assert torch.all(out[:, :4].abs().sum(dim=1) > 0)


def test_optional_auxiliary_losses_train_all_auxiliary_heads():
    batch = 6
    emb = torch.randn(batch, 8, requires_grad=True)
    preds = {
        "embedding": emb,
        "l1": torch.randn(batch, 3, requires_grad=True),
        "l2": torch.randn(batch, 4, requires_grad=True),
        "l3": torch.randn(batch, 5, requires_grad=True),
        "burst": torch.randn(batch, requires_grad=True),
        "snr": torch.randn(batch, requires_grad=True),
        "log_baud": torch.randn(batch, requires_grad=True),
        "outlier": torch.randn(batch, requires_grad=True),
    }
    loss_fn = Signal9Loss(3, 4, 5, emb_dim=8)
    loss = loss_fn(
        preds,
        torch.tensor([0, 1, 2, 0, 1, 2]),
        torch.tensor([0, 1, 2, 3, 0, 1]),
        torch.tensor([0, 1, 2, 3, 4, 0]),
        torch.tensor([1, 1, 1, 0, 1, 0]),
        t_snr=torch.arange(batch, dtype=torch.float32),
        t_log_baud=torch.linspace(6.0, 9.0, batch),
        t_outlier=torch.tensor([0, 0, 1, 0, 1, 0]),
    )
    loss.backward()
    for name in ("l1", "l2", "l3", "snr", "log_baud", "outlier"):
        assert preds[name].grad is not None
        assert preds[name].grad.abs().sum() > 0


def test_signed_8bit_quantizer_saturates_codes_not_physical_amplitude():
    iq = np.array([2.5 + 0.2j, -1.75 - 3.0j, 0.0 + 1.0j], dtype=np.complex64)
    quantized = quantize_signed_8bit(iq)
    assert quantized.dtype == np.complex64
    assert np.isfinite(quantized).all()
    assert np.max(np.abs(quantized.real)) > 1.5
    assert np.max(np.abs(quantized.imag)) > 2.5


def test_training_conditioner_matches_deployment_conditioner():
    trainer = _load_trainer()
    rng = np.random.default_rng(91)
    z = (rng.standard_normal((3, 128)) + 1j * rng.standard_normal((3, 128))).astype(np.complex64)
    z[0, 4] = np.nan + 1j * np.nan
    z[1, 27] *= 100.0
    batch = np.stack([z.real, z.imag], axis=1)
    actual = trainer.condition_classifier_batch(batch)
    for row in range(len(z)):
        expected, _ = condition_for_classifier(z[row])
        expected_iq = np.stack([expected.real, expected.imag])
        np.testing.assert_allclose(actual[row], expected_iq, rtol=2e-6, atol=2e-6)


def test_library_trainer_uses_the_deployment_conditioner():
    z = np.ones(128, dtype=np.complex64)
    z[4] = np.nan + 1j * np.nan
    z[27] = 100.0 + 20.0j
    expected, _ = condition_for_classifier(z)
    sample = make_sample(z, {"mod": 0})
    expected_iq = np.stack([expected.real, expected.imag])
    np.testing.assert_allclose(sample["iq"].numpy(), expected_iq, rtol=2e-6, atol=2e-6)
    assert CONDITIONING_CONTRACT.endswith("rms-v1")


def test_blocking_ring_buffer_ignores_spurious_notification():
    ring = StreamingRingBuffer(capacity_samples=2, overflow_policy=OverflowPolicy.BLOCK)
    assert ring.push(np.array([1, 2], dtype=np.complex64)) == 2
    result = []
    producer = threading.Thread(target=lambda: result.append(
        ring.push(np.array([3], dtype=np.complex64), timeout=1.0)))
    producer.start()
    time.sleep(0.03)
    with ring._lock:
        ring._not_full.notify_all()
    time.sleep(0.03)
    assert producer.is_alive()
    ring.pull(1)
    producer.join(timeout=1.0)
    assert result == [1]


def test_pseudo_label_fraction_is_enforced_per_class():
    predictions = [
        {"track_id": i, "predicted_label": "QPSK", "confidence": 0.99,
         "branch_agreement": True}
        for i in range(100)
    ]
    kept = filter_pseudo_labels(predictions, max_pseudo_fraction=0.20)
    assert len(kept) == 20


def _prepend_bit_slip(payload: bytes, n: int = 3) -> np.ndarray:
    prefix = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)[:n]
    return np.concatenate([prefix, np.unpackbits(np.frombuffer(payload, dtype=np.uint8))])


def test_ccsds_and_ism_accept_non_byte_aligned_bitstreams():
    header = bytes([0, 0, 1, 2, 0, 0])
    payload = b"V5" + b"X" * 118
    content = header + payload
    crc = crc16_ccitt(content)
    framed = frame_ccsds(content + bytes([crc >> 8, crc & 0xFF]), apply_scrambler=True)
    ccsds = deframe_ccsds_tm(_prepend_bit_slip(framed), frame_length=128, has_fecf=True)
    assert any(frame.crc_valid and frame.payload == payload for frame in ccsds)

    ism_payload = b"SENSOR"
    ism_content = bytes([len(ism_payload)]) + ism_payload
    ism_crc = crc16_ccitt(ism_content)
    packet = bytes.fromhex("d391") + ism_content + bytes([ism_crc >> 8, ism_crc & 0xFF])
    ism = deframe_ism(_prepend_bit_slip(packet, 5))
    assert any(frame.crc_valid and frame.payload == ism_payload for frame in ism)


def test_ax25_finds_flags_before_unstuffing_with_bit_slip():
    frame = frame_ax25("APRS", "K1ABC", b"PAYLOAD")
    flag = np.unpackbits(np.frombuffer(b"\x7e", dtype=np.uint8))
    content = np.unpackbits(np.frombuffer(frame[1:-1], dtype=np.uint8))
    stuffed = hdlc_bit_stuff(content)
    slipped = np.concatenate([np.array([1, 0, 1], dtype=np.uint8), flag, stuffed, flag])
    decoded = deframe_ax25(slipped)
    assert len(decoded) == 1
    assert decoded[0].crc_valid
    assert decoded[0].payload == b"PAYLOAD"


def test_hdf_partitions_keep_64_row_groups_disjoint():
    trainer = _load_trainer()
    streamer = trainer.DynamicGrandCorpusStreamer.__new__(trainer.DynamicGrandCorpusStreamer)
    streamer.seed = 123
    streamer.rng = np.random.default_rng(456)
    groups = {}
    for partition in ("train", "cal", "val"):
        rows = streamer._partition_h5_indices(100_000, 2000, partition, "test_source")
        assert len(rows) == 2000
        groups[partition] = set((rows // trainer.H5_SPLIT_GROUP_ROWS).tolist())
        assert len(groups[partition]) <= trainer.H5_IO_FANOUT_GROUPS
    assert groups["train"].isdisjoint(groups["cal"])
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["cal"].isdisjoint(groups["val"])


def test_hardware_capture_sessions_never_cross_partitions():
    trainer = _load_trainer()
    streamer = trainer.DynamicGrandCorpusStreamer.__new__(trainer.DynamicGrandCorpusStreamer)
    streamer.seed = 123
    paths = [Path("root/modrec/usrp/1") / f"{mod}.dat"
             for mod in ("bpsk", "qpsk", "8psk", "qam16", "qam64")]
    paths += [Path("root/modrec/usrp/2/bpsk.dat"),
              Path("root/modrec/usrp/3/bpsk.dat"),
              Path("root/modrec/rtl/1/bpsk.dat")]
    keys = [streamer._hardware_group_key("modrec", path) for path in paths]
    assignments = streamer._balanced_hardware_group_partitions("modrec", sorted(set(keys)))
    assert len({assignments[key] for key in keys[:5]}) == 1
    assert {"train", "cal", "val"}.issubset(set(assignments.values()))


def test_training_expert_contract_has_no_permanent_zero_padding():
    trainer = _load_trainer()
    iq = torch.randn(3, 2, 128)
    features = trainer.extract_training_expert_batch(iq)
    assert features.shape == (3, trainer.TRAIN_EXPERT_DIM)
    assert trainer.TRAIN_EXPERT_DIM == 11


def test_production_multitask_heads_receive_gradients_for_every_trained_task():
    trainer = _load_trainer()
    heads = trainer.SignalInfinityMultiTaskHeads(emb_dim=16, n_mods=5)
    embedding = torch.randn(8, 16, requires_grad=True)
    predictions = heads(embedding)
    labels = torch.arange(8) % 5
    loss = (F.cross_entropy(predictions["mod_logits"], labels)
            + 0.02 * F.smooth_l1_loss(
                predictions["snr_db"], torch.linspace(-5.0, 15.0, 8))
            + 0.05 * F.smooth_l1_loss(
                predictions["log_symrate"], torch.linspace(5.0, 9.0, 8))
            + 0.05 * F.binary_cross_entropy_with_logits(
                predictions["presence_logit"], (labels != 4).float()))
    loss.backward()
    for name in ("mod_head", "snr_head", "symrate_head", "presence_head"):
        grads = [p.grad for p in getattr(heads, name).parameters()]
        assert all(grad is not None for grad in grads)
        assert sum(float(grad.abs().sum()) for grad in grads) > 0.0


def test_onnx_export_marks_iq_sequence_axis_dynamic(tmp_path):
    with patch("torch.onnx.export") as mocked:
        export_onnx(nn.Identity(), tmp_path / "model.onnx")
    dynamic_axes = mocked.call_args.kwargs["dynamic_axes"]
    assert dynamic_axes["iq"][2] == "seq_len"
