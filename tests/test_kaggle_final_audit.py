"""Fail-closed label/provenance tests for the standalone Kaggle trainer."""
from __future__ import annotations

import importlib.util
import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest
import torch


def _trainer():
    script = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_grand_corpus_training.py"
    spec = importlib.util.spec_from_file_location("s9_kaggle_final_audit", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_modrec_only_accepts_exact_documented_stems(tmp_path: Path):
    m = _trainer()
    assert m.MOD_VOCAB[m.resolve_modrec_modulation(tmp_path / "qam16_2mhz.dat")] == "16QAM"
    assert m.MOD_VOCAB[m.resolve_modrec_modulation(tmp_path / "8psk_2mhz.dat")] == "8PSK"
    assert m.resolve_modrec_modulation(tmp_path / "maybe_qam16_capture.dat") == -1


def test_kiwi_requires_matching_sigmf_annotation(tmp_path: Path):
    m = _trainer()
    wav_path = tmp_path / "misleading_QPSK.wav"
    wav_path.touch()
    assert m.resolve_kiwi_sigmf(wav_path)[0] == -1

    sidecar = wav_path.with_suffix(".sigmf-meta")
    sidecar.write_text(json.dumps({
        "global": {"core:datatype": "ci16_le", "core:sample_rate": 12000},
        "annotations": [{"core:sample_start": 0, "core:sample_count": -1,
                         "core:label": "FT8"}],
    }), encoding="utf-8")
    idx, snr, baud = m.resolve_kiwi_sigmf(wav_path)
    assert m.MOD_VOCAB[idx] == "8FSK"
    assert math.isnan(snr) and math.isnan(baud)


def test_kiwi_reader_rejects_mono_audio(tmp_path: Path):
    m = _trainer()
    wav_path = tmp_path / "mono.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(12000)
        wav.writeframes(np.zeros(4096, dtype="<i2").tobytes())
    streamer = object.__new__(m.DynamicGrandCorpusStreamer)
    streamer.input_len = 128
    streamer.rng = np.random.default_rng(1)
    streamer.ram_cache = {}
    streamer.raw_iq_maps = {}
    iq, label, _, _ = streamer._read_hw_sample(
        (wav_path, m.MOD_TO_IDX["BPSK"], wav_path.stat().st_size, ".wav",
         float("nan"), float("nan"))
    )
    assert iq.shape == (2, 128)
    assert label == -1


def test_kiwi_reader_reads_stereo_ci16_iq(tmp_path: Path):
    m = _trainer()
    wav_path = tmp_path / "iq.wav"
    raw = np.column_stack((np.arange(512, dtype=np.int16), -np.arange(512, dtype=np.int16)))
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(12000)
        wav.writeframes(raw.astype("<i2").tobytes())
    streamer = object.__new__(m.DynamicGrandCorpusStreamer)
    streamer.input_len = 128
    streamer.rng = np.random.default_rng(1)
    streamer.ram_cache = {}
    streamer.raw_iq_maps = {}
    iq, label, _, _ = streamer._read_hw_sample(
        (wav_path, m.MOD_TO_IDX["BPSK"], wav_path.stat().st_size, ".wav",
         float("nan"), float("nan"))
    )
    assert iq.shape == (2, 128)
    assert label == m.MOD_TO_IDX["BPSK"]
    np.testing.assert_allclose(iq[1], -iq[0])


def test_inference_rejects_incomplete_kaggle_backbone(tmp_path: Path):
    m = _trainer()
    from signal9.train.inference import MultiTaskClassifier

    backbone = m.SignalInfinityBackbone(stem_dim=128, expert_dim=32, emb_dim=512)
    heads = m.SignalInfinityMultiTaskHeads(emb_dim=512, n_mods=len(m.MOD_VOCAB),
                                           n_fecs=5, n_scramblers=5)
    broken = backbone.state_dict()
    del broken["stem.conv1.weight"]
    path = tmp_path / "broken.ckpt"
    torch.save({
        "format": "signal9-multitask-v1",
        "backbone": broken,
        "heads": heads.state_dict(),
        "class_names": list(m.MOD_VOCAB),
        "input_len": 128,
        "expert_dim": 32,
        "stem_dim": 128,
        "emb_dim": 512,
        "n_fecs": 5,
        "n_scramblers": 5,
    }, path)
    with pytest.raises(ValueError, match="incompatible backbone state"):
        MultiTaskClassifier.load(path)
