"""Regression tests for the AMC_1_2.0 audit (see Improvements.md, Phase 3).

Each test pins one defect that was found and fixed, so it cannot silently
return.  They are deliberately small and synthetic so the file runs in seconds.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from signal9.dsp.constellation import bits_to_symbols, llr_to_soft_symbol, symbols_to_bits
from signal9.dsp.matched_filter import rrc_taps

RNG = np.random.default_rng(2026)


def _crc_frames(n, body=16, order=">I"):
    out = []
    for i in range(n):
        b = bytes([0xAB, 0xCD, i & 0xFF]) + RNG.integers(0, 256, body - 3, dtype=np.uint8).tobytes()
        out.append(b + struct.pack(order, zlib.crc32(b)))
    return np.unpackbits(np.frombuffer(b"".join(out), dtype=np.uint8))


def _rrc_tx(bits, mod="QPSK", sps=8, alpha=0.35, snr_db=25.0):
    s = bits_to_symbols(bits, mod)
    up = np.zeros(len(s) * sps, dtype=np.complex64)
    up[::sps] = s
    x = np.convolve(up, rrc_taps(alpha, sps=sps, span=10)).astype(np.complex64)
    p = float(np.mean(np.abs(x) ** 2))
    n = (RNG.normal(size=len(x)) + 1j * RNG.normal(size=len(x))) * np.sqrt(p / (2 * 10 ** (snr_db / 10)))
    return (x + n).astype(np.complex64)


# ── C1: pipeline.process() NameError ────────────────────────────────────────
def test_process_short_input_does_not_raise():
    from signal9.pipeline import Signal9Engine
    r = Signal9Engine().process(np.zeros(10, dtype=np.complex64))
    assert r["status"] == "INSUFFICIENT_SAMPLES"


# ── C2: LLR sign convention between demapper and FEC decoders ───────────────
def test_llr_bridge_makes_viterbi_decode_correctly():
    from signal9.fec.conv import conv_encode, viterbi_decode
    info = RNG.integers(0, 2, 300).astype(np.uint8)
    llr = symbols_to_bits(bits_to_symbols(conv_encode(info), "QPSK"), "QPSK", hard=False, noise_var=0.01)
    assert np.mean(viterbi_decode(llr, 7, [0o171, 0o133])[:300] == info) < 0.05    # raw: complement
    assert np.array_equal(viterbi_decode(llr_to_soft_symbol(llr), 7, [0o171, 0o133])[:300], info)


def test_blackboard_conv_branch_locks_crc():
    from signal9.config import Config
    from signal9.decide.blackboard import BlackboardArbiter
    from signal9.fec.conv import conv_encode
    payload = b"AUDIT_REGRESSION_CONV_BRANCH"
    info = np.unpackbits(np.frombuffer(payload + struct.pack(">I", zlib.crc32(payload)), dtype=np.uint8))
    syms = bits_to_symbols(conv_encode(info), "QPSK")
    w, _ = BlackboardArbiter(Config(early_exit_on_crc=True)).arbitrate(syms, "QPSK", fec_hint="conv_k7_r12")
    assert w.crc_locked and w.payload == payload


# ── C3: audio WAV analytic conversion ───────────────────────────────────────
@pytest.mark.parametrize("n", [96_000, 307_200, 1_300_000])
def test_audio_to_analytic_is_one_sided(n):
    from signal9.io.audio_path import audio_to_analytic
    t = np.arange(n) / 48000.0
    z = audio_to_analytic(np.sin(2 * np.pi * 1200 * t) + 0.5 * np.sin(2 * np.pi * 2200 * t))
    seg = z[n // 2 - 32768:n // 2 + 32768] * np.hanning(65536)
    Z = np.fft.fft(seg)
    f = np.fft.fftfreq(len(Z))
    rejection = 10 * np.log10(np.sum(np.abs(Z[f > 0]) ** 2) / np.sum(np.abs(Z[f < 0]) ** 2))
    assert rejection > 60.0


def test_iter_wav_iq_routes_mono_audio_through_hilbert():
    from signal9.io.chunker import iter_wav_iq
    x = (np.sin(2 * np.pi * 0.05 * np.arange(8192)) * 20000).astype(np.int16)
    chunk = next(iter_wav_iq(x, block=8192, kind="mono_audio"))
    assert np.abs(chunk.imag).max() > 0.1          # was real + 0j before the fix


# ── C6: all four interleaver families, blind, at an arbitrary phase ─────────
def _hamming_stream(n):
    G = np.array([[1, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 1],
                  [0, 0, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8)
    return ((RNG.integers(0, 2, (n, 4)).astype(np.uint8) @ G) % 2).reshape(-1).astype(np.uint8)


def test_interleaver_roundtrips():
    from signal9.fec import interleaver as il
    x = RNG.integers(0, 2, 6000).astype(np.uint8)
    y = il.strip_convolutional_delay(il.convolutional_deinterleave(il.convolutional_interleave(x, 4, 3), 4, 3), 4, 3)
    assert np.array_equal(y[:5000], x[:5000])
    assert np.array_equal(il.helical_deinterleave(il.helical_interleave(x, 6, 7, 2), 6, 7, 2), x)
    q = il.qpp_permutation(40, 3, 10)
    assert np.array_equal(il.permutation_deinterleave(il.permutation_interleave(x, q), q), x)
    with pytest.raises(ValueError):
        il.qpp_permutation(40, 2, 10)


@pytest.mark.parametrize("family", ["block", "helical", "convolutional", "pseudorandom"])
def test_identify_interleaver_blind_with_phase(family):
    from signal9.fec import interleaver as il
    from signal9.fec.conv import conv_encode
    from signal9.label.interleaver_detect import identify_interleaver
    coded = conv_encode(RNG.integers(0, 2, 6000).astype(np.uint8))
    stream = {"block": lambda c: il.block_interleave(c, 6, 8),
              "helical": lambda c: il.helical_interleave(c, 6, 7, 2),
              "convolutional": lambda c: il.convolutional_interleave(c, 5, 2),
              "pseudorandom": lambda c: il.permutation_interleave(c, il.qpp_permutation(40, 3, 10))}[family](coded)
    x = np.concatenate([RNG.integers(0, 2, 11).astype(np.uint8), stream])     # starts mid-block
    assert identify_interleaver(x)["type"] == family


def test_identify_interleaver_honest_negatives():
    from signal9.label.interleaver_detect import identify_interleaver
    assert identify_interleaver(_hamming_stream(2500))["type"] == "none"
    assert identify_interleaver(RNG.integers(0, 2, 8000).astype(np.uint8))["type"] == "UNRESOLVED"


# ── estimator accuracy ──────────────────────────────────────────────────────
@pytest.mark.parametrize("mod", ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"])
def test_modulation_order_not_always_64qam(mod):
    from signal9.dsp.mod_order import select_modulation
    s = bits_to_symbols(RNG.integers(0, 2, 12000).astype(np.uint8), mod).astype(np.complex128)
    n = (RNG.normal(size=len(s)) + 1j * RNG.normal(size=len(s))) * np.sqrt(10 ** (-2.2) / 2)
    assert select_modulation(s + n)["modulation"] == mod


def test_gaussian_noise_is_not_a_constellation():
    from signal9.dsp.mod_order import select_modulation
    noise = RNG.normal(size=8000) + 1j * RNG.normal(size=8000)
    assert select_modulation(noise)["modulation"] == "UNRESOLVED"     # was "64QAM"


@pytest.mark.parametrize("alpha", [0.2, 0.35, 0.5])
def test_rolloff_estimate(alpha):
    from signal9.dsp.matched_filter import estimate_rolloff
    syms = (RNG.integers(0, 2, 8000) * 2 - 1) + 1j * (RNG.integers(0, 2, 8000) * 2 - 1)
    up = np.zeros(len(syms) * 8, dtype=complex)
    up[::8] = syms
    x = np.convolve(up, rrc_taps(alpha, sps=8, span=12))
    est, _ = estimate_rolloff(x + 0.01 * (RNG.normal(size=len(x)) + 1j * RNG.normal(size=len(x))), 48000.0, 6000.0)
    assert abs(est - alpha) < 0.06


# ── CRC / GF(2) kernels ─────────────────────────────────────────────────────
@pytest.mark.parametrize("shift", range(8))
def test_crc_search_any_bit_alignment(shift):
    from signal9.decide.crc_search import crc32_search_bitstream
    bits = np.concatenate([RNG.integers(0, 2, shift).astype(np.uint8), _crc_frames(6)])
    r = crc32_search_bitstream(bits)
    assert r.found and r.bit_offset == shift and zlib.crc32(r.payload) == r.crc


def test_gf2_rank_matches_dense_reference():
    from signal9.dsp.gf2 import _gf2_rank_dense, gf2_rank
    for _ in range(200):
        m = (RNG.random((int(RNG.integers(1, 200)), int(RNG.integers(1, 150)))) < 0.5).astype(np.uint8)
        assert gf2_rank(m) == _gf2_rank_dense(m)


# ── FEC false positives ─────────────────────────────────────────────────────
def test_uncoded_stream_is_not_reported_as_convolutional():
    from signal9.label.code_id import identify_fec_scheme
    assert identify_fec_scheme(RNG.integers(0, 2, 4000).astype(np.uint8))[0].name == "none"


def test_degenerate_streams_detected_and_zero_rs_rejected():
    from signal9.exploit.chain import _try_rs, stream_degeneracy
    assert stream_degeneracy(np.zeros(4000, dtype=np.uint8)) is not None
    assert stream_degeneracy(np.tile([1, 1, 0], 2000).astype(np.uint8)) is not None
    assert stream_degeneracy(RNG.integers(0, 2, 4000).astype(np.uint8)) is None
    assert _try_rs(np.zeros(8 * 255 * 3, dtype=np.uint8)) is None


# ── FSK demodulation ────────────────────────────────────────────────────────
def test_fsk_demodulation_rate_and_bits():
    from signal9.dsp.fsk import fsk_demodulate
    bits = RNG.integers(0, 2, 3000).astype(np.uint8)
    f = np.repeat(np.where(bits == 1, 1.0, -1.0), 40)
    x = np.exp(1j * 2 * np.pi * np.cumsum(f * 600.0 / 48000.0)).astype(np.complex64)
    r = fsk_demodulate(x, 48000.0)
    assert r.modulation == "2FSK" and abs(r.symbol_rate_hz - 1200.0) < 1.0
    best = min(np.mean((r.hard_bits[:2800] ^ flip) != bits[d:d + 2800]) for d in range(3) for flip in (0, 1))
    assert best == 0.0


# ── End to end ──────────────────────────────────────────────────────────────
def test_chain_conv_coded_qpsk_locks_crc_end_to_end():
    from signal9.exploit import ExploitationChain
    from signal9.fec.conv import conv_encode
    r = ExploitationChain().run(_rrc_tx(conv_encode(_crc_frames(40))), 48000.0)
    assert r.demod.modulation == "QPSK"
    assert abs(r.demod.symbol_rate_hz - 6000.0) < 5.0
    assert r.fec.get("class") == "convolutional"
    assert r.status == "SUCCESS_CRC_LOCKED"
    assert (r.frames.get("crc") or {}).get("model") == "CRC-32/ISO-HDLC"


# ── Training ────────────────────────────────────────────────────────────────
def test_ema_tracks_batchnorm_statistics():
    import torch
    from signal9.train.ema import ModelEMA
    g = torch.Generator().manual_seed(0)
    net = torch.nn.BatchNorm1d(4)
    ema = ModelEMA(net)
    net.train()
    for _ in range(200):
        net(torch.randn(32, 4, generator=g) + 3.0)       # data mean is 3
        ema.update(net)
    # The live running mean converges to 3; the EMA copy must follow it.  The
    # old EMA averaged parameters only, so its running mean stayed at 0.
    assert torch.allclose(ema.ema_model.running_mean, torch.full((4,), 3.0), atol=0.3)


def test_supcon_matches_paper_formula():
    import torch
    import torch.nn.functional as F
    from signal9.train.losses import SupervisedContrastiveLoss
    torch.manual_seed(0)
    z, y, t = torch.randn(10, 8), torch.tensor([0, 0, 1, 1, 1, 2, 2, 0, 3, 1]), 0.1
    zn = F.normalize(z, dim=1)
    tot, cnt = 0.0, 0
    for i in range(10):
        P = [p for p in range(10) if p != i and y[p] == y[i]]
        if not P:
            continue
        den = sum(torch.exp(zn[i] @ zn[a] / t) for a in range(10) if a != i)
        tot += -sum(torch.log(torch.exp(zn[i] @ zn[p] / t) / den) for p in P) / len(P)
        cnt += 1
    assert abs(float(SupervisedContrastiveLoss(t)(z, y)) - float(tot / cnt)) < 1e-4


def test_synthetic_snr_labels_are_true():
    from signal9.train.synth import SynthParams, generate
    p = SynthParams("QPSK", 8, 0.35, 0.004, 10.0, 0.3, False, False)
    clean = generate(SynthParams(**{**p.__dict__, "snr_db": 200.0}), 4096, np.random.default_rng(5))
    noisy = generate(p, 4096, np.random.default_rng(5))
    a = np.vdot(clean, noisy) / np.vdot(clean, clean)
    snr = 10 * np.log10(np.mean(np.abs(a * clean) ** 2) / np.mean(np.abs(noisy - a * clean) ** 2))
    assert abs(snr - 10.0) < 0.5


def test_label_vocabulary():
    from signal9.train.labels import MOD_VOCAB, RML2018_CLASSES_FIXED, mod_index
    assert RML2018_CLASSES_FIXED[0] == "OOK" and RML2018_CLASSES_FIXED[-1] == "OQPSK"
    assert MOD_VOCAB[mod_index("QAM16")] == "16QAM"
    assert MOD_VOCAB[mod_index("PAM4")] == "PAM4"
    with pytest.raises(ValueError):
        mod_index("NOT_A_MODULATION")


def test_pipeline_reports_not_trained_without_checkpoint():
    from signal9.pipeline import _untrained_prediction
    lv = _untrained_prediction(10.0)["levels"]
    assert all(lv[k]["decision_state"] == "NOT_TRAINED" and lv[k]["prediction"] is None for k in ("l1", "l2", "l3"))


# ── GUI ─────────────────────────────────────────────────────────────────────
def test_gui_constructs_and_renders(tmp_path):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from signal9.gui.app import MainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    f = tmp_path / "t.cf32"
    _rrc_tx(_crc_frames(20)).tofile(f)
    from signal9.gui.app import analyze_sync
    rep, art = analyze_sync(str(f), fs=48000.0)
    w.load_results(rep, art)
    app.processEvents()
    assert w.track_list.count() >= 1 and w.param_table.rowCount() > 10
    w.close()


# ── Decoded message output ──────────────────────────────────────────────────
def test_chain_returns_decoded_message_text():
    from signal9.exploit import ExploitationChain
    from signal9.fec.conv import conv_encode
    fr = b""
    for i in range(30):
        body = bytes([0xAB, 0xCD]) + f"Manipal BKL [{i:02d}]".encode().ljust(20)
        fr += body + struct.pack(">I", zlib.crc32(body))
    bits = np.unpackbits(np.frombuffer(fr, dtype=np.uint8))
    r = ExploitationChain().run(_rrc_tx(conv_encode(bits)), 48000.0)
    dec = r.to_dict()["decoded"]
    assert dec["frames_crc_ok"] >= 25
    lines = dec["message_text"].split("\n")
    assert all(l.startswith("Manipal BKL [") for l in lines)      # sync stripped, text kept
    assert all(row["crc_ok"] for row in dec["frames"] if row["payload_text"].startswith("Manipal"))


def test_visible_code_is_not_reported_as_interleaved():
    """Repeated frame content once made a wrong block hypothesis beat 'none'."""
    from signal9.fec.conv import conv_encode
    from signal9.label.interleaver_detect import identify_interleaver
    fr = b""
    for i in range(40):
        body = bytes([0xAB, 0xCD]) + f"Manipal BKL [{i:02d}]".encode().ljust(20)
        fr += body + struct.pack(">I", zlib.crc32(body))
    coded = conv_encode(np.unpackbits(np.frombuffer(fr, dtype=np.uint8)))
    assert identify_interleaver(coded)["type"] == "none"
