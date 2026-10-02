"""Regression cases for the supplied receiver review; all captures are seeded."""
import numpy as np
import pytest
from scipy.signal import fftconvolve, firwin

from signal9.dsp.bit_alignment import align_bytes, reference_ber
from signal9.dsp.constellation import get_constellation
from signal9.dsp.mod_order import score_modulations, select_modulation
from signal9.detect.multiscale import detect_multiscale, inband_snr
from signal9.detect.spectrogram import log_spectrogram


@pytest.mark.parametrize("offset", range(8))
@pytest.mark.parametrize("inverted,bitorder", [(False, "big"), (True, "little")])
def test_byte_alignment_recovers_message(offset, inverted, bitorder):
    message = b"Telemetry received: carrier stable, payload checksum ready.\n" * 5
    bits = np.unpackbits(np.frombuffer(message, dtype=np.uint8), bitorder=bitorder) ^ int(inverted)
    bits = np.r_[np.zeros(offset, dtype=np.uint8), bits]
    result = align_bytes(bits)
    assert result["raw"] == message
    assert (result["bit_offset"], result["inverted"], result["bitorder"]) == (offset, inverted, bitorder)


def test_alignment_never_upgrades_preview_to_crc_evidence():
    from signal9.exploit.chain import extract_decoded
    bits = np.unpackbits(np.frombuffer(b"Hello, world! " * 8, dtype=np.uint8))
    out = extract_decoded(np.r_[np.ones(3, dtype=np.uint8), bits], {})
    assert out["byte_alignment"]["bit_offset"] == 3
    assert out["preview_text"].startswith("Hello, world!")
    assert out["preview_validated"] is False
    assert "message_text" not in out


def test_ber_reports_truncation_and_errors():
    out = reference_ber([0, 1, 1, 0], [1, 0, 0, 1], offset=1)
    assert out["bits_compared"] == 3 and out["bit_errors"] == 1
    assert out["ber"] == pytest.approx(1 / 3) and not out["complete"]


def test_repeating_whole_frame_is_not_a_sync_word():
    from signal9.label.d6_frames import find_sync_pattern
    frame = np.random.default_rng(3).integers(0, 2, 592, dtype=np.uint8)
    assert find_sync_pattern(np.tile(frame, 20), 592) is None


def test_low_snr_grid_cannot_absorb_noise():
    rng = np.random.default_rng(8)
    const = get_constellation("QPSK")
    symbols = rng.choice(const, 8000) + (rng.normal(size=8000) + 1j * rng.normal(size=8000)) * np.sqrt(10 ** (-0.4) / 2)
    scores = {s["mod"]: s for s in score_modulations(symbols, snr_db=4)}
    assert scores["64QAM"]["rejected_by"] == "evm_snr_inconsistent"
    assert select_modulation(symbols, snr_db=4, hint="64QAM")["modulation"] != "64QAM"
    assert select_modulation(symbols, timing_lock=0.1)["reason"] == "timing_not_locked"


def test_low_snr_reports_family_instead_of_wrong_order():
    rng = np.random.default_rng(40)
    s = rng.choice(get_constellation("64QAM"), 16000)
    s += (rng.normal(size=len(s)) + 1j * rng.normal(size=len(s))) * np.sqrt(10 ** (-0.8) / 2)
    result = select_modulation(s, snr_db=8)
    assert result["modulation"] == "UNRESOLVED"
    assert result["modulation_family"] == "QAM"
    assert result["reason"] == "insufficient_snr_for_order"


def test_noise_only_file_does_not_fabricate_fallback_track(tmp_path):
    from signal9.pipeline import Signal9Engine
    rng = np.random.default_rng(42)
    noise = (rng.normal(size=65536) + 1j * rng.normal(size=65536)).astype(np.complex64) * 0.1
    capture = tmp_path / "noise.cf32"
    noise.tofile(capture)
    report = Signal9Engine(checkpoint="").process_file(capture, fs=256000, fmt="cf32", exploit=False)
    assert report["total_signals_detected"] == 0
    assert report["status"] == "NO_SIGNAL_DETECTED"


def test_cli_can_print_missing_rolloff(capsys):
    from main import _print_exploitation
    _print_exploitation({"demodulation": {"family": "linear", "rolloff": None}})
    assert "roll-off n/a" in capsys.readouterr().out


@pytest.mark.parametrize("mod", ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"])
def test_high_snr_order_preserved(mod):
    rng = np.random.default_rng(10)
    s = rng.choice(get_constellation(mod), 6000)
    s = s + (rng.normal(size=len(s)) + 1j * rng.normal(size=len(s))) * np.sqrt(10 ** (-2.5) / 2)
    assert select_modulation(s, snr_db=25)["modulation"] == mod


def wide_capture(seed=0, snr_db=-3):
    rng = np.random.default_rng(seed)
    fs, n, center, bw = 256000, 131072, 45000, 12000
    signal = fftconvolve(rng.normal(size=n) + 1j * rng.normal(size=n),
                         firwin(257, bw / 2, fs=fs), mode="same")
    signal *= np.sqrt((bw / fs) * 10 ** (snr_db / 10) / np.mean(abs(signal) ** 2))
    signal *= np.exp(2j * np.pi * center * np.arange(n) / fs)
    noise = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)
    return signal + noise, fs, center, bw


@pytest.mark.parametrize("seed", range(5))
def test_weak_wide_capture_is_detected(seed):
    x, fs, center, bw = wide_capture(seed)
    f, t, p = log_spectrogram(x, fs)
    boxes = detect_multiscale(p, f, t)
    matches = [b for b in boxes if b["f_lo"] < center < b["f_hi"]]
    assert len(matches) == 1
    assert matches[0]["bandwidth"] > 0.7 * bw
    assert matches[0]["duration"] > 0.8 * len(x) / fs


@pytest.mark.parametrize("seed", range(10))
def test_multiscale_noise_only(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=65536) + 1j * rng.normal(size=65536)
    f, t, p = log_spectrogram(x, 256000)
    assert detect_multiscale(p, f, t) == []


def test_snr_is_integrated_over_declared_band():
    x, fs, center, bw = wide_capture(snr_db=5.7)
    f, t, p = log_spectrogram(x, fs)
    tr = dict(f_lo=center - bw / 2, f_hi=center + bw / 2, t_lo=0, t_hi=len(x) / fs)
    assert inband_snr(p, f, t, tr, [tr]) == pytest.approx(5.7, abs=0.6)


def test_channel_filter_rejects_neighbor_within_zoom_tier():
    from signal9.channelize.channelizer import channelize
    fs = 64000
    n = np.arange(fs)
    x = np.exp(2j * np.pi * 1000 * n / fs) + 3 * np.exp(2j * np.pi * 11000 * n / fs)
    y, rate = channelize(x, fs, f_center=0, bw_hz=14000)
    core = y[300:-300]
    ts = np.arange(300, len(y) - 300) / rate
    wanted = abs(np.mean(core * np.exp(-2j * np.pi * 1000 * ts)))
    neighbor = abs(np.mean(core * np.exp(-2j * np.pi * 11000 * ts)))
    assert wanted > 0.95 and neighbor < 0.01


def test_rolloff_out_of_range_is_not_quality_one():
    from signal9.dsp.matched_filter import estimate_rolloff
    from signal9.exploit.chain import DemodResult
    rng = np.random.default_rng(5)
    x = rng.normal(size=32768) + 1j * rng.normal(size=32768)
    _, quality = estimate_rolloff(x, 64000, 2000)
    assert quality == 0
    assert DemodResult().to_dict()["rolloff"] is None


def test_disagreement_is_reported_and_carrier_offsets_are_added():
    from signal9.report.reporter import Signal9Reporter
    r = Signal9Reporter("test.iq", 64000, fc=100000000)
    track = dict(center_freq=10000, f_lo=5000, f_hi=15000, bandwidth=10000,
                 t_lo=0, t_hi=1, detector_snr_db=8, post_filter_snr_db=4)
    pred = {"levels": {"l2": {"prediction": "BPSK", "decision_state": "DECISION"}}}
    ex = {"demodulation": {"modulation": "64QAM", "cfo_hz": 12, "symbol_rate_hz": 5000}}
    r.add_detection(track, pred, ex)
    r.add_detection({**track, "center_freq": 12000}, pred, ex)
    r.flag_overlapping_rates()
    a = r.detections[0]
    assert a["modulation_decision"]["decision_state"] == "DISAGREEMENT"
    assert a["carrier_frequency_hz"] == 100010012
    assert a["snr_measurement"]["interference_suspected"]
    assert a["duplicate_rate_track_ids"] == [2]
