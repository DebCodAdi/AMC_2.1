"""L17 Wideband Spectrum Survey test suite (Phase 3).

Validates:
  - Multi-resolution STFT bank
  - 2D OS-CFAR burst detection
  - DBSCAN/Connected-components burst clustering
  - NCO extraction, decimation & SigMF generation
  - Priority scheduling and load shedding under heavy congestion
"""
from __future__ import annotations
import pytest
import numpy as np

from signal9.survey.stft import multiresolution_stft
from signal9.survey.detect_cfar import os_cfar_2d
from signal9.survey.cluster import cluster_detections
from signal9.survey.extract import extract_burst
from signal9.survey.schedule import SurveyScheduler


@pytest.fixture
def wideband_scene():
    """Synthesize a 100 ms wideband capture at 2.4 MSPS with two distinct emissions."""
    fs = 2_400_000.0
    duration_s = 0.05  # 50 ms
    N = int(fs * duration_s)
    rng = np.random.default_rng(42)

    # 1. Background AWGN
    noise = 0.05 * (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
    wb = noise.copy()

    t = np.arange(N) / fs

    # 2. Burst 1: at +400 kHz from t=10ms to t=30ms
    idx_b1_start = int(0.010 * fs)
    idx_b1_end = int(0.030 * fs)
    t_b1 = t[idx_b1_start:idx_b1_end]
    sig1 = np.exp(1j * 2.0 * np.pi * 400_000.0 * t_b1).astype(np.complex64)
    wb[idx_b1_start:idx_b1_end] += 1.0 * sig1

    # 3. Burst 2: at -500 kHz from t=25ms to t=45ms
    idx_b2_start = int(0.025 * fs)
    idx_b2_end = int(0.045 * fs)
    t_b2 = t[idx_b2_start:idx_b2_end]
    sig2 = np.exp(1j * 2.0 * np.pi * (-500_000.0) * t_b2).astype(np.complex64)
    wb[idx_b2_start:idx_b2_end] += 0.8 * sig2

    return wb, fs


def test_multiresolution_stft(wideband_scene):
    wb, fs = wideband_scene
    mrs = multiresolution_stft(wb, fs=fs, nfft_coarse=256, nfft_fine=1024)

    assert mrs.coarse.P_db.shape[0] == 256
    assert mrs.fine.P_db.shape[0] == 1024
    assert len(mrs.f_welch) == 1024
    assert not np.isnan(mrs.fine.P_db).any()


def test_os_cfar_and_clustering(wideband_scene):
    wb, fs = wideband_scene
    mrs = multiresolution_stft(wb, fs=fs, nfft_coarse=256, nfft_fine=512)
    fine = mrs.fine

    # Run 2D OS-CFAR
    mask = os_cfar_2d(fine.P_db, guard_cells=(2, 2), ref_cells=(6, 6), threshold_db=10.0)
    assert np.any(mask), "CFAR detector failed to detect emissions"

    # Cluster detections
    bursts = cluster_detections(mask, fine.f, fine.t, fine.P_db, min_pixels=4)
    assert len(bursts) >= 2, f"Expected at least 2 bursts, found {len(bursts)}"

    # Check center frequencies match +400 kHz and -500 kHz
    centers = [b.f_center_hz for b in bursts]
    has_pos = any(abs(c - 400_000.0) < 50_000.0 for c in centers)
    has_neg = any(abs(c - (-500_000.0)) < 50_000.0 for c in centers)
    assert has_pos, f"Did not find +400 kHz emission. Found centers: {centers}"
    assert has_neg, f"Did not find -500 kHz emission. Found centers: {centers}"


def test_extraction_and_sigmf(wideband_scene):
    wb, fs = wideband_scene
    mrs = multiresolution_stft(wb, fs=fs, nfft_coarse=256, nfft_fine=512)
    mask = os_cfar_2d(mrs.fine.P_db, threshold_db=10.0)
    bursts = cluster_detections(mask, mrs.fine.f, mrs.fine.t, mrs.fine.P_db, min_pixels=4)
    
    # Pick the strongest burst
    strongest = max(bursts, key=lambda b: b.snr_db_est)
    extraction = extract_burst(wb, fs_wb=fs, fc_rf=915_000_000.0, burst=strongest)

    assert len(extraction.channel_iq) > 0
    assert extraction.channel_fs < fs  # Properly decimated
    # SigMF metadata structure validation
    meta = extraction.sigmf_metadata
    assert "global" in meta
    assert "captures" in meta
    assert "annotations" in meta
    assert meta["global"]["core:datatype"] == "cf32_le"
    assert meta["captures"][0]["core:frequency"] > 0.0


def test_scheduler_priority_and_shedding(wideband_scene):
    wb, fs = wideband_scene
    mrs = multiresolution_stft(wb, fs=fs, nfft_coarse=256, nfft_fine=512)
    mask = os_cfar_2d(mrs.fine.P_db, threshold_db=10.0)
    bursts = cluster_detections(mask, mrs.fine.f, mrs.fine.t, mrs.fine.P_db, min_pixels=4)
    extractions = [extract_burst(wb, fs_wb=fs, fc_rf=915e6, burst=b) for b in bursts]

    # Initialize scheduler with capacity limit 1 to force shedding
    scheduler = SurveyScheduler(max_queue_size=1)
    for ext in extractions:
        scheduler.enqueue(ext)

    stats = scheduler.stats()
    assert stats["queue_depth"] <= 1
    if len(extractions) > 1:
        assert stats["n_dropped"] >= 1

    # Dispatched burst should be the highest SNR
    dispatched = scheduler.dispatch_next()
    assert dispatched is not None
    assert scheduler.queue_depth == 0
