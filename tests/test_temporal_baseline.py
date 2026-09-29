"""Tests for learned per-frequency power baseline detection."""
import numpy as np

from signal9.detect.temporal_baseline import TemporalBaseline


def test_first_observation_initializes_without_false_alarm():
    baseline = TemporalBaseline()
    p = np.full((16, 8), -90.0)
    assert np.allclose(baseline.update(p), 0.0)
    assert baseline.chunks_seen == 1


def test_new_narrowband_signal_is_not_absorbed():
    baseline = TemporalBaseline(alpha=0.5, guard_db=6.0)
    quiet = np.full((16, 8), -90.0)
    baseline.update(quiet)
    event = quiet.copy()
    event[6:8] = -72.0
    excess = baseline.update(event)
    assert float(excess[6:8].min()) > 15.0
    # The event is outside the update guard, so it cannot poison the floor.
    assert np.allclose(baseline.baseline_db[6:8], -90.0)


def test_ordinary_noise_floor_change_adapts():
    baseline = TemporalBaseline(alpha=0.5, guard_db=6.0)
    baseline.update(np.full((8, 4), -90.0))
    baseline.update(np.full((8, 4), -87.0))
    assert np.allclose(baseline.baseline_db, -88.5)


def test_receiver_wide_gain_change_is_not_a_full_band_event():
    baseline = TemporalBaseline(alpha=0.5, guard_db=6.0)
    baseline.update(np.full((8, 4), -90.0))
    # A uniform +12 dB AGC shift must not be presented to CFAR as an emitter.
    excess = baseline.update(np.full((8, 4), -78.0))
    assert np.allclose(excess, 0.0)
    assert np.allclose(baseline.baseline_db, -84.0)
