"""Regression tests for the classifier's conservative conditioning contract."""
import numpy as np

from signal9.preprocess import condition_for_classifier


def test_repair_and_normalize_are_finite():
    x = np.array([1 + 2j, np.nan + 1j, np.inf + 0j, -1 - 2j], dtype=np.complex64)
    y, audit = condition_for_classifier(x)
    assert np.all(np.isfinite(y))
    assert audit.finite_repaired == 2
    assert np.isclose(np.mean(np.abs(y) ** 2), 1.0)


def test_preserves_real_carrier_dc_component():
    # A post-channelization DC component can be a valid AM/CW carrier.
    x = np.full(4096, 2 + 0j, dtype=np.complex64)
    y, audit = condition_for_classifier(x)
    assert audit.dc_preserved
    assert np.allclose(y, 1 + 0j)


def test_impulsive_contamination_is_gated_and_reduced():
    rng = np.random.default_rng(7)
    x = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(np.complex64)
    x[::100] += 40 + 40j
    y, audit = condition_for_classifier(x)
    assert audit.impulsive_noise
    assert audit.impulsive_mitigation_applied
    assert np.isfinite(y).all()
