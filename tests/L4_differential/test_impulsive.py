"""Differential test for impulsive noise detection and mitigation (Track F-3)."""
from __future__ import annotations
import numpy as np

from signal9.sim.noise import middleton_class_a_noise
from signal9.health.impulsive import detect_impulsive_noise, huber_winsorize, mcc_mma_equalizer


def test_detect_impulsive_vs_gaussian():
    rng = np.random.default_rng(42)
    # Pure Gaussian noise
    gauss = (rng.standard_normal(8192) + 1j * rng.standard_normal(8192)).astype(np.complex64)
    is_imp_g, kurt_g = detect_impulsive_noise(gauss)
    assert not is_imp_g
    assert kurt_g < 2.5

    # Middleton Class-A impulsive noise
    mid = middleton_class_a_noise(8192, total_power=1.0, impulse_index_A=0.02, seed=42)
    is_imp_m, kurt_m = detect_impulsive_noise(mid)
    assert is_imp_m
    assert kurt_m > 3.5


def test_huber_winsorization():
    rng = np.random.default_rng(42)
    clean = np.ones(1000, dtype=np.complex64)
    corrupted = clean.copy()
    # Inject 5 huge spikes
    corrupted[100] = 50.0 + 50j
    corrupted[200] = -40.0 + 30j

    win = huber_winsorize(corrupted, k_sigma=3.0)
    # The spikes should be suppressed
    assert np.abs(win[100]) < 10.0
    assert np.abs(win[200]) < 10.0
    # Clean samples should remain close to 1.0
    assert np.isclose(np.abs(win[500]), 1.0, atol=0.1)
    # Phase must be preserved
    assert np.isclose(np.angle(win[100]), np.angle(corrupted[100]), atol=1e-5)


def test_mcc_mma_convergence():
    rng = np.random.default_rng(42)
    # Simple QPSK constellation
    bits = rng.integers(0, 2, 2000, dtype=np.uint8)
    i_sym = 2.0 * bits[0::2] - 1.0
    q_sym = 2.0 * bits[1::2] - 1.0
    syms = (i_sym + 1j * q_sym) / np.sqrt(2.0)
    # Add impulsive noise
    noise = middleton_class_a_noise(len(syms), total_power=0.01, impulse_index_A=0.05, seed=42)
    rx = syms + noise

    y, w = mcc_mma_equalizer(rx, sps=1, n_taps=11, mu=0.001)
    assert len(y) == len(rx)
    assert not np.isnan(w).any()
