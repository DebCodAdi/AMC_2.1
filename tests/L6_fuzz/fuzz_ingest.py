"""L6 Fuzzing tier — Ingestion and Front-End robustness.

Verifies that no input causes an unhandled exception or crash (DEF-025, DEF-026,
DEF-033, DEF-034). Every input must either return a valid Stage1Out or an
explicit Status failure code (e.g. EMPTY_FILE, INSUFFICIENT_SAMPLES, TOO_CORRUPT,
INVALID_SAMPLES).

Spec reference: §2.4 (L6 tier), DEF-025, DEF-026, DEF-033, DEF-034 (PS26147).
"""
from __future__ import annotations
import pytest
import numpy as np

from signal9.io.resampler import repair_nans, check_nan_density, safe_resample_poly
from signal9.status import Status
from signal9.dsp.cfo import mth_power_cfo
from signal9.dsp.matched_filter import rrc_taps, apply_rrc
from signal9.features.cumulants import compute_cumulants


class TestL6Fuzz:
    """Fuzzing tests across corrupted, random, and pathological inputs."""

    def test_fuzz_all_nan_zero_samples(self):
        """All-NaN or all-zero input must return INVALID_SAMPLES / TOO_CORRUPT without crash."""
        all_nan = np.full(512, np.nan, dtype=np.complex64)
        is_corrupt, frac = check_nan_density(all_nan, limit=0.5)
        assert is_corrupt
        assert frac == 1.0

        all_zero = np.zeros(512, dtype=np.complex64)
        # CFO estimation on all-zero should not raise ZeroDivisionError
        cfo = mth_power_cfo(all_zero, fs=48000.0, M=4)
        assert np.isfinite(cfo)

    def test_fuzz_odd_length_samples(self):
        """Odd-length input (DEF-025) must not crash downstream processors."""
        rng = np.random.default_rng(123)
        odd_len = 1001
        x = (rng.standard_normal(odd_len) + 1j * rng.standard_normal(odd_len)).astype(np.complex64)
        cfo = mth_power_cfo(x, fs=48000.0, M=4)
        assert np.isfinite(cfo)
        cums, status = compute_cumulants(x)
        assert status == "OK" or status in Status.members()

    def test_fuzz_nan_inf_random_bursts(self):
        """Bursts of NaNs and Infs must be safely repaired or gated (DEF-033, DEF-034)."""
        rng = np.random.default_rng(456)
        x = (rng.standard_normal(500) + 1j * rng.standard_normal(500)).astype(np.complex64)
        # Inject NaNs at boundaries and interior
        x[0:5] = np.nan
        x[100:110] = np.nan
        x[-5:] = np.nan
        x[250] = np.inf

        repaired, _ = repair_nans(x)
        assert not np.any(np.isnan(repaired))
        assert not np.any(np.isinf(repaired))

    def test_fuzz_extreme_ratios_resample(self):
        """Extreme P/Q resample ratios (DEF-035) must not cause tap explosions."""
        x = np.ones(128, dtype=np.complex64)
        # Pathological ratio 14400 / 10000 -> GCD reduced and capped to <= 512
        resampled = safe_resample_poly(x, 14400, 10000, max_tap_length=512)
        assert len(resampled) > 0
        assert np.all(np.isfinite(resampled))

    def test_fuzz_zero_length(self):
        """Zero-length input must return empty without crashing."""
        x_empty = np.zeros(0, dtype=np.complex64)
        cfo = mth_power_cfo(x_empty, fs=48000.0, M=4)
        assert cfo == 0.0
        cums, status = compute_cumulants(x_empty)
        assert status == Status.INSUFFICIENT_SAMPLES.value
