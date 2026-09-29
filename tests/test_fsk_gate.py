"""Regression tests for the conservative FSK family gate."""
from __future__ import annotations

import numpy as np

from signal9.dsp.fsk import looks_like_fsk


def test_rectangular_qpsk_is_not_misrouted_to_fsk() -> None:
    """QPSK phase-transition impulses are not FSK tone plateaux."""
    rng = np.random.default_rng(73)
    points = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64) / np.sqrt(2)
    symbols = points[rng.integers(0, len(points), size=1024)]
    iq = np.repeat(symbols, 4).astype(np.complex64)
    assert not looks_like_fsk(iq, 48_000.0)


def test_two_tone_fsk_still_passes_gate() -> None:
    """A balanced 2-FSK waveform has two stable frequency plateaux."""
    rng = np.random.default_rng(74)
    bits = rng.integers(0, 2, size=1024)
    freq = np.repeat(np.where(bits == 1, 3_000.0, -3_000.0), 8)
    phase = np.cumsum(2.0 * np.pi * freq / 48_000.0)
    iq = np.exp(1j * phase).astype(np.complex64)
    assert looks_like_fsk(iq, 48_000.0)
