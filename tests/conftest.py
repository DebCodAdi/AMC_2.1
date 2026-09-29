"""Pytest configuration — fixtures, seeds, corpus loaders.

All tests that need a seeded RNG, a synthetic capture, or access to the
golden corpus import from here.  Nothing in tests/ should call
``np.random`` directly.

Spec reference: R5, §6.1, §6.2 (PS26147).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

# ── Seed fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def global_seed() -> int:
    """Single master seed for the entire test session (DEF-043)."""
    return 42


@pytest.fixture
def rng(global_seed) -> np.random.Generator:
    """Fresh PCG64 Generator for each test.  Tests are deterministic across
    runs when using this fixture."""
    return np.random.default_rng(np.random.SeedSequence(global_seed))


# ── Golden corpus ────────────────────────────────────────────────────────────

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_PAYLOADS = GOLDEN_DIR / "payloads"
GOLDEN_TELEMETRY = GOLDEN_DIR / "telemetry"
GOLDEN_MANIFEST = GOLDEN_DIR / "MANIFEST.json"


@pytest.fixture(scope="session")
def golden_manifest() -> dict:
    """Load the frozen golden manifest once per session."""
    if not GOLDEN_MANIFEST.exists():
        pytest.skip(
            f"Golden manifest not found at {GOLDEN_MANIFEST}. "
            "Run Phase 0 snapshot capture first."
        )
    with open(GOLDEN_MANIFEST, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def golden_entries(golden_manifest) -> list[dict]:
    """List of golden entries from the manifest."""
    return golden_manifest.get("entries", [])


# ── Synthetic capture helpers ─────────────────────────────────────────────────

def make_bpsk(n_sym: int = 1024,
              sps: float = 4.0,
              snr_db: float = 20.0,
              seed: int = 0) -> np.ndarray:
    """Quick BPSK capture for unit tests — no FEC, no shaping."""
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    bits = rng.integers(0, 2, n_sym)
    syms = 2.0 * bits - 1.0 + 0j
    x = np.repeat(syms, int(sps))
    sp = np.mean(np.abs(x) ** 2)
    sigma = np.sqrt(sp / (2 * 10 ** (snr_db / 10)))
    x = x + sigma * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x.astype(np.complex64)


def make_qpsk(n_sym: int = 1024,
              sps: float = 4.0,
              snr_db: float = 20.0,
              seed: int = 0) -> np.ndarray:
    """Quick QPSK capture for unit tests — no FEC, no shaping."""
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    bits = rng.integers(0, 2, 2 * n_sym)
    i = 2.0 * bits[0::2] - 1.0
    q = 2.0 * bits[1::2] - 1.0
    syms = (i + 1j * q) / np.sqrt(2)
    x = np.repeat(syms, int(sps))
    sp = np.mean(np.abs(x) ** 2)
    sigma = np.sqrt(sp / (2 * 10 ** (snr_db / 10)))
    x = x + sigma * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x.astype(np.complex64)


# ── Pytest marks ─────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (run with --slow to include them)",
    )
    config.addinivalue_line(
        "markers",
        "requires_dataset: marks tests that need the 200-file real corpus",
    )
    config.addinivalue_line(
        "markers",
        "requires_gnuradio: marks L4 tests that need GNU Radio",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--slow", action="store_true", default=False,
        help="Run slow tests (L5 Monte-Carlo, L7 chaos)",
    )
    parser.addoption(
        "--with-dataset", action="store_true", default=False,
        help="Enable tests that require the 200-file real corpus",
    )


def pytest_collection_modifyitems(config, items):
    skip_slow = not config.getoption("--slow")
    skip_dataset = not config.getoption("--with-dataset")
    for item in items:
        if "slow" in item.keywords and skip_slow:
            item.add_marker(pytest.mark.skip(reason="Pass --slow to run"))
        if "requires_dataset" in item.keywords and skip_dataset:
            item.add_marker(pytest.mark.skip(reason="Pass --with-dataset to run"))
        if "requires_gnuradio" in item.keywords:
            try:
                import gnuradio  # noqa: F401
            except ImportError:
                item.add_marker(pytest.mark.skip(reason="GNU Radio not installed"))
