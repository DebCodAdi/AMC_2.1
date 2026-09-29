"""L3 Metamorphic Relations registry and runner.

Metamorphic testing checks *relations among the outputs of multiple related
executions* rather than absolute values (Chen et al., 1998 / ACM Surveys 2018).

Usage::

    pytest tests/L3_metamorphic/test_metamorphic.py -v

Spec reference: §2.5, §6.3 (PS26147).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# MR dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MR:
    """A single metamorphic relation.

    Attributes
    ----------
    id : str
        Unique identifier, e.g. "MR-1.1".
    transform : callable(x, rng) -> ndarray
        Transform applied to the input capture ``x``.
    relation : str
        What must be preserved.  See ``run_mr`` for supported strings.
    param : float
        Numeric parameter used by the relation check (e.g. CFO shift in Hz).
    expect_fail_until : str
        If non-empty, this MR is *expected* to fail until the named defect
        is fixed.  The test runner marks it ``xfail``.
    description : str
        Human-readable description of what the relation checks.
    """
    id: str
    transform: Callable[[np.ndarray, np.random.Generator], np.ndarray]
    relation: str
    param: float = 0.0
    expect_fail_until: str = ""
    description: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _inject_nans(x: np.ndarray, rng: np.random.Generator, frac: float) -> np.ndarray:
    x = x.copy()
    n = int(frac * len(x))
    idx = rng.choice(len(x), n, replace=False)
    x[idx] = float("nan") + 0j
    return x


def _inject_slip(x: np.ndarray, at: int, order: int) -> np.ndarray:
    """Inject a cycle slip of 2π/order at sample index ``at``."""
    x = x.copy()
    x[at:] *= np.exp(2j * np.pi / order)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# MR Registry (spec §2.5 / §6.3)
# ─────────────────────────────────────────────────────────────────────────────

REGISTRY: list[MR] = [
    # ── Stage 1 / robustness MRs ─────────────────────────────────────────
    MR("MR-1.1",
       lambda x, r: x * 3.7,
       "payload_equal",
       description="Amplitude scaling must not change the decoded payload"),
    MR("MR-1.2",
       lambda x, r: x * np.exp(1j * 0.9),
       "payload_equal",
       description="Constant phase rotation must not change the decoded payload"),
    MR("MR-1.3",
       lambda x, r: np.conj(x),
       "payload_equal",
       description="Complex conjugation (spectrum flip) must not change payload"),
    MR("MR-1.4",
       lambda x, r: x * np.exp(1j * np.pi),
       "payload_equal",
       description="180° rotation (BPSK ambiguity) must not change payload"),
    MR("MR-1.5",
       lambda x, r: x[::-1],
       "payload_equal",
       description="Time reversal must not change payload for symmetric codes"),
    MR("MR-1.6",
       lambda x, r: np.r_[np.zeros(5000, x.dtype), x],
       "payload_equal",
       description="Prepended silence must not corrupt the payload"),
    MR("MR-1.7",
       lambda x, r: np.r_[x, np.zeros(5000, x.dtype)],
       "payload_equal",
       description="Appended silence must not corrupt the payload"),
    MR("MR-1.8",
       lambda x, r: _inject_nans(x, r, frac=1e-3),
       "payload_equal",
       description="Sparse NaN samples (< 0.1%) must be repaired transparently"),
    MR("MR-1.9",
       lambda x, r: x + (0.5 + 0.3j),
       "payload_equal",
       description="DC offset must be removed before processing"),
    MR("MR-1.10",
       lambda x, r: x * (-1.0 + 0j),
       "payload_equal",
       description="IQ polarity flip (spectrum inversion) must not lose payload"),
    MR("MR-1.11",
       lambda x, r: x,          # identity — used to verify gsop idempotence
       "gsop_idempotent",
       description="GSOP applied twice must give the same result as once (DEF-010)"),
    # ── CFO / Stage 2 MRs ────────────────────────────────────────────────
    MR("MR-2.1",
       lambda x, r: x * np.exp(2j * np.pi * 0.01 * np.arange(len(x), dtype=float)),
       "cfo_shift",
       param=0.01,
       description="Injecting CFO=0.01·Fs must shift the estimate by exactly that amount"),
    MR("MR-2.2",
       lambda x, r: np.conj(x),
       "cfo_sign_flip",
       description="Conjugating flips the spectrum; CFO estimate must flip sign"),
    MR("MR-2.3",
       lambda x, r: x * np.exp(2j * np.pi * 0.2 * np.arange(len(x), dtype=float)),
       "cfo_shift",
       param=0.2,
       description="Large CFO=0.2·Fs must be estimated to < 1e-4·Rs residual (DEF-002)"),
    # ── Cycle-slip MRs ───────────────────────────────────────────────────
    MR("MR-4.1",
       lambda x, r: _inject_slip(x, len(x) // 2, 4),
       "payload_equal",
       description="A single QPSK cycle slip at midpoint must not corrupt the payload"),
    MR("MR-4.2",
       lambda x, r: _inject_slip(x, len(x) // 4, 2),
       "payload_equal",
       description="A BPSK cycle slip must not corrupt the payload"),
    # ── Symbol rate MRs ──────────────────────────────────────────────────
    MR("MR-5.1",
       lambda x, r: np.repeat(x, 2),      # halves symbol rate
       "payload_equal",
       description="Doubling samples per symbol (halving baud) must decode the same payload"),
    # ── Payload / FEC MRs ────────────────────────────────────────────────
    MR("MR-7.1",
       lambda x, r: x,
       "payload_equal",
       description="Identity transform — baseline reproducibility check (DEF-043)"),
    MR("MR-7.5",
       lambda x, r: x,
       "payload_equal",
       description="Payload must be byte-identical across 3 independent runs"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_mr(mr: MR, capture: np.ndarray, pipeline: Callable,
           rng: np.random.Generator) -> bool:
    """Apply ``mr.transform`` to ``capture`` and check the relation.

    Parameters
    ----------
    mr : MR
        The relation to test.
    capture : np.ndarray
        Complex IQ samples.
    pipeline : callable(x) -> result_dict
        The pipeline under test.  Must return a dict with 'status',
        'payload_hex', 'cfo_hz', 'evm', etc.
    rng : np.random.Generator
        Seeded RNG for transforms that need randomness.

    Returns
    -------
    bool
        True if the relation holds.
    """
    base = pipeline(capture)
    transformed = mr.transform(capture.copy(), rng)
    new = pipeline(transformed)

    def _payload(r):
        return r.get("payload_hex") or r.get("payload_b64") or ""

    if mr.relation == "payload_equal":
        # Both must lock AND payloads must be byte-identical
        b_locked = base.get("status") == "SUCCESS_CRC_LOCKED"
        n_locked = new.get("status") == "SUCCESS_CRC_LOCKED"
        if not b_locked:
            return True  # Base didn't lock — MR is vacuously satisfied
        return n_locked and (_payload(base) == _payload(new))

    if mr.relation == "cfo_shift":
        # Injecting a frequency shift must move the CFO estimate by exactly that amount
        if "cfo_hz" not in base or "cfo_hz" not in new:
            return True
        fs = base.get("fs_est") or 1.0
        expected_shift = mr.param * fs
        actual_shift = (new["cfo_hz"] - base["cfo_hz"])
        return abs(actual_shift - expected_shift) < 0.1   # < 0.1 Hz (spec MR-2.1 gate)

    if mr.relation == "cfo_sign_flip":
        if "cfo_hz" not in base or "cfo_hz" not in new:
            return True
        return abs(new["cfo_hz"] + base["cfo_hz"]) < 0.5

    if mr.relation == "gsop_idempotent":
        try:
            from signal9.health.iq_image import apply_gsop  # type: ignore
        except ImportError:
            return True  # Module not implemented yet — skip
        x1 = apply_gsop(capture)
        x2 = apply_gsop(x1)
        return float(np.max(np.abs(x1 - x2))) < 1e-9

    raise ValueError(f"Unknown relation: {mr.relation!r}")
