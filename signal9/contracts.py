"""Stage contracts — typed dataclasses with runtime ``validate()`` methods.

Each stage in the Signal9 pipeline has a *contract*: a typed description
of the data it must produce.  Violating a contract is a hard failure
**independently of end-to-end accuracy** (spec Rule R7, §2.1).

This enforces:
  • C1.x  Stage 1 (Ingestion) guarantees
  • C2.x  Stage 2 (Spectral / resample) guarantees
  • C4.x  Stage 4 (Synchroniser) guarantees

Usage::

    out = Stage1Out(x=samples, fs_hint=None, fmt="cs16",
                   n_repaired=0, mme_ratio=3.2)
    out.validate()   # raises AssertionError with a named tag on failure

The validate() call is cheap (pure NumPy, no copies).  Run it at the end
of every stage in debug/test mode; gate it behind ``__debug__`` in
production if needed.

Spec reference: §2.1 (PS26147).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Ingestion
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Stage1Out:
    """Output contract for Stage 1 (Ingestion).

    Attributes
    ----------
    x : np.ndarray
        1-D complex128 array, fully finite, DC-removed, normalised.
    fs_hint : float | None
        Sample rate in Hz derived from file metadata, or None if unknown.
    fmt : str
        Resolved dtype name (e.g. "cs16", "cf32", "wav_stereo_iq").
    n_repaired : int
        Number of NaN / Inf samples patched during ingestion.
    mme_ratio : float
        Maximum-to-median eigenvalue ratio from the MME detector.
    """

    x: np.ndarray
    fs_hint: float | None
    fmt: str
    n_repaired: int
    mme_ratio: float

    # ── Contract invariants ───────────────────────────────────────────────
    def validate(self) -> None:
        """Raise ``AssertionError`` with a tag if any contract is violated."""
        assert self.x.ndim == 1, "C1.0 x must be 1-D"
        assert self.x.dtype == np.complex128, (
            f"C1.0 x must be complex128, got {self.x.dtype}"
        )
        assert len(self.x) > 0, "C1.0 x must be non-empty"
        assert np.isfinite(self.x).all(), "C1.1 non-finite sample escaped ingestion"
        assert abs(np.mean(self.x)) < 1e-3, (
            f"C1.2 DC not removed (mean={abs(np.mean(self.x)):.4e})"
        )
        std = np.std(self.x)
        assert 0.5 < std < 2.0, (
            f"C1.3 power not normalised (std={std:.4f}, expected 0.5…2.0)"
        )
        if self.fs_hint is not None:
            assert self.fs_hint > 0, "C1.4 fs_hint must be positive"
        assert self.n_repaired >= 0, "C1.5 n_repaired must be non-negative"
        assert self.mme_ratio >= 0.0, "C1.6 mme_ratio must be non-negative"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Spectral / Resample
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Stage2Out:
    """Output contract for Stage 2 (Spectral analysis + resampling).

    Attributes
    ----------
    x2 : np.ndarray
        1-D complex128 at exactly 2 SPS.
    fs : float
        Estimated sample rate in Hz.
    rs : float
        Estimated symbol rate in baud.
    sps_pre : float
        SPS before resampling (= fs / rs).
    cfo_hz : float
        Coarse CFO estimate in Hz.
    alpha : float
        Best-estimate roll-off factor ∈ [0, 1].
    fs_candidates : list[tuple[float, float]]
        Ranked (fs, log-likelihood) pairs.  At least one entry required
        (DEF-011 fix: never collapse to a hard single value until
        the blackboard resolves ambiguity).
    """

    x2: np.ndarray
    fs: float
    rs: float
    sps_pre: float
    cfo_hz: float
    alpha: float
    fs_candidates: list

    def validate(self) -> None:
        assert self.x2.ndim == 1, "C2.0 x2 must be 1-D"
        assert np.isfinite(self.x2).all(), "C2.1 non-finite sample in resampled signal"
        assert self.fs > 0, "C2.5 fs must be positive"
        assert self.rs > 0, "C2.5 rs must be positive"
        sps_actual = self.fs / self.rs
        assert abs(sps_actual - 2.0) < 1e-4, (
            f"C2.2 output is not 2 SPS (fs={self.fs:.1f}, rs={self.rs:.1f}, "
            f"sps={sps_actual:.6f})"
        )
        assert abs(self.cfo_hz) <= 0.5 * self.fs, (
            f"C2.3 CFO estimate {self.cfo_hz:.1f} Hz exceeds Nyquist "
            f"(±{0.5*self.fs:.1f} Hz)"
        )
        assert 0.0 <= self.alpha <= 1.0, (
            f"C2.4 roll-off α must be ∈ [0,1], got {self.alpha}"
        )
        assert len(self.fs_candidates) >= 1, (
            "C2.6 fs_candidates must contain at least one entry"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Synchroniser
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Stage4Out:
    """Output contract for Stage 4 (Timing & carrier synchronisation).

    Attributes
    ----------
    sym : np.ndarray
        1-D complex array of synchronised symbols at 1 SPS.
    slip_indices : list[int]
        Sample indices at which cycle slips were detected (DEF-003).
        An empty list means no slips detected.
    evm : float
        Error-vector magnitude ∈ [0, 2].  0 = perfect; 1 = noise-equivalent.
    snr_db : float
        Estimated SNR in dB.  May be −∞ for very short arrays.
    """

    sym: np.ndarray
    slip_indices: list
    evm: float
    snr_db: float

    def validate(self) -> None:
        assert self.sym.ndim == 1, "C4.0 sym must be 1-D"
        assert len(self.sym) > 0, "C4.0 sym must be non-empty"
        assert np.isfinite(self.sym).all(), "C4.1 non-finite symbol escaped synchroniser"
        assert 0.0 <= self.evm <= 2.0, (
            f"C4.2 EVM={self.evm:.4f} out of range [0, 2]"
        )
        n = len(self.sym)
        for idx in self.slip_indices:
            assert 0 <= idx < n, (
                f"C4.3 slip index {idx} out of bounds for sym length {n}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry schema
# ─────────────────────────────────────────────────────────────────────────────

def empty_telemetry() -> dict:
    """Return a fully-specified telemetry dict with sentinel values.

    Every pipeline result must include all of these keys.  Missing keys
    are a contract violation caught by L0_contracts tests.

    Spec reference: §2.2 — Failure Attribution Waterfall columns.
    """
    return {
        # ── Per-stage estimates ───────────────────────────────────────────
        "fs_est": None,           # Hz, float or None
        "fs_conf": None,          # 0..1, float or None
        "rs_est": None,           # baud, float or None
        "alpha_est": None,        # roll-off, float or None
        "cfo_est": None,          # Hz, float or None
        "snr_est": None,          # dB, float or None
        "evm": None,              # dimensionless, float or None
        "n_slips": None,          # int ≥ 0 or None
        # ── Branch / arbiter ─────────────────────────────────────────────
        "n_branches_locked": 0,   # int
        "branch_name": None,      # str or None — the winning branch
        # ── Timing ───────────────────────────────────────────────────────
        "t_stage_ms": {},         # dict[str, float] — ms per named stage
        "t_total_ms": None,       # float or None
        # ── Failure attribution ───────────────────────────────────────────
        "failure_stage": None,    # int 1..7 or None
        "failure_reason": None,   # str or None
        # ── Confidence ───────────────────────────────────────────────────
        "confidence": None,       # float 0..1 or None
        "plausibility_score": None,  # float or None (DEF-006)
    }


def validate_result_schema(result: dict) -> list[str]:
    """Return a list of schema-violation strings.  Empty list = valid.

    Checks that every key from ``empty_telemetry()`` is present and that
    the ``status`` field is a legal Status member.
    """
    from signal9.status import Status

    errors: list[str] = []
    required = set(empty_telemetry().keys()) | {"status", "file"}
    missing = required - set(result.keys())
    for k in sorted(missing):
        errors.append(f"Missing required key: '{k}'")

    if "status" in result:
        try:
            Status(result["status"])
        except ValueError:
            errors.append(f"Unknown status value: '{result['status']}'")

    return errors
