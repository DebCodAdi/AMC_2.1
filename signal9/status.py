"""Closed status enumeration for the Signal9 pipeline.

Every result dict produced by the pipeline must carry a ``status`` field
whose value is one of these members.  String comparisons against bare
literals are forbidden in src/ — import this enum instead.

Spec reference: §9.3 (PS26147).
"""
from __future__ import annotations
from enum import Enum


class Status(str, Enum):
    """All possible pipeline outcomes.  The set is closed — adding a new
    condition requires adding it here AND updating the schema test in
    L0_contracts.
    """

    # ── Success ────────────────────────────────────────────────────────────
    SUCCESS_CRC_LOCKED = "SUCCESS_CRC_LOCKED"
    """CRC-32 (or variant) was validated over the decoded payload."""

    # ── Soft success / near-miss ────────────────────────────────────────────
    DEMOD_NO_CRC = "DEMOD_NO_CRC"
    """Demodulation produced symbols but no branch locked a CRC.
    The plausibility score may still be high — check ``confidence``."""

    # ── Detection / signal-level failures ──────────────────────────────────
    NO_SIGNAL_DETECTED = "NO_SIGNAL_DETECTED"
    """MME or energy test found no signal above the noise floor."""

    UNMODULATED_CARRIER = "UNMODULATED_CARRIER"
    """A pure CW tone was detected but no modulation was found."""

    UNKNOWN_MODULATION = "UNKNOWN_MODULATION"
    """AMC reject option fired: Mahalanobis distance exceeds χ²₆(0.999)."""

    # ── File / format failures ──────────────────────────────────────────────
    EMPTY_FILE = "EMPTY_FILE"
    """File contains zero bytes."""

    INSUFFICIENT_SAMPLES = "INSUFFICIENT_SAMPLES"
    """File contains fewer than MIN_SAMPLES after format parsing."""

    INSUFFICIENT_SAMPLES_FOR_DETECTION = "INSUFFICIENT_SAMPLES_FOR_DETECTION"
    """Sample count is below the minimum for reliable MME (P < 20·L)."""

    INVALID_SAMPLES = "INVALID_SAMPLES"
    """All samples are NaN, all-zero, or otherwise degenerate."""

    TOO_CORRUPT = "TOO_CORRUPT"
    """NaN density exceeds tolerance (> 50%); spline repair refused."""

    CLIPPED_ADC_SATURATION = "CLIPPED_ADC_SATURATION"
    """Hard-clipping fraction exceeds the reporting threshold."""

    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    """File format could not be identified or is explicitly unsupported."""

    # ── Unsupported features ────────────────────────────────────────────────
    UNSUPPORTED_SCRAMBLER = "UNSUPPORTED_SCRAMBLER"
    """An additive or self-synchronising scrambler was detected.
    Combinatorial descrambling is not implemented."""

    UNSUPPORTED_FEC = "UNSUPPORTED_FEC"
    """The identified or suspected FEC type (turbo, tail-biting, RSC, …)
    is not in the supported decoder set."""

    UNSUPPORTED_INTERLEAVER = "UNSUPPORTED_INTERLEAVER"
    """The interleaver seed or geometry cannot be recovered (e.g. PRNG
    with unknown seed)."""

    # ── Synchronisation failures ────────────────────────────────────────────
    PLL_UNLOCKED = "PLL_UNLOCKED"
    """The Costas or decision-directed PLL failed to acquire.
    Residual CFO likely exceeds ω_max."""

    # ── Arbiter / branch-pool failures ─────────────────────────────────────
    ALL_BRANCHES_FAILED = "ALL_BRANCHES_FAILED"
    """Every hypothesis branch raised an exception or returned no result."""

    # ── Resource / system failures ──────────────────────────────────────────
    TIMEOUT = "TIMEOUT"
    """A branch or the full pipeline exceeded the per-file wall-clock budget."""

    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    """A branch exceeded its memory or CPU quota and was killed."""

    # ──────────────────────────────────────────────────────────────────────
    @classmethod
    def is_success(cls, s: "Status") -> bool:
        """Return True only for the hard-success status."""
        return s is cls.SUCCESS_CRC_LOCKED

    @classmethod
    def is_failure(cls, s: "Status") -> bool:
        return s not in (cls.SUCCESS_CRC_LOCKED, cls.DEMOD_NO_CRC)

    @classmethod
    def members(cls) -> list[str]:
        """Return every member value as a sorted list of strings."""
        return sorted(m.value for m in cls)
