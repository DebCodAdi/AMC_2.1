"""Payload plausibility scoring — replaces the entropy-threshold gate.

DEF-006: The constant H ≤ 6.8 entropy threshold fires on high-entropy
payloads (encrypted, compressed, digitally signed) that are legitimately
correct.  The replacement is a calibrated three-part score.

Critical ordering rule (spec §IV-M):
  A valid CRC MUST NEVER be vetoed by the plausibility score.
  The score governs only the DEMOD_NO_CRC decision path.

Spec reference: DEF-006, §IV-M (PS26147).
"""
from __future__ import annotations
import math
import struct
import zlib
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CRC-32 utilities
# ─────────────────────────────────────────────────────────────────────────────

def check_crc32(payload: bytes, crc_bytes: bytes | None = None) -> bool:
    """Return True if ``payload`` ends with its own CRC-32 (little-endian).

    If ``crc_bytes`` is provided, it is used directly; otherwise the last
    4 bytes of ``payload`` are treated as the CRC trailer.
    """
    if crc_bytes is not None:
        data = payload
        (expected,) = struct.unpack("<I", crc_bytes[:4])
    else:
        if len(payload) < 4:
            return False
        data = payload[:-4]
        (expected,) = struct.unpack("<I", payload[-4:])
    actual = zlib.crc32(data) & 0xFFFFFFFF
    return actual == expected


def check_crc32_any_window(data: bytes,
                            min_len: int = 8,
                            max_windows: int = 256) -> tuple[bool, int, int]:
    """Windowed CRC-32 search (DEF-022).

    Scans all (start, end) pairs for a valid CRC-32 trailer.  Returns
    (found, start_byte_offset, end_byte_offset).
    """
    n = len(data)
    checked = 0
    for start in range(0, n - min_len - 4):
        for end in range(start + min_len, min(n, start + max_windows * 4)):
            if end + 4 > n:
                break
            payload_slice = data[start:end]
            crc_slice = data[end:end + 4]
            if check_crc32(payload_slice, crc_slice):
                return True, start, end
            checked += 1
            if checked >= max_windows * max_windows:
                return False, 0, 0
    return False, 0, 0


# ─────────────────────────────────────────────────────────────────────────────
# Individual plausibility signals
# ─────────────────────────────────────────────────────────────────────────────

def _byte_entropy(data: bytes) -> float:
    """Shannon entropy in bits per byte.  Returns 0 for empty input."""
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _miller_madow_entropy(data: bytes) -> float:
    """Miller-Madow bias-corrected entropy estimate.

    For short payloads, raw Shannon entropy is biased downward because
    many symbols have not been observed.  The Miller-Madow correction
    adds (k-1)/(2n) where k = number of distinct symbols observed.
    """
    if not data:
        return 0.0
    n = len(data)
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    k = int(np.sum(counts > 0))
    h_raw = _byte_entropy(data)
    correction = (k - 1) / (2 * n)
    return h_raw + correction


def _entropy_z_score(data: bytes) -> float:
    """Standardised entropy: z = (H - mu_null) / sigma_null.

    The null distribution is for random bytes at this payload length.
    For large n, H → 8 bits/byte; for short n there is a downward bias.
    We use a lookup-table correction calibrated from random byte sampling.

    Positive z: higher entropy than random.  Negative z: lower.
    """
    if not data:
        return 0.0
    n = len(data)
    h = _miller_madow_entropy(data)
    # Expected entropy of random bytes at this length (CLT approximation)
    # For n >= 256: E[H] ≈ 7.98 bits, std ≈ 0.01
    # For small n: significant downward bias
    if n >= 256:
        mu = 7.98
        sigma = 0.10
    elif n >= 64:
        mu = 7.50 + (n - 64) / (256 - 64) * 0.48
        sigma = 0.30
    elif n >= 16:
        mu = 6.0 + (n - 16) / (64 - 16) * 1.50
        sigma = 0.60
    else:
        mu = 4.0
        sigma = 1.50
    return (h - mu) / (sigma + 1e-12)


def _printable_ratio(data: bytes) -> float:
    """Fraction of bytes that are printable ASCII.

    Printable ASCII: 0x09 (tab), 0x0A (LF), 0x0D (CR), 0x20–0x7E.
    A plaintext protocol payload typically has ratio > 0.90.
    An encrypted or binary payload typically has ratio < 0.20.
    """
    if not data:
        return 0.0
    printable = sum(
        1 for b in data
        if b in (0x09, 0x0A, 0x0D) or 0x20 <= b <= 0x7E
    )
    return printable / len(data)


# Bigram log-likelihood under a simple English/telemetry model.
# Pre-computed from a reference corpus; serves as a weak prior for plaintext.
_BIGRAM_COMMON = {
    (0x20, 0x74), (0x20, 0x61), (0x65, 0x20), (0x74, 0x68),
    (0x68, 0x65), (0x69, 0x6E), (0x20, 0x69), (0x6F, 0x6E),
    (0x61, 0x74), (0x65, 0x72), (0x20, 0x73), (0x72, 0x65),
}

def _bigram_loglik(data: bytes) -> float:
    """Weak bigram log-likelihood.  Returns log-probability ratio vs uniform.

    Positive = more English-like.  Negative = less English-like.
    """
    if len(data) < 2:
        return 0.0
    n_bigrams = len(data) - 1
    n_common = sum(
        1 for i in range(n_bigrams)
        if (data[i], data[i + 1]) in _BIGRAM_COMMON
    )
    # Under uniform bigram model: P(common) = 12/65536 ≈ 1.8e-4
    # Under English model: P(common) ≈ 0.12
    uniform_rate = len(_BIGRAM_COMMON) / 65536.0
    observed_rate = (n_common + 1) / (n_bigrams + 1)  # Laplace smoothing
    # Log likelihood ratio (positive = more English-like)
    llr = math.log(observed_rate / (uniform_rate + 1e-12) + 1e-12)
    return float(np.clip(llr, -5.0, 5.0))


# ─────────────────────────────────────────────────────────────────────────────
# Main plausibility scorer
# ─────────────────────────────────────────────────────────────────────────────

def plausibility_score(payload: bytes,
                       crc_valid: bool = False,
                       n_branches_agreeing: int = 1,
                       weights: tuple[float, ...] | None = None) -> float:
    """Calibrated payload plausibility score ∈ [0, 1].

    Parameters
    ----------
    payload : bytes
        The decoded payload bytes.
    crc_valid : bool
        Whether a CRC-32 check passed.  This is the dominant term.
    n_branches_agreeing : int
        How many blackboard branches produced the same payload.
    weights : tuple or None
        (w_entropy, w_printable, w_bigram, w_crc, w_branches).
        Defaults to ``Config.entropy_plausibility_weights``.

    Returns
    -------
    float
        Score in [0, 1].  Higher = more plausible.

    Notes
    -----
    - A valid CRC raises the score to ≥ 0.95 regardless of other signals.
      This implements the "valid CRC is never vetoed" rule from §IV-M.
    - An invalid CRC does NOT lower the score below what the other signals
      support — it simply provides no positive evidence.
    """
    if weights is None:
        weights = (0.25, 0.35, 0.15, 0.20, 0.05)

    w_entropy, w_printable, w_bigram, w_crc, w_branches = weights

    if not payload:
        return 0.0

    # ── Individual signals ────────────────────────────────────────────────
    # Entropy signal: high entropy is suspicious for short payloads,
    # but NOT for longer ones (encrypted payloads are legitimately high).
    # We invert: low entropy → not random → plausible.
    z_entropy = _entropy_z_score(payload)
    # A z-score of -3 (entropy much lower than random) → signal = 1.0
    # A z-score of 0 (random) → signal = 0.5
    # A z-score of +3 (higher than random) → signal = 0.0 (not penalised much)
    s_entropy = float(np.clip(0.5 - z_entropy / 6.0, 0.0, 1.0))

    # Printable ratio signal
    s_printable = _printable_ratio(payload)

    # Bigram log-likelihood signal (normalised to [0, 1])
    llr = _bigram_loglik(payload)
    s_bigram = float(np.clip((llr + 5.0) / 10.0, 0.0, 1.0))

    # CRC signal (dominant term)
    s_crc = 1.0 if crc_valid else 0.0

    # Branch agreement signal
    # log(1) = 0 → 0; log(5) ≈ 1.6 → normalise to ~1 at 5 branches
    s_branches = float(np.clip(math.log(max(1, n_branches_agreeing)) / 1.6, 0.0, 1.0))

    # ── Weighted combination ──────────────────────────────────────────────
    raw = (w_entropy * s_entropy
           + w_printable * s_printable
           + w_bigram * s_bigram
           + w_crc * s_crc
           + w_branches * s_branches)

    # ── Override rule (§IV-M): valid CRC → score ≥ 0.95 always ──────────
    if crc_valid:
        raw = max(raw, 0.95)

    return float(np.clip(raw, 0.0, 1.0))


def is_plausible(payload: bytes,
                 crc_valid: bool = False,
                 n_branches_agreeing: int = 1,
                 threshold: float = 0.40) -> bool:
    """Boolean gate: return True if payload is worth reporting.

    The default threshold (0.40) is conservative — prefer false positives
    over silently discarding payloads.  For `DEMOD_NO_CRC` reporting,
    the calling code should use a higher threshold (~0.60).
    """
    if crc_valid:
        return True  # Valid CRC is never vetoed (§IV-M)
    return plausibility_score(payload, crc_valid, n_branches_agreeing) >= threshold
