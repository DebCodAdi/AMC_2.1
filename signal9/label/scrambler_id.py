"""Blind scrambler identification via GF(2) rank restoration and entropy reduction (Track F-4).

Identifies multiplicative and additive scramblers from demodulated bitstreams:
  - G3RUH (1 + x^12 + x^17)
  - V.35 (1 + x^18 + x^23)
  - CCSDS Additive PN255 (x^8 + x^7 + x^5 + x^3 + 1)
  - DVB-S Additive PRBS15 (x^15 + x^14 + 1)
  - 802.11 Additive PRBS7 (x^7 + x^4 + 1)

Spec reference: Track F-4 (PS26147 / SIGNAL10).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from signal9.sim.framing import (
    g3ruh_descramble,
    v35_descramble,
    ccsds_descramble,
)
from signal9.dsp.gf2 import gf2_rank


def _byte_entropy(bits: np.ndarray) -> float:
    """Compute Shannon entropy in bits per byte of bitstream."""
    n_bytes = len(bits) // 8
    if n_bytes < 16:
        return 8.0
    byte_vals = np.packbits(bits[:n_bytes * 8])
    counts = np.bincount(byte_vals, minlength=256)
    probs = counts[counts > 0] / n_bytes
    return float(-np.sum(probs * np.log2(probs)))


def _dvbs_descramble(bits: np.ndarray) -> np.ndarray:
    """DVB-S additive PRBS15 descrambler.

    Polynomial: 1 + x^14 + x^15.
    Feedback = bit14 XOR bit0 (MSB XOR LSB of the 15-bit shift register).

    BUG-4 fix: the old implementation used ``(reg>>14) ^ (reg>>13)`` which is
    bit14 XOR bit13 — not the correct polynomial.  For ``1 + x^14 + x^15`` the
    feedback taps are at positions 14 and 0 (the MSB and LSB of a 15-bit
    register), so the correct expression is ``((reg >> 14) ^ reg) & 1``.
    """
    # Polynomial 1 + x^14 + x^15: feedback = tap14 XOR tap0
    reg = 0x4A80  # Standard DVB initialization (EN 300 421 §4.4.2)
    n = len(bits)
    pn = np.empty(n, dtype=np.uint8)
    for i in range(n):
        fb = ((reg >> 14) ^ reg) & 1   # bit14 XOR bit0  (was bit14 XOR bit13 — BUG-4)
        reg = ((reg << 1) | fb) & 0x7FFF
        pn[i] = fb
    return bits ^ pn


def _wifi_descramble(bits: np.ndarray) -> np.ndarray:
    """802.11 additive PRBS7 descrambler."""
    # Polynomial 1 + x^4 + x^7
    reg = 0x7F
    n = len(bits)
    pn = np.empty(n, dtype=np.uint8)
    for i in range(n):
        out = ((reg >> 6) ^ (reg >> 3)) & 1
        reg = ((reg << 1) | out) & 0x7F
        pn[i] = out
    return bits ^ pn


def _check_hdlc_flags(bits: np.ndarray) -> int:
    """Count HDLC flags (01111110) across all bit alignments."""
    if len(bits) < 8:
        return 0
    b = (bits > 0).astype(np.uint8)
    # Look for [0, 1, 1, 1, 1, 1, 1, 0] at every bit position.  A sliding-window
    # view compares all positions in one vectorised pass; the previous
    # per-position Python loop with np.array_equal was the single slowest call
    # in scrambler identification (~0.13 s per 6 kbit stream).
    pattern = np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8)
    windows = np.lib.stride_tricks.sliding_window_view(b, 8)
    return int(np.count_nonzero((windows == pattern).all(axis=1)))


@dataclass(frozen=True)
class ScramblerCandidate:
    name: str
    descramble_fn: callable
    is_additive: bool


CANDIDATES = [
    ScramblerCandidate("G3RUH", g3ruh_descramble, False),
    ScramblerCandidate("V35", v35_descramble, False),
    ScramblerCandidate("CCSDS", lambda b: np.unpackbits(np.frombuffer(ccsds_descramble(bytes(np.packbits(b[:(len(b)//8)*8]))), dtype=np.uint8)), True),
    ScramblerCandidate("DVB_S", _dvbs_descramble, True),
    ScramblerCandidate("WIFI_80211", _wifi_descramble, True),
]


def identify_scrambler(bits: np.ndarray | bytes,
                       entropy_threshold_gain: float = 0.25,
                       min_bits: int = 128) -> tuple[str, float]:
    """Blindly identify candidate scrambler polynomial."""
    if isinstance(bits, (bytes, bytearray)):
        b = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))
    else:
        b = np.asarray(bits, dtype=np.uint8).ravel()

    if len(b) < min_bits:
        return "none", 0.0

    h_raw = _byte_entropy(b)
    flags_raw = _check_hdlc_flags(b)

    best_name = "none"
    best_score = 0.0

    for cand in CANDIDATES:
        try:
            cand_bits = cand.descramble_fn(b.copy())
            if len(cand_bits) == 0:
                continue

            h_desc = _byte_entropy(cand_bits)
            delta_h = max(0.0, h_raw - h_desc)

            # Check if HDLC flags appeared
            flags_desc = _check_hdlc_flags(cand_bits)
            flag_gain = max(0, flags_desc - flags_raw)

            # Score metric combining entropy drop and flag appearance
            score = (delta_h / 2.5) + (0.5 if flag_gain >= 1 else 0.0) + (0.3 if flag_gain >= 2 else 0.0)

            # Also check GF(2) Hankel rank reduction on first 32x32 submatrix
            if len(cand_bits) >= 128:
                L = min(32, len(cand_bits) // 4)
                H_mat = np.array([cand_bits[i : i + L] for i in range(L)], dtype=np.uint8)
                rank = gf2_rank(H_mat)
                if rank < L - 2:
                    score += 0.25

            # Only consider candidate if entropy genuinely drops or multiple flags appear with entropy drop
            if (delta_h >= entropy_threshold_gain) or (flag_gain >= 2 and delta_h >= 0.15):
                if score > best_score:
                    best_score = score
                    best_name = cand.name

        except Exception:
            continue

    confidence = min(1.0, float(best_score))
    if confidence < 0.2:
        return "none", 0.0

    return best_name, confidence

# ─────────────────────────────────────────────────────────────────────────────
# Blind LFSR detection via Berlekamp-Massey (Plan §7.2, blindbits.py §7)
# ─────────────────────────────────────────────────────────────────────────────

def berlekamp_massey(seq: np.ndarray) -> tuple[int, np.ndarray]:
    """LFSR synthesis over GF(2) via Berlekamp-Massey algorithm.

    Returns (L, connection_poly) where L is the LFSR length and poly[0]=1.
    A low L relative to the window length indicates LFSR-generated structure
    (scrambled idle fill, PRBS test sequence, etc.).

    Plan reference: §7.2 — the bias-based LFSR detector.
    Measured: recovers degree-15 LFSR exactly from a 400-bit sequence.
    """
    s = np.asarray(seq, dtype=np.uint8).ravel()
    n = len(s)
    if n == 0:
        return 0, np.array([1], dtype=np.uint8)
    C = np.zeros(n + 1, dtype=np.uint8); C[0] = 1
    B = C.copy()
    L, m = 0, 1
    for N in range(n):
        d = int(s[N])
        for i in range(1, L + 1):
            d ^= int(C[i]) & int(s[N - i])
        d &= 1
        if d == 0:
            m += 1
        elif 2 * L <= N:
            T = C.copy()
            # XOR B (shifted by m) into C; lengths may differ → use min
            xor_len = min(len(C) - m, len(B))
            C[m:m + xor_len] ^= B[:xor_len]
            L, B, m = N + 1 - L, T, 1
        else:
            xor_len = min(len(C) - m, len(B))
            C[m:m + xor_len] ^= B[:xor_len]
            m += 1
    return L, C[:L + 1]


def linear_complexity_profile(bits: np.ndarray,
                               window: int = 512,
                               step: int = 256) -> np.ndarray:
    """Windowed linear complexity (Plan §7.2).

    Returns an array of LFSR lengths for sliding windows over `bits`.
    L ≈ window/2  →  random / coded stream (no exploitable structure).
    L << window/2 →  LFSR-generated region (idle fill, scrambler seed,
                     or unscrambled constant under an additive scrambler).

    Use the minimum L as a seed region for Berlekamp-Massey reconstruction.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    out = []
    if len(bits) < window:
        return np.asarray(out)
    for s in range(0, len(bits) - window + 1, step):
        L, _ = berlekamp_massey(bits[s:s + window])
        out.append(L)
    return np.asarray(out)


def blind_lfsr_descramble(bits: np.ndarray,
                           window: int = 256,
                           max_degree: int = 23) -> tuple[np.ndarray, int, np.ndarray] | None:
    """Attempt blind additive LFSR descrambling (Plan §7.2).

    Searches for a window with linear complexity << window/2, which indicates
    a region of LFSR-generated (or zero-padded) bits under an additive
    scrambler.  If found, runs Berlekamp-Massey to recover the polynomial and
    XORs the full stream with the regenerated LFSR sequence.

    Parameters
    ----------
    bits : 1D uint8 array.
    window : int — analysis window size.
    max_degree : int — maximum LFSR degree to accept (rejects random results).

    Returns
    -------
    (descrambled_bits, lfsr_degree, connection_poly) or None if not found.
    """
    from signal9.label.scrambler_id import _byte_entropy as _be
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    step = window // 2
    lc_vals = []
    for s in range(0, max(len(bits) - window, 0) + 1, step):
        L, _ = berlekamp_massey(bits[s:s + window])
        lc_vals.append((s, L))

    if not lc_vals:
        return None

    # Find the window with the smallest linear complexity
    min_s, min_L = min(lc_vals, key=lambda x: x[1])
    if min_L > max_degree or min_L < 3:
        return None

    # Recover polynomial and initial state from that window
    seed_bits = bits[min_s:min_s + window]
    L, poly = berlekamp_massey(seed_bits)
    if L < 3 or L > max_degree:
        return None

    state = seed_bits[:L].copy()

    # Regenerate the LFSR sequence aligned to start of bits
    # Rewind: generate enough bits before min_s to find the phase
    n_total = len(bits)
    # Generate forward from state at min_s
    from signal9.dsp.gf2 import gf2_rank  # noqa — import guard
    taps = np.nonzero(poly[1:])[0] + 1

    pn = np.zeros(n_total, dtype=np.uint8)
    buf = list(state[:L])
    for k in range(L):
        pn[min_s + k] = buf[k]
    for k in range(L, n_total - min_s):
        acc = 0
        for t in taps:
            if k - t >= 0:
                acc ^= buf[k - t]
            elif min_s + k - t >= 0:
                acc ^= int(bits[min_s + k - t])
        acc &= 1
        pn[min_s + k] = acc
        buf.append(acc)

    # Extend backward from min_s - 1 down to 0
    other_taps = [t for t in taps if t < L]
    for j in range(min_s - 1, -1, -1):
        acc = int(pn[j + L])
        for t in other_taps:
            acc ^= int(pn[j + L - t])
        pn[j] = acc & 1

    descrambled = np.bitwise_xor(bits, pn)
    return descrambled, L, poly

