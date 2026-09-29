"""Differential demapping and cycle-slip mitigation (Track H / IV-L).

Implements:
  - IV-L.1: Track H differential demapping (rotation-invariant by construction).
    Encodes phase difference Δφ between consecutive symbols: Δφ = angle(s[n] * conj(s[n-1])).
    A cycle slip destroys exactly 1 symbol, not the rest of the frame.
  - IV-L.2: Segmented rotation search for detected cycle slip locations.
    Splits symbol stream at detected slip indices and searches rotations independently
    up to 4 segments (256 combinations).
  - D2 bit normalisation helpers (Plan §4 / blindbits.py §4):
    - differential_decode(): XOR successive bits (removes differential encoding).
    - nrzi_decode(): NRZ-I → NRZ; transition = 0 in most standards.
    - manchester_decode(): IEEE 802.3 biphase, returns (bits, valid_fraction).

Spec reference: IV-L, DEF-003, DEF-024 (PS26147).
"""
from __future__ import annotations
import itertools
import numpy as np


def differential_demodulate(symbols: np.ndarray,
                            mod_type: str = "QPSK") -> np.ndarray:
    """Differential demapping: decode phase difference Δφ between consecutive symbols.

    Parameters
    ----------
    symbols : 1D complex array of symbols (e.g. at 1 SPS after timing recovery)
    mod_type : str ('BPSK', 'QPSK', '8PSK', '16QAM')

    Returns
    -------
    diff_symbols : 1D complex array of length len(symbols) - 1
        Rotated to baseband constellation based on phase transition.
    """
    symbols = np.asarray(symbols, dtype=np.complex64).ravel()
    if len(symbols) < 2:
        return np.zeros(0, dtype=np.complex64)

    # Phase difference between adjacent symbols: z[n] = s[n] * conj(s[n-1])
    z = symbols[1:] * np.conj(symbols[:-1])

    mod = mod_type.upper().replace("-", "")
    if mod == "BPSK":
        # Phase difference 0 -> +1, pi -> -1
        # Real part of z is positive for same phase, negative for phase flip
        return np.sign(z.real).astype(np.complex64)

    elif mod == "QPSK":
        # DQPSK: nominal transitions are {0, pi/2, pi, 3pi/2} (or pi/4 DQPSK)
        # Normalize magnitude to unit circle
        mag = np.abs(z) + 1e-12
        z_norm = z / mag
        # Rotate by pi/4 so quadrants align with (+-1 +- 1j)/sqrt(2)
        # Or standard QPSK: snap phase to nearest multiple of pi/2
        phase = np.angle(z_norm)
        quad = np.round(phase / (np.pi / 2.0))
        diff_sym = np.exp(1j * quad * (np.pi / 2.0)) / np.sqrt(2.0)
        return diff_sym.astype(np.complex64)

    elif mod == "8PSK":
        mag = np.abs(z) + 1e-12
        z_norm = z / mag
        phase = np.angle(z_norm)
        octant = np.round(phase / (np.pi / 4.0))
        diff_sym = np.exp(1j * octant * (np.pi / 4.0))
        return diff_sym.astype(np.complex64)

    else:
        # Default: normalize z
        mag = np.abs(z) + 1e-12
        return (z / mag).astype(np.complex64)


def segmented_rotation_search(symbols: np.ndarray,
                              slip_indices: list[int] | np.ndarray,
                              eval_fn,
                              max_segments: int = 4) -> tuple[np.ndarray, float, tuple[int, ...]]:
    """Segmented rotation search across detected cycle slip locations.

    Parameters
    ----------
    symbols : 1D complex array
    slip_indices : list of integer sample indices where slips occurred
    eval_fn : callable(candidate_symbols) -> float score (higher is better)
    max_segments : int, default 4 (caps complexity at 4^4 = 256 combinations)

    Returns
    -------
    (best_symbols, best_score, best_rotations)
    """
    symbols = np.asarray(symbols, dtype=np.complex64)
    N = len(symbols)
    if N == 0:
        return symbols, -float("inf"), ()

    # Filter slip indices
    valid_slips = sorted([int(s) for s in slip_indices if 0 < s < N])
    if len(valid_slips) == 0:
        # Just evaluate 4 standard global rotations {0, 90, 180, 270}
        rot_factors = [1.0 + 0j, 0.0 + 1j, -1.0 + 0j, 0.0 - 1j]
        best_s = symbols
        best_score = -float("inf")
        best_rot = (0,)
        for r_idx, r in enumerate(rot_factors):
            candidate = symbols * r
            score = eval_fn(candidate)
            if score > best_score:
                best_score = score
                best_s = candidate
                best_rot = (r_idx,)
        return best_s, best_score, best_rot

    # Cap number of segments at max_segments
    if len(valid_slips) >= max_segments:
        # Take the most significant slips (e.g. evenly spaced or largest phase hop)
        step = len(valid_slips) / (max_segments - 1)
        valid_slips = [valid_slips[int(i * step)] for i in range(max_segments - 1)]

    # Define segment boundaries: [0, s1, s2, ..., N]
    boundaries = [0] + valid_slips + [N]
    n_segs = len(boundaries) - 1

    rot_factors = [1.0 + 0j, 0.0 + 1j, -1.0 + 0j, 0.0 - 1j]
    best_candidate = symbols.copy()
    best_score = -float("inf")
    best_rots: tuple[int, ...] = tuple([0] * n_segs)

    # Search combinations of rotations per segment
    for rots in itertools.product(range(4), repeat=n_segs):
        candidate = np.empty_like(symbols)
        for seg_idx in range(n_segs):
            start = boundaries[seg_idx]
            end = boundaries[seg_idx + 1]
            r = rot_factors[rots[seg_idx]]
            candidate[start:end] = symbols[start:end] * r

        score = eval_fn(candidate)
        if score > best_score:
            best_score = score
            best_candidate = candidate
            best_rots = rots

    return best_candidate, best_score, best_rots


# ─────────────────────────────────────────────────────────────────────────────
# D2 Bit-level normalisation helpers (Plan §4, blindbits.py §4)
# ─────────────────────────────────────────────────────────────────────────────

def differential_decode(bits: np.ndarray) -> np.ndarray:
    """XOR successive hard bits: out[n] = bits[n] XOR bits[n-1].

    Removes differential encoding applied at the transmitter (e.g. DBPSK/DQPSK
    after the Costas loop has resolved the carrier phase).  Output length is
    len(bits) - 1.

    Plan §4 / blindbits.py §4: ``differential_decode(bits) = bits[1:] ^ bits[:-1]``
    """
    b = np.asarray(bits, dtype=np.uint8).ravel()
    if len(b) < 2:
        return np.zeros(0, dtype=np.uint8)
    return np.bitwise_xor(b[1:], b[:-1])


def nrzi_decode(bits: np.ndarray, invert: bool = False) -> np.ndarray:
    """NRZ-I to NRZ conversion.

    In NRZ-I (Non-Return-to-Zero Inverted) a *transition* encodes a logical 0
    (most standards: USB, RS-232 alt mode, MIL-STD-1553).  A non-transition
    encodes a logical 1.  ``invert=True`` swaps the convention for systems where
    a transition encodes logical 1.

    Parameters
    ----------
    bits : 1D uint8 array of NRZ-I hard bits.
    invert : bool
        False (default): transition → 0, no transition → 1.
        True: transition → 1, no transition → 0.

    Returns
    -------
    decoded : uint8 array, length len(bits) - 1.

    Plan §4 / blindbits.py §4:
        ``d = bits[1:] ^ bits[:-1]; return d ^ 1 if not invert else d``
    """
    b = np.asarray(bits, dtype=np.uint8).ravel()
    if len(b) < 2:
        return np.zeros(0, dtype=np.uint8)
    d = np.bitwise_xor(b[1:], b[:-1])
    return (d ^ 1).astype(np.uint8) if not invert else d.astype(np.uint8)


def manchester_decode(bits: np.ndarray, ieee: bool = True) -> tuple[np.ndarray, float]:
    """Biphase Manchester decoding.

    Pairs consecutive bits: in IEEE 802.3 convention (``ieee=True``) each
    (0,1) cell encodes a logical 1 and (1,0) encodes a logical 0.  In the
    Thomas/GE convention (``ieee=False``) the polarity is reversed.

    Invalid cells (both bits equal) are detected and excluded from the valid
    fraction count.

    Parameters
    ----------
    bits : 1D uint8 array.  Must be even length; any trailing odd bit is ignored.
    ieee : bool
        True  (default) → IEEE 802.3: cell bit1 = data bit.
        False           → Thomas/GE:  cell bit0 = data bit.

    Returns
    -------
    (decoded_bits, valid_fraction)
        decoded_bits : uint8 array of length len(bits) // 2.
        valid_fraction : fraction of cells that were valid (opposite polarity).
            1.0 = all cells valid; < 0.9 typically indicates a wrong clock phase
            or wrong encoding.

    Plan §4 / blindbits.py §4:
        ``b = bits[:len(bits)//2*2].reshape(-1, 2)``
        ``valid = (b[:,0] != b[:,1])``
        ``out = b[:,1] if ieee else b[:,0]``
    """
    b = np.asarray(bits, dtype=np.uint8).ravel()
    n_pairs = len(b) // 2
    if n_pairs == 0:
        return np.zeros(0, dtype=np.uint8), 0.0
    pairs = b[: n_pairs * 2].reshape(n_pairs, 2)
    valid = (pairs[:, 0] != pairs[:, 1])
    out = pairs[:, 1] if ieee else pairs[:, 0]
    return out.astype(np.uint8), float(valid.mean())
