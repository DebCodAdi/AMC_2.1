"""Constellation mapping and demapping with Gray coding.

Implements:
  - gray_map(mod_type, index): Gray-coded constellation symbol lookup (KAT-GRAY-1).
  - bits_to_symbols(bits, mod_type): Bitstream to complex symbol modulation.
  - symbols_to_bits(symbols, mod_type, hard=True): Symbol demapper (hard or soft LLR).

Spec reference: KAT-GRAY-1, DEF-017, IV-L (PS26147).
"""
from __future__ import annotations
import numpy as np


# Standard 2-bit Gray levels for PAM/QAM:
# 00 (0) -> -3, 01 (1) -> -1, 11 (3) -> +1, 10 (2) -> +3
_GRAY_LEVELS_4 = {0: -3.0, 1: -1.0, 3: +1.0, 2: +3.0}

# Standard 3-bit Gray levels for 64-QAM:
# 000(0)->-7, 001(1)->-5, 011(3)->-3, 010(2)->-1, 110(6)->+1, 111(7)->+3, 101(5)->+5, 100(4)->+7
_GRAY_LEVELS_8 = {0: -7.0, 1: -5.0, 3: -3.0, 2: -1.0, 6: 1.0, 7: 3.0, 5: 5.0, 4: 7.0}


def gray_map(mod_type: str, index: int) -> complex:
    """Return the complex constellation symbol for index with Gray mapping.

    Parameters
    ----------
    mod_type : str
        Modulation type: 'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'.
    index : int
        Symbol index in 0 .. M-1.

    Returns
    -------
    complex
        Normalized complex constellation symbol (unit average power).
    """
    mod = mod_type.upper().replace("-", "")

    if mod == "BPSK":
        # 0 -> +1, 1 -> -1 (or 0 -> -1, 1 -> +1)
        g = index ^ (index >> 1)
        return complex(-1.0 if (g & 1) else 1.0, 0.0)

    elif mod == "QPSK":
        # Gray encoded QPSK
        g = index ^ (index >> 1)
        g_I = (g >> 1) & 1
        g_Q = g & 1
        i_val = -1.0 if g_I else 1.0
        q_val = -1.0 if g_Q else 1.0
        return complex(i_val / np.sqrt(2.0), q_val / np.sqrt(2.0))

    elif mod == "8PSK":
        # 8-PSK Gray code on circle
        g = index ^ (index >> 1)
        theta = 2.0 * np.pi * g / 8.0
        return complex(np.cos(theta), np.sin(theta))

    elif mod in ("16QAM", "QAM16"):
        # 16-QAM: 4 bits, 2 bits for I, 2 bits for Q
        g = index ^ (index >> 1)
        g_I = (g >> 2) & 3
        g_Q = g & 3
        i_val = _GRAY_LEVELS_4[g_I] / np.sqrt(10.0)
        q_val = _GRAY_LEVELS_4[g_Q] / np.sqrt(10.0)
        return complex(i_val, q_val)

    elif mod in ("64QAM", "QAM64"):
        # 64-QAM: 6 bits, 3 bits for I, 3 bits for Q
        g = index ^ (index >> 1)
        g_I = (g >> 3) & 7
        g_Q = g & 7
        i_val = _GRAY_LEVELS_8[g_I] / np.sqrt(42.0)
        q_val = _GRAY_LEVELS_8[g_Q] / np.sqrt(42.0)
        return complex(i_val, q_val)

    else:
        raise ValueError(f"Unsupported modulation type: {mod_type}")


def get_constellation(mod_type: str) -> np.ndarray:
    """Return array of all M constellation symbols for given mod_type."""
    mod = mod_type.upper().replace("-", "")
    M_map = {"BPSK": 2, "QPSK": 4, "8PSK": 8, "16QAM": 16, "QAM16": 16, "64QAM": 64, "QAM64": 64}
    M = M_map.get(mod, 4)
    return np.array([gray_map(mod_type, i) for i in range(M)], dtype=np.complex128)


def bits_to_symbols(bits: np.ndarray, mod_type: str) -> np.ndarray:
    """Modulate binary bitstream into complex constellation symbols.

    Parameters
    ----------
    bits : 1D array of uint8 (0 or 1)
    mod_type : str ('BPSK', 'QPSK', '8PSK', '16QAM', '64QAM')

    Returns
    -------
    symbols : 1D complex64 array
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    mod = mod_type.upper().replace("-", "")
    bps_map = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "16QAM": 4, "QAM16": 4, "64QAM": 6, "QAM64": 6}
    bps = bps_map.get(mod, 2)

    # Pad with zeros if length not multiple of bps
    rem = len(bits) % bps
    if rem != 0:
        bits = np.pad(bits, (0, bps - rem))

    # Pack bits into integer indices
    reshaped = bits.reshape(-1, bps)
    # Convert binary to integer
    powers = 1 << np.arange(bps)[::-1]
    # Here bits represent the binary value before Gray mapping
    indices = reshaped.dot(powers)

    # Map to constellation
    constellation = get_constellation(mod_type)
    symbols = constellation[indices]
    return symbols.astype(np.complex64)


def symbols_to_bits(symbols: np.ndarray,
                    mod_type: str,
                    hard: bool = True,
                    noise_var: float = 0.1) -> np.ndarray:
    """Demap complex symbols to bits (hard decisions or soft LLRs).

    Parameters
    ----------
    symbols : 1D complex array
    mod_type : str
    hard : bool, default True
        If True, returns uint8 array of {0, 1}.
        If False, returns float32 array of log-likelihood ratios (LLRs) in the
        textbook convention ``L(b) = log( P(b=0|y) / P(b=1|y) )``, i.e.
        **positive LLR means the bit is more likely 0**.  This matches
        blindbits.py ``symbols_to_llr`` and the plan's Appendix C.
        The FEC decoders in ``signal9.fec`` use the opposite (soft-symbol)
        convention, so route LLRs through :func:`llr_to_soft_symbol` before
        handing them to ``viterbi_decode`` / ``ldpc_decode``.
    noise_var : float, default 0.1
        Estimated noise variance sigma^2 for soft LLR computation (DEF-017).

    Returns
    -------
    bits or llrs : 1D array
    """
    symbols = np.asarray(symbols, dtype=np.complex128).ravel()
    mod = mod_type.upper().replace("-", "")
    bps_map = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "16QAM": 4, "QAM16": 4, "64QAM": 6, "QAM64": 6}
    bps = bps_map.get(mod, 2)
    M = 1 << bps
    const = get_constellation(mod_type)

    if hard:
        # Minimum Euclidean distance
        # dists shape: (len(symbols), M)
        diffs = symbols[:, None] - const[None, :]
        dists = np.abs(diffs) ** 2
        nearest = np.argmin(dists, axis=1)

        # Unpack indices to bits
        # (len(symbols), bps)
        out_bits = np.zeros((len(symbols), bps), dtype=np.uint8)
        for b in range(bps):
            shift = bps - 1 - b
            out_bits[:, b] = (nearest >> shift) & 1
        return out_bits.ravel()
    else:
        # Soft LLR: LLR(b_i) = log(P(b_i=0)/P(b_i=1))
        # max-log approximation with noise variance scaling (DEF-017)
        sigma2 = max(noise_var, 1e-4)
        llrs = np.zeros((len(symbols), bps), dtype=np.float32)
        diffs = np.abs(symbols[:, None] - const[None, :]) ** 2  # |y - s|^2

        for b in range(bps):
            shift = bps - 1 - b
            mask0 = [((i >> shift) & 1) == 0 for i in range(M)]
            mask1 = [((i >> shift) & 1) == 1 for i in range(M)]

            min_dist0 = np.min(diffs[:, mask0], axis=1)
            min_dist1 = np.min(diffs[:, mask1], axis=1)
            # LLR = (min_dist1 - min_dist0) / sigma2
            llr_raw = (min_dist1 - min_dist0) / sigma2
            # Clip LLRs to [-8.0, 8.0] per DEF-017
            llrs[:, b] = np.clip(llr_raw, -8.0, 8.0).astype(np.float32)

        return llrs.ravel()


# ─────────────────────────────────────────────────────────────────────────────
# Soft-value convention bridge
# ─────────────────────────────────────────────────────────────────────────────

def llr_to_soft_symbol(llr: np.ndarray) -> np.ndarray:
    """Convert demapper LLRs into the soft-symbol convention the FEC decoders use.

    Two conventions are in play in this codebase and they are exact opposites:

    * :func:`symbols_to_bits` (``hard=False``) returns the textbook LLR
      ``log(P(b=0)/P(b=1))`` -- **positive means bit 0**.  This is also what
      ``blindbits.symbols_to_llr`` returns and what the plan's Appendix C
      specifies.
    * :func:`signal9.fec.conv.viterbi_decode` and
      :func:`signal9.fec.ldpc.ldpc_decode` score against ``2*bit - 1``, so for
      them **positive means bit 1**.  That is the soft-symbol convention used by
      ``commpy``'s ``decoding_type="unquantized"`` Viterbi, which is the
      differential oracle those decoders are validated against, so it is kept.

    Feeding one directly into the other silently decodes the bit-complement of
    the true information sequence: the decoder still finds a valid path through
    the trellis, no exception is raised, and every downstream CRC check simply
    fails.  Measured on a noiseless rate-1/2 K=7 stream, information-bit
    agreement was 0.0 without this conversion and 1.0 with it.

    The conversion is a negation, kept as a named function so the convention
    change is visible at the call site instead of being an unexplained minus
    sign that a later refactor can drop.
    """
    return -np.asarray(llr, dtype=np.float64)
