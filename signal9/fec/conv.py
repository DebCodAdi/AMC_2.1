"""Convolutional codec and Viterbi decoder.

Implements:
  - conv_encode(bits, K, gens): Rate-1/N convolutional encoder.
  - viterbi_decode(llr, K, gens, tb_depth=96): Soft-decision Viterbi decoder.
  - DEF-018: Metric normalisation (subtract min path metric) to prevent float overflow.
  - Puncturing / depuncturing for rate 2/3, 3/4, 5/6, 7/8 (DEF-013).
    Patterns are corrected per BUG-2 (period alignment) and BUG-5 (7/8 added):
    each pattern has even length so g1/g2 pairs stay aligned with period boundaries.

Spec reference: KAT-CONV-1..4, DEF-018, DEF-013 (PS26147).
"""
from __future__ import annotations
import numpy as np


def conv_encode(bits: np.ndarray,
                K: int = 7,
                gens: list[int] | tuple[int, ...] | None = None,
                generators: list[int] | tuple[int, ...] | None = None) -> np.ndarray:
    if gens is None:
        gens = generators if generators is not None else [0o171, 0o133]

    """Encode binary bitstream using rate 1/len(gens) convolutional code.

    Parameters
    ----------
    bits : 1D array of uint8 (0 or 1)
    K : int
        Constraint length (e.g. 7).
    gens : sequence of int
        Generator polynomials in octal (e.g. [0o171, 0o133]).

    Returns
    -------
    encoded : 1D uint8 array
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    n_gens = len(gens)
    gen_masks = [int(g) for g in gens]
    state = 0
    mask = (1 << K) - 1

    out = np.zeros(len(bits) * n_gens, dtype=np.uint8)
    for i, b in enumerate(bits):
        state = ((state << 1) | int(b)) & mask
        for g_idx, g in enumerate(gen_masks):
            # Parity of selected taps
            out[i * n_gens + g_idx] = bin(state & g).count("1") % 2

    return out


try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

if _HAS_NUMBA:
    # cache=True persists the compiled kernel to __pycache__.  Without it every
    # new process -- each CLI run, each GUI launch -- paid ~5 s of JIT
    # compilation on its first decode, which dominated short captures.
    @numba.njit(fastmath=True, cache=True)
    def _viterbi_numba_core(llr: np.ndarray,
                            n_gens: int,
                            n_steps: int,
                            n_states: int,
                            next_state: np.ndarray,
                            branch_sym: np.ndarray,
                            tb_prev: np.ndarray,
                            tb_bit: np.ndarray) -> np.ndarray:
        pm = np.full(n_states, 1e9, dtype=np.float64)
        pm[0] = 0.0
        for t in range(n_steps):
            r = llr[t * n_gens : (t + 1) * n_gens]
            new_pm = np.full(n_states, 1e9, dtype=np.float64)
            for s in range(n_states):
                if pm[s] >= 1e8:
                    continue
                for b in range(2):
                    ns = next_state[s, b]
                    metric = pm[s]
                    for g in range(n_gens):
                        diff = r[g] - branch_sym[s, b, g]
                        metric += diff * diff
                    if metric < new_pm[ns]:
                        new_pm[ns] = metric
                        tb_prev[t, ns] = s
                        tb_bit[t, ns] = b
            min_metric = np.min(new_pm)
            pm = new_pm - min_metric

        best_state = int(np.argmin(pm))
        decoded = np.zeros(n_steps, dtype=np.uint8)
        curr_state = best_state
        for t in range(n_steps - 1, -1, -1):
            decoded[t] = tb_bit[t, curr_state]
            curr_state = tb_prev[t, curr_state]
        return decoded


def viterbi_decode(llr: np.ndarray,
                   K: int,
                   gens: list[int] | tuple[int, ...],
                   tb_depth: int = 96) -> np.ndarray:
    """Soft-decision Viterbi decoder with DEF-018 path metric normalisation.

    Soft-value convention: branch metrics are squared distances to ``2*bit - 1``,
    so a **positive** input value votes for **bit 1** (the soft-symbol convention
    of ``commpy``'s ``decoding_type="unquantized"`` decoder, against which this
    implementation is verified bit-exactly in ``tests/L4_differential``).

    This is the opposite sign to the textbook LLRs produced by
    :func:`signal9.dsp.constellation.symbols_to_bits` with ``hard=False``.
    Convert with :func:`signal9.dsp.constellation.llr_to_soft_symbol` first --
    passing raw demapper LLRs here decodes the bit-complement silently.
    """
    llr = np.asarray(llr, dtype=np.float64).ravel()
    n_gens = len(gens)
    n_steps = len(llr) // n_gens
    if n_steps == 0:
        return np.zeros(0, dtype=np.uint8)

    n_states = 1 << (K - 1)
    state_mask = n_states - 1
    gen_masks = [int(g) for g in gens]

    # Precompute branch outputs for (state, input_bit)
    branch_sym = np.zeros((n_states, 2, n_gens), dtype=np.float64)
    next_state = np.zeros((n_states, 2), dtype=np.int32)
    for s in range(n_states):
        for b in (0, 1):
            reg = ((s << 1) | b)
            next_state[s, b] = reg & state_mask
            for g_idx, g in enumerate(gen_masks):
                bit = bin(reg & g).count("1") % 2
                branch_sym[s, b, g_idx] = 2.0 * bit - 1.0

    # Traceback table
    tb_prev = np.zeros((n_steps, n_states), dtype=np.int32)
    tb_bit = np.zeros((n_steps, n_states), dtype=np.uint8)

    if _HAS_NUMBA:
        return _viterbi_numba_core(
            llr, n_gens, n_steps, n_states, next_state, branch_sym, tb_prev, tb_bit
        )

    # Pure Python fallback
    pm = np.full(n_states, 1e9, dtype=np.float64)
    pm[0] = 0.0

    for t in range(n_steps):
        r = llr[t * n_gens : (t + 1) * n_gens]
        new_pm = np.full(n_states, 1e9, dtype=np.float64)

        for s in range(n_states):
            if pm[s] >= 1e8:
                continue
            for b in (0, 1):
                ns = next_state[s, b]
                expected = branch_sym[s, b]
                metric = pm[s] + float(np.sum((r - expected) ** 2))
                if metric < new_pm[ns]:
                    new_pm[ns] = metric
                    tb_prev[t, ns] = s
                    tb_bit[t, ns] = b

        min_metric = float(np.min(new_pm))
        pm = new_pm - min_metric

    best_state = int(np.argmin(pm))
    decoded = np.zeros(n_steps, dtype=np.uint8)
    curr_state = best_state
    for t in range(n_steps - 1, -1, -1):
        decoded[t] = tb_bit[t, curr_state]
        curr_state = tb_prev[t, curr_state]

    return decoded


# ─────────────────────────────────────────────────────────────────────────────
# Puncturing and Depuncturing with Phase Sweeping (DEF-013, EC-FEC-010..013)
# ─────────────────────────────────────────────────────────────────────────────

# Standard puncture patterns: 1 = transmitted, 0 = punctured (erasure)
# Each pattern is applied to the interleaved (g1,g2,g1,g2,...) coded stream.
# Pattern length must be EVEN so g1/g2 pairs align with period boundaries.
#
# Rate derivation: for a rate-1/2 mother code with 2 coded bits per input bit,
#   rate = (input_bits) / (transmitted_bits)
#              = (period/2) / sum(pattern == 1)
#
#   "2/3": period=4,  sum=3  → rate (4/2)/3 = 2/3  ✓   (was [1,1,0] len=3  ✗ BUG-2)
#   "3/4": period=6,  sum=4  → rate (6/2)/4 = 3/4  ✓   (was [1,1,0,1,1,0] → 4/6≡2/3 ✗ BUG-2)
#   "5/6": period=10, sum=6  → rate (10/2)/6 = 5/6 ✓   (was wrong pattern BUG-2)
#   "7/8": period=14, sum=8  → rate (14/2)/8 = 7/8 ✓   (was missing BUG-5)
_PUNCTURE_PATTERNS = {
    "2/3": np.array([1, 1, 1, 0], dtype=np.uint8),
    "3/4": np.array([1, 1, 0, 1, 1, 0], dtype=np.uint8),
    "5/6": np.array([1, 1, 0, 1, 1, 0, 0, 1, 1, 0], dtype=np.uint8),
    "7/8": np.array([1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8),
}


def puncture(encoded_bits: np.ndarray, rate: str = "2/3") -> np.ndarray:
    """Apply puncturing pattern to rate-1/2 convolutional stream."""
    pat = _PUNCTURE_PATTERNS.get(rate)
    if pat is None:
        return encoded_bits
    p_len = len(pat)
    n_blocks = len(encoded_bits) // p_len
    if n_blocks == 0:
        return encoded_bits

    usable = n_blocks * p_len
    reshaped = encoded_bits[:usable].reshape(n_blocks, p_len)
    keep_mask = (pat == 1)
    punctured = reshaped[:, keep_mask].ravel()
    return punctured.astype(np.uint8)


def depuncture(punctured_llrs: np.ndarray,
               rate: str = "2/3",
               phase: int = 0) -> np.ndarray:
    """Insert erasures (LLR=0.0) at punctured positions aligned with phase (DEF-013).

    Parameters
    ----------
    punctured_llrs : 1D array of soft values
    rate : str ('2/3', '3/4', '5/6')
    phase : int in [0, n_phases - 1] (puncture alignment phase)

    Returns
    -------
    1D array of LLRs expanded to rate-1/2 format with 0.0 at erasures
    """
    pat = _PUNCTURE_PATTERNS.get(rate)
    if pat is None:
        return punctured_llrs

    p_len = len(pat)
    n_transmitted_per_period = int(np.sum(pat == 1))

    # Roll pattern by phase
    pat_phased = np.roll(pat, -phase)
    keep_indices = np.where(pat_phased == 1)[0]

    n_blocks = len(punctured_llrs) // n_transmitted_per_period
    if n_blocks == 0:
        return punctured_llrs

    out = np.zeros(n_blocks * p_len, dtype=np.float64)
    usable_in = n_blocks * n_transmitted_per_period
    in_blocks = punctured_llrs[:usable_in].reshape(n_blocks, n_transmitted_per_period)
    out_blocks = out.reshape(n_blocks, p_len)

    out_blocks[:, keep_indices] = in_blocks
    return out


def invert_ccsds_g2(llrs: np.ndarray) -> np.ndarray:
    """Invert the second generator output (G2) for CCSDS convolutional codes (EC-FEC-006)."""
    out = np.asarray(llrs, dtype=np.float64).copy()
    # Invert odd-indexed LLRs (G2 branch)
    out[1::2] = -out[1::2]
    return out

