"""Low-Density Parity-Check (LDPC) codec (KAT-LDPC-1..2).

Implements:
  - IEEE 802.11n rate-1/2 N=648, K=324 parity check matrix H with lifting factor Z=27.
  - ldpc_encode(data, n=648, k=324): Systematic LDPC encoder.
  - ldpc_decode(llr, n=648, k=324, max_iter=50): Normalized Min-Sum (NMS) decoder (DEF-016).
  - ldpc_parity_check(cw, n=648, k=324): Syndrome computation.
  - Flags non-convergence instead of silently returning corrupted input (DEF-016).

Spec reference: KAT-LDPC-1..2, DEF-016 (PS26147).
"""
from __future__ import annotations
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# IEEE 802.11n Rate-1/2 Base Prototype Matrix (12 x 24)
# ─────────────────────────────────────────────────────────────────────────────

_H_BASE_R12 = [
    [0, -1, -1, -1, 0, 0, -1, -1, 0, -1, -1, 0, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [22, 0, -1, -1, 17, -1, 0, 0, 12, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [6, -1, 0, -1, 10, -1, -1, -1, 24, -1, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, -1, -1, 0, 20, -1, -1, -1, 25, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
    [23, -1, -1, -1, 3, -1, -1, -1, 0, -1, 9, 11, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1],
    [24, -1, 23, 1, 17, -1, 3, -1, 10, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
    [25, -1, -1, -1, 8, -1, -1, -1, 7, 18, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
    [13, 24, -1, -1, 0, -1, 8, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
    [7, 20, -1, 16, 22, 10, -1, -1, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1],
    [11, -1, -1, -1, 19, -1, -1, -1, 13, -1, 3, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
    [25, -1, 8, -1, 23, 18, -1, 14, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
    [3, -1, -1, -1, 16, -1, -1, 2, 25, 5, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]
]

def _build_h_matrix(n: int = 648, k: int = 324) -> np.ndarray:
    """Expand base matrix with lifting factor Z."""
    Z = n // 24  # 27 for n=648
    m_blocks = 12
    n_blocks = 24
    H = np.zeros((m_blocks * Z, n_blocks * Z), dtype=np.uint8)
    for i in range(m_blocks):
        for j in range(n_blocks):
            shift = _H_BASE_R12[i][j]
            if shift >= 0:
                shift = shift % Z
                for z in range(Z):
                    H[i * Z + z, j * Z + ((z + shift) % Z)] = 1
    return H

# Cached expanded H matrix and parity inverse
_H_648_324 = _build_h_matrix(648, 324)

def _invert_gf2(A: np.ndarray) -> np.ndarray:
    n = len(A)
    M = np.hstack([A.copy() % 2, np.eye(n, dtype=np.uint8)])
    for i in range(n):
        piv = np.where(M[i:, i] == 1)[0]
        if len(piv) == 0:
            raise ValueError("Singular matrix in GF(2)")
        p = piv[0] + i
        if p != i:
            M[[i, p]] = M[[p, i]]
        for j in range(n):
            if j != i and M[j, i] == 1:
                M[j] ^= M[i]
    return M[:, n:]

_H_PAR_INV_648_324 = _invert_gf2(_H_648_324[:, 324:])


# ─────────────────────────────────────────────────────────────────────────────
# Parity Check & Encode
# ─────────────────────────────────────────────────────────────────────────────

def ldpc_parity_check(codeword: np.ndarray, n: int = 648, k: int = 324) -> np.ndarray:
    """Compute syndrome s = H * c (mod 2).

    Parameters
    ----------
    codeword : 1D array of uint8 of length n

    Returns
    -------
    syndrome : 1D array of length (n - k)
    """
    cw = np.asarray(codeword, dtype=np.uint8).ravel()[:n]
    if len(cw) < n:
        cw = np.pad(cw, (0, n - len(cw)))
    return np.dot(_H_648_324, cw) % 2


def ldpc_encode(data: np.ndarray, n: int = 648, k: int = 324) -> np.ndarray:
    """Systematically encode information bits into an LDPC codeword.

    Parameters
    ----------
    data : 1D array of uint8 of length k (bits 0 or 1)

    Returns
    -------
    codeword : 1D uint8 array of length n
    """
    bits = np.asarray(data, dtype=np.uint8).ravel()
    if len(bits) < k:
        bits = np.pad(bits, (0, k - len(bits)))
    elif len(bits) > k:
        bits = bits[:k]

    # Systematic: info bits + parity bits
    H_info = _H_648_324[:, :k]
    rhs = np.dot(H_info, bits) % 2
    parity = np.dot(_H_PAR_INV_648_324, rhs) % 2
    return np.concatenate([bits, parity]).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Normalized Min-Sum Decoder (DEF-016)
# ─────────────────────────────────────────────────────────────────────────────

def ldpc_decode(llr: np.ndarray,
                n: int = 648,
                k: int = 324,
                max_iter: int = 50,
                nms_alpha: float = 0.75) -> tuple[np.ndarray, bool, int, int]:
    """Decode soft LLRs using Normalized Min-Sum (NMS) algorithm (DEF-016).

    Parameters
    ----------
    llr : 1D array of soft values (length n)
        Convention: value < 0 indicates bit 0, value > 0 indicates bit 1 --
        the soft-symbol convention, which is the opposite sign to the textbook
        LLRs returned by
        :func:`signal9.dsp.constellation.symbols_to_bits` (``hard=False``).
        Convert with
        :func:`signal9.dsp.constellation.llr_to_soft_symbol` before calling.
    n : int, default 648
    k : int, default 324
    max_iter : int, default 50
    nms_alpha : float, default 0.75
        Normalization factor for check-node updates (DEF-016).

    Returns
    -------
    (decoded_bits, converged, n_iter, syndrome_weight)
        decoded_bits : length k uint8 array
        converged : bool
        n_iter : int (0 if initial hard decisions had 0 syndrome)
        syndrome_weight : int (sum of syndrome bits)
    """
    llr_arr = np.asarray(llr, dtype=np.float64).ravel()[:n]
    if len(llr_arr) < n:
        llr_arr = np.pad(llr_arr, (0, n - len(llr_arr)))

    # Initial hard decisions: llr < 0 -> bit 0, llr > 0 -> bit 1
    c_hat = (llr_arr > 0).astype(np.uint8)
    synd = np.dot(_H_648_324, c_hat) % 2
    sw = int(np.sum(synd))
    if sw == 0:
        return c_hat[:k], True, 0, 0

    # Build adjacency lists for fast message passing
    M_checks, N_vars = _H_648_324.shape
    # Check connections
    check_nodes = [np.where(_H_648_324[c, :] == 1)[0] for c in range(M_checks)]
    # Variable connections
    var_nodes = [np.where(_H_648_324[:, v] == 1)[0] for v in range(N_vars)]

    # Messages: M_cv (check to var), L_vc (var to check)
    # Store in sparse representation
    # cv_msg[c, idx] corresponds to edge (c, check_nodes[c][idx])
    cv_msg = [np.zeros(len(check_nodes[c]), dtype=np.float64) for c in range(M_checks)]

    for it in range(1, max_iter + 1):
        # 1. Variable-to-check update: L_vc = L_v + sum_{c' != c} M_c'v
        # Precompute total LLR for each variable
        L_total = llr_arr.copy()
        for c in range(M_checks):
            v_indices = check_nodes[c]
            for idx, v in enumerate(v_indices):
                L_total[v] += cv_msg[c][idx]

        # Check early termination
        c_hat = (L_total > 0).astype(np.uint8)
        synd = np.dot(_H_648_324, c_hat) % 2
        sw = int(np.sum(synd))
        if sw == 0:
            return c_hat[:k], True, it, 0

        # 2. Check-to-variable update (Normalized Min-Sum)
        for c in range(M_checks):
            v_indices = check_nodes[c]
            deg = len(v_indices)
            # Incoming variable LLRs for this check node: L_total[v] - cv_msg[c][idx]
            v_msgs = np.array([L_total[v] - cv_msg[c][idx] for idx, v in enumerate(v_indices)])
            signs = np.sign(v_msgs)
            signs[signs == 0] = 1.0
            abs_msgs = np.abs(v_msgs)

            total_sign = np.prod(signs)
            for idx in range(deg):
                # Min over all other edges
                # mask out current index
                other_mins = np.min(np.delete(abs_msgs, idx))
                edge_sign = total_sign * signs[idx]
                cv_msg[c][idx] = nms_alpha * edge_sign * other_mins

    # Failed to converge in max_iter
    return c_hat[:k], False, max_iter, sw
