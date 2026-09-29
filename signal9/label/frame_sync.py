"""Algebraic frame synchronisation via soft-syndrome LLRs (IV-I).

Replaces blind brute-force bit-slip sweeping (DEF-007) with algebraic
syndrome-based frame synchronisation (Imad, Sicot & Houcke, 2009).

For each candidate offset τ:
  Computes the soft syndrome log-likelihood:
    LLR(s_i) = 2 · atanh( ∏_{j} tanh( LLR(y_{τ+j}) / 2 ) )
  The true frame boundary maximizes the likelihood that all syndromes are zero.

Spec reference: IV-I, DEF-007 (PS26147).
"""
from __future__ import annotations
import numpy as np


def find_frame_boundary_syndrome(llrs: np.ndarray,
                                  H: np.ndarray,
                                  max_search: int = 128) -> tuple[int, float]:
    """Find the frame start offset τ using soft syndrome likelihood.

    Parameters
    ----------
    llrs : 1D array of channel LLRs
    H : 2D parity-check matrix (M x N) over GF(2)
    max_search : int, maximum offset to test

    Returns
    -------
    (best_offset, confidence_score)
        best_offset : integer index in [0, max_search - 1]
        confidence_score : log-likelihood ratio advantage of best over median
    """
    llrs = np.asarray(llrs, dtype=np.float64).ravel()
    M, N = H.shape
    search_range = min(max_search, len(llrs) - N)
    if search_range <= 0:
        return 0, 0.0

    # Build adjacency list for check equations
    check_nodes = [np.where(H[i, :] == 1)[0] for i in range(M)]

    scores = np.zeros(search_range, dtype=np.float64)

    for tau in range(search_range):
        window = llrs[tau : tau + N]
        # tanh(LLR / 2)
        tanh_vals = np.tanh(np.clip(window / 2.0, -10.0, 10.0))
        total_ll = 0.0

        for edges in check_nodes:
            if len(edges) == 0:
                continue
            # Product of tanh values for check equation
            prod_tanh = float(np.prod(tanh_vals[edges]))
            # Probability that syndrome is 0: P(s=0) = (1 + prod_tanh) / 2
            p_zero = max(1e-12, 0.5 * (1.0 + prod_tanh))
            total_ll += np.log(p_zero)

        scores[tau] = total_ll

    best_tau = int(np.argmax(scores))
    median_score = float(np.median(scores))
    confidence = float(scores[best_tau] - median_score)

    return best_tau, confidence


def find_frame_boundary_hard(bits: np.ndarray,
                             H: np.ndarray,
                             max_search: int = 128) -> tuple[int, int]:
    """Find frame start offset using hard syndrome weight (number of zero syndromes)."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    M, N = H.shape
    search_range = min(max_search, len(bits) - N)
    if search_range <= 0:
        return 0, 0

    best_tau = 0
    min_synd_weight = M + 1

    for tau in range(search_range):
        block = bits[tau : tau + N]
        synd = np.dot(H, block) % 2
        weight = int(np.sum(synd))
        if weight < min_synd_weight:
            min_synd_weight = weight
            best_tau = tau
            if weight == 0:
                break

    return best_tau, min_synd_weight
