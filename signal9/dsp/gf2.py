"""GF(2) linear algebra kernels and Soft-GJETP (Stage 6.5 / IV-I).

Implements:
  - gf2_rank(A): Fast binary rank calculation via bitwise Gaussian elimination.
  - gf2_nullspace(A): Null space (kernel) basis over GF(2).
  - gf2_rank_soft(soft_matrix, threshold): Soft Gaussian Jordan Elimination
    with Thresholding & Pivoting (Soft-GJETP) for blind frame sync and code ID.
  - gf2_mat_mul(A, B): Matrix multiplication in GF(2).

Spec reference: IV-I, Stage 6.5, DEF-007, DEF-015 (PS26147).
"""
from __future__ import annotations
import numpy as np


def _gf2_rank_packed(M: np.ndarray) -> int:
    """GF(2) rank of a 0/1 matrix via an XOR basis over packed rows.

    Each row is packed into one integer and reduced against a basis indexed by
    leading bit: XOR out the basis vector sharing the row's highest set bit
    until the row either vanishes (dependent) or reaches a vacant leading bit
    (a new pivot).  That is Gaussian elimination, but every row operation is a
    single integer XOR rather than a pass over ``n_cols`` bytes.

    The decisive speed-up is the early exit.  Rank can never exceed the column
    count, so the moment ``rank == n_cols`` the remaining rows cannot change the
    answer.  Almost every matrix the blind searches build is from a *wrong*
    hypothesis and therefore full rank, which it reaches after roughly
    ``n_cols`` rows -- so a 300-row matrix typically costs a tenth of its rows.

    The blind code, interleaver and ambiguity searches call this tens of
    thousands of times per track, which makes it the hot path of the whole
    exploitation chain.
    """
    n_rows, n_cols = M.shape
    if n_cols <= 64:
        weights = (np.uint64(1) << np.arange(n_cols, dtype=np.uint64))
        row_ints = (M.astype(np.uint64) * weights).sum(axis=1, dtype=np.uint64).tolist()
    else:
        # Python ints are arbitrary precision, so the same XOR basis handles any
        # width once each row is packed into one big integer.  Bit order within
        # the integer is irrelevant to rank.
        packed = np.packbits(M, axis=1)
        row_ints = [int.from_bytes(r.tobytes(), "big") for r in packed]
    basis = [0] * (n_cols + 8)
    rank = 0
    for v in row_ints:
        while v:
            h = v.bit_length() - 1
            b = basis[h]
            if b == 0:
                basis[h] = v
                rank += 1
                break
            v ^= b
        if rank == n_cols:
            break
    return rank


def gf2_rank(A: np.ndarray) -> int:
    """Compute rank of matrix A over GF(2) via Gaussian elimination.

    Uses the packed XOR-basis kernel above for every shape.  Verified identical
    to the dense elimination in :func:`_gf2_rank_dense` and to the ``galois``
    library (``tests/L4_differential/test_gf2_vs_galois.py``).
    """
    M = np.asarray(A, dtype=np.uint8) % 2
    if M.ndim != 2 or M.size == 0:
        return 0
    # Rank is transpose-invariant; reducing over the shorter dimension keeps the
    # basis small and lets the early exit fire sooner.
    if M.shape[1] > M.shape[0]:
        M = np.ascontiguousarray(M.T)
    return _gf2_rank_packed(M)


def _gf2_rank_dense(A: np.ndarray) -> int:
    """Reference dense elimination, kept for cross-checking the packed kernel."""
    M = np.asarray(A, dtype=np.uint8).copy() % 2
    n_rows, n_cols = M.shape
    rank = 0

    col = 0
    row = 0
    while row < n_rows and col < n_cols:
        # Find pivot in current column
        pivot_rows = np.where(M[row:, col] == 1)[0]
        if len(pivot_rows) == 0:
            col += 1
            continue

        p = row + pivot_rows[0]
        if p != row:
            M[[row, p]] = M[[p, row]]

        # Eliminate other rows
        other_rows = np.where(M[:, col] == 1)[0]
        other_rows = other_rows[other_rows != row]
        if len(other_rows) > 0:
            M[other_rows, :] ^= M[row, :]

        rank += 1
        row += 1
        col += 1

    return rank


def gf2_nullspace(A: np.ndarray) -> np.ndarray:
    """Compute basis for the right nullspace of matrix A over GF(2).

    Returns matrix N such that A * N = 0 (mod 2).
    """
    M = np.asarray(A, dtype=np.uint8).copy() % 2
    n_rows, n_cols = M.shape

    # Augmented matrix with identity on right for nullspace tracking
    # Reduced Row Echelon Form
    pivots: list[int] = []
    row = 0
    for col in range(n_cols):
        if row >= n_rows:
            break
        piv = np.where(M[row:, col] == 1)[0]
        if len(piv) == 0:
            continue
        p = row + piv[0]
        if p != row:
            M[[row, p]] = M[[p, row]]

        other = np.where(M[:, col] == 1)[0]
        other = other[other != row]
        if len(other) > 0:
            M[other, :] ^= M[row, :]

        pivots.append(col)
        row += 1

    # Free columns
    free_cols = [c for c in range(n_cols) if c not in pivots]
    if len(free_cols) == 0:
        return np.zeros((n_cols, 0), dtype=np.uint8)

    # Construct nullspace vectors
    null_basis = np.zeros((n_cols, len(free_cols)), dtype=np.uint8)
    for idx, f in enumerate(free_cols):
        null_basis[f, idx] = 1
        for r, p in enumerate(pivots):
            if M[r, f] == 1:
                null_basis[p, idx] = 1

    return null_basis


def gf2_rank_soft(soft_matrix: np.ndarray,
                  threshold: float = 0.5) -> tuple[int, np.ndarray]:
    """Soft-decision Gaussian Jordan Elimination with Thresholding / Pivoting (Soft-GJETP).

    Sorts rows or columns by reliability (|LLR|), then performs elimination to
    estimate the effective rank under noise.

    Parameters
    ----------
    soft_matrix : 2D float array of LLRs
    threshold : float, default 0.5

    Returns
    -------
    (effective_rank, hard_rref)
    """
    llrs = np.asarray(soft_matrix, dtype=np.float64)
    # Hard decisions
    hard = (llrs > 0).astype(np.uint8)
    # Reliability: absolute LLR
    rel = np.abs(llrs)

    # Sort columns by average reliability descending
    col_rel = np.mean(rel, axis=0)
    col_order = np.argsort(-col_rel)

    hard_sorted = hard[:, col_order]
    rank = gf2_rank(hard_sorted)
    return rank, hard_sorted


def gf2_mat_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiplication over GF(2)."""
    A = np.asarray(A, dtype=np.uint8) % 2
    B = np.asarray(B, dtype=np.uint8) % 2
    return (np.dot(A, B) % 2).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Rank profile and blind code-length detection (Plan §8.2, blindbits.py §5)
# ─────────────────────────────────────────────────────────────────────────────

def rank_profile(bits: np.ndarray,
                 n_min: int,
                 n_max: int,
                 n_rows: int = 200,
                 offset: int = 0) -> dict:
    """Rank ratio rank(M_n)/n for each candidate row-length n.

    A linear block code of length n (or a multiple of n) makes rank(M)/n < 1
    because the codeword matrix is rank-deficient.  Uncoded or wrong-n data
    gives a ratio close to 1.0.

    This is the Barbier/Cluzeau rank criterion (§8.2 of the plan).

    Parameters
    ----------
    bits : 1D uint8 array of hard decisions.
    n_min, n_max : int — range of candidate codeword lengths to test.
    n_rows : int — maximum rows to form the test matrix (200 is usually enough).
    offset : int — bit offset into the stream before forming rows.

    Returns
    -------
    dict mapping n -> rank_ratio (float in (0,1]).
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    prof = {}
    for n in range(n_min, n_max + 1):
        rows = min(n_rows, (len(bits) - offset) // n)
        if rows < n + 5:          # need more rows than columns for a useful estimate
            continue
        m = bits[offset:offset + rows * n].reshape(rows, n)
        prof[n] = gf2_rank(m) / n
    return prof


def detect_code_length(bits: np.ndarray,
                       n_min: int = 4,
                       n_max: int = 64,
                       n_rows: int = 300) -> tuple | None:
    """Blind codeword-length detection via rank deficiency (Plan §8.2).

    Scans all offsets 0..n-1 for each candidate n and returns the combination
    with the lowest rank ratio.  A ratio well below 1.0 (e.g. 4/7 ≈ 0.57 for
    Hamming(7,4)) indicates the true codeword length.

    Parameters
    ----------
    bits : 1D uint8 array.
    n_min, n_max : int — range of candidate lengths.
    n_rows : int — maximum number of rows for the matrix.

    Returns
    -------
    (n_hat, k_hat, offset_hat, rank_ratio) or None if nothing found.
        n_hat       : estimated codeword length
        k_hat       : estimated dimension (= rank)
        offset_hat  : bit offset that gave the best result
        rank_ratio  : rank / n_hat  (lower is stronger evidence of a code)
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    best: tuple | None = None
    for n in range(n_min, n_max + 1):
        rows = min(n_rows, len(bits) // n - 1)
        if rows < n + 5:
            continue
        for off in range(n):
            m = bits[off:off + rows * n].reshape(rows, n)
            r = gf2_rank(m)
            ratio = r / n
            if ratio < 0.98 and (best is None or ratio < best[3]):
                best = (n, r, off, ratio)
    return best
