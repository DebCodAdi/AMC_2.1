"""Blind algebraic interleaver detection (IV-H).

Implements rank-deficiency and Kolmogorov-Smirnov testing for blind
identification of block interleaver geometry (R x C).

Under the true period P = R (or C), the matrix formed by reshaping the
intercepted bitstream exhibits linear dependency (rank deficiency) due to
the underlying code constraints.

Coverage of problem-statement requirement (iii): the block detector below is
joined by convolutional (Forney), helical/diagonal and semi-blind pseudo-random
detectors, and by identify_interleaver() which scores all four families with the
same GF(2) rank statistic and returns the winner (or an explicit UNRESOLVED).

Plan additions (Blind_Exploitation_Chain_Plan.md §8.2):
  - detect_block_interleaver_depth(): depth-scan search in front of a known code
    length (blindbits.py reference algorithm).  Returns per-depth rank ratios so
    the search engine can pick the minimum unambiguously.
  - detect_block_interleaver_blind(): no prior code_n required — calls
    detect_code_length() first, then sweeps depths.
  - Both operate on LLR arrays as well as hard bits (permutation is index-only).

Spec reference: IV-H, DEF-007, DEF-014 (PS26147).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from signal9.dsp.gf2 import gf2_rank, detect_code_length
from signal9.fec.interleaver import (
    convolutional_deinterleave,
    convolutional_interleaver_delay,
    strip_convolutional_delay,
    helical_deinterleave,
    permutation_deinterleave,
    prng_permutation,
    qpp_permutation,
)


@dataclass(frozen=True)
class InterleaverCandidate:
    R: int
    C: int
    rank_deficiency: int
    score: float


def block_interleave(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Write rows-first, read columns-first (classic block interleaver)."""
    bits = np.asarray(bits)
    n = len(bits) // (rows * cols) * (rows * cols)
    b = bits[:n].reshape(-1, rows, cols)
    return b.transpose(0, 2, 1).reshape(-1)


def block_deinterleave(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Inverse of block_interleave (write columns-first, read rows-first)."""
    bits = np.asarray(bits)
    n = len(bits) // (rows * cols) * (rows * cols)
    b = bits[:n].reshape(-1, cols, rows)
    return b.transpose(0, 2, 1).reshape(-1)


def detect_block_interleaver(bits: np.ndarray,
                              candidate_geometries: list[tuple[int, int]] | None = None) -> list[InterleaverCandidate]:
    """Blindly identify block interleaver dimensions (R x C).

    Parameters
    ----------
    bits : 1D array of intercepted bits
    candidate_geometries : list of (R, C) tuples to test
        Default: [(4, 8), (8, 4), (8, 8), (16, 16), (32, 32)]

    Returns
    -------
    list of InterleaverCandidate sorted by score descending
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if candidate_geometries is None:
        candidate_geometries = [
            (4, 8), (8, 4), (3, 5), (5, 3), (8, 8),
            (16, 16), (16, 32), (32, 16), (17, 13)
        ]

    results: list[InterleaverCandidate] = []
    N = len(bits)

    for R, C in candidate_geometries:
        block_len = R * C
        n_blocks = N // block_len
        if n_blocks < 4:
            continue

        # Deinterleave candidate
        sample_blocks = min(n_blocks, 64)
        usable_len = sample_blocks * block_len
        data = bits[:usable_len].reshape(sample_blocks, C, R)
        deint = np.transpose(data, (0, 2, 1)).reshape(sample_blocks * R, C)

        # Compute rank of matrix
        r = gf2_rank(deint)
        expected_rank = min(deint.shape[0], deint.shape[1])
        deficiency = max(0, expected_rank - r)

        # Autocorrelation lag check at lag R
        lag = R
        if N > lag + 100:
            corr = float(np.mean(bits[:-lag] == bits[lag:]))
            bias = abs(corr - 0.5)
        else:
            bias = 0.0

        score = float(deficiency * 10.0 + bias * 20.0)
        results.append(InterleaverCandidate(
            R=R,
            C=C,
            rank_deficiency=deficiency,
            score=score
        ))

    results.sort(key=lambda c: c.score, reverse=True)
    return results


def detect_block_interleaver_depth(bits: np.ndarray,
                                   code_n: int,
                                   depth_max: int = 32,
                                   n_rows: int = 300) -> tuple[int | None, dict[int, float]]:
    """Depth search for a block interleaver in front of a known (n,k) block code.

    For the true depth D the de-interleaved stream regains the code's rank
    deficiency.  We de-interleave under each hypothesis and score with the rank
    criterion — lower rank ratio is stronger evidence.

    This is the algorithm from blindbits.py / Plan §8.2 (Burel-Gautier / Sicot-
    Houcke-Barbier rank criterion).

    Parameters
    ----------
    bits : 1D uint8 array (hard decisions or LLR signs converted to bits).
    code_n : int — known codeword length in bits.
    depth_max : int — maximum interleaver depth to search.
    n_rows : int — rows for the rank matrix.

    Returns
    -------
    (best_depth, scores)
        best_depth : int or None
        scores : dict {depth -> rank_ratio}
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    scores: dict[int, float] = {}
    for d in range(2, depth_max + 1):
        de = block_deinterleave(bits, d, code_n)
        rows = min(n_rows, len(de) // code_n)
        if rows < code_n + 5:
            continue
        m = de[:rows * code_n].reshape(rows, code_n)
        scores[d] = gf2_rank(m) / code_n
    if not scores:
        return None, {}
    best = min(scores, key=scores.__getitem__)
    return best, scores


def detect_block_interleaver_blind(bits: np.ndarray,
                                   n_min: int = 4,
                                   n_max: int = 64,
                                   depth_max: int = 32,
                                   n_rows: int = 300) -> dict:
    """Fully blind block interleaver detection: no prior code length required.

    1. Calls detect_code_length() to find the most likely code length.
    2. Calls detect_block_interleaver_depth() to find the interleaver depth.

    Returns a dict with keys: code_n, k, offset, depth, rank_ratio, scores.
    Returns an empty dict if no code structure is found.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    result = detect_code_length(bits, n_min=n_min, n_max=n_max, n_rows=n_rows)
    if result is None:
        return {}
    n_hat, k_hat, off_hat, ratio = result
    # Search depth with the detected code length
    best_depth, depth_scores = detect_block_interleaver_depth(
        bits[off_hat:], code_n=n_hat, depth_max=depth_max, n_rows=n_rows
    )
    return {
        "code_n": n_hat,
        "k": k_hat,
        "offset": off_hat,
        "depth": best_depth,
        "rank_ratio": ratio,
        "scores": depth_scores,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shared rank scoring
# ─────────────────────────────────────────────────────────────────────────────

def _rank_ratio(bits: np.ndarray, code_n: int, n_rows: int = 200) -> float:
    """Rank of the reshaped GF(2) matrix divided by the codeword length.

    This is the Burel-Gautier / Sicot-Houcke-Barbier criterion that every
    estimator below shares.  A linear (n, k) block code fills a codeword matrix
    with rank k < n, so a correctly de-interleaved stream scores about k/n while
    every wrong hypothesis destroys the linear structure and scores about 1.0.
    ``1.0`` is returned when there is not enough data to form a matrix taller
    than it is wide, because a short matrix is rank-deficient for trivial
    reasons and would otherwise look like a spurious detection.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if code_n <= 0:
        return 1.0
    rows = min(int(n_rows), len(bits) // code_n)
    if rows < code_n + 5:
        return 1.0
    m = bits[:rows * code_n].reshape(rows, code_n)
    return float(gf2_rank(m)) / float(code_n)


# ─────────────────────────────────────────────────────────────────────────────
# Convolutional (Forney) interleaver detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_convolutional_interleaver(bits: np.ndarray,
                                     code_n: int,
                                     branches_max: int = 16,
                                     delay_max: int = 8,
                                     n_rows: int = 200) -> tuple[tuple[int, int] | None, dict]:
    """Blind (B, M) search for a Forney convolutional interleaver.

    Scans branch counts ``B`` in ``2..branches_max`` and delay increments ``M``
    in ``1..delay_max``, de-interleaves under each hypothesis, strips the
    constant pipeline delay, and scores the result with the rank criterion.  The
    lowest rank ratio wins, matching the block-interleaver detector so the two
    scores are directly comparable in :func:`identify_interleaver`.

    A convolutional interleaver has no frame boundary, so there is no offset to
    search -- the pipeline delay is deterministic in ``(B, M)``.  That is what
    makes this a two-parameter scan rather than the three-parameter one the
    block and helical families need.

    Returns ``((B, M), scores)`` or ``(None, scores)`` when nothing beat the
    no-structure baseline.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    scores: dict[tuple[int, int], float] = {}
    for B in range(2, int(branches_max) + 1):
        for M in range(1, int(delay_max) + 1):
            de = convolutional_deinterleave(bits, B, M)
            de = strip_convolutional_delay(de, B, M)
            # Drop a second pipeline delay from the tail: those samples were
            # still flushing when the capture ended and are zero fill.
            d = convolutional_interleaver_delay(B, M)
            if len(de) > d:
                de = de[:len(de) - d]
            if len(de) < code_n * (code_n + 5):
                continue
            scores[(B, M)] = _rank_ratio(de, code_n, n_rows)
    if not scores:
        return None, {}
    best = min(scores, key=scores.__getitem__)
    return best, scores


# ─────────────────────────────────────────────────────────────────────────────
# Helical / diagonal interleaver detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_helical_interleaver(bits: np.ndarray,
                               code_n: int,
                               rows_max: int = 12,
                               cols_max: int = 12,
                               slope_max: int = 4,
                               n_rows: int = 200) -> tuple[tuple[int, int, int] | None, dict]:
    """Blind ``(rows, cols, slope)`` search for a helical / diagonal interleaver.

    Same rank criterion as the block and convolutional detectors.  ``slope``
    starts at 1 because slope 0 is exactly the block interleaver, which
    :func:`detect_block_interleaver_depth` already covers -- keeping the families
    disjoint is what lets :func:`identify_interleaver` report a single answer
    instead of two equally-scored ones.

    Returns ``((rows, cols, slope), scores)`` or ``(None, scores)``.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    scores: dict[tuple[int, int, int], float] = {}
    for r in range(2, int(rows_max) + 1):
        for c in range(2, int(cols_max) + 1):
            block = r * c
            if len(bits) < block * 4:
                continue
            for sl in range(1, min(int(slope_max), c - 1) + 1):
                de = helical_deinterleave(bits, r, c, sl)
                scores[(r, c, sl)] = _rank_ratio(de, code_n, n_rows)
    if not scores:
        return None, {}
    best = min(scores, key=scores.__getitem__)
    return best, scores


# ─────────────────────────────────────────────────────────────────────────────
# Pseudo-random interleaver: semi-blind library search
# ─────────────────────────────────────────────────────────────────────────────

# Standard 3GPP LTE turbo interleaver (K, f1, f2) triples from 36.212
# Table 5.1.3-3, covering the short-block end of the table that a blind receiver
# can realistically test within a bounded budget.
_QPP_LIBRARY: list[tuple[int, int, int]] = [
    (40, 3, 10), (48, 7, 12), (56, 19, 42), (64, 7, 16),
    (72, 7, 18), (80, 11, 20), (88, 5, 22), (96, 11, 24),
    (104, 7, 26), (112, 41, 84), (120, 103, 90), (128, 15, 32),
    (160, 21, 120), (192, 23, 48), (256, 15, 32), (320, 21, 120),
]


def detect_pseudorandom_interleaver(bits: np.ndarray,
                                    code_n: int,
                                    prng_sizes: tuple[int, ...] = (32, 64, 128, 256),
                                    prng_seeds: tuple[int, ...] = (),
                                    n_rows: int = 200) -> tuple[dict | None, dict]:
    """Semi-blind pseudo-random interleaver identification against a library.

    A pseudo-random (turbo-style) interleaver is a full permutation of the block.
    Recovering an arbitrary one from an intercepted stream is not tractable at
    useful block lengths -- the literature (Cote & Sendrier, ISIT 2010) treats it
    as an expensive special case, and the project plan's honest-limits section
    says so explicitly.  What *is* tractable is the semi-blind case: the
    transmitter used a standard permutation, so we test the standard library.

    Two families are tested:

    * 3GPP LTE quadratic permutation polynomials (36.212 Table 5.1.3-3).
    * Seeded PRNG permutations, for streams this toolchain generated itself --
      only when ``prng_seeds`` is supplied, since guessing a PRNG seed blindly
      is not a search anyone can bound.

    Returns ``(best, scores)`` where ``best`` is a dict describing the winning
    permutation, or ``(None, scores)`` when nothing in the library fits.  A
    caller that gets ``None`` should report ``PSEUDORANDOM_UNRESOLVED`` rather
    than fall back to a weaker family: an unrecognised permutation is a genuine
    negative result, not a block interleaver.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    scores: dict[str, float] = {}
    best_key: str | None = None
    best_desc: dict | None = None
    best_score = 1.0

    def _try(name: str, perm: np.ndarray, desc: dict) -> None:
        nonlocal best_key, best_desc, best_score
        if len(bits) < len(perm) * 4:
            return
        de = permutation_deinterleave(bits, perm)
        sc = _rank_ratio(de, code_n, n_rows)
        scores[name] = sc
        if sc < best_score:
            best_key, best_desc, best_score = name, desc, sc

    for K, f1, f2 in _QPP_LIBRARY:
        try:
            perm = qpp_permutation(K, f1, f2)
        except ValueError:
            continue
        _try(f"qpp_K{K}_f1{f1}_f2{f2}", perm,
             {"type": "pseudorandom", "family": "qpp", "size": K, "f1": f1, "f2": f2})

    for size in prng_sizes:
        for seed in prng_seeds:
            perm = prng_permutation(size, seed)
            _try(f"prng_n{size}_s{seed}", perm,
                 {"type": "pseudorandom", "family": "prng", "size": size, "seed": seed})

    if best_desc is not None:
        best_desc = dict(best_desc, rank_ratio=best_score, name=best_key)
    return best_desc, scores


# ─────────────────────────────────────────────────────────────────────────────
# Unified interleaver identification (PS requirement iii)
# ─────────────────────────────────────────────────────────────────────────────

def _local_structure(bits: np.ndarray, n_min: int = 4, n_max: int = 32,
                     n_rows: int = 200) -> tuple[float, int]:
    """Lowest rank ratio over short row widths at offset 0, and the width that gave it.

    Used after a candidate de-interleave to ask "did this restore the code?".
    Offset 0 is enough here because the caller has already aligned the stream
    to an interleaver block boundary, and interleavers operate on whole coded
    blocks -- so codeword (or coded-symbol) boundaries line up with it.
    """
    best, best_n = 1.0, 0
    for n in range(n_min, n_max + 1):
        r = _rank_ratio(bits, n, n_rows)
        if r < best:
            best, best_n = r, n
    return best, best_n


def find_interleaver_block(bits: np.ndarray,
                           w_min: int = 33,
                           w_max: int = 256,
                           n_rows: int = 200) -> tuple[int, list[int], float] | None:
    """Block size ``W`` and candidate phases of a block-type interleaver.

    Block, helical and pseudo-random interleavers all apply one fixed
    permutation to each W-bit block.  Build the matrix whose rows are exactly
    those blocks and the permutation becomes a column permutation -- which
    leaves GF(2) rank unchanged.  So at the right ``(W, phase)`` the matrix is as
    rank-deficient as the un-interleaved code, whatever the permutation was,
    while at any other width or phase the interleaving has destroyed the
    structure.

    That decouples the two unknowns: first locate the block, then identify the
    permutation inside it.  Knowing the phase matters because a captured stream
    never starts on an interleaver block boundary -- de-interleaving from bit 0
    mixes two blocks in every window and no permutation hypothesis can fit.

    ``w_min`` defaults above the short widths that :func:`_local_structure`
    already covers, so a code that is visible without de-interleaving is not
    mistaken for an interleaver block.

    Returns ``(W, phases, rank_ratio)`` -- the phases tied at the minimum rank
    ratio, best first -- or None.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    for W in range(int(w_min), int(w_max) + 1):
        if len(bits) < (W + 5) * W + W:
            break
        ratios = np.array([_rank_ratio(bits[ph:], W, n_rows) for ph in range(W)])
        r_min = float(ratios.min())
        if r_min < _INTERLEAVER_RANK_THRESHOLD:
            # Multiples of the true block size are also rank-deficient, so the
            # first width that shows structure is the fundamental.  Phases a
            # bit or two off the true boundary cost only a bit or two of rank
            # and often tie with it, so every phase at the minimum is returned
            # and the permutation test -- which is sharp about phase -- decides.
            tied = [int(ph) for ph in np.flatnonzero(ratios <= r_min + 1.0 / W + 1e-9)]
            return W, tied, r_min
    return None


# Row widths probed when asking whether a de-interleave restored a code whose
# length is unknown: the common block lengths (Hamming 7/15/31, Golay 23/24)
# plus even widths where a rate-1/2 convolutional code becomes rank-deficient.
_PROBE_WIDTHS = (7, 8, 12, 14, 15, 16, 21, 23, 24, 28, 31, 32)


def _probe_structure(bits: np.ndarray, n_rows: int = 200) -> float:
    """Cheap code-agnostic structure probe: lowest rank ratio over _PROBE_WIDTHS."""
    return min(_rank_ratio(bits, n, n_rows) for n in _PROBE_WIDTHS)


def detect_convolutional_interleaver_blind(bits: np.ndarray,
                                           branches_max: int = 12,
                                           delay_max: int = 17,
                                           n_rows: int = 200) -> tuple[dict | None, float]:
    """Blind (B, M, commutator phase) search that does not need the code length.

    A captured stream starts at an arbitrary commutator position, and
    de-interleaving from the wrong branch applies every branch the wrong delay,
    so the phase has to be searched alongside B and M.  Each hypothesis is
    scored by whether it restores short-width code structure (see
    :func:`_probe_structure`).  The defaults reach the DVB-S/T setting
    (B=12, M=17).

    Returns ``(params, ratio)``; params is None when nothing restored structure.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    best: dict | None = None
    best_r = 1.0
    for B in range(2, int(branches_max) + 1):
        for M in range(1, int(delay_max) + 1):
            d = convolutional_interleaver_delay(B, M)
            if len(bits) < 2 * d + 32 * 40:
                break
            for ph in range(B):
                de = strip_convolutional_delay(convolutional_deinterleave(bits[ph:], B, M), B, M)
                if len(de) > d:
                    de = de[:len(de) - d]
                r = _probe_structure(de, n_rows)
                if r < best_r - 1e-9:
                    best_r = r
                    best = {"type": "convolutional", "branches": B, "delay": M, "phase": ph}
    return best, best_r


def _identify_within_block(bits: np.ndarray, W: int, phase: int,
                           prng_seeds: tuple[int, ...], n_rows: int) -> list[tuple[float, dict]]:
    """Score every permutation family of block size ``W`` by restored local structure."""
    aligned = bits[phase:]
    out: list[tuple[float, dict]] = []

    for R in range(2, W // 2 + 1):
        if W % R:
            continue
        C = W // R
        if C < 2:
            continue
        de = block_deinterleave(aligned, R, C)
        r, n = _local_structure(de, n_max=min(32, W - 1), n_rows=n_rows)
        out.append((r, {"type": "block", "rows": R, "cols": C, "depth": R, "code_n": C,
                        "phase": int(phase), "local_code_n": n}))
        for slope in range(1, min(4, C - 1) + 1):
            deh = helical_deinterleave(aligned, R, C, slope)
            rh, nh = _local_structure(deh, n_max=min(32, W - 1), n_rows=n_rows)
            out.append((rh, {"type": "helical", "rows": R, "cols": C, "slope": slope,
                             "phase": int(phase), "local_code_n": nh}))

    for K, f1, f2 in _QPP_LIBRARY:
        if K != W:
            continue
        try:
            perm = qpp_permutation(K, f1, f2)
        except ValueError:
            continue
        dq = permutation_deinterleave(aligned, perm)
        rq, nq = _local_structure(dq, n_max=min(32, W - 1), n_rows=n_rows)
        out.append((rq, {"type": "pseudorandom", "family": "qpp", "size": K, "f1": f1,
                         "f2": f2, "phase": int(phase), "local_code_n": nq}))

    for seed in prng_seeds:
        perm = prng_permutation(W, seed)
        dp = permutation_deinterleave(aligned, perm)
        rp, npp = _local_structure(dp, n_max=min(32, W - 1), n_rows=n_rows)
        out.append((rp, {"type": "pseudorandom", "family": "prng", "size": W, "seed": seed,
                         "phase": int(phase), "local_code_n": npp}))
    return out


# A hypothesis must beat this rank ratio to count as evidence of structure.
# Random or wrongly de-interleaved data gives a full-rank matrix (ratio ~1.0);
# a real (n, k) code gives k/n.  0.95 leaves headroom for the handful of rows a
# noisy stream loses without admitting pure chance.
_INTERLEAVER_RANK_THRESHOLD = 0.95


def identify_interleaver(bits: np.ndarray,
                         code_n: int | None = None,
                         n_min: int = 4,
                         n_max: int = 32,
                         n_rows: int = 200,
                         prng_seeds: tuple[int, ...] = (),
                         budget_families: tuple[str, ...] = ("none", "block", "convolutional", "helical", "pseudorandom"),
                         ) -> dict:
    """Identify which interleaver family a bitstream went through, and its parameters.

    This is the entry point for problem-statement requirement (iii).  It scores
    every family with the same statistic -- the GF(2) rank ratio of the
    de-interleaved codeword matrix -- so the results are directly comparable, and
    returns the family with the strongest rank deficiency.

    ``"none"`` is always in the candidate set and is scored first: many real HF
    modes (RTTY, PSK31, FT8) carry no interleaver at all, and a detector that
    cannot return "none" will always invent one.

    Parameters
    ----------
    bits : 1D uint8 array of hard decisions.
    code_n : known codeword length, or None to estimate it with
        :func:`signal9.dsp.gf2.detect_code_length` first.
    prng_seeds : seeds to test for the PRNG branch of the pseudo-random search.
        Empty by default -- see :func:`detect_pseudorandom_interleaver`.
    budget_families : which families to spend time on, for a bounded search.

    Returns
    -------
    dict with ``type`` (one of ``none``, ``block``, ``convolutional``,
    ``helical``, ``pseudorandom``, ``PSEUDORANDOM_UNRESOLVED``, ``UNRESOLVED``),
    the winning parameters, ``rank_ratio``, ``code_n``, and ``scores``
    (best ratio per family, so a caller can show the runner-up).
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()

    if code_n is None:
        # Blind path.  First ask whether the stream carries a block-type
        # interleaver at all: locate its block size and phase from rank
        # deficiency at the block width, then identify the permutation inside.
        # This works for convolutional codes as well as block codes, because it
        # never needs to know the code's own length.
        local_raw, _ = _local_structure(bits, n_max=n_max, n_rows=n_rows)
        blk = None
        # A code that is already visible at short widths is not interleaved --
        # and it is rank-deficient at every multiple of its own length too, so
        # the block search would "find" a spurious interleaver block at one of
        # those multiples.  Only look for a block when the short-width structure
        # is absent, which is exactly the signature interleaving leaves.
        if (local_raw >= _INTERLEAVER_RANK_THRESHOLD
                and any(f in budget_families for f in ("block", "helical", "pseudorandom"))):
            blk = find_interleaver_block(bits, w_min=n_max + 1, n_rows=n_rows)

        # A convolutional interleaver has no block period, so it is searched
        # separately; it also destroys short-width structure, so it is only
        # worth the cost when that structure is missing.
        conv_blind, conv_r = None, 1.0
        if local_raw >= _INTERLEAVER_RANK_THRESHOLD and "convolutional" in budget_families:
            conv_blind, conv_r = detect_convolutional_interleaver_blind(bits, n_rows=n_rows)
        conv_ok = (conv_blind is not None and conv_r < _INTERLEAVER_RANK_THRESHOLD
                   and conv_r + 0.02 < local_raw)

        if blk is not None:
            W, phases, w_ratio = blk
            phase = phases[0]
            # The rank-deficient period can be a multiple of the true block: an
            # odd block on a rate-1/2 code alternates which coded bit each block
            # starts on, so the matrix only becomes consistent at twice the
            # block size.  Every divisor of W is therefore a candidate block,
            # tested at every tied phase (reduced modulo that block size).
            cands = []
            for W0 in range(4, W + 1):
                if W % W0:
                    continue
                for ph0 in sorted({ph % W0 for ph in phases}):
                    cands.extend(_identify_within_block(bits, W0, ph0, prng_seeds, n_rows))
            cands = [c for c in cands if c[1]["type"] in budget_families]
            if cands:
                ratio, params = min(cands, key=lambda c: c[0])
                # The winner has to restore structure that the raw stream does
                # not already show at the same short widths -- and beat the best
                # convolutional hypothesis, which competes on the same footing.
                if (ratio < _INTERLEAVER_RANK_THRESHOLD and ratio + 0.02 < local_raw
                        and not (conv_ok and conv_r < ratio)):
                    out = dict(params)
                    out["rank_ratio"] = round(float(ratio), 4)
                    out["block_size"] = int(params.get("size") or
                                            (params.get("rows", 0) * params.get("cols", 0)) or W)
                    out["block_rank_ratio"] = round(float(w_ratio), 4)
                    out["scores"] = {"none": round(float(local_raw), 4),
                                     out["type"]: round(float(ratio), 4)}
                    return out
            if conv_ok:
                out = dict(conv_blind)
                out["rank_ratio"] = round(float(conv_r), 4)
                out["scores"] = {"none": round(float(local_raw), 4),
                                 "convolutional": round(float(conv_r), 4)}
                return out
            # A block-periodic structure exists but no permutation in the
            # library restores the code: the honest answer is that the block is
            # found and the permutation is not.
            return {"type": "PSEUDORANDOM_UNRESOLVED", "block_size": int(W),
                    "phase": int(phase), "block_rank_ratio": round(float(w_ratio), 4),
                    "rank_ratio": round(float(local_raw), 4),
                    "reason": "interleaver block located but no permutation hypothesis restored the code",
                    "scores": {"none": round(float(local_raw), 4)}}

        if conv_ok:
            out = dict(conv_blind)
            out["rank_ratio"] = round(float(conv_r), 4)
            out["scores"] = {"none": round(float(local_raw), 4),
                             "convolutional": round(float(conv_r), 4)}
            return out

        found = detect_code_length(bits, n_min=n_min, n_max=n_max, n_rows=n_rows)
        if found is None:
            # No linear structure anywhere: either uncoded, scrambled, or
            # encrypted.  Saying so is more useful than guessing a geometry.
            return {"type": "UNRESOLVED", "code_n": None, "rank_ratio": 1.0,
                    "reason": "no codeword length with rank deficiency found",
                    "scores": {}}
        code_n = int(found[0])

    family_best: dict[str, tuple[float, dict]] = {}

    if "none" in budget_families:
        none_ratio = _rank_ratio(bits, code_n, n_rows)
        family_best["none"] = (none_ratio, {"type": "none"})
        # The code is already visible without de-interleaving, so the stream is
        # not interleaved: a real interleaver destroys exactly this structure.
        # Comparing ratios instead let a wrong hypothesis win whenever it lined
        # up repeated frame content (identical text in every frame) and looked
        # a little *more* structured than the truth -- measured on an AFSK
        # recording: "block" 0.656 vs "none" 0.719, and FEC then failed.
        if none_ratio < _INTERLEAVER_RANK_THRESHOLD:
            return {"type": "none", "code_n": int(code_n), "rank_ratio": round(none_ratio, 4),
                    "scores": {"none": round(none_ratio, 4)}}

    if "block" in budget_families:
        depth, depth_scores = detect_block_interleaver_depth(
            bits, code_n=code_n, depth_max=min(32, max(2, len(bits) // (code_n * 8))), n_rows=n_rows)
        if depth is not None:
            family_best["block"] = (depth_scores[depth],
                                    {"type": "block", "depth": int(depth), "code_n": int(code_n)})

    if "convolutional" in budget_families:
        bm, conv_scores = detect_convolutional_interleaver(bits, code_n=code_n, n_rows=n_rows)
        if bm is not None:
            family_best["convolutional"] = (conv_scores[bm],
                                            {"type": "convolutional", "branches": int(bm[0]), "delay": int(bm[1])})

    if "helical" in budget_families:
        rcs, hel_scores = detect_helical_interleaver(bits, code_n=code_n, n_rows=n_rows)
        if rcs is not None:
            family_best["helical"] = (hel_scores[rcs],
                                      {"type": "helical", "rows": int(rcs[0]),
                                       "cols": int(rcs[1]), "slope": int(rcs[2])})

    if "pseudorandom" in budget_families:
        pr, pr_scores = detect_pseudorandom_interleaver(
            bits, code_n=code_n, prng_seeds=prng_seeds, n_rows=n_rows)
        if pr is not None:
            family_best["pseudorandom"] = (float(pr["rank_ratio"]), pr)

    scores = {k: round(v[0], 4) for k, v in family_best.items()}
    if not family_best:
        return {"type": "UNRESOLVED", "code_n": int(code_n), "rank_ratio": 1.0, "scores": scores}

    winner = min(family_best, key=lambda k: family_best[k][0])
    ratio, params = family_best[winner]

    if ratio > _INTERLEAVER_RANK_THRESHOLD:
        # Nothing showed rank deficiency.  If the pseudo-random library was
        # searched and missed, say so specifically -- that is a different and
        # more informative negative than "no structure at all".
        kind = "PSEUDORANDOM_UNRESOLVED" if "pseudorandom" in budget_families else "UNRESOLVED"
        return {"type": kind, "code_n": int(code_n), "rank_ratio": round(ratio, 4),
                "reason": "no hypothesis reached the rank-deficiency threshold",
                "scores": scores}

    out = dict(params)
    out["rank_ratio"] = round(ratio, 4)
    out["code_n"] = int(code_n)
    out["scores"] = scores
    return out


def deinterleave_with(stream: np.ndarray, params: dict) -> np.ndarray:
    """Apply the inverse of the interleaver described by ``params``.

    ``params`` is what :func:`identify_interleaver` returns.  Works on hard bits
    and on LLR arrays alike, because every de-interleaver is an index
    permutation.  An unknown or unresolved type passes the stream through
    unchanged rather than raising, so a partial chain can still report what it
    did recover.
    """
    stream = np.asarray(stream)
    kind = params.get("type", "none")
    # Blind detection reports where the first interleaver block starts; the
    # partial block before it cannot be de-interleaved and is dropped.
    phase = int(params.get("phase", 0) or 0)
    if phase and kind in ("block", "helical", "pseudorandom"):
        stream = stream[phase:]
    if kind == "block":
        rows = int(params.get("rows", params.get("depth")))
        cols = int(params.get("cols", params.get("code_n")))
        return block_deinterleave(stream, rows, cols)
    if kind == "convolutional":
        B, M = int(params["branches"]), int(params["delay"])
        if phase:
            stream = stream[phase:]
        return strip_convolutional_delay(convolutional_deinterleave(stream, B, M), B, M)
    if kind == "helical":
        return helical_deinterleave(stream, int(params["rows"]), int(params["cols"]), int(params["slope"]))
    if kind == "pseudorandom":
        if params.get("family") == "qpp":
            perm = qpp_permutation(int(params["size"]), int(params["f1"]), int(params["f2"]))
        else:
            perm = prng_permutation(int(params["size"]), int(params["seed"]))
        return permutation_deinterleave(stream, perm)
    return stream
