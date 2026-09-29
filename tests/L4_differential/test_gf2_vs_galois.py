"""Differential testing of GF(2) linear algebra vs galois library (SIGNAL10 §G.1 / Rule R9)."""
from __future__ import annotations
import pytest
import numpy as np
import galois

from signal9.dsp.gf2 import gf2_rank, gf2_nullspace

GF2 = galois.GF(2)


@pytest.mark.parametrize("shape", [(8, 8), (32, 16), (64, 63), (128, 64), (200, 255), (1, 1), (1, 50), (50, 1)])
@pytest.mark.parametrize("seed", range(5))
def test_rank_matches_galois(shape, seed):
    rng = np.random.default_rng(seed)
    M = rng.integers(0, 2, shape).astype(np.uint8)
    expected_rank = int(np.linalg.matrix_rank(GF2(M)))
    actual_rank = gf2_rank(M)
    assert actual_rank == expected_rank, f"Rank mismatch for shape {shape}, seed {seed}: got {actual_rank}, want {expected_rank}"


def test_rank_deficient_construction():
    """A matrix built to be rank-deficient must be reported as such."""
    rng = np.random.default_rng(0)
    B = rng.integers(0, 2, (40, 64)).astype(np.uint8)
    # Append 10 linearly dependent rows
    dependent = (B[:10] ^ B[10:20])
    M = np.vstack([B, dependent])
    actual_rank = gf2_rank(M)
    expected_rank = int(np.linalg.matrix_rank(GF2(M)))
    assert actual_rank == expected_rank == 40
    assert actual_rank < M.shape[0]


def test_nullspace_is_orthogonal():
    """A * N == 0 (mod 2) for the computed nullspace basis."""
    rng = np.random.default_rng(42)
    M = rng.integers(0, 2, (30, 64)).astype(np.uint8)
    N = gf2_nullspace(M)
    if N.size > 0:
        prod = (M @ N) % 2
        assert np.all(prod == 0), "Nullspace basis failed orthogonality test (M @ N % 2 != 0)"
