"""Differential testing of Reed-Solomon codec vs reedsolo (SIGNAL10 §G.1 / Rule R9)."""
from __future__ import annotations
import pytest
import numpy as np
reedsolo = pytest.importorskip("reedsolo")
from signal9.fec.rs import rs_encode, rs_decode


@pytest.mark.parametrize("n,k", [(255, 223), (204, 188)])
@pytest.mark.parametrize("seed", range(5))
def test_rs_encode_bit_exact(n, k, seed):
    """Signal9 RS encoding must match reedsolo reference exactly."""
    nsym = n - k
    rsc = reedsolo.RSCodec(nsym, c_exp=8, fcr=0, prim=0x11D)

    rng = np.random.default_rng(seed)
    msg = rng.integers(0, 256, k).astype(np.uint8)

    ref_cw = np.frombuffer(rsc.encode(bytes(msg)), dtype=np.uint8)
    ours_cw = rs_encode(msg, n=n, k=k, fcr=0)

    assert np.array_equal(ours_cw, ref_cw), f"RS encoding mismatch for ({n},{k}), seed {seed}"


@pytest.mark.parametrize("n_errs", [0, 4, 8, 16])
def test_rs_decode_agreement(n_errs):
    """Signal9 RS decoder must correct up to t errors matching reference."""
    n, k = 255, 223
    rsc = reedsolo.RSCodec(32, c_exp=8, fcr=0, prim=0x11D)

    rng = np.random.default_rng(42)
    for _ in range(5):
        msg = rng.integers(0, 256, k).astype(np.uint8)
        cw = rs_encode(msg, n=n, k=k, fcr=0)

        # Inject errors
        corrupted = cw.copy()
        if n_errs > 0:
            err_pos = rng.choice(n, n_errs, replace=False)
            corrupted[err_pos] ^= rng.integers(1, 256, n_errs).astype(np.uint8)

        dec_ours, errs_corrected = rs_decode(corrupted, n=n, k=k, fcr=0)
        assert errs_corrected == n_errs
        assert np.array_equal(dec_ours, msg)

        # Also verify reedsolo can decode it identically
        dec_ref = np.frombuffer(rsc.decode(bytes(corrupted))[0], dtype=np.uint8)
        assert np.array_equal(dec_ours, dec_ref)


def test_rs_uncorrectable_error_flagged():
    """Errors exceeding t=16 must be flagged (n_err == -1) and not silently corrupted."""
    n, k = 255, 223
    rng = np.random.default_rng(99)
    msg = rng.integers(0, 256, k).astype(np.uint8)
    cw = rs_encode(msg, n=n, k=k, fcr=0)

    # 20 errors > t=16
    corrupted = cw.copy()
    err_pos = rng.choice(n, 20, replace=False)
    corrupted[err_pos] ^= rng.integers(1, 256, 20).astype(np.uint8)

    _, n_err = rs_decode(corrupted, n=n, k=k, fcr=0)
    assert n_err == -1, f"Expected uncorrectable error flag (-1), got {n_err}"
