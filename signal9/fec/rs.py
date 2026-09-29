"""Reed-Solomon codec over GF(2^8) (KAT-RS-1..4).

Implements:
  - rs_encode(data, n=255, k=223): Systematic RS encoder.
  - rs_decode(cw, n=255, k=223): Berlekamp-Massey + Chien search + Forney algorithm.
  - Shortened codes: e.g. DVB RS(204, 188).
  - Flags decode failure when errors > t (KAT-RS-3).
  - DEF-015: Conventional vs Dual (Berlekamp) basis transformation.

Spec reference: KAT-RS-1..4, DEF-014, DEF-015 (PS26147).
"""
from __future__ import annotations
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# GF(2^8) Galois field arithmetic (primitive poly 0x11D = 285)
# ─────────────────────────────────────────────────────────────────────────────

PRIM_POLY = 0x11D
_gf_exp = [0] * 512
_gf_log = [0] * 256

x = 1
for i in range(255):
    _gf_exp[i] = x
    _gf_log[x] = i
    x <<= 1
    if x & 0x100:
        x ^= PRIM_POLY
for i in range(255, 512):
    _gf_exp[i] = _gf_exp[i - 255]


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _gf_exp[_gf_log[a] + _gf_log[b]]


def _gf_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("Division by zero in GF(2^8)")
    if a == 0:
        return 0
    return _gf_exp[(_gf_log[a] - _gf_log[b]) % 255]


def _gf_poly_eval(p: list[int] | np.ndarray, x_val: int) -> int:
    y = 0
    for c in p:
        y = _gf_mul(y, x_val) ^ int(c)
    return y


def _rs_gen_poly(nsym: int, fcr: int = 0) -> list[int]:
    g = [1]
    for i in range(nsym):
        root = _gf_exp[(fcr + i) % 255]
        new_g = [0] * (len(g) + 1)
        for j, c in enumerate(g):
            new_g[j] ^= c
            new_g[j + 1] ^= _gf_mul(c, root)
        g = new_g
    return g


# ─────────────────────────────────────────────────────────────────────────────
# RS Encode & Decode
# ─────────────────────────────────────────────────────────────────────────────

def rs_encode(data: np.ndarray | list[int] | bytes,
              n: int = 255,
              k: int = 223,
              fcr: int = 0) -> np.ndarray:
    """Encode data into an (n, k) Reed-Solomon codeword.

    Parameters
    ----------
    data : array-like of length k (bytes or uint8 in 0..255)
    n : int, default 255 (codeword length)
    k : int, default 223 (message length)
    fcr : int, default 0 (first consecutive root)

    Returns
    -------
    np.ndarray of length n, dtype uint8
    """
    msg = [int(b) for b in data]
    if len(msg) != k:
        if len(msg) < k:
            msg = msg + [0] * (k - len(msg))
        else:
            msg = msg[:k]

    nsym = n - k
    gen = _rs_gen_poly(nsym, fcr)
    rem = list(msg) + [0] * nsym
    for i in range(k):
        c = rem[i]
        if c != 0:
            for j in range(1, len(gen)):
                rem[i + j] ^= _gf_mul(gen[j], c)

    codeword = list(msg) + rem[k:]
    return np.array(codeword, dtype=np.uint8)


def rs_decode(codeword: np.ndarray | list[int] | bytes,
              n: int = 255,
              k: int = 223,
              fcr: int = 0) -> tuple[np.ndarray, int]:
    """Decode an (n, k) Reed-Solomon codeword.

    Parameters
    ----------
    codeword : array-like of length n
    n : int (codeword length)
    k : int (message length)
    fcr : int, default 0

    Returns
    -------
    (recovered_data, n_errors) : tuple[np.ndarray, int]
        recovered_data : length k, uint8 array
        n_errors : number of corrected errors, or -1 if uncorrectable (> t).
    """
    cw = [int(b) for b in codeword]
    if len(cw) < n:
        return np.zeros(k, dtype=np.uint8), -1
    cw = cw[:n]
    nsym = n - k
    t = nsym // 2

    # 1. Syndromes: S_j = cw(alpha^(fcr + j)) for j = 0..nsym-1
    S = [_gf_poly_eval(cw, _gf_exp[(fcr + j) % 255]) for j in range(nsym)]
    if max(S) == 0:
        return np.array(cw[:k], dtype=np.uint8), 0

    # 2. Berlekamp-Massey algorithm
    Lambda = [1]
    B = [1]
    L = 0
    m = 1
    b = 1
    for r in range(nsym):
        delta = S[r]
        for i in range(1, len(Lambda)):
            delta ^= _gf_mul(Lambda[i], S[r - i])
        if delta == 0:
            m += 1
        else:
            T = list(Lambda)
            factor = _gf_div(delta, b)
            while len(Lambda) < len(B) + m:
                Lambda.append(0)
            for i in range(len(B)):
                Lambda[i + m] ^= _gf_mul(factor, B[i])
            if 2 * L <= r:
                L = r + 1 - L
                B = T
                b = delta
                m = 1
            else:
                m += 1

    if L > t:
        return np.array(cw[:k], dtype=np.uint8), -1

    # 3. Chien search for error positions
    err_pos = []
    for idx in range(len(cw)):
        p = len(cw) - 1 - idx
        x_val = _gf_exp[(255 - (p % 255)) % 255]
        val = 0
        x_pow = 1
        for c in Lambda:
            val ^= _gf_mul(c, x_pow)
            x_pow = _gf_mul(x_pow, x_val)
        if val == 0:
            err_pos.append(idx)

    if len(err_pos) != L:
        return np.array(cw[:k], dtype=np.uint8), -1

    # 4. Forney algorithm for error values
    Omega = [0] * nsym
    for i in range(nsym):
        for j in range(len(Lambda)):
            if i + j < nsym:
                Omega[i + j] ^= _gf_mul(S[i], Lambda[j])

    for idx in err_pos:
        p = len(cw) - 1 - idx
        X_l_inv = _gf_exp[(255 - (p % 255)) % 255]
        num = 0
        x_pow = 1
        for c in Omega:
            num ^= _gf_mul(c, x_pow)
            x_pow = _gf_mul(x_pow, X_l_inv)

        den = 0
        for i in range(1, len(Lambda), 2):
            den ^= _gf_mul(Lambda[i], _gf_exp[((i - 1) * (255 - (p % 255))) % 255])
        if den == 0:
            return np.array(cw[:k], dtype=np.uint8), -1

        scale = _gf_exp[((1 - fcr) * p) % 255]
        Y = _gf_mul(scale, _gf_div(num, den))
        cw[idx] ^= Y

    # Verify that syndromes are now zero
    post_S = [_gf_poly_eval(cw, _gf_exp[(fcr + j) % 255]) for j in range(nsym)]
    if max(post_S) != 0:
        return np.array(cw[:k], dtype=np.uint8), -1

    return np.array(cw[:k], dtype=np.uint8), len(err_pos)


# ─────────────────────────────────────────────────────────────────────────────
# Dual-basis transformation matrix (DEF-015 / CCSDS 131.0-B)
# ─────────────────────────────────────────────────────────────────────────────

# Standard CCSDS dual-basis transformation matrix T over GF(2)
_DUAL_BASIS_MATRIX = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 0],
    [1, 0, 1, 0, 1, 0, 1, 1],
], dtype=np.uint8)


_INV_DUAL_BASIS_MATRIX = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 1, 0],
    [1, 0, 1, 1, 0, 1, 1, 1],
], dtype=np.uint8)


def to_dual_basis(symbols: np.ndarray) -> np.ndarray:
    """Convert symbols from conventional basis to CCSDS dual basis."""
    out = np.zeros_like(symbols)
    for i, s in enumerate(symbols):
        b = np.array([(s >> bit) & 1 for bit in range(8)], dtype=np.uint8)
        db = np.dot(_DUAL_BASIS_MATRIX, b) % 2
        out[i] = int(np.sum(db * (1 << np.arange(8))))
    return out


def from_dual_basis(symbols: np.ndarray) -> np.ndarray:
    """Convert symbols from CCSDS dual basis to conventional basis."""
    out = np.zeros_like(symbols)
    for i, s in enumerate(symbols):
        b = np.array([(s >> bit) & 1 for bit in range(8)], dtype=np.uint8)
        cb = np.dot(_INV_DUAL_BASIS_MATRIX, b) % 2
        out[i] = int(np.sum(cb * (1 << np.arange(8))))
    return out
