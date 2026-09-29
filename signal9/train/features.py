"""Expert (hand-crafted) feature vector fed to the backbone's fusion layer.

One function, used identically by every training dataset and by inference, so
the network sees the same feature distribution everywhere.
Populates all 32 dimensions with Swami & Sadler higher-order cumulants (up to 8th order)
and physical envelope / instantaneous frequency / spectral statistics.
"""
from __future__ import annotations

import math
from functools import lru_cache
import numpy as np

from ..features.cumulants import cumulant_features

EXPERT_DIM = 32
_CLIP = 10.0


def _get_partitions(collection):
    """Generates all set partitions of a sequence."""
    if len(collection) == 1:
        yield [collection]
        return
    first = collection[0]
    for smaller in _get_partitions(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n+1:]
        yield [[first]] + smaller


@lru_cache(maxsize=32)
def _precompute_partition_rules(p: int, q: int):
    """Precomputes combinatorial partition coefficients and block signatures for C_{p+q, q}."""
    elements = [(+1, i) for i in range(p)] + [(-1, i + p) for i in range(q)]
    rules = []
    for part in _get_partitions(elements):
        num_blocks = len(part)
        coeff = math.factorial(num_blocks - 1) * ((-1) ** (num_blocks - 1))
        block_signatures = []
        for block in part:
            pos = sum(1 for sign, _ in block if sign == 1)
            neg = sum(1 for sign, _ in block if sign == -1)
            block_signatures.append((pos, neg))
        rules.append((coeff, tuple(block_signatures)))
    return rules


def compute_cumulant(x: np.ndarray, p: int, q: int) -> complex:
    """Computes joint complex cumulant C_{p+q, q} from zero-mean samples x."""
    rules = _precompute_partition_rules(p, q)
    moments = {}
    for a in range(p + 1):
        for b in range(q + 1):
            if a == 0 and b == 0:
                moments[(a, b)] = 1.0
            else:
                moments[(a, b)] = complex(np.mean((x ** a) * (np.conj(x) ** b)))
    cumulant = 0.0 + 0.0j
    for coeff, block_sigs in rules:
        term = coeff
        for sig in block_sigs:
            term *= moments[sig]
        cumulant += term
    return cumulant


def extract_hoc_dict(iq: np.ndarray) -> dict[str, float]:
    """Returns a dictionary of named higher-order cumulants and physical metrics."""
    x = np.asarray(iq, dtype=np.complex64).ravel()
    if x.size < 32:
        return {"c42": 0.0, "c63": 0.0, "c84": 0.0, "c80": 0.0, "cv": 0.0, "kurt": 0.0}
    p = float(np.mean(np.abs(x) ** 2))
    if p <= 0 or not np.isfinite(p):
        return {"c42": 0.0, "c63": 0.0, "c84": 0.0, "c80": 0.0, "cv": 0.0, "kurt": 0.0}
    xc = (x - np.mean(x)) / np.sqrt(p)
    a = np.abs(xc)
    return {
        "c42": float(compute_cumulant(xc, 2, 2).real),
        "c63": float(compute_cumulant(xc, 3, 3).real),
        "c84": float(compute_cumulant(xc, 4, 4).real),
        "c80": float(np.abs(compute_cumulant(xc, 8, 0))),
        "cv": float(np.std(a) / (np.mean(a) + 1e-12)),
        "kurt": float(np.mean(a ** 4)),
    }


def expert_features(iq: np.ndarray, full: bool = False, legacy_papr: bool = False) -> np.ndarray:
    """Return the float32[EXPERT_DIM] expert feature vector for one IQ window.

    If full=False (default), populates indices 0..10 and keeps 11..31 as zeros
    for 100% strict mathematical backward-compatibility with pretrained checkpoint weights.
    If full=True, populates all 32 dimensions with 6th/8th-order cumulants and higher moments.
    """
    x = np.asarray(iq, dtype=np.complex64).ravel()
    out = np.zeros(EXPERT_DIM, dtype=np.float32)
    if x.size < 32:
        return out
    p = float(np.mean(np.abs(x) ** 2))
    if p <= 0 or not np.isfinite(p):
        return out
    x_norm = x / np.sqrt(p)
    xc = x_norm - np.mean(x_norm)

    # 0..4: Classical cumulants (backward-compatible with pretrained checkpoints)
    # This vector is part of the neural checkpoint input contract.  The
    # vectorised Kaggle trainer computes cumulants over the complete conditioned
    # window, so inference must do the same.  Robust outlier handling already
    # happens in ``condition_for_classifier``; trimming again here changes the
    # moments and caused a material train/serve feature skew.
    out[0:5] = cumulant_features(x_norm, trim_frac=0.0)

    # 5..10: Envelope & Frequency statistics
    a = np.abs(x_norm)
    mean_a = float(np.mean(a)) + 1e-12
    out[5] = float(np.std(a) / mean_a)
    out[6] = float(np.mean(a ** 4))
    diff_prod = x_norm[1:] * np.conj(x_norm[:-1])
    fi = np.angle(diff_prod) / (2.0 * np.pi)
    std_fi = float(np.std(fi))
    mean_fi = float(np.mean(np.abs(fi)))
    out[7] = std_fi
    out[8] = mean_fi
    spec = np.abs(np.fft.fft(x_norm)) ** 2 + 1e-12
    mean_spec = float(np.mean(spec))
    out[9] = float(np.exp(np.mean(np.log(spec))) / (mean_spec + 1e-12))
    papr_db = float(10.0 * np.log10(np.max(a ** 2) + 1e-12))
    out[10] = papr_db / 10.0 if legacy_papr else papr_db

    if not full:
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=_CLIP, neginf=-_CLIP), -_CLIP, _CLIP)

    # 11..13: 6th-order cumulants
    out[11] = float(np.abs(compute_cumulant(xc, 6, 0)))
    out[12] = float(np.abs(compute_cumulant(xc, 5, 1)))
    out[13] = float(np.abs(compute_cumulant(xc, 4, 2)))

    # 14..18: 8th-order Swami & Sadler cumulants
    out[14] = float(np.abs(compute_cumulant(xc, 8, 0)))
    out[15] = float(np.abs(compute_cumulant(xc, 7, 1)))
    out[16] = float(np.abs(compute_cumulant(xc, 6, 2)))
    out[17] = float(np.abs(compute_cumulant(xc, 5, 3)))
    out[18] = float(compute_cumulant(xc, 4, 4).real)

    # 19..20: Rotational symmetry circular moments
    out[19] = float(np.abs(np.mean(xc ** 8)))
    out[20] = float(np.abs(np.mean(xc ** 4)))

    # 21..22: Higher-order statistical shapes
    std_a = float(np.std(a)) + 1e-12
    out[21] = float(np.mean((a - mean_a) ** 3) / (std_a ** 3))
    out[22] = float(np.mean((fi - np.mean(fi)) ** 4) / ((std_fi + 1e-12) ** 4))

    # 23..26: Spectral descriptors
    freqs = np.linspace(0.0, 1.0, len(spec))
    centroid = float(np.sum(freqs * spec) / np.sum(spec))
    out[23] = centroid
    cumsum = np.cumsum(spec)
    roll90 = float(freqs[np.searchsorted(cumsum, 0.90 * cumsum[-1])])
    out[24] = roll90
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spec) / np.sum(spec)))
    out[25] = spread
    out[26] = float(np.max(spec) / (np.median(spec) + 1e-12))

    # 27..31: Time-domain correlations and dynamics
    out[27] = float(np.mean(x_norm.real * x_norm.imag))
    out[28] = float(np.var(a ** 2))
    out[29] = float(np.mean(np.abs(np.diff(np.unwrap(np.angle(x_norm)))) ** 2))
    out[30] = float(np.mean(np.abs(np.diff(np.signbit(fi - np.mean(fi))))))
    out[31] = float(np.mean(diff_prod.real))

    return np.clip(np.nan_to_num(out, nan=0.0, posinf=_CLIP, neginf=-_CLIP), -_CLIP, _CLIP)
