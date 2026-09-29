"""Blind modulation-order selection from a recovered symbol cloud.

Why this module exists
----------------------
The obvious way to pick a constellation is "whichever gives the lowest EVM".
That is wrong, and wrong in a way that always fails the same direction: EVM is
the distance to the *nearest* constellation point, and a higher-order
constellation has more points, so its nearest point is never further away.  EVM
therefore falls monotonically with order and an argmin over it selects 64-QAM
for every input -- QPSK, BPSK and pure noise alike.  (Measured: a clean QPSK
track at 25 dB scores -25.0 dB under QPSK and -19.4 dB under 64-QAM, so the
argmin is not even close.)

Three measurements are needed to bracket the true order, because each one only
rules out one kind of mistake:

**Usage** -- the entropy of the hard-decision symbol histogram, normalised by
``log2(M)``.  With random payload bits every point of the true constellation is
hit about equally, so usage approaches 1.0; an over-ordered hypothesis leaves
most of its points empty (QPSK scored as 64-QAM: 4 points of 64, usage 0.33).
This kills over-ordering at useful SNR, where the clusters stay separated.

**EVM** -- kills *under*-ordering, which usage cannot see: BPSK scored against a
QPSK cloud uses both of its points, so its usage is a perfect 1.0, but it fits
them 20 dB worse.

**Residual Gaussianity** -- kills over-ordering at low SNR, where noise smears
symbols across the whole finer grid and usage saturates at 1.0 anyway.  Under
the true constellation the residual is the channel noise, i.e. circular complex
Gaussian, whose kurtosis ``E|r|^4 / (E|r|^2)^2`` is exactly 2.  Under an
over-ordered hypothesis the residual is dominated by the fixed offset between
the real symbol and the nearest point of a finer grid, which is far more
concentrated: measured 1.40 for 64-QAM against 16-QAM data at 15 dB, versus
1.88 for the true 16-QAM hypothesis.

**Parsimony** settles what is left.  When two surviving hypotheses fit within
``_PARSIMONY_TOL_DB`` of each other, the simpler one wins and the result is
flagged ambiguous, rather than silently promoting the higher order.

Honest limit: high-order QAM below roughly 12 dB in-band SNR is genuinely
ambiguous -- the clusters overlap, so 16-QAM and 64-QAM become hard to separate
by any measurement on the symbol cloud alone.  ``select_modulation`` reports
``ambiguous=True`` and returns the full score table instead of pretending.
"""
from __future__ import annotations

import numpy as np

from .constellation import get_constellation
from .snr_estimate import evm_db

# Candidate orders, smallest first, so ties resolve toward the simpler model.
DEFAULT_CANDIDATES: tuple[str, ...] = ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM")

# Fraction of log2(M) the symbol-usage entropy must reach for a hypothesis to
# count as exercised by the data.  Random data over the true constellation
# approaches 1.0; the margin below absorbs histogram noise over a few thousand
# symbols and payloads that are not perfectly uniform (ASCII, headers, idle
# fill).  0.90 still rejects the tightest real confusion.
_USAGE_THRESHOLD = 0.90

# A hypothesis whose EVM is worse than this never counts as a lock, whatever
# its usage: the constellation did not open.
_MIN_EVM_DB = -6.0

# Residual kurtosis below this means the residual is a fixed grid offset rather
# than noise -- the signature of an over-ordered hypothesis.  Circular complex
# Gaussian noise gives exactly 2.0.
_MIN_RESIDUAL_KURTOSIS = 1.60

# Symbol-cloud kurtosis above this is treated as noise (see select_modulation).
_MAX_CLOUD_KURTOSIS = 1.80

# Two hypotheses fitting within this many dB of each other are treated as
# equally good, and the simpler (lower-order) one is chosen.
_PARSIMONY_TOL_DB = 3.0


def _normalised(symbols: np.ndarray) -> np.ndarray | None:
    symbols = np.asarray(symbols, dtype=np.complex128).ravel()
    if symbols.size == 0:
        return None
    power = float(np.mean(np.abs(symbols) ** 2))
    if power <= 0 or not np.isfinite(power):
        return None
    return symbols / np.sqrt(power)


def constellation_usage(symbols: np.ndarray, mod_type: str) -> float:
    """Entropy of the hard-decision symbol histogram, normalised by ``log2(M)``.

    1.0 when every constellation point is used equally often; roughly
    ``log2(M_true)/log2(M_hypothesis)`` when the data exercises only a
    sub-constellation.
    """
    y = _normalised(symbols)
    const = get_constellation(mod_type)
    M = len(const)
    if y is None or M < 2:
        return 0.0
    nearest = np.argmin(np.abs(y[:, None] - const[None, :]) ** 2, axis=1)
    counts = np.bincount(nearest, minlength=M).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / np.log2(M))


def residual_kurtosis(symbols: np.ndarray, mod_type: str) -> float:
    """Kurtosis ``E|r|^4 / (E|r|^2)^2`` of the residual to the nearest point.

    2.0 for circular complex Gaussian noise, i.e. for the true constellation.
    Well below 2.0 when the residual is a deterministic offset to a grid finer
    than the signal's own.
    """
    y = _normalised(symbols)
    const = get_constellation(mod_type)
    if y is None or len(const) < 2:
        return 0.0
    nearest = np.argmin(np.abs(y[:, None] - const[None, :]) ** 2, axis=1)
    r = y - const[nearest]
    m2 = float(np.mean(np.abs(r) ** 2))
    if m2 <= 0 or not np.isfinite(m2):
        return 0.0
    return float(np.mean(np.abs(r) ** 4) / (m2 * m2))


def score_modulations(symbols: np.ndarray,
                      candidates: tuple[str, ...] = DEFAULT_CANDIDATES) -> list[dict]:
    """Score every candidate constellation, one record per hypothesis.

    Each record carries ``mod``, ``evm_db``, ``usage``, ``residual_kurtosis``,
    ``plausible`` and, when rejected, ``rejected_by`` -- so the report can show
    *why* a hypothesis lost, not only which one won.
    """
    symbols = np.asarray(symbols, dtype=np.complex128).ravel()
    out: list[dict] = []
    if symbols.size < 16:
        return out
    angles = np.linspace(0, 2.0 * np.pi, 16, endpoint=False)
    for mod in candidates:
        try:
            const = get_constellation(mod)
            M = len(const)
            best_e = 999.0
            best_rot = symbols
            best_angle = 0.0
            for a in angles:
                s_rot = symbols * np.exp(-1j * a)
                d = float(evm_db(s_rot, const))
                if d < best_e:
                    best_e = d
                    best_rot = s_rot
                    best_angle = a
            e = best_e
            u = constellation_usage(best_rot, mod)
            k = residual_kurtosis(best_rot, mod)
        except Exception:
            continue

        max_possible_usage = min(1.0, np.log2(max(2, symbols.size)) / np.log2(M))
        usage_thresh = min(_USAGE_THRESHOLD, 0.85 * max_possible_usage)

        reject = None
        if e > _MIN_EVM_DB:
            reject = "evm"           # under-ordered: constellation never opened
        elif u < usage_thresh:
            reject = "usage"         # over-ordered: most points never used
        elif k < _MIN_RESIDUAL_KURTOSIS:
            reject = "residual"      # over-ordered: residual is a grid offset
        out.append({
            "mod": mod,
            "evm_db": round(e, 2),
            "usage": round(u, 3),
            "residual_kurtosis": round(k, 3),
            "best_phase_rad": float(best_angle),
            "plausible": reject is None,
            "rejected_by": reject,
        })
    return out


def select_modulation(symbols: np.ndarray,
                      candidates: tuple[str, ...] = DEFAULT_CANDIDATES,
                      hint: str | None = None) -> dict:
    """Pick the constellation the symbol cloud actually came from.

    ``hint`` (typically the neural classifier's label) only breaks ties between
    hypotheses that are already plausible on the measurements; it can never
    promote one the usage, EVM or residual test rejected.  That ordering keeps a
    mis-trained or out-of-distribution classifier from overriding physics.

    Returns a dict with ``modulation`` (``"UNRESOLVED"`` when nothing was
    plausible), ``evm_db``, ``ambiguous``, ``runner_up`` and the full
    ``scores`` table.
    """
    scores = score_modulations(symbols, candidates)
    # A cloud with Gaussian statistics is noise, not a constellation.  Every
    # PSK/QAM cloud has E|y|^4/(E|y|^2)^2 <= 1.38 without noise (1.0 for PSK,
    # 1.32 for 16QAM, 1.38 for 64QAM) and only approaches the Gaussian value 2
    # as SNR falls to 0 dB (64QAM is still ~1.64 at 5 dB).  Without this test a
    # noise-like cloud passed every other check as "64QAM": it fills the finer
    # grid (usage ~1), its residual is Gaussian, and its EVM against 64 points
    # is about -12 dB.
    y = _normalised(symbols)
    cloud_kurtosis = float(np.mean(np.abs(y) ** 4)) if y is not None else 2.0
    if cloud_kurtosis > _MAX_CLOUD_KURTOSIS:
        return {"modulation": "UNRESOLVED", "evm_db": 0.0, "ambiguous": False,
                "runner_up": None, "scores": scores,
                "reason": f"symbol cloud is Gaussian (kurtosis {cloud_kurtosis:.2f}): noise, not a constellation"}
    plausible = [s for s in scores if s["plausible"]]
    if not plausible:
        return {"modulation": "UNRESOLVED", "evm_db": 0.0, "ambiguous": False,
                "runner_up": None, "scores": scores}

    best_fit = min(s["evm_db"] for s in plausible)
    # Parsimony: everything fitting within tolerance of the best is "as good",
    # so take the lowest-order member of that set.  `candidates` is ordered
    # smallest-first, so the first match is the simplest.
    within = [s for s in plausible if s["evm_db"] <= best_fit + _PARSIMONY_TOL_DB]
    chosen = within[0]

    if hint:
        h = hint.upper().replace("-", "")
        for s in within:
            if s["mod"] == h:
                chosen = s
                break

    others = [s for s in within if s["mod"] != chosen["mod"]]
    return {
        "modulation": chosen["mod"],
        "evm_db": float(chosen["evm_db"]),
        "best_phase_rad": float(chosen.get("best_phase_rad", 0.0)),
        # More than one hypothesis fitting within tolerance is a real ambiguity
        # (typically high-order QAM at low SNR); saying so beats a false
        # certainty the downstream bit stages would inherit.
        "ambiguous": len(within) > 1,
        "runner_up": others[0]["mod"] if others else None,
        "scores": scores,
    }
