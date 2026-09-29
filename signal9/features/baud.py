"""Multi-estimator baud-rate fusion.

DEF-011: The existing baud-rate estimator uses only the envelope non-linearity
and a single peak pick.  Three independent estimators are now fused:

  1. **Envelope autocorrelation** (NL-AC) — original, robust for AM/PM bursts
  2. **Spectral Correlation Function (SCF / CAFE)** — resolves ambiguity
     between Rs and 2Rs from |x|² spectral lines; handles very high SPS
  3. **Zero-crossing rate on baseband phase** — reliable for constant-envelope
     signals (FSK, GMSK) at any SPS

Fusion gate: two estimators agreeing within ±0.1% → high confidence.
If only one estimate exists, it is returned with low confidence.

The function returns a sorted list of (rs_hz, log_likelihood) pairs so the
blackboard can carry multiple hypotheses (Config.fs_candidates_k).

Spec reference: DEF-011, §IV-A (PS26147).
"""
from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Estimator 1: Envelope non-linearity autocorrelation (NL-AC)
# ─────────────────────────────────────────────────────────────────────────────

def _nlac_baud(x: np.ndarray, fs: float,
               min_baud: float, max_baud: float,
               nlin: int = 2) -> list[tuple[float, float]]:
    """Return up to 3 (baud_hz, prominence) candidates from the NL-AC estimator."""
    # Both |x|² and |x|⁴ nonlinearities
    env2 = np.abs(x) ** 2
    env2 -= env2.mean()
    env4 = np.abs(x) ** 4
    env4 -= env4.mean()
    env = env2 + 0.1 * env4

    n = int(2 ** np.floor(np.log2(len(env))))
    if n < 256:
        return []
    psd = np.abs(np.fft.rfft(env[:n])) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    mask = (freqs >= min_baud) & (freqs <= max_baud)
    if not np.any(mask):
        return []

    vf = freqs[mask]
    vp = psd[mask]
    med = float(np.median(vp)) + 1e-20

    # Find top-3 peaks
    candidates = []
    vp_copy = vp.copy()
    for _ in range(3):
        pk = int(np.argmax(vp_copy))
        prom = vp_copy[pk] / med
        if prom < 3.0:
            break
        candidates.append((float(vf[pk]), float(np.log(prom))))
        # Suppress ±5% window around this peak
        window = int(0.05 * vf[pk] / (fs / n)) + 1
        lo = max(0, pk - window)
        hi = min(len(vp_copy), pk + window)
        vp_copy[lo:hi] = 0.0

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Estimator 2: SCF (Spectral Correlation Function) at cyclic frequency α
# ─────────────────────────────────────────────────────────────────────────────

def _scf_baud(x: np.ndarray, fs: float,
              min_baud: float, max_baud: float) -> list[tuple[float, float]]:
    """Simplified CAFE (Cyclic Autocorrelation Function Estimate).

    Computes the cyclic PSD at α = 1/Ts = Rs.  For a linearly modulated
    signal with symbol rate Rs, |x(t)|² has a spectral line at α=Rs.

    This is equivalent to the NL-AC estimator but uses a finer resolution
    and includes the Rs/2 harmonic disambiguation step.
    """
    n = len(x)
    if n < 512:
        return []

    # Use FFT of |x|² — CAFE at α = k/T
    env2 = np.abs(x) ** 2
    env2 -= env2.mean()

    # Windowed FFT for sidelobe suppression
    win = np.hanning(n)
    psd = np.abs(np.fft.rfft(env2 * win)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    mask = (freqs >= min_baud) & (freqs <= max_baud)
    if not np.any(mask):
        return []

    vf = freqs[mask]
    vp = psd[mask]
    med = float(np.median(vp)) + 1e-20

    # Highest peak
    pk = int(np.argmax(vp))
    prom = float(vp[pk]) / med
    if prom < 4.0:
        return []

    rs_cand = float(vf[pk])

    # Disambiguation: check if Rs/2 has a stronger line (indicating we found 2Rs)
    half = rs_cand / 2.0
    if half > min_baud:
        half_mask = np.abs(freqs - half) < 0.01 * half
        if half_mask.any():
            half_psd = float(psd[half_mask].max())
            full_psd = float(psd[np.abs(freqs - rs_cand) < 0.01 * rs_cand].max() + 1e-20)
            if half_psd / full_psd > 2.0:
                rs_cand = half  # The true Rs is Rs/2

    return [(rs_cand, float(np.log(max(prom, 1e-6))))]


# ─────────────────────────────────────────────────────────────────────────────
# Estimator 3: Zero-crossing rate on instantaneous phase
# ─────────────────────────────────────────────────────────────────────────────

def _zcr_baud(x: np.ndarray, fs: float,
              min_baud: float, max_baud: float) -> list[tuple[float, float]]:
    """Zero-crossing rate on the high-passed instantaneous frequency.

    Effective for constant-envelope signals (FSK, GMSK, CPM) where the
    envelope estimator produces no spectral lines.

    Not reliable for QAM — use the NL-AC / SCF estimators for those.
    """
    if len(x) < 256:
        return []

    # Instantaneous frequency (unwrapped phase derivative)
    phase = np.unwrap(np.angle(x))
    instfreq = np.diff(phase) * fs / (2.0 * np.pi)

    # High-pass at min_baud/4 to remove slow drift
    from scipy.signal import butter, filtfilt
    try:
        b, a = butter(2, min_baud / 4.0 / (fs / 2.0), btype="high")
        instfreq_hp = filtfilt(b, a, instfreq)
    except Exception:
        instfreq_hp = instfreq - np.mean(instfreq)

    # Count zero crossings
    signs = np.sign(instfreq_hp)
    zc = np.sum(np.diff(signs) != 0)
    rs_est = float(zc) / (2.0 * len(instfreq_hp) / fs)

    if not (min_baud <= rs_est <= max_baud):
        return []

    return [(rs_est, 1.0)]  # confidence always 1.0 (one estimate)


# ─────────────────────────────────────────────────────────────────────────────
# Three-way fusion
# ─────────────────────────────────────────────────────────────────────────────

def estimate_baud_rate(x: np.ndarray,
                       fs: float,
                       min_baud: float = 10.0,
                       max_baud: float | None = None) -> float:
    """Return the best single baud-rate estimate in Hz.

    Backward-compatible API: returns a float (0.0 if no estimate).
    For the multi-hypothesis blackboard use ``baud_candidates()`` instead.
    """
    cands = baud_candidates(x, fs, min_baud, max_baud)
    if not cands:
        return 0.0
    return cands[0][0]


def baud_candidates(x: np.ndarray,
                    fs: float,
                    min_baud: float = 10.0,
                    max_baud: float | None = None,
                    fusion_tol: float = 0.001,
                    top_k: int = 3) -> list[tuple[float, float]]:
    """Return top-k (baud_hz, log_likelihood) pairs.

    Three estimators (NL-AC, SCF, ZCR) are run independently and their
    candidates are fused:
      - If two estimators agree within ±fusion_tol (0.1%), their ll is summed.
      - Candidates that appear in only one estimator are kept but penalised.

    Parameters
    ----------
    x : complex array
        IQ samples at sample rate ``fs``.
    fs : float
        Sample rate in Hz.
    min_baud : float
        Minimum baud rate to consider.
    max_baud : float or None
        Maximum baud rate (default: fs/2).
    fusion_tol : float
        Relative tolerance for considering two estimates as "agreeing".
    top_k : int
        Number of candidates to return.

    Returns
    -------
    list of (baud_hz, log_likelihood)
        Sorted by log_likelihood descending.  May be empty.
    """
    if len(x) < 256:
        return []

    max_baud = max_baud or fs / 2.0

    nl_cands = _nlac_baud(x, fs, min_baud, max_baud)
    scf_cands = _scf_baud(x, fs, min_baud, max_baud)
    zcr_cands = _zcr_baud(x, fs, min_baud, max_baud)

    all_cands: list[tuple[float, float]] = nl_cands + scf_cands + zcr_cands

    if not all_cands:
        return []

    # Cluster agreeing candidates: merge any two within fusion_tol of each other
    clusters: list[list[tuple[float, float]]] = []
    used = [False] * len(all_cands)

    for i, (fi, li) in enumerate(all_cands):
        if used[i]:
            continue
        cluster = [(fi, li)]
        for j, (fj, lj) in enumerate(all_cands):
            if j == i or used[j]:
                continue
            if abs(fi - fj) / (fi + 1e-12) <= fusion_tol:
                cluster.append((fj, lj))
                used[j] = True
        used[i] = True
        clusters.append(cluster)

    # Score each cluster: sum of log-likelihoods + bonus for multi-estimator agreement
    merged: list[tuple[float, float]] = []
    for cluster in clusters:
        freqs = [f for f, _ in cluster]
        lls = [ll for _, ll in cluster]
        f_mean = float(np.mean(freqs))
        ll_total = float(np.sum(lls))
        if len(cluster) >= 2:
            ll_total += 2.0  # Multi-estimator agreement bonus
        merged.append((f_mean, ll_total))

    merged.sort(key=lambda t: t[1], reverse=True)
    return merged[:top_k]
