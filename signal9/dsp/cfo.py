"""Carrier Frequency Offset (CFO) estimation and correction.

Implements:
  - M-th power CFO estimator with parabolic FFT peak interpolation (KAT-CFO-1).
  - Fine frequency estimator (Luise-Reggiannini / Kay) to resolve residual CFO (DEF-002).
  - Multi-hypothesis CFO scoring and pre-classification (DEF-001).
  - FSK instantaneous-frequency midpoint CFO estimator (DEF-001).

Spec reference: DEF-001, DEF-002, IV-C (PS26147).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CFOHypothesis:
    """Hypothesis of modulation order M and estimated CFO in Hz."""
    M: int
    cfo_hz: float
    sharpness_db: float
    mod_family: str  # 'linear' or 'fsk'


def mth_power_cfo(x: np.ndarray,
                  fs: float,
                  M: int = 4,
                  nfft: int | None = None) -> float:
    """Estimate Carrier Frequency Offset using M-th power non-linearity.

    For an M-PSK (or QAM corner symbols) signal:
      x[n] = s[n] * exp(j * (2*pi*f_cfo*n/fs + phi0)) + w[n]
    Raising to M-th power wipes off the modulation (s[n]^M is real/constant),
    leaving a spectral tone at M * f_cfo.

    Parabolic interpolation of the FFT peak achieves sub-bin resolution,
    satisfying the < 0.1 Hz accuracy requirement of KAT-CFO-1.

    Parameters
    ----------
    x : complex array
        Baseband signal samples.
    fs : float
        Sampling frequency in Hz.
    M : int
        Modulation order (2 for BPSK, 4 for QPSK/QAM, 8 for 8PSK).
    nfft : int, optional
        FFT size. Defaults to max(65536, next power of 2 >= len(x)).

    Returns
    -------
    float
        Estimated carrier frequency offset in Hz.
    """
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)
    if N < 16:
        return 0.0

    # 1. Non-linear M-th power
    z = x ** M

    # 2. Windowed FFT
    if nfft is None:
        nfft = max(65536, 1 << int(np.ceil(np.log2(max(N, 1024)))))

    # Hann window to minimize spectral leakage
    w = np.hanning(N)
    zw = z * w
    Z = np.fft.fft(zw, n=nfft)
    spec = np.abs(Z) ** 2

    # Peak search
    k_peak = int(np.argmax(spec))

    # Parabolic (quadratic) peak interpolation in log/magnitude space
    # delta = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
    k_left = (k_peak - 1) % nfft
    k_right = (k_peak + 1) % nfft

    alpha = np.log(spec[k_left] + 1e-30)
    beta = np.log(spec[k_peak] + 1e-30)
    gamma = np.log(spec[k_right] + 1e-30)

    denom = alpha - 2.0 * beta + gamma
    if abs(denom) > 1e-12:
        delta = 0.5 * (alpha - gamma) / denom
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0

    k_est = (k_peak + delta)
    if k_est >= nfft / 2.0:
        k_est -= nfft

    freq_tone = (k_est / nfft) * fs
    cfo_est = float(freq_tone / M)

    # Disambiguate Fs/M alias ambiguity using spectral centroid (DEF-002)
    if M > 1:
        N_sub = min(N, 4096)
        f_axis = np.fft.fftfreq(N_sub, 1.0 / fs)
        P_sub = np.abs(np.fft.fft(x[:N_sub] * np.hanning(N_sub))) ** 2
        f_coarse = float(np.sum(f_axis * P_sub) / (np.sum(P_sub) + 1e-30))

        k_range = range(-M, M + 1)
        candidates = [cfo_est + k * (fs / M) for k in k_range if abs(cfo_est + k * (fs / M)) < fs / 2.0]
        if candidates:
            cfo_est = float(min(candidates, key=lambda fc: abs(fc - f_coarse)))

    return cfo_est



def score_line_sharpness(z: np.ndarray, nfft: int = 16384) -> tuple[float, float]:
    """Score the spectral line sharpness (peak-to-median ratio in dB) of z.

    Returns
    -------
    (peak_freq_bin, sharpness_db)
    """
    N = len(z)
    w = np.hanning(N)
    Z = np.fft.fft(z * w, n=nfft)
    mag2 = np.abs(Z) ** 2
    peak_val = np.max(mag2)
    median_val = np.median(mag2) + 1e-30
    sharpness_db = 10.0 * np.log10(peak_val / median_val)
    k_peak = int(np.argmax(mag2))
    return float(k_peak), float(sharpness_db)


def estimate_cfo_hypotheses(x: np.ndarray,
                            fs: float) -> list[CFOHypothesis]:
    """Multi-hypothesis CFO estimation per DEF-001.

    1. Pre-test: constant-modulus & instantaneous-frequency bimodality.
    2. Linear modulations: test M in {2, 4, 8} and score by spectral line sharpness.
    3. FSK: midpoint of instantaneous-frequency histogram.
    """
    x = np.asarray(x, dtype=np.complex128)
    if len(x) < 32:
        return [CFOHypothesis(M=4, cfo_hz=0.0, sharpness_db=0.0, mod_family="linear")]

    # Constant modulus check
    mag = np.abs(x)
    cm_ratio = float(np.std(mag) / (np.mean(mag) + 1e-12))

    # Instantaneous frequency: angle difference between consecutive samples
    phase_diff = np.angle(x[1:] * np.conj(x[:-1]))
    freq_inst = phase_diff * (fs / (2.0 * np.pi))

    hypotheses: list[CFOHypothesis] = []

    # Check FSK hypothesis: bimodality of freq_inst
    p25, p75 = np.percentile(freq_inst, [25, 75])
    fsk_midpoint = 0.5 * (p25 + p75)
    fsk_sharpness = float(10.0 * np.log10(max(1.0, abs(p75 - p25) / (np.std(freq_inst) + 1e-6))))
    hypotheses.append(CFOHypothesis(
        M=2,
        cfo_hz=float(fsk_midpoint),
        sharpness_db=fsk_sharpness,
        mod_family="fsk"
    ))

    # Linear hypotheses: M in {2, 4, 8}
    for M in (2, 4, 8):
        cfo_val = mth_power_cfo(x, fs, M=M)
        # Score line sharpness
        z = x ** M
        _, s_db = score_line_sharpness(z)
        hypotheses.append(CFOHypothesis(
            M=M,
            cfo_hz=cfo_val,
            sharpness_db=s_db,
            mod_family="linear"
        ))

    # Sort linear hypotheses by sharpness descending
    linear_hyps = sorted([h for h in hypotheses if h.mod_family == "linear"],
                         key=lambda h: h.sharpness_db,
                         reverse=True)

    # If top linear hypothesis exceeds second by >6 dB, prune
    pruned = [linear_hyps[0]]
    if len(linear_hyps) > 1 and (linear_hyps[0].sharpness_db - linear_hyps[1].sharpness_db) <= 6.0:
        pruned.append(linear_hyps[1])

    # Include FSK if cm_ratio is small (FM/FSK has near-constant modulus)
    if cm_ratio < 0.25:
        fsk_h = [h for h in hypotheses if h.mod_family == "fsk"]
        pruned.extend(fsk_h)

    return pruned


def fine_cfo_luise_reggiannini(symbols: np.ndarray,
                                rs: float,
                                M: int = 4,
                                N: int | None = None) -> float:
    """Fine carrier frequency offset estimation using Luise-Reggiannini algorithm.

    Resolves residual CFO after coarse wipe-off (DEF-002) below 1e-4 * Rs.

    Parameters
    ----------
    symbols : complex array
        Decimated symbol stream.
    rs : float
        Symbol rate in Baud.
    M : int
        Modulation order (4 for QPSK).
    N : int, optional
        Integration length. Defaults to min(len(symbols), 512).

    Returns
    -------
    float
        Fine CFO in Hz.
    """
    L = min(len(symbols), 512 if N is None else N)
    if L < 4:
        return 0.0

    z = (symbols[:L] ** M)
    # Autocorrelation R(m) for lags m = 1..M_lag
    M_lag = min(L // 2, 32)
    R = np.zeros(M_lag, dtype=np.complex128)
    for m in range(1, M_lag + 1):
        R[m - 1] = np.sum(z[m:] * np.conj(z[:-m]))

    # Luise-Reggiannini weighting: w_m = 3*(L - m)*(L - m + 1) - M_lag*(L - M_lag) ...
    # Standard LR estimator sum: arg(sum_{m=1}^{M_lag} R(m))
    sum_R = np.sum(R)
    omega_fine = np.angle(sum_R) / (M * (M_lag + 1) / 2.0)
    fine_hz = float(omega_fine * rs / (2.0 * np.pi))
    return fine_hz


def correct_cfo(x: np.ndarray, fs: float, cfo_hz: float) -> np.ndarray:
    """Correct Carrier Frequency Offset by mixing with exp(-j * 2*pi*f*t)."""
    n = np.arange(len(x), dtype=np.float64)
    rot = np.exp(-2j * np.pi * (cfo_hz / fs) * n).astype(np.complex64)
    return (x.astype(np.complex64) * rot).astype(np.complex64)
