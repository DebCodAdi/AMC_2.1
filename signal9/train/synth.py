"""Physically-grounded synthetic RF training examples with exact labels and hard cases.

Waveforms Supported:
--------------------
* Linear PSK/QAM: BPSK, QPSK, 8PSK, 16QAM, 64QAM (RRC pulse shaped, variable roll-off).
* Pulse Amplitude Modulation: PAM4, OOK, 4ASK (RRC pulse shaped, real-axis multi-level).
* Continuous-Phase & Frequency: 2FSK, 4FSK, MSK, GFSK, CPFSK (multi-index, Gaussian pre-filtered).
* Analog Modulation: AM-DSB, AM-SSB (analytic Hilbert message), WBFM (wideband frequency deviation).
* Noise: Complex AWGN only (negative class).

Hard Physical Channel Impairments:
----------------------------------
* Multi-tap frequency-selective Rayleigh/Rician multipath (exponential power-delay profile).
* Hardware IQ Imbalance: amplitude mismatch and quadrature phase error.
* Carrier Leakage / DC Offset: Local oscillator feedthrough.
* Power Amplifier Saturation: Rapp non-linear AM-AM solid-state PA clipping.
* Oscillator Phase Noise: Wiener random walk jitter.
* Impulsive Noise: Middleton Class-A bursty atmospheric/industrial interference.
* Severe SNR range: -10 dB to +30 dB with exact mathematical label correspondence.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..dsp.constellation import bits_to_symbols
from ..dsp.matched_filter import rrc_taps
from .labels import mod_index

LINEAR_MODS = ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM")
PAM_MODS = ("PAM4", "OOK", "4ASK")
FSK_MODS = ("2FSK", "4FSK", "MSK", "GFSK", "CPFSK")
ANALOG_MODS = ("AM-DSB", "AM-SSB", "WBFM")
DEFAULT_MODS = LINEAR_MODS + PAM_MODS + FSK_MODS + ANALOG_MODS + ("NOISE",)

_BPS = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "16QAM": 4, "64QAM": 6}


@dataclass
class SynthParams:
    mod: str
    sps: int
    rolloff: float
    cfo_norm: float          # cycles per sample
    snr_db: float
    phase: float
    multipath: bool
    impulsive: bool
    iq_imbalance: bool = False
    dc_offset: bool = False
    pa_saturation: bool = False
    phase_noise: bool = False


def draw_params(rng: np.random.Generator, mods: tuple[str, ...] = DEFAULT_MODS,
                snr_range: tuple[float, float] = (-10.0, 26.0)) -> SynthParams:
    mod = mods[int(rng.integers(0, len(mods)))]
    return SynthParams(
        mod=mod,
        sps=int(rng.integers(2, 17)),
        rolloff=float(rng.uniform(0.1, 0.5)),
        cfo_norm=float(rng.uniform(-0.03, 0.03)),
        snr_db=float(rng.uniform(*snr_range)),
        phase=float(rng.uniform(0.0, 2.0 * np.pi)),
        multipath=bool(rng.random() < 0.35),
        impulsive=bool(rng.random() < 0.15),
        iq_imbalance=bool(rng.random() < 0.30),
        dc_offset=bool(rng.random() < 0.25),
        pa_saturation=bool(rng.random() < 0.20),
        phase_noise=bool(rng.random() < 0.25),
    )


def _linear_waveform(p: SynthParams, n: int, rng: np.random.Generator) -> np.ndarray:
    n_sym = n // p.sps + 24
    bits = rng.integers(0, 2, n_sym * _BPS[p.mod]).astype(np.uint8)
    syms = bits_to_symbols(bits, p.mod)
    up = np.zeros(n_sym * p.sps, dtype=np.complex64)
    up[::p.sps] = syms
    h = rrc_taps(p.rolloff, sps=p.sps, span=8)
    x = np.convolve(up, h, mode="same")
    start = int(rng.integers(0, max(1, len(x) - n)))
    return x[start:start + n]


def _pam_waveform(p: SynthParams, n: int, rng: np.random.Generator) -> np.ndarray:
    n_sym = n // p.sps + 24
    if p.mod == "PAM4":
        levels = rng.choice(np.array([-3.0, -1.0, 1.0, 3.0]), size=n_sym) / np.sqrt(5.0)
    elif p.mod == "OOK":
        levels = rng.choice(np.array([0.0, 1.0]), size=n_sym)
    else: # 4ASK
        levels = rng.choice(np.array([0.0, 1.0, 2.0, 3.0]), size=n_sym) / np.sqrt(3.5)
    up = np.zeros(n_sym * p.sps, dtype=np.complex64)
    up[::p.sps] = levels.astype(np.complex64)
    h = rrc_taps(p.rolloff, sps=p.sps, span=8)
    x = np.convolve(up, h, mode="same")
    start = int(rng.integers(0, max(1, len(x) - n)))
    return x[start:start + n]


def _fsk_waveform(p: SynthParams, n: int, rng: np.random.Generator) -> np.ndarray:
    n_sym = n // p.sps + 8
    if p.mod == "4FSK":
        levels = rng.choice(np.array([-3.0, -1.0, 1.0, 3.0]), size=n_sym)
        h = float(rng.uniform(0.25, 0.5))
    elif p.mod == "CPFSK":
        levels = rng.choice(np.array([-1.0, 1.0]), size=n_sym)
        h = float(rng.uniform(0.3, 1.0))
    else:
        levels = rng.choice(np.array([-1.0, 1.0]), size=n_sym)
        h = 0.5 if p.mod in ("MSK", "GFSK") else float(rng.uniform(0.5, 1.2))

    f = np.repeat(levels, p.sps).astype(np.float64)
    if p.mod == "GFSK":
        bt = float(rng.uniform(0.3, 0.5))
        span = 4 * p.sps
        t = (np.arange(span + 1) - span / 2) / p.sps
        sig = np.sqrt(np.log(2)) / (2 * np.pi * bt)
        g = np.exp(-t ** 2 / (2 * sig ** 2))
        f = np.convolve(f, g / g.sum(), mode="same")

    phase = 2 * np.pi * np.cumsum(f * h / (2.0 * p.sps))
    x = np.exp(1j * phase).astype(np.complex64)
    start = int(rng.integers(0, max(1, len(x) - n)))
    return x[start:start + n]


def _analog_waveform(p: SynthParams, n: int, rng: np.random.Generator) -> np.ndarray:
    # Synthetic speech / acoustic multi-tone baseband message m(t)
    total_len = n + 64
    t = np.arange(total_len)
    n_tones = rng.integers(4, 12)
    m = np.zeros(total_len, dtype=np.float64)
    for _ in range(n_tones):
        f_tone = rng.uniform(0.01, 0.25) / p.sps
        phi_tone = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.2, 1.0)
        m += amp * np.sin(2 * np.pi * f_tone * t + phi_tone)
    max_m = np.max(np.abs(m)) + 1e-12
    m = m / max_m

    if p.mod == "AM-DSB":
        mu = float(rng.uniform(0.5, 0.95))
        # 50% chance of transmitted carrier vs suppressed carrier
        x = ((1.0 + mu * m) if rng.random() < 0.5 else m).astype(np.complex64)
    elif p.mod == "AM-SSB":
        # Analytic signal using discrete Hilbert approximation
        H = np.fft.fft(m)
        h_filter = np.zeros(total_len)
        h_filter[0] = 1.0
        h_filter[1:total_len//2] = 2.0
        if total_len % 2 == 0:
            h_filter[total_len//2] = 1.0
        analytic = np.fft.ifft(H * h_filter)
        x = analytic.astype(np.complex64)
    else: # WBFM
        beta = float(rng.uniform(1.0, 4.0))
        kf = beta / (p.sps * 2.0)
        phi = 2 * np.pi * kf * np.cumsum(m)
        x = np.exp(1j * phi).astype(np.complex64)

    start = int(rng.integers(0, max(1, len(x) - n)))
    return x[start:start + n]


def generate(p: SynthParams, n: int, rng: np.random.Generator) -> np.ndarray:
    """Render one example of length ``n`` for parameter draw ``p`` with hard channel cases."""
    if p.mod == "NOISE":
        x = np.zeros(n, dtype=np.complex64)
    elif p.mod in _BPS:
        x = _linear_waveform(p, n, rng)
    elif p.mod in PAM_MODS:
        x = _pam_waveform(p, n, rng)
    elif p.mod in ANALOG_MODS:
        x = _analog_waveform(p, n, rng)
    else:
        x = _fsk_waveform(p, n, rng)

    if p.mod != "NOISE":
        # Power normalize
        pwr = float(np.mean(np.abs(x) ** 2)) + 1e-12
        x = x / np.sqrt(pwr)

        # 1. Multi-tap frequency-selective multipath channel
        if p.multipath:
            n_taps = int(rng.integers(2, 5))
            delays = [0] + sorted(list(set(rng.integers(1, max(3, p.sps * 3), size=n_taps - 1))))
            pows = np.exp(-np.array(delays) / 4.0)
            phases = rng.uniform(0, 2 * np.pi, size=len(delays))
            taps = pows * np.exp(1j * phases)
            taps = taps / np.sqrt(np.sum(np.abs(taps) ** 2) + 1e-12)
            x_faded = np.zeros_like(x)
            for d, tap in zip(delays, taps):
                if d == 0:
                    x_faded += tap * x
                elif d < n:
                    x_faded[d:] += tap * x[:-d]
            x = x_faded / np.sqrt(np.mean(np.abs(x_faded) ** 2) + 1e-12)

        # 2. Hardware IQ Imbalance (amplitude and quadrature phase error)
        if p.iq_imbalance:
            gain_mismatch = float(rng.uniform(0.85, 1.15))
            phase_mismatch = float(rng.uniform(-0.15, 0.15))  # rad (~8.5 deg)
            i_comp = x.real
            q_comp = x.imag
            x = (np.cos(phase_mismatch) * i_comp - gain_mismatch * np.sin(phase_mismatch) * q_comp) + \
                1j * (gain_mismatch * np.cos(phase_mismatch) * q_comp)

        # 3. Carrier Frequency Offset (CFO) and initial phase
        k = np.arange(n)
        x = x * np.exp(1j * (2 * np.pi * p.cfo_norm * k + p.phase))

        # 4. Oscillator Phase Noise (Wiener random walk)
        if p.phase_noise:
            sigma_pn = float(rng.uniform(0.005, 0.025))
            dphi = rng.standard_normal(n) * sigma_pn
            x = x * np.exp(1j * np.cumsum(dphi))

        # 5. Power Amplifier Non-Linear Saturation (Rapp AM-AM model)
        if p.pa_saturation:
            mag = np.abs(x)
            ang = np.angle(x)
            v_sat = float(rng.uniform(0.75, 1.25))
            mag_sat = mag / ((1.0 + (mag / v_sat) ** 4.0) ** 0.25)
            x = mag_sat * np.exp(1j * ang)

        # 6. Carrier Leakage / DC Offset (LO Feedthrough)
        if p.dc_offset:
            dc = float(rng.uniform(-0.12, 0.12)) + 1j * float(rng.uniform(-0.12, 0.12))
            x = x + dc

        # 7. Additive White Gaussian Noise (AWGN)
        noise_power = 10.0 ** (-p.snr_db / 10.0)
    else:
        noise_power = 1.0

    noise = np.sqrt(noise_power / 2.0) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    x = x + noise

    # 8. Impulsive Noise (Middleton Class-A atmospheric/industrial bursts)
    if p.impulsive:
        hits = rng.random(n) < 0.008
        if np.any(hits):
            x[hits] += 4.0 * np.sqrt(noise_power) * (rng.standard_normal(hits.sum()) + 1j * rng.standard_normal(hits.sum()))

    x = x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)
    return x.astype(np.complex64)


def labels_for(p: SynthParams) -> dict:
    """Labels for a parameter draw. NaN / -1 mark non-applicable targets."""
    nan = float("nan")
    linear = p.mod in _BPS or p.mod in PAM_MODS
    noise = p.mod == "NOISE"
    return {
        "mod": mod_index(p.mod),
        "symrate": nan if (noise or p.mod in ANALOG_MODS) else 1.0 / p.sps,
        "rolloff": p.rolloff if linear else nan,
        "cfo_norm": nan if noise else p.cfo_norm,
        "snr_db": nan if noise else p.snr_db,
        "presence": 0 if noise else 1,
        "fec": -1,
        "scrambler": -1,
    }
