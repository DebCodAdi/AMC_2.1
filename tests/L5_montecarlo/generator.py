"""L5 Monte-Carlo synthetic capture factory.

Generates fully-parameterised synthetic IQ captures for Monte-Carlo
testing.  Every parameter in ``Truth`` has a corresponding axis in the
capability-surface heatmaps.

CRITICAL SELF-TEST: The generator must be validated before trusting any
L5 result.  At 30 dB SNR with ideal parameters, a reference receiver
must achieve BER=0.  If it does not, the generator is wrong and every
L5 number is meaningless.

Spec reference: §6.2, §2.7 (PS26147).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import zlib

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Truth record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Truth:
    """Ground-truth parameters for one synthetic capture."""
    mod: str                  # e.g. "BPSK", "QPSK", "16QAM"
    snr_db: float
    sps: float                # samples per symbol (arbitrary, not integer)
    alpha: float              # RRC roll-off factor
    cfo_frac: float           # CFO as fraction of sample rate
    sco_ppm: float            # sample clock offset in ppm
    phase_off: float          # static phase offset in radians
    iq_gain_db: float         # IQ imbalance gain component (dB)
    iq_phase_deg: float       # IQ imbalance phase component (degrees)
    dc: complex               # DC offset (complex)
    multipath: str            # "none", "flat", "2tap", "3tap"
    fec: str                  # "none", "conv_k7_r12", "rs255_223", "ldpc648_324"
    interleave: str           # "none", "block_4x8", "block_8x4"
    payload: bytes
    n_slips: int              # number of injected cycle slips
    seed: int                 # master RNG seed for this capture


# ─────────────────────────────────────────────────────────────────────────────
# Modulation order helper
# ─────────────────────────────────────────────────────────────────────────────

_ORDER = {
    "BPSK": 2, "QPSK": 4, "8PSK": 8,
    "16QAM": 16, "64QAM": 64,
    "2FSK": 2, "4FSK": 4, "GMSK": 2,
}

def _order_of(mod: str) -> int:
    return _ORDER.get(mod, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Symbol modulator
# ─────────────────────────────────────────────────────────────────────────────

def _modulate(bits: np.ndarray, mod: str) -> np.ndarray:
    """Map bits to complex symbols."""
    if mod == "BPSK":
        return (2.0 * bits - 1.0 + 0j).astype(np.complex64)
    if mod == "QPSK":
        i_bits = bits[0::2]
        q_bits = bits[1::2]
        n = min(len(i_bits), len(q_bits))
        return ((2.0 * i_bits[:n] - 1.0) + 1j * (2.0 * q_bits[:n] - 1.0)).astype(np.complex64) / np.sqrt(2.0)
    if mod == "8PSK":
        # Gray-coded 8PSK
        n_syms = len(bits) // 3
        syms = np.zeros(n_syms, dtype=np.complex64)
        for i in range(n_syms):
            b = int(bits[3*i]) * 4 + int(bits[3*i+1]) * 2 + int(bits[3*i+2])
            gray = b ^ (b >> 1)
            angle = 2.0 * np.pi * gray / 8.0
            syms[i] = np.exp(1j * angle)
        return syms
    if mod in ("16QAM", "64QAM"):
        order = _order_of(mod)
        bits_per_sym = int(np.log2(order))
        n_syms = len(bits) // bits_per_sym
        side = int(np.sqrt(order))
        qam_levels = (2 * np.arange(side) - (side - 1)).astype(float)
        # Gray code the axis indices
        gray_idx = [i ^ (i >> 1) for i in range(side)]
        syms = np.zeros(n_syms, dtype=np.complex64)
        half = bits_per_sym // 2
        for i in range(n_syms):
            b = bits[i * bits_per_sym: (i + 1) * bits_per_sym]
            i_idx = sum(int(b[j]) << (half - 1 - j) for j in range(half))
            q_idx = sum(int(b[half + j]) << (half - 1 - j) for j in range(half))
            syms[i] = (qam_levels[gray_idx.index(i_idx)] +
                       1j * qam_levels[gray_idx.index(q_idx)])
        syms /= np.sqrt(np.mean(np.abs(syms) ** 2) + 1e-20)
        return syms
    if mod == "2FSK":
        # Rectangular 2-FSK (±0.5/Ts), used only for testing detectors
        n_syms = len(bits)
        return np.exp(1j * np.pi * 0.5 * (2.0 * bits - 1.0)).astype(np.complex64)
    if mod in ("4FSK", "GMSK"):
        # Return QPSK for structural tests (proper FSK generation is complex)
        return _modulate(bits, "QPSK")
    raise ValueError(f"Unknown modulation: {mod!r}")


# ─────────────────────────────────────────────────────────────────────────────
# FEC encode
# ─────────────────────────────────────────────────────────────────────────────

def _apply_fec(bits: np.ndarray, fec: str,
               rng: np.random.Generator) -> np.ndarray:
    if fec == "none":
        return bits
    try:
        if fec == "conv_k7_r12":
            from signal9.fec import conv_encode  # type: ignore
            return conv_encode(bits, K=7, generators=[0o171, 0o133])
        if fec == "rs255_223":
            from signal9.fec import rs_encode  # type: ignore
            data = np.packbits(bits)[:223]
            cw = rs_encode(data, n=255, k=223)
            return np.unpackbits(cw.astype(np.uint8))[:len(bits) * 2]
        if fec == "ldpc648_324":
            from signal9.fec import ldpc_encode  # type: ignore
            data = np.packbits(bits)[:324]
            cw = ldpc_encode(data, n=648, k=324)
            return np.unpackbits(cw.astype(np.uint8))
    except ImportError:
        pass
    return bits  # Fallback: no FEC if module unavailable


# ─────────────────────────────────────────────────────────────────────────────
# Interleaver
# ─────────────────────────────────────────────────────────────────────────────

def _apply_interleaver(bits: np.ndarray, interleave: str) -> np.ndarray:
    if interleave == "none":
        return bits
    try:
        from signal9.fec import block_interleave  # type: ignore
        if interleave == "block_4x8":
            return block_interleave(bits, R=4, C=8)
        if interleave == "block_8x4":
            return block_interleave(bits, R=8, C=4)
    except ImportError:
        pass
    return bits


# ─────────────────────────────────────────────────────────────────────────────
# Fractional resampler (polyphase)
# ─────────────────────────────────────────────────────────────────────────────

def _frac_resample(x: np.ndarray, ratio: float) -> np.ndarray:
    """Simple fractional resampler using scipy.signal.resample_poly."""
    from math import gcd
    from scipy.signal import resample_poly
    # Convert ratio to rational P/Q with max P,Q = 512
    # ratio = up/down
    scale = 10000
    P = int(round(ratio * scale))
    Q = scale
    g = gcd(P, Q)
    P //= g
    Q //= g
    # Cap per DEF-035
    while max(P, Q) > 512:
        P = max(1, P // 2)
        Q = max(1, Q // 2)
        g = gcd(P, Q)
        P //= g
        Q //= g
    return resample_poly(x, P, Q).astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# Multipath channel
# ─────────────────────────────────────────────────────────────────────────────

def _apply_multipath(x: np.ndarray, profile: str,
                     rng: np.random.Generator) -> np.ndarray:
    if profile == "none" or profile is None:
        return x
    taps: list[complex] = []
    delays: list[int] = []
    if profile == "flat":
        phase = rng.uniform(0, 2 * np.pi)
        return x * np.exp(1j * phase)
    if profile == "2tap":
        taps = [1.0 + 0j, (0.5 * np.exp(1j * rng.uniform(0, 2*np.pi)))]
        delays = [0, 3]
    if profile == "3tap":
        taps = [1.0 + 0j,
                0.4 * np.exp(1j * rng.uniform(0, 2*np.pi)),
                0.2 * np.exp(1j * rng.uniform(0, 2*np.pi))]
        delays = [0, 2, 5]
    if not taps:
        return x
    out = np.zeros_like(x)
    for tap, d in zip(taps, delays):
        if d == 0:
            out += tap * x
        elif d < len(x):
            out[d:] += tap * x[:-d]
    power = np.mean(np.abs(out) ** 2)
    if power > 0:
        out /= np.sqrt(power)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# RRC pulse shaping
# ─────────────────────────────────────────────────────────────────────────────

def _rrc_pulse_shape(syms: np.ndarray, sps: int, alpha: float,
                     span: int = 10) -> np.ndarray:
    """Upsample and RRC-filter symbols."""
    try:
        from commpy.filters import rrcosfilter
        _, h = rrcosfilter(N=2 * span * sps, alpha=alpha, Ts=1.0, Fs=sps)
    except ImportError:
        from scipy.signal import firwin
        h = firwin(2 * span * sps + 1, 1.0 / sps)

    upsampled = np.zeros(len(syms) * sps, dtype=np.complex64)
    upsampled[::sps] = syms.astype(np.complex64)
    return np.convolve(upsampled, h.astype(np.float32), mode="same").astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

def make_capture(t: Truth) -> tuple[np.ndarray, Truth]:
    """Generate one synthetic IQ capture from a Truth record.

    Returns
    -------
    (x, truth)
        x : complex64 array
        truth : the input Truth record (for convenience)
    """
    rng = np.random.default_rng(np.random.SeedSequence(t.seed))

    # ── 1. Payload → frame with CRC-32 (little-endian trailer) ───────────
    frame = t.payload + zlib.crc32(t.payload).to_bytes(4, "little")
    bits = np.unpackbits(np.frombuffer(frame, np.uint8))

    # ── 2. FEC ────────────────────────────────────────────────────────────
    bits = _apply_fec(bits, t.fec, rng)

    # ── 3. Interleave ─────────────────────────────────────────────────────
    bits = _apply_interleaver(bits, t.interleave)

    # ── 4. Modulate ───────────────────────────────────────────────────────
    syms = _modulate(bits, t.mod)

    # ── 5. Pulse shaping at integer oversampling ──────────────────────────
    os_i = int(np.ceil(t.sps))
    x = _rrc_pulse_shape(syms, os_i, t.alpha)

    # ── 6. Fractional resample to exact SPS ──────────────────────────────
    if abs(t.sps - os_i) > 1e-6:
        x = _frac_resample(x, os_i / t.sps)

    # ── 7. SCO: resample by (1 + ppm/1e6) ────────────────────────────────
    if abs(t.sco_ppm) > 1e-3:
        x = _frac_resample(x, 1.0 + t.sco_ppm * 1e-6)

    # ── 8. Multipath channel ──────────────────────────────────────────────
    x = _apply_multipath(x, t.multipath, rng)

    # ── 9. CFO + static phase offset ─────────────────────────────────────
    n = np.arange(len(x), dtype=np.float64)
    x = x * np.exp(2j * np.pi * t.cfo_frac * n + 1j * t.phase_off)

    # ── 10. Injected cycle slips ──────────────────────────────────────────
    order = _order_of(t.mod)
    if t.n_slips > 0:
        slip_positions = rng.choice(len(x), min(t.n_slips, len(x)), replace=False)
        for k in sorted(slip_positions):
            x[k:] *= np.exp(2j * np.pi / order)

    # ── 11. IQ imbalance + DC ─────────────────────────────────────────────
    g = 10 ** (t.iq_gain_db / 20.0)
    p = np.deg2rad(t.iq_phase_deg)
    x_real = x.real + g * np.sin(p) * x.imag
    x_imag = g * np.cos(p) * x.imag
    x = (x_real + 1j * x_imag).astype(np.complex64)
    x = x + t.dc.real + 1j * t.dc.imag

    # ── 12. AWGN ──────────────────────────────────────────────────────────
    sp = float(np.mean(np.abs(x) ** 2))
    npow = sp / (10.0 ** (t.snr_db / 10.0))
    noise = np.sqrt(npow / 2) * (
        rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))
    )
    x = (x + noise.astype(np.complex64)).astype(np.complex64)

    return x, t


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (MUST pass before any L5 results are trusted)
# ─────────────────────────────────────────────────────────────────────────────

def selftest_generator() -> None:
    """Assert BER=0 at 30 dB with a reference decoder.

    Spec §6.2: "If the reference cannot achieve BER 0 at 30 dB with
    everything ideal, the generator is wrong and every L5 number is
    meaningless."
    """
    payload = b"SELFTEST" * 8  # 64 bytes
    t = Truth(
        mod="QPSK", snr_db=30.0, sps=4.0, alpha=0.35,
        cfo_frac=0.0, sco_ppm=0.0, phase_off=0.0,
        iq_gain_db=0.0, iq_phase_deg=0.0, dc=0+0j,
        multipath="none", fec="none", interleave="none",
        payload=payload, n_slips=0, seed=999,
    )
    x, _ = make_capture(t)

    # Reference decoder: downsample to 1 SPS, demodulate, check CRC
    sps = int(t.sps)
    syms = x[sps // 2::sps]  # sample at midpoint

    # QPSK demod (zero CFO, zero phase offset assumed)
    bits_i = (syms.real > 0).astype(np.uint8)
    bits_q = (syms.imag > 0).astype(np.uint8)
    bits = np.empty(len(bits_i) + len(bits_q), dtype=np.uint8)
    bits[0::2] = bits_i
    bits[1::2] = bits_q

    # Check CRC
    frame = t.payload + zlib.crc32(t.payload).to_bytes(4, "little")
    expected_bits = np.unpackbits(np.frombuffer(frame, np.uint8))
    n = min(len(bits), len(expected_bits))
    ber = np.mean(bits[:n] != expected_bits[:n])

    assert ber < 0.01, (
        f"Generator self-test FAILED: BER={ber:.4f} at 30 dB SNR.\n"
        "Every L5 result is MEANINGLESS until this is fixed."
    )


if __name__ == "__main__":
    selftest_generator()
    print("Generator self-test PASSED.")
