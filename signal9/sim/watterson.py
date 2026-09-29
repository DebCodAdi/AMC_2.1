"""Watterson HF ionospheric skywave channel simulation (ITU-R F.1487).

Models high-frequency (HF, 3-30 MHz) ionospheric multi-path propagation
using tapped delay lines with bi-Gaussian Doppler fading on each path.

Reference: ITU-R Recommendation F.1487, 'Testing of HF modems on
ionospheric simulators'.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal


@dataclass(frozen=True)
class WattersonPath:
    """Single magneto-ionic path parameters."""
    delay_ms: float
    gain_linear: float = 1.0
    doppler_spread_hz: float = 1.0  # 2-sigma Doppler spread
    freq_shift_hz: float = 0.0      # Frequency offset / ionospheric drift


@dataclass(frozen=True)
class WattersonProfile:
    """Standard ITU-R F.1487 profiles."""
    name: str
    paths: tuple[WattersonPath, ...]


PROFILES: dict[str, WattersonProfile] = {
    "quiet": WattersonProfile(
        name="quiet",
        paths=(
            WattersonPath(delay_ms=0.0, gain_linear=1.0, doppler_spread_hz=0.1, freq_shift_hz=0.0),
            WattersonPath(delay_ms=0.5, gain_linear=0.7, doppler_spread_hz=0.1, freq_shift_hz=0.05),
        )
    ),
    "moderate": WattersonProfile(
        name="moderate",
        paths=(
            WattersonPath(delay_ms=0.0, gain_linear=1.0, doppler_spread_hz=1.0, freq_shift_hz=0.0),
            WattersonPath(delay_ms=1.0, gain_linear=0.7, doppler_spread_hz=1.0, freq_shift_hz=0.2),
        )
    ),
    "disturbed": WattersonProfile(
        name="disturbed",
        paths=(
            WattersonPath(delay_ms=0.0, gain_linear=1.0, doppler_spread_hz=2.0, freq_shift_hz=0.0),
            WattersonPath(delay_ms=2.0, gain_linear=0.5, doppler_spread_hz=2.0, freq_shift_hz=0.5),
        )
    ),
    "flutter": WattersonProfile(
        name="flutter",
        paths=(
            WattersonPath(delay_ms=0.0, gain_linear=1.0, doppler_spread_hz=10.0, freq_shift_hz=0.0),
            WattersonPath(delay_ms=3.0, gain_linear=0.5, doppler_spread_hz=10.0, freq_shift_hz=1.0),
        )
    ),
}


def _generate_doppler_process(n_samples: int,
                              fs: float,
                              doppler_spread_hz: float,
                              rng: np.random.Generator) -> np.ndarray:
    """Generate complex Gaussian fading process with Gaussian Doppler spectrum."""
    if doppler_spread_hz <= 0 or n_samples == 0:
        return np.ones(n_samples, dtype=np.complex64)

    # Standard deviation in normalized frequency
    # Doppler spread is defined as 2*sigma
    sigma_f = (doppler_spread_hz / 2.0) / fs
    cutoff = min(0.49, max(1e-5, 3.0 * sigma_f))

    # Design Gaussian lowpass filter or FIR
    # Filter white complex Gaussian noise
    w_raw = (rng.standard_normal(n_samples + 256) + 1j * rng.standard_normal(n_samples + 256)).astype(np.complex64)
    
    # Gaussian filter kernel
    filt_len = min(255, max(15, int(fs / max(0.1, doppler_spread_hz) * 0.5)))
    if filt_len % 2 == 0:
        filt_len += 1
    t = np.arange(-(filt_len // 2), filt_len // 2 + 1)
    kernel = np.exp(-0.5 * (t / max(1.0, (filt_len / 6.0))) ** 2)
    kernel /= np.sum(kernel)

    faded = signal.convolve(w_raw, kernel, mode="same")
    faded = faded[128 : 128 + n_samples]
    
    # Normalize unit power
    power = np.mean(np.abs(faded) ** 2)
    if power > 0:
        faded /= np.sqrt(power)
    return faded.astype(np.complex64)


def watterson_channel(iq: np.ndarray,
                      fs: float = 48000.0,
                      profile: str | WattersonProfile = "moderate",
                      seed: int | None = None) -> np.ndarray:
    """Apply Watterson HF ionospheric skywave channel to complex baseband IQ.

    Parameters
    ----------
    iq : np.ndarray
        Complex baseband signal.
    fs : float
        Sample rate in Hz.
    profile : str or WattersonProfile
        Standard profile name ('quiet', 'moderate', 'disturbed', 'flutter')
        or custom WattersonProfile.
    seed : int, optional
        Seed for reproducible fading generation.

    Returns
    -------
    y : np.ndarray
        Faded output signal with identical length as iq.
    """
    iq = np.asarray(iq, dtype=np.complex64).ravel()
    N = len(iq)
    if N == 0:
        return iq

    if isinstance(profile, str):
        if profile not in PROFILES:
            raise ValueError(f"Unknown Watterson profile {profile!r}. Available: {list(PROFILES.keys())}")
        prof = PROFILES[profile]
    else:
        prof = profile

    rng = np.random.default_rng(seed)
    y = np.zeros(N, dtype=np.complex64)

    t = np.arange(N, dtype=np.float64) / fs

    for p in prof.paths:
        delay_samples = int(np.round(p.delay_ms * 1e-3 * fs))
        
        # Two magneto-ionic components (O and X waves)
        g1 = _generate_doppler_process(N, fs, p.doppler_spread_hz, rng)
        g2 = _generate_doppler_process(N, fs, p.doppler_spread_hz, rng)

        # Apply differential frequency shift
        if p.freq_shift_hz != 0:
            c1 = np.exp(1j * 2.0 * np.pi * p.freq_shift_hz * t).astype(np.complex64)
            c2 = np.exp(-1j * 2.0 * np.pi * p.freq_shift_hz * t).astype(np.complex64)
            h_p = (g1 * c1 + g2 * c2) / np.sqrt(2.0)
        else:
            h_p = (g1 + g2) / np.sqrt(2.0)

        # Delayed signal
        if delay_samples < N:
            if delay_samples == 0:
                s_delayed = iq
            else:
                s_delayed = np.r_[np.zeros(delay_samples, dtype=np.complex64), iq[:-delay_samples]]
            
            y += (p.gain_linear * h_p * s_delayed).astype(np.complex64)

    # Normalize total energy to conserve average input power
    p_in = np.mean(np.abs(iq) ** 2) + 1e-12
    p_out = np.mean(np.abs(y) ** 2) + 1e-12
    y *= np.sqrt(p_in / p_out)

    return y.astype(np.complex64)
