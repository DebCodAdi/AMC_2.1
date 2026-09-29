"""VHF/UHF Multipath Fading (Rayleigh and Rician with Jakes Doppler spectrum)."""
from __future__ import annotations
import numpy as np

def jakes_fading_process(n_samples: int, fs: float, max_doppler_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Generate Rayleigh fading process with classical Jakes (U-shaped) Doppler spectrum."""
    if max_doppler_hz <= 0.1:
        return np.ones(n_samples, dtype=np.complex64)

    fs_f = max(20.0 * max_doppler_hz, 20.0)
    nf = int(np.ceil(n_samples / fs * fs_f)) + 16
    w = (rng.standard_normal(nf) + 1j * rng.standard_normal(nf)) / np.sqrt(2.0)
    freqs = np.fft.fftfreq(nf, 1.0 / fs_f)
    
    # Jakes spectrum: S(f) ~ 1 / sqrt(1 - (f/fd)^2) for |f| < fd
    ratio = np.clip(np.abs(freqs) / max_doppler_hz, 0.0, 0.999)
    H = np.where(np.abs(freqs) <= max_doppler_hz, 1.0 / np.sqrt(1.0 - ratio ** 2 + 1e-4), 0.0)
    
    tap = np.fft.ifft(np.fft.fft(w) * np.sqrt(H))
    tap /= np.sqrt(np.mean(np.abs(tap) ** 2) + 1e-12)

    t_sig = np.arange(n_samples) / fs
    t_fade = np.arange(nf) / fs_f
    tap_i = np.interp(t_sig, t_fade, tap.real) + 1j * np.interp(t_sig, t_fade, tap.imag)
    return tap_i.astype(np.complex64)

def rician_channel(x: np.ndarray,
                   fs: float,
                   k_factor_db: float = 6.0,
                   max_doppler_hz: float = 10.0,
                   delay_samples: list[int] | None = None,
                   tap_gains_db: list[float] | None = None,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply Rician (or Rayleigh if k_factor_db -> -inf) multipath fading channel."""
    rng = rng or np.random.default_rng()
    delays = delay_samples or [0, 5, 12]
    gains = tap_gains_db or [0.0, -3.0, -6.0]

    k_lin = 10.0 ** (k_factor_db / 10.0) if k_factor_db > -40.0 else 0.0
    los_amp = np.sqrt(k_lin / (k_lin + 1.0))
    scat_amp = np.sqrt(1.0 / (k_lin + 1.0))

    y = np.zeros_like(x, dtype=np.complex128)
    for idx, (d, g_db) in enumerate(zip(delays, gains)):
        fade = jakes_fading_process(len(x), fs, max_doppler_hz, rng)
        # Tap 0 has LOS component if Rician
        if idx == 0 and k_lin > 0:
            tap_total = los_amp + scat_amp * fade
        else:
            tap_total = fade
        
        xd = np.roll(x, d)
        if d > 0:
            xd[:d] = 0
        y += (10.0 ** (g_db / 20.0)) * tap_total * xd

    norm = np.sqrt(sum(10.0 ** (g / 10.0) for g in gains))
    y = y / (norm if norm > 0 else 1.0)
    return y.astype(x.dtype)
