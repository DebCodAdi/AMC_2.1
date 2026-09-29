"""Two-path Watterson HF ionospheric skywave channel model (ITU-R F.1487 / CCIR-520)."""
from __future__ import annotations
import numpy as np

WATTERSON_PRESETS = {
    "good": {"delay_ms": 0.5, "doppler_hz": 0.1, "gains_db": (0.0, -3.0)},
    "moderate": {"delay_ms": 1.0, "doppler_hz": 0.5, "gains_db": (0.0, 0.0)},
    "poor": {"delay_ms": 2.0, "doppler_hz": 1.0, "gains_db": (0.0, 0.0)},
    "flutter": {"delay_ms": 3.0, "doppler_hz": 5.0, "gains_db": (0.0, 0.0)},
}

def _gauss_doppler_process(n_samples: int, fs: float, doppler_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a complex Rayleigh fading process with Gaussian Doppler spectrum."""
    fs_f = max(20.0 * doppler_hz, 10.0)
    nf = int(np.ceil(n_samples / fs * fs_f)) + 16
    w = (rng.standard_normal(nf) + 1j * rng.standard_normal(nf)) / np.sqrt(2.0)
    sigma = max(doppler_hz / 2.0, 1e-3)
    freqs = np.fft.fftfreq(nf, 1.0 / fs_f)
    H = np.exp(-0.5 * (freqs / sigma) ** 2)
    tap = np.fft.ifft(np.fft.fft(w) * np.sqrt(H))
    tap /= np.sqrt(np.mean(np.abs(tap) ** 2) + 1e-12)
    
    # Interpolate to signal sampling rate
    t_sig = np.arange(n_samples) / fs
    t_fade = np.arange(nf) / fs_f
    tap_i = np.interp(t_sig, t_fade, tap.real) + 1j * np.interp(t_sig, t_fade, tap.imag)
    return tap_i

def watterson_channel(x: np.ndarray,
                      fs: float,
                      preset: str = "moderate",
                      delay_ms: float | None = None,
                      doppler_hz: float | None = None,
                      gains_db: tuple[float, float] | None = None,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply two-path Watterson HF channel with Gaussian Doppler fading.
    Power is normalized so that output expected power equals input power.
    """
    rng = rng or np.random.default_rng()
    spec = WATTERSON_PRESETS.get(preset, WATTERSON_PRESETS["moderate"])
    d_ms = delay_ms if delay_ms is not None else spec["delay_ms"]
    d_hz = doppler_hz if doppler_hz is not None else spec["doppler_hz"]
    g_db = gains_db if gains_db is not None else spec["gains_db"]

    delays = [0, int(round(d_ms * 1e-3 * fs))]
    y = np.zeros_like(x, dtype=np.complex128)
    
    for d, g in zip(delays, g_db):
        tap = _gauss_doppler_process(len(x), fs, d_hz, rng)
        xd = np.roll(x, d)
        if d > 0:
            xd[:d] = 0
        y += (10.0 ** (g / 20.0)) * tap * xd

    # Normalize total expected power
    norm_factor = np.sqrt(sum(10.0 ** (g / 10.0) for g in g_db))
    y = y / (norm_factor if norm_factor > 0 else 1.0)
    return y.astype(x.dtype)
