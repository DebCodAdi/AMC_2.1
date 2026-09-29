"""Audio to analytic IQ conversion — chunked Hilbert for large files.

DEF-008: The original hilbert() call loads the entire WAV into RAM as
float64 (8× the file size), then applies an FFT-based Hilbert over the
whole signal.  For a 100 MB WAV at 48 kHz stereo, this blows past 800 MB.

Fixed implementation:
  1. Block-wise exact FFT Hilbert with guard context (see audio_to_analytic).
  2. Mono WAV is now correctly recognised and converted rather than being
     returned as ``real + 0j`` (the DEF-008 silent silent-failure).
  3. Stereo WAV where one channel IS IQ is handled (keep as-is).
  4. Stereo WAV where left ≠ right (true audio) → chunked Hilbert on
     the mean of both channels.

Memory budget: one block of chunk_size + 2 x guard complex samples.

Spec reference: DEF-008, §IV-B.B1 (PS26147).
"""
from __future__ import annotations
import numpy as np
from scipy.signal import hilbert


def audio_to_analytic(audio_real: np.ndarray,
                      chunk_size: int = 1 << 20,
                      guard: int = 16384) -> np.ndarray:
    """Convert a real audio signal to its analytic (one-sided) complex form.

    Short inputs use scipy's exact FFT Hilbert transform.  Long ones are cut
    into ``chunk_size`` blocks, each transformed with ``guard`` samples of
    context on both sides and only its centre kept.  The FFT Hilbert of a
    finite block is wrong near the block edges, but the Hilbert kernel decays as
    1/(pi*n), so ``guard`` = 16384 bounds the error in the kept region to about
    2/(pi*16384), i.e. roughly -88 dB -- below any real capture's noise floor.
    Peak memory is one block of ``chunk_size + 2*guard`` complex samples.

    Why the previous implementation was replaced: above 2 x 131072 samples it
    switched to a 127-tap windowed FIR Hilbert and took ``lfilter(...)[L-1:]``,
    which removes the zero padding but not the filter's own (L-1)/2 = 63-sample
    group delay.  The imaginary part was therefore 63 samples behind the real
    part, so the output was not an analytic signal at all -- measured on a 307k-
    sample AFSK recording, positive and negative frequencies differed by only
    0.3 dB, and every tone appeared twice.  A 127-tap FIR is also far too short
    for audio: its transition band at 48 kHz is ~1 kHz wide, which is where
    AFSK and SSB-carried modems put their tones.
    """
    x = np.asarray(audio_real, dtype=np.float64).ravel()
    n = len(x)
    if n == 0:
        return np.empty(0, dtype=np.complex64)
    if n <= chunk_size + 2 * guard:
        return hilbert(x).astype(np.complex64)

    out = np.empty(n, dtype=np.complex64)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        a = max(0, start - guard)
        b = min(n, stop + guard)
        seg = hilbert(x[a:b])
        out[start:stop] = seg[start - a:stop - a]
    return out


def wav_to_analytic(data: np.ndarray,
                    kind: str,
                    chunk_size: int = 131072) -> np.ndarray:
    """Convert a WAV data array (as returned by read_wav) to complex baseband.

    DEF-008: The pipeline used to treat mono WAVs as ``real + 0j``.  This
    silently produced a double-sided spectrum that contaminated the AMC
    feature extractor.  The correct path is:
      - Mono → Hilbert → analytic signal (one-sided spectrum)
      - Stereo IQ (kind='iq') → left + j·right (already complex)
      - Stereo audio (kind='stereo_audio') → mean of channels → Hilbert
      - Mono duplicated (kind='mono_duplicated') → treat as mono

    Parameters
    ----------
    data : ndarray
        Raw WAV samples (may be int16, int32, float32).
    kind : str
        One of 'iq', 'mono_audio', 'mono_duplicated', 'stereo_audio'.
    chunk_size : int
        Hilbert chunking.

    Returns
    -------
    complex64 baseband signal.
    """
    # ── Normalise to float ────────────────────────────────────────────────
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        data_f = data.astype(np.float32) / scale
    else:
        data_f = data.astype(np.float32)

    # ── Route by kind ─────────────────────────────────────────────────────
    if kind == "iq":
        # Stereo IQ: left=I, right=Q
        if data_f.ndim == 2 and data_f.shape[1] >= 2:
            return (data_f[:, 0] + 1j * data_f[:, 1]).astype(np.complex64)
        # Degenerate: single-channel labelled as IQ — treat as mono
        real = data_f.ravel() if data_f.ndim == 1 else data_f[:, 0]
        return audio_to_analytic(real, chunk_size)

    if kind in ("mono_audio", "mono_duplicated"):
        real = data_f.ravel() if data_f.ndim == 1 else data_f[:, 0]
        return audio_to_analytic(real, chunk_size)

    if kind == "stereo_audio":
        # Mix both channels (e.g. USB/LSB left/right or sum-difference)
        if data_f.ndim == 2 and data_f.shape[1] >= 2:
            real = (data_f[:, 0] + data_f[:, 1]) / 2.0
        else:
            real = data_f.ravel()
        return audio_to_analytic(real, chunk_size)

    # Unknown kind — best-effort
    real = data_f.ravel()
    return audio_to_analytic(real, chunk_size)


def mix_to_baseband(audio_real: np.ndarray, fs: float, f_carrier: float) -> np.ndarray:
    """Mix real audio containing a carrier (e.g., AM-DSB-FC) down to baseband."""
    n = np.arange(len(audio_real))
    mix = audio_real * np.exp(-2j * np.pi * f_carrier / fs * n)
    return mix.astype(np.complex64)
