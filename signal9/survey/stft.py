"""Multi-resolution STFT bank for wideband spectrum monitoring (Phase 3).

Provides simultaneous coarse (high time resolution) and fine (high frequency resolution)
spectrograms to capture both fast transient bursts and narrow continuous carriers.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal


@dataclass(frozen=True)
class SpectrogramData:
    f: np.ndarray        # Frequency bins in Hz
    t: np.ndarray        # Time points in seconds
    P_db: np.ndarray     # Log power in dB (shape: [len(f), len(t)])


@dataclass(frozen=True)
class MultiresolutionSpectrogram:
    coarse: SpectrogramData   # High time resolution (e.g. nfft=256)
    fine: SpectrogramData     # High frequency resolution (e.g. nfft=2048)
    psd_welch: np.ndarray     # Average PSD across capture in dB
    f_welch: np.ndarray       # PSD frequency axis


def _compute_stft(iq: np.ndarray,
                  fs: float,
                  nperseg: int,
                  noverlap: int,
                  window: str = "hann") -> SpectrogramData:
    f, t, Zxx = signal.stft(
        iq,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        boundary=None,
    )
    # FFT shift to center DC
    f = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(Zxx, axes=0)
    P_db = 10.0 * np.log10(np.abs(Zxx) ** 2 + 1e-15)
    return SpectrogramData(f=f, t=t, P_db=P_db.astype(np.float32))


def multiresolution_stft(iq: np.ndarray,
                         fs: float = 48000.0,
                         nfft_coarse: int = 256,
                         nfft_fine: int = 2048,
                         overlap_frac: float = 0.5) -> MultiresolutionSpectrogram:
    """Compute multi-resolution STFT bank over complex baseband IQ.

    Parameters
    ----------
    iq : 1D complex64 array
    fs : float
        Wideband sampling rate in Hz.
    nfft_coarse : int, default 256
        FFT size for fast burst/transient analysis.
    nfft_fine : int, default 2048
        FFT size for fine frequency channel resolution.
    overlap_frac : float, default 0.5

    Returns
    -------
    MultiresolutionSpectrogram
    """
    iq = np.asarray(iq, dtype=np.complex64).ravel()
    N = len(iq)
    if N < nfft_coarse:
        # Pad with zeros if necessary
        iq = np.pad(iq, (0, nfft_coarse - N))
        N = len(iq)

    actual_fine = min(nfft_fine, N)
    if actual_fine % 2 != 0:
        actual_fine -= 1

    overlap_coarse = int(nfft_coarse * overlap_frac)
    overlap_fine = int(actual_fine * overlap_frac)

    coarse = _compute_stft(iq, fs, nfft_coarse, overlap_coarse)
    fine = _compute_stft(iq, fs, actual_fine, overlap_fine)

    # Average Welch PSD
    f_w, p_w = signal.welch(iq, fs=fs, nperseg=actual_fine, return_onesided=False)
    f_w = np.fft.fftshift(f_w)
    p_w = np.fft.fftshift(p_w)
    psd_db = (10.0 * np.log10(p_w + 1e-15)).astype(np.float32)

    return MultiresolutionSpectrogram(
        coarse=coarse,
        fine=fine,
        psd_welch=psd_db,
        f_welch=f_w.astype(np.float32),
    )
