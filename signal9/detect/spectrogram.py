"""STFT log-power spectrogram computation."""
from __future__ import annotations
import numpy as np
from scipy import signal as sps

def log_spectrogram(x: np.ndarray, fs: float, nfft: int = 1024, hop: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the Blackman-Harris STFT log-power spectrogram.
    Returns (freqs_hz, times_sec, P_db).
    """
    hop = hop or (nfft // 2)
    f, t, S = sps.stft(
        x,
        fs=fs,
        nperseg=nfft,
        noverlap=nfft - hop,
        return_onesided=False,
        boundary=None,
        padded=False,
        window="blackmanharris"
    )
    # Shift zero frequency to center
    P = np.fft.fftshift(np.abs(S) ** 2, axes=0)
    f = np.fft.fftshift(f)
    P_db = 10.0 * np.log10(P + 1e-20)
    return f, t, P_db


def resize_spectrogram(P: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Area-average a 2-D spectrogram to ``(out_h, out_w)`` using NumPy only.

    The spectral CNN branch takes a fixed-size input while ``log_spectrogram`` is
    computed at the frequency resolution the analysis needs (nfft=1024), so the
    two have to be reconciled.  This is an integer-bucket area average -- the
    same operator as ``cv2.INTER_AREA`` for downscaling, which is the right
    choice for power spectrograms because it averages energy instead of point
    sampling it (nearest/bilinear sampling of a line spectrum can miss a narrow
    carrier entirely).  Implementing it here keeps OpenCV out of the runtime
    dependency set.
    """
    P = np.asarray(P, dtype=np.float32)
    if P.ndim != 2 or P.size == 0:
        return np.zeros((out_h, out_w), dtype=np.float32)

    h, w = P.shape
    # Bucket every source row/column into exactly one destination cell, then take
    # the mean per bucket.  np.add.reduceat needs monotonically increasing edges,
    # so upscaling falls back to nearest-neighbour index mapping.
    def _axis_mean(arr: np.ndarray, n_out: int, axis: int) -> np.ndarray:
        n_in = arr.shape[axis]
        if n_in == n_out:
            return arr
        if n_in < n_out:
            idx = np.clip((np.arange(n_out) * n_in) // n_out, 0, n_in - 1)
            return np.take(arr, idx, axis=axis)
        edges = (np.arange(n_out) * n_in) // n_out
        counts = np.diff(np.append(edges, n_in)).astype(np.float32)
        summed = np.add.reduceat(arr, edges, axis=axis)
        shape = [1] * arr.ndim
        shape[axis] = n_out
        return summed / counts.reshape(shape)

    out = _axis_mean(P, out_h, axis=0)
    out = _axis_mean(out, out_w, axis=1)
    return np.ascontiguousarray(out, dtype=np.float32)
