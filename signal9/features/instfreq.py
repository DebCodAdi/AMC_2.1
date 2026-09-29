"""Instantaneous frequency features for FSK tone counting."""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks

def fsk_tone_count(iq: np.ndarray, fs: float) -> tuple[int, float]:
    """Estimate number of FSK tones and spacing by finding peaks in the instantaneous frequency histogram."""
    # Matched LPF logic (basic moving average for smoothing phase)
    iq_smooth = np.convolve(iq, np.ones(5)/5, mode='same')
    
    phase = np.unwrap(np.angle(iq_smooth))
    inst_f = np.diff(phase) * (fs / (2 * np.pi))
    
    # Histogram of instantaneous frequency
    hist, bin_edges = np.histogram(inst_f, bins=100, density=True)
    
    # Smooth histogram slightly
    hist_smooth = np.convolve(hist, np.ones(3)/3, mode='same')
    
    # Find peaks in histogram
    peaks, _ = find_peaks(hist_smooth, prominence=np.max(hist_smooth)*0.1)
    
    tone_count = int(len(peaks))
    
    # Estimate spacing
    if tone_count >= 2:
        peak_freqs = bin_edges[peaks]
        spacing = float(np.mean(np.diff(peak_freqs)))
    else:
        spacing = 0.0
        
    return tone_count, spacing
