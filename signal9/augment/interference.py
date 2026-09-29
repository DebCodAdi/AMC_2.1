"""Interference injection: adjacent-channel donor signals, spurs, and birdies."""
from __future__ import annotations
import numpy as np

def inject_donor_interferer(x: np.ndarray,
                            donor: np.ndarray,
                            sir_db: float = 10.0,
                            freq_offset_hz: float = 0.0,
                            fs: float = 1.0) -> np.ndarray:
    """Inject a donor signal as adjacent or co-channel interference at specified SIR (Signal-to-Interference Ratio)."""
    if len(donor) < len(x):
        donor = np.tile(donor, int(np.ceil(len(x) / len(donor))))
    donor_seg = donor[:len(x)]
    
    # Frequency shift donor if requested
    if freq_offset_hz != 0.0:
        n = np.arange(len(x))
        donor_seg = donor_seg * np.exp(2j * np.pi * freq_offset_hz / fs * n)
        
    p_sig = np.mean(np.abs(x) ** 2)
    p_donor = np.mean(np.abs(donor_seg) ** 2) + 1e-12
    desired_p_donor = p_sig / (10.0 ** (sir_db / 10.0))
    
    scaled_donor = donor_seg * np.sqrt(desired_p_donor / p_donor)
    return (x + scaled_donor).astype(x.dtype)

def add_spur(x: np.ndarray,
             fs: float,
             spur_freq_hz: float,
             spur_sir_db: float = 15.0) -> np.ndarray:
    """Inject a continuous CW spur / receiver birdie at specified offset."""
    n = np.arange(len(x))
    spur = np.exp(2j * np.pi * spur_freq_hz / fs * n)
    p_sig = np.mean(np.abs(x) ** 2)
    desired_spur_power = p_sig / (10.0 ** (spur_sir_db / 10.0))
    return (x + spur * np.sqrt(desired_spur_power)).astype(x.dtype)
