"""Semi-synthetic signal generator (Modem audio -> Analytic baseband -> Channel -> Impairments)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy.signal import hilbert, resample_poly
from fractions import Fraction
from signal9.augment import random_chain

def make_semisynth_sample(clean_audio_mono: np.ndarray,
                          audio_fs: float,
                          target_tier_fs: float = 12000.0,
                          band: str = "HF") -> np.ndarray:
    """Transform clean audio waveform into a realistic OTA RF capture:
    1. Hilbert transform to create analytic baseband
    2. Polyphase rational resample to tier rate
    3. Apply physical channel and hardware impairment chain
    """
    analytic = hilbert(clean_audio_mono.astype(np.float32))
    
    # Resample to target tier rate
    ratio = Fraction(int(target_tier_fs), int(round(audio_fs))).limit_denominator(2048)
    baseband = resample_poly(analytic, ratio.numerator, ratio.denominator).astype(np.complex64)
    
    # Apply full impairment chain
    impaired = random_chain(baseband, fs=target_tier_fs, sig_bw_hz=min(3000.0, target_tier_fs * 0.4), band=band)
    return impaired
