"""JT9 decoder interface for FT8 and FT4 signals (WSJT-X suite)."""
from __future__ import annotations
import re
import shutil
import subprocess
import tempfile
import datetime as dt
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy import signal as sps
from scipy.io import wavfile

LINE_REGEX = re.compile(r"^\s*(\d{6})\s+(-?\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+~\s+(.*)$")

def iq_to_usb_audio(iq: np.ndarray, fs: float, lo: float = 100.0, hi: float = 3100.0, out_fs: int = 12000) -> np.ndarray:
    """Demodulate IQ to Upper Sideband (USB) audio for JT9 processing:
    1. Shift audio passband [lo, hi] to center
    2. Butterworth bandpass filter
    3. Re-shift to positive audio frequencies
    4. Rational polyphase resample to out_fs (12 kHz)
    5. Convert to int16 PCM
    """
    n = np.arange(len(iq))
    mid = 0.5 * (lo + hi)
    z = iq * np.exp(-2j * np.pi * mid / fs * n)
    sos = sps.butter(8, 0.5 * (hi - lo), fs=fs, output="sos")
    z = sps.sosfiltfilt(sos, z)
    z = z * np.exp(2j * np.pi * mid / fs * n)
    audio = np.real(z)
    
    r = Fraction(out_fs, int(round(fs))).limit_denominator(2000)
    audio_resampled = sps.resample_poly(audio, r.numerator, r.denominator)
    max_amp = np.max(np.abs(audio_resampled)) + 1e-12
    return (audio_resampled / max_amp * 0.8 * 32767.0).astype(np.int16)

def decode_ft8_slots(iq: np.ndarray, fs: float, t_start_utc: dt.datetime | None = None, fc_hz: float = 14074000.0):
    """Scan 15-second FT8 time slots and run jt9 -8.
    Yields GOLD annotations. NEVER stores decoded payload content (privacy hygiene).
    """
    jt9_path = shutil.which("jt9")
    if not jt9_path:
        return

    audio = iq_to_usb_audio(iq, fs)
    afs = 12000
    t_start = t_start_utc or dt.datetime.now(dt.timezone.utc)
    first_sec = (15 - (t_start.second % 15) - t_start.microsecond / 1e6) % 15
    s = int(first_sec * afs)

    while s + 15 * afs <= len(audio):
        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / "slot.wav"
            wavfile.write(str(wav_path), afs, audio[s:s + 15 * afs])
            try:
                out = subprocess.run(
                    [jt9_path, "-8", "-a", td, "-t", td, str(wav_path)],
                    capture_output=True, text=True, timeout=30
                ).stdout
                for line in out.splitlines():
                    m = LINE_REGEX.match(line)
                    if m:
                        snr = int(m.group(2))
                        dt_s = float(m.group(3))
                        f_audio = int(m.group(4))
                        t0 = s / afs + 0.5 + dt_s
                        yield {
                            "label_l3": "FT8",
                            "tier": "GOLD",
                            "snr_db": snr,
                            "t_lo": float(t0),
                            "t_hi": float(t0 + 12.64),
                            "f_lo": float(fc_hz + f_audio),
                            "f_hi": float(fc_hz + f_audio + 50.0),
                            "verified": True,
                        }
            except Exception:
                pass
        s += 15 * afs
