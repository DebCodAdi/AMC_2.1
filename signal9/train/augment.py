"""Online RF physical impairment augmentations with weak/strong pairs (SIGNAL INFINITY §2.3)."""
from __future__ import annotations
import numpy as np


def quantize_signed_8bit(iq: np.ndarray) -> np.ndarray:
    """Quantize complex IQ through a symmetric signed 8-bit ADC model.

    The gain maps the largest I/Q component to full scale.  Saturation occurs
    in integer code space; undoing that gain must not impose an unrelated
    ``[-1, 1]`` limit on the returned physical samples.
    """
    x = np.asarray(iq)
    if x.size == 0:
        return x.astype(np.complex64)
    peak_component = max(float(np.max(np.abs(x.real))),
                         float(np.max(np.abs(x.imag))), 1e-6)
    scale = 127.0 / peak_component
    i_codes = np.clip(np.round(x.real * scale), -127.0, 127.0)
    q_codes = np.clip(np.round(x.imag * scale), -127.0, 127.0)
    return (i_codes / scale + 1j * q_codes / scale).astype(np.complex64)


class OnlineRFAugmenter:
    """Online physical impairment augmentation for baseband IQ signals."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def weak_augment(self, iq: np.ndarray) -> np.ndarray:
        """Weak augmentation: random phase rotation and gentle gain scaling."""
        phase = self.rng.uniform(0, 2 * np.pi)
        gain = 10.0 ** (self.rng.uniform(-3.0, 3.0) / 20.0)
        return (iq * np.exp(1j * phase) * gain).astype(np.complex64)

    def strong_augment(self, iq: np.ndarray, fs: float = 48000.0) -> np.ndarray:
        """Strong augmentation: phase, CFO, SCO, Watterson multipath, Middleton noise, ADC quantization."""
        out = iq.copy()
        N = len(out)
        t = np.arange(N) / fs

        # 1. Random carrier frequency offset (CFO)
        cfo = float(self.rng.uniform(-0.1 * fs, 0.1 * fs))
        out = out * np.exp(1j * 2 * np.pi * cfo * t)

        # 2. Random phase rotation
        phase = float(self.rng.uniform(0, 2 * np.pi))
        out = out * np.exp(1j * phase)

        # 3. Simple multipath 2-tap Rayleigh fade
        if self.rng.random() < 0.7:
            delay = int(self.rng.integers(1, 8))
            alpha = float(self.rng.uniform(0.1, 0.5)) * np.exp(1j * self.rng.uniform(0, 2 * np.pi))
            echo = np.roll(out, delay) * alpha
            out = out + echo

        # 4. Middleton Class-A impulsive noise mixture
        if self.rng.random() < 0.5:
            # Add occasional heavy-tailed impulsive hits
            impulse_mask = self.rng.random(N) < 0.01
            n_imp = np.sum(impulse_mask)
            if n_imp > 0:
                impulse_noise = 3.0 * (self.rng.standard_normal(n_imp) + 1j * self.rng.standard_normal(n_imp))
                out[impulse_mask] += impulse_noise

        # 5. AWGN background noise
        snr_db = float(self.rng.uniform(0.0, 25.0))
        sigma = np.sqrt(1.0 / (2.0 * (10.0 ** (snr_db / 10.0))))
        noise = sigma * (self.rng.standard_normal(N) + 1j * self.rng.standard_normal(N))
        out = out + noise

        # 6. RTL-SDR 8-bit quantization with DC offset
        if self.rng.random() < 0.4:
            out = quantize_signed_8bit(out)

        # 7. Hardware IQ Imbalance (amplitude mismatch & quadrature phase error)
        if self.rng.random() < 0.35:
            gain_mismatch = float(self.rng.uniform(0.85, 1.15))
            phase_mismatch = float(self.rng.uniform(-0.15, 0.15))
            i_comp = out.real
            q_comp = out.imag
            out = (np.cos(phase_mismatch) * i_comp - gain_mismatch * np.sin(phase_mismatch) * q_comp) + \
                  1j * (gain_mismatch * np.cos(phase_mismatch) * q_comp)

        # 8. Carrier Leakage / DC Offset (LO feedthrough)
        if self.rng.random() < 0.25:
            dc = float(self.rng.uniform(-0.08, 0.08)) + 1j * float(self.rng.uniform(-0.08, 0.08))
            out = out + dc

        # 9. Power Amplifier Non-Linear Saturation (Rapp AM-AM clipping)
        if self.rng.random() < 0.20:
            mag = np.abs(out)
            ang = np.angle(out)
            v_sat = float(self.rng.uniform(0.75, 1.25))
            mag_sat = mag / ((1.0 + (mag / v_sat) ** 4.0) ** 0.25)
            out = mag_sat * np.exp(1j * ang)

        return out.astype(np.complex64)
