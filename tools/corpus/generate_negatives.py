"""Bucket F negative and unknown signal generator for open-set training (SIGNAL INFINITY §1.1, §1.3)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


NEGATIVE_CLASSES = [
    "noise", "empty", "cw", "chirp", "radar_pulse", "ofdm_sim", "spur"
]


def generate_negative_burst(kind: str,
                            n_samples: int = 4096,
                            fs: float = 48000.0,
                            seed: int | None = None) -> np.ndarray:
    """Generate out-of-class or non-communication signal."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs

    if kind == "empty":
        # Thermal noise floor at -40 dB
        return (1e-4 * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))).astype(np.complex64)

    elif kind == "cw":
        # Continuous wave unmodulated tone
        fc = float(rng.uniform(-0.4 * fs, 0.4 * fs))
        tone = np.exp(1j * (2 * np.pi * fc * t + rng.uniform(0, 2*np.pi)))
        noise = 0.05 * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
        return (tone + noise).astype(np.complex64)

    elif kind == "chirp":
        # Linear frequency modulated radar chirp
        f0 = float(rng.uniform(-0.3 * fs, 0.0))
        f1 = float(rng.uniform(0.1 * fs, 0.4 * fs))
        k = (f1 - f0) / (n_samples / fs)
        chirp = np.exp(1j * (2 * np.pi * (f0 * t + 0.5 * k * t**2)))
        noise = 0.05 * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
        return (chirp + noise).astype(np.complex64)

    elif kind == "radar_pulse":
        # Periodic high-power pulse
        signal = np.zeros(n_samples, dtype=np.complex64)
        pulse_len = int(rng.integers(64, 512))
        start = int(rng.integers(0, max(1, n_samples - pulse_len)))
        fc = float(rng.uniform(-0.3 * fs, 0.3 * fs))
        carrier = np.exp(1j * 2 * np.pi * fc * t[start : start + pulse_len])
        signal[start : start + pulse_len] = carrier
        noise = 0.05 * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
        return (signal + noise).astype(np.complex64)

    elif kind == "spur":
        # Hardware local oscillator leakage or ADC spurious tone
        spur = 0.8 * np.exp(1j * rng.uniform(0, 2*np.pi)) * np.ones(n_samples, dtype=np.complex64)
        noise = 0.1 * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
        return (spur + noise).astype(np.complex64)

    elif kind == "ofdm_sim":
        # Synthetic multicarrier OFDM burst (not single carrier)
        n_carriers = 64
        symbols = rng.choice([-1, 1], size=(n_samples // n_carriers, n_carriers)) + 1j * rng.choice([-1, 1], size=(n_samples // n_carriers, n_carriers))
        ofdm = np.fft.ifft(symbols, axis=1).flatten()
        if len(ofdm) < n_samples:
            ofdm = np.pad(ofdm, (0, n_samples - len(ofdm)))
        else:
            ofdm = ofdm[:n_samples]
        noise = 0.05 * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
        return (ofdm + noise).astype(np.complex64)

    else:
        # Standard Gaussian noise
        return (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)).astype(np.complex64)


def main():
    parser = argparse.ArgumentParser(description="Generate Bucket F negative and unknown signals")
    parser.add_argument("--n", type=int, default=1000, help="Number of negative samples to generate")
    parser.add_argument("--out", type=str, default="data/corpus/shards/F_neg", help="Output directory")
    args = parser.parse_args()

    out_p = Path(args.out)
    out_p.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    for i in range(args.n):
        kind = rng.choice(NEGATIVE_CLASSES)
        iq = generate_negative_burst(kind, seed=42 + i)
        fn = out_p / f"neg_{kind}_{i:05d}.raw"
        iq.tofile(fn)

    print(f"Generated {args.n} Bucket F negative samples in {out_p}.")


if __name__ == "__main__":
    main()
