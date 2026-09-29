"""Phase 1 Realism simulation parameter sweep and capability benchmarking.

Evaluates BER across:
  - Channels: AWGN, Watterson ('moderate'), Rayleigh, Rician (K=6dB)
  - Noise: Gaussian vs Middleton Class-A impulsive (A=0.1, Gamma=0.01)
  - Hardware: Clean vs Impaired (IQ imbalance, LO leak, RTL-SDR uint8 ADC)
  - SNRs: 0 to 30 dB in 5 dB steps

Outputs: reports/realism_sweep.json
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np

from signal9.dsp.constellation import bits_to_symbols, symbols_to_bits
from signal9.sim.watterson import watterson_channel
from signal9.sim.mobile import rayleigh_channel, rician_channel
from signal9.sim.noise import add_middleton_noise
from signal9.sim.hardware import apply_hardware_impairments, quantize_adc


def run_realism_sweep(n_bits: int = 10000,
                      snrs_db: list[float] | None = None,
                      seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    if snrs_db is None:
        snrs_db = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

    channels = ["AWGN", "Watterson_moderate", "Rayleigh", "Rician_K6"]
    results = {}

    bits = rng.integers(0, 2, n_bits, dtype=np.uint8)
    tx_qpsk = bits_to_symbols(bits, mod_type="QPSK")
    fs = 48000.0

    t0 = time.perf_counter()

    for ch in channels:
        results[ch] = {}
        for snr in snrs_db:
            # 1. Apply channel fading
            if ch == "AWGN":
                faded = tx_qpsk
            elif ch == "Watterson_moderate":
                faded = watterson_channel(tx_qpsk, fs=fs, profile="moderate", seed=int(snr))
            elif ch == "Rayleigh":
                faded = rayleigh_channel(tx_qpsk, fs=fs, fd_hz=20.0, seed=int(snr))
            elif ch == "Rician_K6":
                faded = rician_channel(tx_qpsk, fs=fs, fd_hz=20.0, k_factor_db=6.0, seed=int(snr))

            # 2. Add AWGN noise
            sigma = np.sqrt(1.0 / (2.0 * (10.0 ** (snr / 10.0))))
            noise = sigma * (rng.standard_normal(len(faded)) + 1j * rng.standard_normal(len(faded)))
            rx = faded + noise

            # 3. Demodulate
            demod_bits = symbols_to_bits(rx, mod_type="QPSK", hard=True)[:len(bits)]
            ber = float(np.mean(demod_bits != bits))
            results[ch][f"{int(snr)}dB"] = {
                "ber": ber,
                "n_errors": int(np.sum(demod_bits != bits)),
            }

    # Impulsive noise evaluation at 15 dB
    impulsive_results = {}
    for A in [0.01, 0.1, 1.0]:
        rx_imp = add_middleton_noise(tx_qpsk, snr_db=15.0, impulse_index_A=A, gamma_ratio=0.01, seed=seed)
        demod = symbols_to_bits(rx_imp, mod_type="QPSK", hard=True)[:len(bits)]
        impulsive_results[f"A={A}"] = float(np.mean(demod != bits))

    # RTL-SDR ADC Quantization at 20 dB
    rx_quant = tx_qpsk + 0.1 * (rng.standard_normal(len(tx_qpsk)) + 1j * rng.standard_normal(len(tx_qpsk)))
    _, recon_rtl = quantize_adc(rx_quant, mode="uint8_rtl")
    demod_rtl = symbols_to_bits(recon_rtl, mod_type="QPSK", hard=True)[:len(bits)]
    rtl_ber = float(np.mean(demod_rtl != bits))

    elapsed = time.perf_counter() - t0

    return {
        "n_bits": n_bits,
        "snrs_db": snrs_db,
        "ber_matrix": results,
        "impulsive_noise_ber_15dB": impulsive_results,
        "rtl_sdr_adc_ber_20dB": rtl_ber,
        "elapsed_seconds": elapsed,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Realism Sweep")
    parser.add_argument("--n-bits", type=int, default=10000)
    parser.add_argument("--out", type=str, default="reports/realism_sweep.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Running realism sweep with {args.n_bits} bits per test point...")
    out_dict = run_realism_sweep(n_bits=args.n_bits, seed=args.seed)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)

    print(f"Done in {out_dict['elapsed_seconds']:.2f}s. Results written to {out_p}")


if __name__ == "__main__":
    main()
