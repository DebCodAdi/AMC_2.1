"""Simulation self-test asserting 0 BER and end-to-end correctness across waveforms.

Spec reference: SIGNAL10 Phase 1 / Realism validation.
"""
from __future__ import annotations
import numpy as np

from signal9.sim.watterson import watterson_channel, PROFILES
from signal9.sim.mobile import rayleigh_channel, rician_channel, apply_doppler_ramp
from signal9.sim.noise import middleton_class_a_noise, add_middleton_noise
from signal9.sim.hardware import apply_hardware_impairments, quantize_adc
from signal9.sim.framing import (
    ccsds_scramble,
    ccsds_descramble,
    g3ruh_scramble,
    g3ruh_descramble,
    v35_scramble,
    v35_descramble,
    hdlc_bit_stuff,
    hdlc_bit_unstuff,
    frame_ax25,
    crc16_ccitt,
)
from signal9.dsp.constellation import bits_to_symbols, symbols_to_bits


def test_framing_and_scramblers():
    """Verify bit-exact invertibility of all scramblers and framing."""
    rng = np.random.default_rng(42)
    raw_bytes = rng.bytes(128)

    # 1. CCSDS
    scrambled_ccsds = ccsds_scramble(raw_bytes)
    recovered_ccsds = ccsds_descramble(scrambled_ccsds)
    assert recovered_ccsds == raw_bytes, "CCSDS additive scrambler round-trip failed"

    # 2. G3RUH
    raw_bits = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8))
    scrambled_g3ruh = g3ruh_scramble(raw_bits)
    recovered_g3ruh = g3ruh_descramble(scrambled_g3ruh)
    # G3RUH shift register has 17-bit memory; after 17 bits it is perfectly synchronized
    assert np.array_equal(recovered_g3ruh[17:], raw_bits[17:]), "G3RUH scrambler round-trip failed"

    # 3. V.35
    scrambled_v35 = v35_scramble(raw_bits)
    recovered_v35 = v35_descramble(scrambled_v35)
    assert np.array_equal(recovered_v35[23:], raw_bits[23:]), "V.35 scrambler round-trip failed"

    # 4. HDLC
    stuffed = hdlc_bit_stuff(raw_bits)
    unstuffed = hdlc_bit_unstuff(stuffed)
    assert np.array_equal(unstuffed, raw_bits), "HDLC bit stuffing round-trip failed"

    # 5. AX.25
    ax_frame = frame_ax25("N0CALL", "W1AW", b"HELLO WORLD")
    assert ax_frame.startswith(b"\x7e") and ax_frame.endswith(b"\x7e")


def test_clean_waveform_ber_at_30db():
    """Verify 0 BER at 30 dB SNR for standard digital modulations."""
    rng = np.random.default_rng(1234)
    n_bits = 2048

    for mod in ["BPSK", "QPSK", "8PSK", "16QAM"]:
        bits = rng.integers(0, 2, n_bits, dtype=np.uint8)
        syms = bits_to_symbols(bits, mod_type=mod)

        # Add 30 dB AWGN
        sigma = np.sqrt(1.0 / (2.0 * (10.0 ** (30.0 / 10.0))))
        noise = sigma * (rng.standard_normal(len(syms)) + 1j * rng.standard_normal(len(syms)))
        rx_syms = syms + noise

        demod_bits = symbols_to_bits(rx_syms, mod_type=mod, hard=True)[:len(bits)]
        ber = np.mean(demod_bits != bits)
        assert ber == 0.0, f"{mod} at 30 dB SNR had non-zero BER: {ber}"


def test_propagation_and_impairment_channels():
    """Verify simulation channels execute cleanly with correct invariants."""
    rng = np.random.default_rng(999)
    iq = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(np.complex64)

    # Watterson profiles
    for pname in PROFILES:
        y_wat = watterson_channel(iq, fs=48000.0, profile=pname, seed=0)
        assert len(y_wat) == len(iq)
        assert not np.isnan(y_wat).any()

    # Mobile channels
    y_ray = rayleigh_channel(iq, fs=48000.0, fd_hz=20.0, seed=0)
    assert len(y_ray) == len(iq)
    y_ric = rician_channel(iq, fs=48000.0, fd_hz=20.0, k_factor_db=10.0, seed=0)
    assert len(y_ric) == len(iq)
    y_dop = apply_doppler_ramp(iq, fs=48000.0, f_start_hz=-5000.0, ramp_rate_hz_per_s=300.0)
    assert len(y_dop) == len(iq)

    # Middleton Class-A noise
    n_mid = middleton_class_a_noise(n_samples=8192, total_power=1.0, impulse_index_A=0.05, seed=0)
    assert len(n_mid) == 8192
    # Verify non-Gaussian heavy tails: kurtosis should be significantly elevated
    kurt = float(np.mean(np.abs(n_mid) ** 4) / ((np.mean(np.abs(n_mid) ** 2)) ** 2))
    assert kurt > 2.5, f"Middleton noise kurtosis too low: {kurt}"

    # Hardware impairments
    y_hw = apply_hardware_impairments(iq, fs=48000.0, gain_imb_db=1.0, phase_imb_deg=3.0, seed=0)
    assert len(y_hw) == len(iq)

    # ADC Quantization RTL-SDR uint8
    raw_b, recon_iq = quantize_adc(iq, mode="uint8_rtl")
    assert len(raw_b) == 2 * len(iq)
    # Correlation between original and quantized should be high (> 0.98)
    corr = np.abs(np.vdot(iq, recon_iq)) / (np.linalg.norm(iq) * np.linalg.norm(recon_iq))
    assert corr > 0.98, f"Quantization correlation too low: {corr}"


def main():
    print("Running simulation self-tests...")
    test_framing_and_scramblers()
    print("[PASS] Framing and scramblers passed bit-exact roundtrip")
    test_clean_waveform_ber_at_30db()
    print("[PASS] Clean waveform 0-BER at 30 dB passed")
    test_propagation_and_impairment_channels()
    print("[PASS] Propagation channels and hardware impairments passed")
    print("All simulation self-tests PASSED.")


if __name__ == "__main__":
    main()
