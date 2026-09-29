"""L15 Real-World Channel and Impairment Validation Test Suite (SIGNAL10 Phase 1).

Spec reference: Phase 1 (Realism Program), R1, R5, R14.
"""
from __future__ import annotations
import pytest
import numpy as np

from signal9.sim.watterson import watterson_channel, PROFILES, WattersonProfile
from signal9.sim.mobile import rayleigh_channel, rician_channel, apply_doppler_ramp
from signal9.sim.noise import middleton_class_a_noise, add_middleton_noise
from signal9.sim.hardware import (
    apply_iq_imbalance,
    apply_lo_leakage,
    apply_phase_noise,
    apply_spurs,
    apply_agc_steps,
    apply_sample_dropouts,
    quantize_adc,
    apply_hardware_impairments,
)
from signal9.sim.framing import (
    ccsds_scramble,
    ccsds_descramble,
    g3ruh_scramble,
    g3ruh_descramble,
    v35_scramble,
    v35_descramble,
    hdlc_bit_stuff,
    hdlc_bit_unstuff,
    frame_ccsds,
    frame_ax25,
    crc16_ccitt,
)
from signal9.dsp.constellation import bits_to_symbols, symbols_to_bits


class TestWattersonChannel:
    @pytest.mark.parametrize("profile_name", list(PROFILES.keys()))
    def test_itu_profiles_stability(self, profile_name):
        rng = np.random.default_rng(42)
        x = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(np.complex64)
        y = watterson_channel(x, fs=48000.0, profile=profile_name, seed=1)
        assert len(y) == len(x)
        assert not np.isnan(y).any()
        assert not np.isinf(y).any()
        # Conserves power within 3 dB tolerance
        p_in = np.mean(np.abs(x) ** 2)
        p_out = np.mean(np.abs(y) ** 2)
        ratio_db = 10.0 * np.log10(p_out / p_in)
        assert abs(ratio_db) < 3.0


class TestMobileChannels:
    def test_rayleigh_fading_properties(self):
        rng = np.random.default_rng(42)
        x = np.ones(16384, dtype=np.complex64)
        y = rayleigh_channel(x, fs=48000.0, fd_hz=50.0, seed=2)
        assert len(y) == len(x)
        # Envelope of y should have Rayleigh-like distribution (mean / std ratio ~ 1.91)
        env = np.abs(y)
        ratio = np.mean(env) / (np.std(env) + 1e-9)
        assert 1.2 < ratio < 2.5

    def test_rician_k_factor(self):
        rng = np.random.default_rng(42)
        x = np.ones(8192, dtype=np.complex64)
        # Low K (mostly diffuse) vs High K (mostly specular)
        y_low = rician_channel(x, fs=48000.0, fd_hz=50.0, k_factor_db=-10.0, seed=3)
        y_high = rician_channel(x, fs=48000.0, fd_hz=50.0, k_factor_db=20.0, seed=3)
        var_low = np.var(np.abs(y_low))
        var_high = np.var(np.abs(y_high))
        assert var_high < var_low

    def test_doppler_ramp_rate(self):
        fs = 48000.0
        x = np.ones(48000, dtype=np.complex64)  # 1 second of CW
        # 350 Hz/s ramp starting at 0 Hz
        y = apply_doppler_ramp(x, fs=fs, f_start_hz=0.0, ramp_rate_hz_per_s=350.0)
        # Instantaneous frequency at t=0.5s should be ~175 Hz
        phase = np.unwrap(np.angle(y))
        inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
        assert abs(inst_freq[len(inst_freq) // 2] - 175.0) < 5.0


class TestMiddletonNoise:
    def test_heavy_tailed_kurtosis(self):
        # A = 0.05 gives sparse strong impulses -> high kurtosis
        n = middleton_class_a_noise(16384, total_power=1.0, impulse_index_A=0.05, seed=4)
        kurt = float(np.mean(np.abs(n) ** 4) / ((np.mean(np.abs(n) ** 2)) ** 2))
        assert kurt > 3.0  # Gaussian complex kurtosis is 2.0; heavy tail is > 3.0

    def test_gaussian_limit(self):
        # A = 100 approaches Gaussian -> kurtosis near 2.0
        n = middleton_class_a_noise(16384, total_power=1.0, impulse_index_A=100.0, seed=5)
        kurt = float(np.mean(np.abs(n) ** 4) / ((np.mean(np.abs(n) ** 2)) ** 2))
        assert abs(kurt - 2.0) < 0.3


class TestHardwareImpairments:
    def test_rtl_sdr_quantization(self):
        rng = np.random.default_rng(42)
        x = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(np.complex64)
        raw_b, recon = quantize_adc(x, mode="uint8_rtl")
        assert raw_b.dtype == np.uint8
        assert 0 <= np.min(raw_b) <= np.max(raw_b) <= 255
        # 127.5 offset should place mean of zero-mean noise near 127/128
        assert 120 < np.mean(raw_b) < 135

    def test_phase_noise_and_spurs(self):
        fs = 48000.0
        x = np.ones(4096, dtype=np.complex64)
        y_pn = apply_phase_noise(x, phase_noise_std_deg=2.0, seed=6)
        assert np.isclose(np.mean(np.abs(y_pn)), 1.0, atol=1e-3)
        y_spur = apply_spurs(x, fs=fs, spur_offsets_hz=[1000.0], spur_level_db=-20.0)
        assert len(y_spur) == len(x)


class TestFramingAndScramblers:
    def test_ccsds_exact(self):
        msg = b"MISSION DATA PACKET 12345678"
        framed = frame_ccsds(msg, apply_scrambler=True)
        assert framed[:4] == bytes.fromhex("1acffc1d")
        descrambled = ccsds_descramble(framed[4:])
        assert descrambled == msg

    def test_ax25_framing(self):
        frame = frame_ax25("APRS", "W3XYZ", b"Test Packet", dest_ssid=0, src_ssid=1)
        assert frame[0] == 0x7E and frame[-1] == 0x7E
        # Address fields are 14 bytes, control 1, pid 1, data 11, crc 2, flags 2 = 31 bytes
        assert len(frame) >= 30
