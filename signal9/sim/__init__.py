"""Realistic RF propagation channels, hardware front-end impairments, and framing.

Implements Signal 10 Phase 1 real-world simulation program:
  - Watterson HF ionospheric skywave model (ITU-R F.1487)
  - Rayleigh / Rician fading & LEO Doppler dynamic ramp
  - Middleton Class-A impulsive noise
  - Hardware impairments (LO leakage, IQ imbalance, ADC quantization, phase noise, AGC)
  - Standard framing & scramblers (CCSDS, G3RUH, V.35, AX.25)
"""
from __future__ import annotations

from .watterson import watterson_channel, WattersonProfile
from .mobile import rayleigh_channel, rician_channel, apply_doppler_ramp
from .noise import middleton_class_a_noise
from .hardware import apply_hardware_impairments, quantize_adc
from .framing import (
    ccsds_scramble,
    ccsds_descramble,
    g3ruh_scramble,
    g3ruh_descramble,
    v35_scramble,
    v35_descramble,
    frame_ccsds,
    frame_ax25,
    frame_ax25_bits,
)

__all__ = [
    "watterson_channel",
    "WattersonProfile",
    "rayleigh_channel",
    "rician_channel",
    "apply_doppler_ramp",
    "middleton_class_a_noise",
    "apply_hardware_impairments",
    "quantize_adc",
    "ccsds_scramble",
    "ccsds_descramble",
    "g3ruh_scramble",
    "g3ruh_descramble",
    "v35_scramble",
    "v35_descramble",
    "frame_ccsds",
    "frame_ax25",
    "frame_ax25_bits",
]
