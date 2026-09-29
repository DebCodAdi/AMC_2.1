"""
Hardware and channel impairment simulation matching Blind_Signal_Classifier.
"""
from .augment import (
    random_chain,
    watterson_channel,
    rician_channel,
    rapp_pa,
    adc_quantize,
    add_awgn_snr,
    inject_noise_bank
)

__all__ = [
    "random_chain",
    "watterson_channel",
    "rician_channel",
    "rapp_pa",
    "adc_quantize",
    "add_awgn_snr",
    "inject_noise_bank"
]
