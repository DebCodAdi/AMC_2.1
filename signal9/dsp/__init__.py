# signal9/dsp/__init__.py
"""Digital signal processing modules for Signal9.

Modules:
  matched_filter  — RRC filter construction and roll-off estimation (IV-B)
  cfo             — CFO estimation (M-th power, Luise-Reggiannini) (DEF-001, DEF-002)
  constellation   — Gray mapping and constellation demapping (KAT-GRAY-1, DEF-017)
  differential    — Track H differential demapping (IV-L / DEF-003, DEF-024)
  timing_recovery — Closed-loop Gardner timing recovery with Farrow (DEF-012)
  snr_estimate    — Three-way NDA SNR estimation (IV-F)
  equalizer       — CMA/MMA equaliser with divergence guard (IV-D / DEF-036, DEF-037)
  gf2             — GF(2) linear algebra and Soft-GJETP rank kernels (Stage 6.5 / IV-I)
"""
from signal9.dsp.matched_filter import rrc_taps, estimate_rolloff, apply_rrc
from signal9.dsp.cfo import (
    mth_power_cfo,
    estimate_cfo_hypotheses,
    fine_cfo_luise_reggiannini,
    correct_cfo,
)
from signal9.dsp.constellation import bits_to_symbols, symbols_to_bits, gray_map
from signal9.dsp.differential import differential_demodulate, segmented_rotation_search
from signal9.dsp.timing_recovery import gardner_timing_recovery
from signal9.dsp.snr_estimate import estimate_snr, estimate_snr_m2m4, estimate_snr_ssme, estimate_snr_svr
from signal9.dsp.equalizer import adaptive_equalize
from signal9.dsp.gf2 import gf2_rank, gf2_nullspace, gf2_rank_soft, gf2_mat_mul

__all__ = [
    "rrc_taps",
    "estimate_rolloff",
    "apply_rrc",
    "mth_power_cfo",
    "estimate_cfo_hypotheses",
    "fine_cfo_luise_reggiannini",
    "correct_cfo",
    "bits_to_symbols",
    "symbols_to_bits",
    "gray_map",
    "differential_demodulate",
    "segmented_rotation_search",
    "gardner_timing_recovery",
    "estimate_snr",
    "estimate_snr_m2m4",
    "estimate_snr_ssme",
    "estimate_snr_svr",
    "adaptive_equalize",
    "gf2_rank",
    "gf2_nullspace",
    "gf2_rank_soft",
    "gf2_mat_mul",
]
