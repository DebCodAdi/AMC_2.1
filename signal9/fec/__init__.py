# signal9/fec/__init__.py
"""Forward Error Correction (FEC) modules.

Includes:
  - Convolutional codec (conv_encode, viterbi_decode, puncture, depuncture, invert_ccsds_g2)
  - Reed-Solomon codec (rs_encode, rs_decode, to_dual_basis, from_dual_basis)
  - Block interleaver (block_interleave, block_deinterleave)
  - LDPC codec (ldpc_encode, ldpc_decode, ldpc_parity_check)
"""
from signal9.fec.interleaver import block_interleave, block_deinterleave
from signal9.fec.conv import (
    conv_encode,
    viterbi_decode,
    puncture,
    depuncture,
    invert_ccsds_g2,
)
from signal9.fec.rs import rs_encode, rs_decode, to_dual_basis, from_dual_basis
from signal9.fec.ldpc import ldpc_encode, ldpc_decode, ldpc_parity_check

__all__ = [
    "block_interleave",
    "block_deinterleave",
    "conv_encode",
    "viterbi_decode",
    "puncture",
    "depuncture",
    "invert_ccsds_g2",
    "rs_encode",
    "rs_decode",
    "to_dual_basis",
    "from_dual_basis",
    "ldpc_encode",
    "ldpc_decode",
    "ldpc_parity_check",
]
