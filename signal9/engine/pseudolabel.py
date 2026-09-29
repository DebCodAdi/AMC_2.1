"""Oracle-verified pseudo-labelling enforcing Rule R16 (SIGNAL INFINITY §3.2).

Rule R16: Data enters the corpus with a label only if (a) human/externally labelled,
or (b) an independent oracle validates it. Model confidence alone is NEVER sufficient.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Admission:
    label: str | None
    weight: float
    verified_by: str
    unlabelled: bool = False


def admit_pseudo_label(prediction: dict, sample_iq: np.ndarray | None = None) -> Admission:
    """Admit pseudo-labels strictly through independent verification tiers (Rule R16)."""
    crc_valid = bool(prediction.get("crc_valid", False))
    protocol_valid = bool(prediction.get("protocol_valid", False))
    ext_decoder_agrees = bool(prediction.get("ext_decoder_agrees", False))
    branches_agree = int(prediction.get("n_branches_agreeing", 0))
    syncword = bool(prediction.get("syncword_found", False))
    evm = float(prediction.get("evm", 1.0))
    pred_label = prediction.get("label", "UNKNOWN")

    # TIER 1: Hard verification (CRC + Protocol or external oracle) -> weight 1.0
    if crc_valid and protocol_valid:
        return Admission(label=pred_label, weight=1.0, verified_by="crc+protocol")
    if ext_decoder_agrees:
        return Admission(label=pred_label, weight=1.0, verified_by="external_oracle")
    if crc_valid:
        return Admission(label=pred_label, weight=1.0, verified_by="crc32")

    # TIER 2: Structural multi-branch verification (independent of classifier head) -> weight 0.5
    if branches_agree >= 3 and syncword and evm < 0.15:
        return Admission(label=pred_label, weight=0.5, verified_by="multi_branch")

    # TIER 3: Physical consistency verification (cyclic line match) -> weight 0.3
    pred_symrate = float(prediction.get("symrate", 0.0))
    meas_symrate = float(prediction.get("measured_symrate", 0.0))
    if pred_symrate > 0 and meas_symrate > 0:
        if abs(pred_symrate - meas_symrate) / pred_symrate < 1e-3:
            return Admission(label=pred_label, weight=0.3, verified_by="physical_consistency")

    # NO ADMISSION on model confidence alone (Rule R16)
    return Admission(label=None, weight=0.0, verified_by="rejected_unverified", unlabelled=True)
