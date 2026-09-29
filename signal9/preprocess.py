"""Conservative conditioning immediately before modulation classification.

The detector/channelizer owns wideband corrections.  This module owns only
operations that are safe on an already isolated emitter.  In particular it
does *not* subtract the mean: after a target is mixed to baseband, DC can be a
real carrier (AM, CW, OOK), not merely receiver LO leakage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from .health.impulsive import detect_impulsive_noise, huber_winsorize


@dataclass(frozen=True)
class ConditioningAudit:
    """Machine-readable record of the transformations seen by the model."""
    finite_repaired: int = 0
    impulsive_noise: bool = False
    impulsive_kurtosis: float = 2.0
    impulsive_mitigation_applied: bool = False
    rms_before: float = 0.0
    rms_after: float = 0.0
    dc_preserved: bool = True
    equalization_recommended: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def condition_for_classifier(iq: np.ndarray) -> tuple[np.ndarray, ConditioningAudit]:
    """Repair capture artifacts and RMS-normalize one channelized emitter.

    Only clear impulsive contamination is winsorized.  Blind equalization is
    deliberately not applied here because the modulation is unknown at this
    point; it is recommended only for a constant-envelope, non-impulsive
    candidate and remains a later hypothesis-specific operation.
    """
    x = np.asarray(iq, dtype=np.complex64).ravel().copy()
    if x.size == 0:
        return x, ConditioningAudit()

    good = np.isfinite(x.real) & np.isfinite(x.imag)
    repaired = int(x.size - np.count_nonzero(good))
    if repaired:
        # NumPy supports a lexicographic complex median, but I and Q are
        # independent ADC coordinates.  Component-wise medians are the
        # meaningful robust replacement and are invariant to the other
        # component's ordering.
        replacement = (np.median(x[good].real) + 1j * np.median(x[good].imag)
                       if np.any(good) else 0.0j)
        x[~good] = replacement

    rms_before = float(np.sqrt(np.mean(np.abs(x) ** 2)))
    impulsive, kurtosis = detect_impulsive_noise(x)
    mitigated = False
    # Gaussian noise has complex kurtosis 2; this threshold keeps normal QAM
    # amplitude variation intact while rejecting isolated ADC/RF bursts.
    if impulsive:
        x = huber_winsorize(x)
        mitigated = True

    rms_after = float(np.sqrt(np.mean(np.abs(x) ** 2)))
    if rms_after > 1e-12:
        x /= rms_after

    mag2 = np.abs(x) ** 2
    papr_db = 10.0 * np.log10(float(np.max(mag2)) / (float(np.mean(mag2)) + 1e-12))
    env_kurtosis = float(np.mean(mag2 ** 2)) / (float(np.mean(mag2)) ** 2 + 1e-12)
    # A recommendation, never a blind mutation: FSK/CPM may also have a
    # constant envelope, so modulation-specific recovery makes the decision.
    equalizer_ok = bool(not impulsive and env_kurtosis < 1.15 and papr_db < 2.5)
    audit = ConditioningAudit(
        finite_repaired=repaired,
        impulsive_noise=bool(impulsive),
        impulsive_kurtosis=float(kurtosis),
        impulsive_mitigation_applied=mitigated,
        rms_before=rms_before,
        rms_after=rms_after,
        equalization_recommended=equalizer_ok,
    )
    return x.astype(np.complex64), audit
