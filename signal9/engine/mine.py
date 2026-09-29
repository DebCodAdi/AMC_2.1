"""Failure miner classifying predictions into the 9 failure modes (SIGNAL INFINITY §3.1)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import numpy as np


@dataclass
class FailureRecord:
    sample_id: str
    failure_mode: str       # 'F-A', 'F-B', 'F-C', 'F-D', 'F-E', 'F-F', 'F-G', 'F-H', 'F-I'
    pred_mod: str
    true_mod: str
    confidence: float
    crc_valid: bool
    plausibility: float
    latency_ms: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_outcome(outcome: dict[str, Any]) -> FailureRecord | None:
    """Classify model outcome into failure taxonomy mode."""
    pred_mod = outcome.get("pred_mod", "UNKNOWN")
    true_mod = outcome.get("true_mod", "UNKNOWN")
    conf = float(outcome.get("confidence", 0.0))
    crc = bool(outcome.get("crc_valid", False))
    plaus = float(outcome.get("plausibility", 0.0))
    lat = float(outcome.get("latency_ms", 0.0))
    is_neg = bool(outcome.get("is_negative", False))
    sample_id = outcome.get("sample_id", "unknown")

    # F-H: Slow (> 1500 ms)
    if lat > 1500.0:
        return FailureRecord(sample_id, "F-H", pred_mod, true_mod, conf, crc, plaus, lat, "Latency budget violation")

    # F-E: Unknown-not-rejected (Bucket F sample got confident known label)
    if is_neg and conf > 0.7 and pred_mod not in ("UNKNOWN", "NOISE", "UNKNOWN_MODULATION"):
        return FailureRecord(sample_id, "F-E", pred_mod, true_mod, conf, crc, plaus, lat, "Negative received confident known label")

    # F-F: Known-wrongly-rejected (valid signal tagged as UNKNOWN)
    if not is_neg and pred_mod in ("UNKNOWN", "UNKNOWN_MODULATION") and conf > 0.7:
        return FailureRecord(sample_id, "F-F", pred_mod, true_mod, conf, crc, plaus, lat, "Known signal rejected as unknown")

    # If correct label
    if pred_mod == true_mod or crc:
        # F-B: Correct but uncalibrated (conf < 0.5)
        if conf < 0.5:
            return FailureRecord(sample_id, "F-B", pred_mod, true_mod, conf, crc, plaus, lat, "Correct but low confidence")
        return None

    # F-A: Confidently wrong (oracle/truth says X, model says Y, conf > 0.9) - HIGHEST PRIORITY
    if conf > 0.9:
        return FailureRecord(sample_id, "F-A", pred_mod, true_mod, conf, crc, plaus, lat, "Confidently wrong prediction")

    # F-C: Near-miss (CRC failed but plausibility high)
    if plaus > 0.7:
        return FailureRecord(sample_id, "F-C", pred_mod, true_mod, conf, crc, plaus, lat, "Near-miss: high plausibility but no lock")

    # Standard misclassification
    return FailureRecord(sample_id, "F-D", pred_mod, true_mod, conf, crc, plaus, lat, "Misclassified outcome")


def mine_failures(outcomes: list[dict[str, Any]]) -> list[FailureRecord]:
    failures = []
    for o in outcomes:
        rec = classify_outcome(o)
        if rec:
            failures.append(rec)
    return failures
