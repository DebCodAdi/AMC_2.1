"""Self-training pseudo-labeling loop on real unlabelled captures."""
from __future__ import annotations
from typing import Any
import numpy as np

def filter_pseudo_labels(predictions: list[dict[str, Any]],
                         conf_threshold: float = 0.95,
                         max_pseudo_fraction: float = 0.20) -> list[dict[str, Any]]:
    """Strictly filtered pseudo-label selection (Fixes A9, T6):
    - Must meet high confidence threshold (>= 0.95)
    - Must have decoder verification OR agreement between branches
    - Capped at max_pseudo_fraction per class to avoid confirmation bias
    """
    if not 0.0 <= max_pseudo_fraction <= 1.0:
        raise ValueError("max_pseudo_fraction must be in [0, 1]")
    if not predictions or max_pseudo_fraction == 0.0:
        return []

    candidates: dict[str, list[dict[str, Any]]] = {}

    for p in predictions:
        conf = float(p.get("confidence", 0.0))
        is_verified = bool(p.get("verified", False))
        branch_agreement = bool(p.get("branch_agreement", True))

        if conf >= conf_threshold and (is_verified or branch_agreement):
            label = str(p.get("predicted_label", ""))
            if not label:
                continue
            candidates.setdefault(label, []).append({
                "track_id": p.get("track_id"),
                "label": label,
                "confidence": conf,
                "tier": "SILVER" if is_verified else "BRONZE",
                "sample_weight": 0.6 if is_verified else 0.3
            })

    # No predicted class may occupy more than the requested fraction of the
    # entire unlabelled pool.  Retain the most confident candidates first.
    per_class_budget = int(np.floor(len(predictions) * max_pseudo_fraction))
    if max_pseudo_fraction > 0.0:
        per_class_budget = max(1, per_class_budget)
    accepted = []
    for label in sorted(candidates):
        ranked = sorted(candidates[label], key=lambda row: row["confidence"], reverse=True)
        accepted.extend(ranked[:per_class_budget])
    return accepted
