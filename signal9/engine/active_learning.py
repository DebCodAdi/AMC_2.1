"""Information-theoretic active learning queue ranking (SIGNAL INFINITY §3.5)."""
from __future__ import annotations
import numpy as np


def compute_acquisition_priority(sample_meta: dict) -> float:
    """Calculate acquisition priority score for human/oracle review.
    
    Balances uncertainty, classical-learned disagreement, novelty, and duplicate penalty.
    """
    is_real = 1.0 if sample_meta.get("is_real_capture", False) else 0.0
    disagreement = 1.0 if sample_meta.get("classical_learned_disagreement", False) else 0.0
    predictive_entropy = float(sample_meta.get("entropy", 0.0))
    prototype_dist = float(sample_meta.get("distance_to_prototype", 0.0))
    is_duplicate = 1.0 if sample_meta.get("is_near_duplicate", False) else 0.0

    priority = (
        3.0 * is_real
        + 2.0 * disagreement
        + 2.0 * predictive_entropy
        + 1.5 * prototype_dist
        - 2.0 * is_duplicate
    )
    return float(priority)


def rank_active_learning_queue(candidates: list[dict], top_k: int = 100) -> list[dict]:
    scored = []
    for c in candidates:
        score = compute_acquisition_priority(c)
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]
