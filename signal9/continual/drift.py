"""Deployment drift detectors and early-warning alarm system (SIGNAL INFINITY §4.5)."""
from __future__ import annotations
import numpy as np


def compute_mmd(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    """Maximum Mean Discrepancy (MMD) with RBF kernel between live and reference embeddings."""
    if len(x) == 0 or len(y) == 0:
        return 0.0
    # Pairwise squared distances
    xx = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    yy = np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    xy = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)

    k_xx = np.exp(-gamma * xx).mean()
    k_yy = np.exp(-gamma * yy).mean()
    k_xy = np.exp(-gamma * xy).mean()

    return float(np.sqrt(max(0.0, k_xx + k_yy - 2 * k_xy)))


def evaluate_drift_alarms(live_embeddings: np.ndarray,
                          ref_embeddings: np.ndarray,
                          classical_preds: list[str],
                          learned_preds: list[str],
                          rejected_count: int,
                          total_count: int) -> dict[str, Any]:
    """Evaluate leading-indicator drift signals before accuracy degrades."""
    # 1. Embedding MMD
    mmd_score = compute_mmd(live_embeddings[:100], ref_embeddings[:100])

    # 2. Classical vs Learned disagreement ratio
    disagreements = sum(1 for c, l in zip(classical_preds, learned_preds) if c != l)
    disagreement_ratio = disagreements / max(1, len(classical_preds))

    # 3. Reject rate (sudden rise indicates new unmapped signals in the RF band)
    reject_rate = rejected_count / max(1, total_count)

    # Trigger levels
    alarms = []
    if mmd_score > 0.35:
        alarms.append("HIGH_EMBEDDING_DRIFT")
    if disagreement_ratio > 0.25:
        alarms.append("HIGH_MODEL_DISAGREEMENT")
    if reject_rate > 0.20:
        alarms.append("NEW_BAND_SIGNALS_DETECTED")

    # Severe action: automatic fallback to classical path (Rule R21)
    fallback_to_classical = (disagreement_ratio > 0.40 or mmd_score > 0.60)

    return {
        "mmd": float(mmd_score),
        "disagreement_ratio": float(disagreement_ratio),
        "reject_rate": float(reject_rate),
        "alarms": alarms,
        "trigger_classical_fallback": fallback_to_classical,
    }
