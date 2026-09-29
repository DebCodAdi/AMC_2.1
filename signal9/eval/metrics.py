"""Evaluation metrics: Top-k accuracy, ECE (right-closed bins), AUROC, and FPR@95TPR."""
from __future__ import annotations
import numpy as np

def compute_topk_accuracy(probs: np.ndarray, targets: np.ndarray, k: int = 1) -> float:
    """Compute Top-k classification accuracy."""
    if len(targets) == 0:
        return 0.0
    topk_preds = np.argsort(probs, axis=-1)[:, -k:]
    correct = np.any(topk_preds == targets[:, None], axis=-1)
    return float(np.mean(correct))

def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE).
    Fixes B18: Uses np.digitize with right-closed last bin so confidence = 1.0 is not dropped.
    """
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == targets)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    # Digitize: bins 1..n_bins
    bin_indices = np.digitize(confidences, bin_boundaries, right=True)
    # Clamp to ensure confidence=0.0 lands in bin 1
    bin_indices = np.clip(bin_indices, 1, n_bins)

    ece = 0.0
    n_total = len(confidences)

    for b in range(1, n_bins + 1):
        in_bin = (bin_indices == b)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            avg_acc = np.mean(accuracies[in_bin])
            avg_conf = np.mean(confidences[in_bin])
            ece += np.abs(avg_acc - avg_conf) * (np.sum(in_bin) / n_total)

    return float(ece)

def open_set_metrics(known_energies: np.ndarray, unknown_energies: np.ndarray) -> dict[str, float]:
    """Compute AUROC and FPR at 95% TPR for open-set detection.
    In known distributions, energy is lower (more negative).
    In unknown distributions, energy is higher (closer to 0).
    """
    scores = np.concatenate([known_energies, unknown_energies])
    labels = np.concatenate([np.zeros(len(known_energies)), np.ones(len(unknown_energies))])

    # Rank-based AUROC computation
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    n_pos = float(len(unknown_energies))
    n_neg = float(len(known_energies))
    if n_pos == 0 or n_neg == 0:
        return {"auroc": 1.0, "fpr_at_95_tpr": 0.0}

    sum_ranks_pos = np.sum(ranks[labels == 1])
    auroc = (sum_ranks_pos - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg)

    # Threshold for 95% TPR on known acceptance (or unknown rejection)
    # 95% of knowns must fall below threshold
    thr_95 = np.percentile(known_energies, 95.0)
    # Fraction of unknowns accepted (false positive rate)
    fpr = float(np.mean(unknown_energies <= thr_95))

    return {
        "auroc": float(auroc),
        "fpr_at_95_tpr": fpr,
    }
