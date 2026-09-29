"""Open-set recognition and decision rejection logic."""
from __future__ import annotations
import numpy as np

def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Computes free energy score: E(x) = -T * logsumexp(logits / T).
    In-distribution samples exhibit low energy (large negative values).
    Out-of-distribution (novel/unknown) samples exhibit high energy (closer to 0).
    """
    scaled = logits / temperature
    max_val = np.max(scaled, axis=-1, keepdims=True)
    lse = max_val + np.log(np.sum(np.exp(scaled - max_val), axis=-1, keepdims=True) + 1e-20)
    return (-temperature * lse).squeeze(-1)

def prototype_distance(embed: np.ndarray, prototypes: np.ndarray) -> float:
    """Computes minimum cosine distance to known class prototypes."""
    embed_norm = embed / (np.linalg.norm(embed) + 1e-12)
    proto_norms = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-12)
    cos_sim = np.dot(proto_norms, embed_norm)
    return float(1.0 - np.max(cos_sim))

def evaluate_decision_state(prob_top1: float,
                            energy: float,
                            snr_db: float,
                            proto_dist: float = 0.0,
                            energy_threshold: float = -5.0,
                            dist_threshold: float = 0.4,
                            prob_threshold: float = 0.60,
                            min_snr_db: float = 3.0) -> str:
    """Determines decision state:
    - 'NO_DECISION': SNR too low or signal too weak for reliable inference
    - 'UNKNOWN': Novel/unsupported signal (adequate SNR but high energy, high dist, or low conf)
    - 'DECISION': Confident classification within known taxonomy
    """
    if snr_db < min_snr_db:
        return "NO_DECISION"
    if energy > energy_threshold or proto_dist > dist_threshold or prob_top1 < prob_threshold:
        return "UNKNOWN"
    return "DECISION"
