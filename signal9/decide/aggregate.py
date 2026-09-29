"""Track prediction aggregation and hierarchical decision resolution."""
from __future__ import annotations
import numpy as np
from .calibrate import calibrate_logits
from ..models.openset import energy_score, evaluate_decision_state

def aggregate_track_predictions(crop_logits_dict: list[dict[str, np.ndarray]],
                                crop_snrs: list[float],
                                temperatures: dict[str, float] | None = None,
                                class_names: dict[str, list[str]] | None = None) -> dict:
    """Aggregate predictions across multiple crops of a signal track.
    Returns calibrated top predictions, probabilities, and decision states for L1, L2, L3.
    """
    if not crop_logits_dict:
        return {}

    temps = temperatures or {"l1": 1.0, "l2": 1.0, "l3": 1.0}
    weights = 10.0 ** (np.clip(np.array(crop_snrs), -10.0, 40.0) / 10.0)
    w_sum = np.sum(weights)
    norm_weights = weights / (w_sum if w_sum > 0 else 1.0)

    results: dict = {
        "track_snr_db": float(np.mean(crop_snrs)),
        "levels": {},
    }

    for lvl in ("l1", "l2", "l3"):
        logits_arr = np.stack([c[lvl] for c in crop_logits_dict], axis=0)  # (K, C)
        # Weighted mean logits
        track_logits = np.sum(logits_arr * norm_weights[:, None], axis=0)  # (C,)
        
        t = temps.get(lvl, 1.0)
        probs = calibrate_logits(track_logits, t)
        energy = float(energy_score(track_logits, t))
        
        top1_idx = int(np.argmax(probs))
        top1_prob = float(probs[top1_idx])
        
        # Decision state
        state = evaluate_decision_state(
            prob_top1=top1_prob,
            energy=energy,
            snr_db=results["track_snr_db"],
            energy_threshold=-5.0,
            prob_threshold=0.55,
            min_snr_db=3.0
        )
        
        lvl_classes = (class_names or {}).get(lvl, [])
        top1_label = lvl_classes[top1_idx] if top1_idx < len(lvl_classes) else f"class_{top1_idx}"

        results["levels"][lvl] = {
            "prediction": top1_label if state == "DECISION" else state,
            "confidence": top1_prob,
            "decision_state": state,
            "energy": energy,
            "top1_index": top1_idx,
        }

    return results
