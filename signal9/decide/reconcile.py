"""Conservative reconciliation of independent neural and DSP decisions."""


def reconcile_modulation(neural: dict, demod: dict) -> dict:
    measured = demod.get("modulation", "UNRESOLVED")
    predicted = neural.get("prediction")
    net_decided = neural.get("decision_state") == "DECISION" and predicted not in (None, "UNKNOWN", "NOISE")
    dsp_decided = measured != "UNRESOLVED" and not demod.get("order_ambiguous", False)
    candidates = [s["mod"] for s in demod.get("order_scores", []) if s.get("plausible")]
    def family(mod):
        return "QAM" if "QAM" in mod else "PSK" if "PSK" in mod else "FSK" if mod in ("MSK", "GMSK") or "FSK" in mod else None
    families = {family(m) for m in candidates}
    if net_decided and dsp_decided and predicted != measured:
        return {"modulation": "UNRESOLVED", "decision_state": "DISAGREEMENT",
                "neural": predicted, "measured": measured, "family": None}
    if dsp_decided:
        return {"modulation": measured, "decision_state": "DECISION", "family": family(measured),
                "source": "agreement" if net_decided else "measurement"}
    # A network cannot settle an explicitly ambiguous measured order.
    if net_decided and not candidates and not demod.get("order_ambiguous"):
        return {"modulation": predicted, "decision_state": "DECISION", "family": family(predicted), "source": "neural"}
    return {"modulation": "UNRESOLVED", "decision_state": "UNKNOWN",
            "family": demod.get("modulation_family") or (next(iter(families)) if len(families) == 1 else None),
            "order_candidates": candidates}
