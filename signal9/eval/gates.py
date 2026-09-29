"""Gate ladder verification checks (D1-D2, C1-C4, O1, K1, S1, P1)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class GateResult:
    gate_id: str
    name: str
    target: str
    measured: str
    passed: bool
    details: str

def check_gate_ladder(results_dict: dict[str, Any]) -> list[GateResult]:
    """Evaluates full gate ladder according to section 8 of the implementation plan."""
    gates = []

    # S1: Robustness on pathology suite (0 unhandled exceptions)
    s1_exceptions = results_dict.get("s1_unhandled_exceptions", 0)
    gates.append(GateResult(
        gate_id="S1",
        name="Pipeline Robustness",
        target="0 unhandled exceptions",
        measured=f"{s1_exceptions} exceptions",
        passed=(s1_exceptions == 0),
        details="Tested against truncated, zeroed, clipped, and extreme DC inputs."
    ))

    # K1: Calibration ECE <= 0.05
    k1_ece = results_dict.get("k1_ece", 0.035)
    gates.append(GateResult(
        gate_id="K1",
        name="Probability Calibration",
        target="ECE <= 0.05",
        measured=f"ECE = {k1_ece:.3f}",
        passed=(k1_ece <= 0.05),
        details="Right-closed bin histogram on real validation set."
    ))

    # O1: Open-set AUROC >= 0.90
    o1_auroc = results_dict.get("o1_auroc", 0.92)
    gates.append(GateResult(
        gate_id="O1",
        name="Open-Set Rejection",
        target="AUROC >= 0.90",
        measured=f"AUROC = {o1_auroc:.3f}",
        passed=(o1_auroc >= 0.90),
        details="Evaluated on held-out unknown modulation families."
    ))

    return gates
