"""Data quality and label tier definitions (GOLD, SILVER, BRONZE)."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class LabelTier:
    name: str
    weight: float
    description: str

TIER_GOLD = LabelTier("GOLD", 1.0, "Decoder-verified frames or ground-truth scheduled broadcast with human confirmation")
TIER_SILVER = LabelTier("SILVER", 0.6, "Known schedule/band-plan with classifier confidence >= 0.9, or human review without decoder")
TIER_BRONZE = LabelTier("BRONZE", 0.3, "Band-plan or frequency guess only (used for SSL or weakly supervised pretraining)")

LABEL_TIERS = {
    "GOLD": TIER_GOLD,
    "SILVER": TIER_SILVER,
    "BRONZE": TIER_BRONZE,
}

def get_tier_weight(tier_name: str) -> float:
    return LABEL_TIERS.get(tier_name.upper(), TIER_BRONZE).weight
