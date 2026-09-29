"""Safety regressions for deployment inference behavior."""
from pathlib import Path


def test_heuristics_are_opt_in_and_fsk_gated():
    source = (Path(__file__).resolve().parents[1] / "signal9" / "train" / "inference.py").read_text(encoding="utf-8")
    assert "apply_heuristics: bool = False" in source
    assert "if apply_heuristics and is_fsk" in source
    assert "conf = min(0.95, conf + 0.35)" not in source


def test_unvalidated_prototype_and_alignment_paths_are_disabled_by_default():
    source = (Path(__file__).resolve().parents[1] / "signal9" / "train" / "inference.py").read_text(encoding="utf-8")
    assert 'ck.get("prototype_weight", 0.0)' in source
    assert 'ck.get("alignment_preprocess", False)' in source
    assert "self.prototype_weight * cos_sim" in source
    assert "+ 3.5 * cos_sim" not in source
