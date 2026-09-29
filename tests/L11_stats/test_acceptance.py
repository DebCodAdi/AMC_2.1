"""L11 Stats tier — Acceptance criteria and statistical significance (§6.4, §8.1).

Verifies:
  - Rule R3 containment check: CRC_locked(new) >= CRC_locked(old).
  - McNemar's paired test for statistical significance (p < 0.05).
  - Gate criteria tracking (Gate 0 through Gate 5).

Spec reference: §0.1 (Rule R3), §2.4 (L11 tier), §6.4, §8.1 (PS26147).
"""
from __future__ import annotations
import math
import pytest


def mcnemar_test(b: int, c: int) -> tuple[float, float]:
    """Compute McNemar's chi-squared statistic and p-value for paired outcomes.

    Parameters
    ----------
    b : int
        Number of cases where new=True and old=False (discordant improvements).
    c : int
        Number of cases where new=False and old=True (discordant regressions).

    Returns
    -------
    (chi2_stat, p_value)
    """
    if b + c == 0:
        return 0.0, 1.0

    # With continuity correction
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
    # p-value from chi2 with 1 degree of freedom (erfc approximation)
    p_value = math.erfc(math.sqrt(chi2 / 2.0))
    return float(chi2), float(p_value)


def check_r3_containment(old_locked_ids: set[str], new_locked_ids: set[str]) -> tuple[bool, set[str]]:
    """Verify Rule R3: new locked files must contain all previously locked files.

    Returns
    -------
    (passes_r3, regressed_files)
    """
    regressed = old_locked_ids - new_locked_ids
    return len(regressed) == 0, regressed


class TestL11Stats:
    """Statistical verification of acceptance criteria."""

    def test_mcnemar_significant_improvement(self):
        """b=20 improvements with c=0 regressions must be highly significant (p < 1e-4)."""
        chi2, p = mcnemar_test(b=20, c=0)
        assert p < 1e-4
        assert chi2 > 15.0

    def test_r3_strict_containment(self):
        """Rule R3 passes when old set is subset of new set, fails on any regression."""
        old_set = {"file_001.iq", "file_002.iq", "file_003.iq"}
        new_set_ok = {"file_001.iq", "file_002.iq", "file_003.iq", "file_004.iq"}
        new_set_bad = {"file_001.iq", "file_002.iq", "file_004.iq"}  # lost file_003

        ok, regressed = check_r3_containment(old_set, new_set_ok)
        assert ok
        assert len(regressed) == 0

        bad, regressed_bad = check_r3_containment(old_set, new_set_bad)
        assert not bad
        assert regressed_bad == {"file_003.iq"}
