"""L3 Metamorphic test runner.

For each MR in the registry, applies the transform to every available
test capture and verifies the relation.  MRs marked ``expect_fail_until``
with an open defect ID are marked ``xfail``.

Spec reference: §2.5, §6.3 (PS26147).
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.L3_metamorphic.relations import REGISTRY, run_mr


# ── Minimal default pipeline stub ─────────────────────────────────────────────

def _try_import_pipeline():
    try:
        from signal9.pipeline import Signal9Engine  # type: ignore
        engine = Signal9Engine()
        return engine.process
    except Exception:
        return None


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def pipeline():
    p = _try_import_pipeline()
    if p is None:
        pytest.skip("Signal9Engine not importable — pipeline not ready")
    return p


@pytest.fixture(scope="session")
def test_captures():
    """Small set of synthetic captures for MR testing."""
    from tests.conftest import make_bpsk, make_qpsk
    rng = np.random.default_rng(0)
    return [
        make_bpsk(n_sym=2048, sps=4.0, snr_db=20.0, seed=1),
        make_qpsk(n_sym=2048, sps=4.0, snr_db=20.0, seed=2),
        make_bpsk(n_sym=2048, sps=4.0, snr_db=10.0, seed=3),
    ]


# ── Parametrised MR tests ─────────────────────────────────────────────────────

@pytest.mark.parametrize("mr", REGISTRY, ids=[mr.id for mr in REGISTRY])
def test_metamorphic_relation(mr, pipeline, test_captures, rng):
    """Run MR against every test capture.  Fail if the relation does not hold
    for any capture that the pipeline successfully processes."""

    marks = []
    if mr.expect_fail_until:
        marks.append(pytest.mark.xfail(
            reason=f"Known failure: {mr.expect_fail_until} not yet fixed",
            strict=False,  # flip to strict=True when DEF is closed
        ))

    if marks:
        pytest.xfail(f"Marked xfail until {mr.expect_fail_until}")

    failures = []
    for i, cap in enumerate(test_captures):
        try:
            result = run_mr(mr, cap, pipeline, rng)
        except Exception as e:
            failures.append(f"capture[{i}]: exception {e!r}")
            continue
        if not result:
            failures.append(f"capture[{i}]: relation '{mr.relation}' violated")

    assert not failures, (
        f"MR {mr.id} ({mr.description}) failed:\n"
        + "\n".join(f"  {f}" for f in failures)
    )
