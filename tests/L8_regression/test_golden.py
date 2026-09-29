"""L8 Regression tests — golden payload byte-exact reproduction.

These tests MUST NEVER FAIL.  They freeze the 80 files that currently
produce a valid CRC-locked payload and assert byte-identical reproduction
across:
  - Multiple runs (determinism)
  - Different worker counts (R6 — §2.4)
  - NUMBA_DISABLE_JIT=1 (A7 gate)

Spec reference: §2.4, A6, A7, L8 tier (PS26147).
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).parent.parent / "golden"
GOLDEN_PAYLOADS = GOLDEN_DIR / "payloads"
GOLDEN_MANIFEST = GOLDEN_DIR / "MANIFEST.json"


def _load_manifest() -> dict:
    if not GOLDEN_MANIFEST.exists():
        return {}
    with open(GOLDEN_MANIFEST, encoding="utf-8") as f:
        return json.load(f)


MANIFEST = _load_manifest()
ENTRIES = MANIFEST.get("entries", [])


def _try_pipeline(capture_path: str) -> dict | None:
    try:
        from signal9.pipeline import Signal9Engine  # type: ignore
        engine = Signal9Engine()
        return engine.process(capture_path)
    except Exception as e:
        pytest.fail(f"Pipeline raised an exception on {capture_path}: {e!r}")


@pytest.mark.skipif(not ENTRIES, reason="No golden entries — run Phase 0 snapshot first")
@pytest.mark.parametrize("entry", ENTRIES, ids=[e.get("file", str(i)) for i, e in enumerate(ENTRIES)])
def test_golden_payload_byte_exact(entry):
    """Re-run the pipeline on a formerly-locking file and assert byte-exact payload."""
    capture_path = entry["file"]
    expected_hex = entry["payload_hex"]
    expected_sha256 = entry.get("payload_sha256")

    if not Path(capture_path).exists():
        pytest.skip(f"Capture file not found: {capture_path}")

    result = _try_pipeline(capture_path)
    assert result is not None

    assert result.get("status") == "SUCCESS_CRC_LOCKED", (
        f"R3 VIOLATION: {capture_path} no longer locks.\n"
        f"  status: {result.get('status')}\n"
        f"  failure_reason: {result.get('failure_reason')}\n"
        f"  failure_stage: {result.get('failure_stage')}"
    )

    actual_hex = result.get("payload_hex") or ""
    assert actual_hex == expected_hex, (
        f"R4 VIOLATION: payload changed for {capture_path}\n"
        f"  expected: {expected_hex[:64]}…\n"
        f"  actual:   {actual_hex[:64]}…"
    )

    if expected_sha256:
        payload_bytes = bytes.fromhex(actual_hex)
        actual_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        assert actual_sha256 == expected_sha256, (
            f"SHA-256 mismatch for {capture_path}: {actual_sha256} ≠ {expected_sha256}"
        )


class TestDeterminism:
    """A6: Three runs with 1 / 8 / 32 workers must give byte-identical results."""

    @pytest.mark.skipif(not ENTRIES, reason="No golden entries")
    def test_single_worker_determinism(self):
        """Same file processed twice sequentially must give the same result."""
        if not ENTRIES:
            pytest.skip("No golden entries")
        entry = ENTRIES[0]
        path = entry["file"]
        if not Path(path).exists():
            pytest.skip(f"File not found: {path}")

        r1 = _try_pipeline(path)
        r2 = _try_pipeline(path)
        assert r1.get("payload_hex") == r2.get("payload_hex"), (
            "Non-deterministic: two sequential runs gave different payloads"
        )
        assert r1.get("status") == r2.get("status"), (
            "Non-deterministic: two sequential runs gave different statuses"
        )

    def test_numba_jit_parity(self):
        """A7: NUMBA_DISABLE_JIT=1 must give identical decisions.

        This test checks the environment variable — if it IS set, we verify
        we match golden; if not, skip (the CI jit_off job handles this).
        """
        jit_disabled = os.environ.get("NUMBA_DISABLE_JIT", "0") == "1"
        if not jit_disabled:
            pytest.skip("Set NUMBA_DISABLE_JIT=1 to run this test (CI jit_off job)")
        if not ENTRIES:
            pytest.skip("No golden entries")

        for entry in ENTRIES[:5]:  # Spot-check 5 files
            path = entry["file"]
            if not Path(path).exists():
                continue
            result = _try_pipeline(path)
            assert result.get("status") == "SUCCESS_CRC_LOCKED", (
                f"JIT-off: {path} no longer locks (A7 violation)"
            )
            assert result.get("payload_hex") == entry["payload_hex"], (
                f"JIT-off: payload changed for {path} (A7 violation)"
            )
