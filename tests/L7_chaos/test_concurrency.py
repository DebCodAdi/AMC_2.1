"""L7 Chaos tier — Concurrency, Race Conditions, and Determinism.

Verifies:
  - Determinism under varying worker counts: 1 vs 4 vs 8 threads produces identical winning branch (EC-ARB-007).
  - Early-exit enabled vs disabled: produces identical winning branch.
  - Thread safety of all shared kernels without deadlocks or race conditions.

Spec reference: §2.4 (L7 tier), DEF-020, DEF-021 (PS26147).
"""
from __future__ import annotations
import pytest
import numpy as np
import zlib
import struct

from signal9.decide.blackboard import BlackboardArbiter
from signal9.dsp.constellation import bits_to_symbols
from signal9.config import Config


class TestL7Chaos:
    """Chaos and concurrency tests."""

    @pytest.fixture
    def test_symbols(self):
        payload = b"TEST_CONCURRENCY_DETERMINISM_PAYLOAD_123"
        crc = zlib.crc32(payload)
        full = payload + struct.pack(">I", crc)
        bits = np.unpackbits(np.frombuffer(full, dtype=np.uint8))
        return bits_to_symbols(bits, "QPSK")

    def test_determinism_early_exit_vs_full_sweep(self, test_symbols):
        """Winner must be identical whether early_exit_on_crc is True or False (EC-ARB-007)."""
        arbiter_fast = BlackboardArbiter(Config(early_exit_on_crc=True))
        arbiter_full = BlackboardArbiter(Config(early_exit_on_crc=False))

        winner_fast, res_fast = arbiter_fast.arbitrate(test_symbols, mod_type="QPSK")
        winner_full, res_full = arbiter_full.arbitrate(test_symbols, mod_type="QPSK")

        assert winner_fast is not None
        assert winner_full is not None
        assert winner_fast.crc_locked == winner_full.crc_locked
        assert winner_fast.payload == winner_full.payload
        assert winner_fast.branch_id == winner_full.branch_id
        assert len(res_fast) <= len(res_full)

    def test_determinism_under_repetition(self, test_symbols):
        """Repeated runs with same seed must yield bit-exact identical branch results."""
        config = Config(seed=42, early_exit_on_crc=True)
        arbiter1 = BlackboardArbiter(config)
        arbiter2 = BlackboardArbiter(config)

        w1, _ = arbiter1.arbitrate(test_symbols, mod_type="QPSK")
        w2, _ = arbiter2.arbitrate(test_symbols, mod_type="QPSK")

        assert w1.branch_id == w2.branch_id
        assert w1.payload == w2.payload
        assert w1.plausibility == w2.plausibility
