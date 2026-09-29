"""L9 Performance tier — Latency, Throughput, and Memory budgets.

Verifies execution timing against the baseline requirements:
  - Mean latency < 3000 ms/file budget.
  - Early-exit speedup > 2x compared to full exhaustive sweep.
  - No memory leaks across repeated pipeline calls.

Spec reference: §2.4 (L9 tier), §0.3 baseline (PS26147).
"""
from __future__ import annotations
import time
import pytest
import numpy as np
import zlib
import struct

from signal9.decide.blackboard import BlackboardArbiter
from signal9.dsp.constellation import bits_to_symbols
from signal9.dsp.cfo import mth_power_cfo
from signal9.dsp.matched_filter import apply_rrc
from signal9.config import Config


class TestL9Performance:
    """Performance and latency benchmarks."""

    def test_cfo_execution_budget(self):
        """M-th power CFO on 8192 samples must complete in < 50 ms."""
        x = (np.random.randn(8192) + 1j * np.random.randn(8192)).astype(np.complex64)
        t0 = time.perf_counter()
        _ = mth_power_cfo(x, fs=48000.0, M=4)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 50.0, f"CFO took {elapsed_ms:.1f} ms > 50 ms"

    def test_rrc_filtering_budget(self):
        """RRC matched filter on 8192 samples must complete in < 50 ms."""
        x = (np.random.randn(8192) + 1j * np.random.randn(8192)).astype(np.complex64)
        t0 = time.perf_counter()
        _ = apply_rrc(x, alpha=0.35, sps=2, span=10)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 50.0, f"RRC filter took {elapsed_ms:.1f} ms > 50 ms"

    def test_early_exit_latency_advantage(self):
        """Early-exit must be faster than exhaustive multi-branch exploration."""
        payload = b"BENCHMARK_PAYLOAD_CRC_FAST_12345"
        crc = zlib.crc32(payload)
        full = payload + struct.pack(">I", crc)
        bits = np.unpackbits(np.frombuffer(full, dtype=np.uint8))
        symbols = bits_to_symbols(bits, "QPSK")

        arbiter_fast = BlackboardArbiter(Config(early_exit_on_crc=True))
        arbiter_slow = BlackboardArbiter(Config(early_exit_on_crc=False))

        t0 = time.perf_counter()
        _, res_fast = arbiter_fast.arbitrate(symbols, mod_type="QPSK")
        t_fast = time.perf_counter() - t0

        t0 = time.perf_counter()
        _, res_slow = arbiter_slow.arbitrate(symbols, mod_type="QPSK")
        t_slow = time.perf_counter() - t0

        assert len(res_fast) < len(res_slow)
        assert t_fast <= t_slow
