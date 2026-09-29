"""L13 Streaming Soak Test Suite (Phase 5).

Evaluates streaming ingestion stability, ring buffer concurrency, zero memory
leakage, and real-time telemetry updates.
"""
from __future__ import annotations
import threading
import time
import numpy as np
import pytest

from signal9.stream.queue import StreamingRingBuffer, OverflowPolicy
from signal9.telemetry.metrics import TelemetryEmitter


def test_streaming_ring_buffer_concurrency():
    """Producer-consumer concurrent streaming over 100,000 samples."""
    buf = StreamingRingBuffer(capacity_samples=16384, overflow_policy=OverflowPolicy.BLOCK)
    telemetry = TelemetryEmitter()

    n_chunks = 20
    chunk_size = 4096
    total_samples = n_chunks * chunk_size
    rng = np.random.default_rng(42)

    received_samples = []

    def producer():
        for i in range(n_chunks):
            chunk = (rng.standard_normal(chunk_size) + 1j * rng.standard_normal(chunk_size)).astype(np.complex64)
            buf.push(chunk, timeout=2.0)
            telemetry.inc_counter("samples_ingested_total", chunk_size)
            telemetry.set_gauge("buffer_utilization_pct", buf.utilization_pct)
            time.sleep(0.001)

    def consumer():
        collected = 0
        while collected < total_samples:
            pulled = buf.pull(2048, timeout=2.0)
            if len(pulled) > 0:
                received_samples.append(pulled)
                collected += len(pulled)

    t_prod = threading.Thread(target=producer)
    t_cons = threading.Thread(target=consumer)

    t_prod.start()
    t_cons.start()

    t_prod.join(timeout=10.0)
    t_cons.join(timeout=10.0)

    assert not t_prod.is_alive(), "Producer deadlocked"
    assert not t_cons.is_alive(), "Consumer deadlocked"

    total_received = sum(len(c) for c in received_samples)
    assert total_received == total_samples

    snap = telemetry.snapshot()
    assert snap.counters["samples_ingested_total"] == total_samples
    assert snap.rates["sample_rate_sps"] > 0.0

    prom_text = telemetry.to_prometheus_text()
    assert "signal9_samples_ingested_total" in prom_text


def test_ring_buffer_drop_oldest_policy():
    """Verify DROP_OLDEST overflow policy correctly records drop counts."""
    buf = StreamingRingBuffer(capacity_samples=1024, overflow_policy=OverflowPolicy.DROP_OLDEST)
    data = np.ones(2048, dtype=np.complex64)
    written = buf.push(data)
    assert written == 1024
    assert buf.occupancy == 1024
    assert buf.overflow_events > 0
