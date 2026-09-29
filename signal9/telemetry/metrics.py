"""Structured real-time telemetry and metrics emitter (Phase 5).

Provides thread-safe Prometheus/OpenTelemetry compatible instrumentation
for real-time pipeline monitoring.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import threading
import time


@dataclass
class TelemetrySnapshot:
    timestamp: float
    counters: dict[str, int]
    gauges: dict[str, float]
    rates: dict[str, float]


class TelemetryEmitter:
    """Thread-safe telemetry metrics collector."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {
            "samples_ingested_total": 0,
            "bursts_detected_total": 0,
            "packets_decoded_total": 0,
            "crc_locked_total": 0,
            "crc_failed_total": 0,
            "buffer_overflows_total": 0,
        }
        self._gauges: dict[str, float] = {
            "current_snr_db": 0.0,
            "cfo_residual_hz": 0.0,
            "evm_current": 0.0,
            "buffer_utilization_pct": 0.0,
            "processing_latency_ms": 0.0,
        }
        self._last_snapshot_time = time.perf_counter()
        self._last_samples = 0

    def inc_counter(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def snapshot(self) -> TelemetrySnapshot:
        now = time.perf_counter()
        with self._lock:
            dt = max(1e-6, now - self._last_snapshot_time)
            curr_samples = self._counters.get("samples_ingested_total", 0)
            sample_rate = (curr_samples - self._last_samples) / dt

            self._last_snapshot_time = now
            self._last_samples = curr_samples

            return TelemetrySnapshot(
                timestamp=now,
                counters=dict(self._counters),
                gauges=dict(self._gauges),
                rates={"sample_rate_sps": sample_rate},
            )

    def to_prometheus_text(self) -> str:
        snap = self.snapshot()
        lines = []
        for k, v in snap.counters.items():
            lines.append(f"# TYPE signal9_{k} counter")
            lines.append(f"signal9_{k} {v}")
        for k, v in snap.gauges.items():
            lines.append(f"# TYPE signal9_{k} gauge")
            lines.append(f"signal9_{k} {v:.4f}")
        for k, v in snap.rates.items():
            lines.append(f"# TYPE signal9_{k} gauge")
            lines.append(f"signal9_{k} {v:.2f}")
        return "\n".join(lines) + "\n"
