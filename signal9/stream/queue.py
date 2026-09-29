"""Zero-allocation thread-safe circular ring buffer for real-time IQ streaming (Phase 5).

Supports SPSC/MPSC producer-consumer streams with configurable overflow policies
and backpressure detection.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import threading
import numpy as np


class OverflowPolicy(str, Enum):
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"


@dataclass(frozen=True)
class StreamBlock:
    sequence_id: int
    data: np.ndarray
    timestamp_s: float
    is_gap: bool = False


class StreamingRingBuffer:
    """Thread-safe circular ring buffer holding complex IQ samples."""

    def __init__(self,
                 capacity_samples: int = 1_048_576,  # 1 MSamples (~4MB cf32)
                 overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST):
        self.capacity = capacity_samples
        self.policy = overflow_policy
        self._buf = np.zeros(self.capacity, dtype=np.complex64)
        self._head = 0  # Write pointer
        self._tail = 0  # Read pointer
        self._size = 0  # Current occupancy in samples
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

        self.total_written = 0
        self.total_read = 0
        self.dropped_samples = 0
        self.overflow_events = 0

    def push(self, samples: np.ndarray, timeout: float | None = None) -> int:
        """Push a block of complex IQ samples into the ring buffer.

        Returns number of samples written.
        """
        samples = np.asarray(samples, dtype=np.complex64).ravel()
        n = len(samples)
        if n == 0:
            return 0

        with self._lock:
            if n > self.capacity:
                overflow_excess = n - self.capacity
                self.overflow_events += 1
                self.dropped_samples += overflow_excess
                samples = samples[-self.capacity:]
                n = self.capacity

            available_space = self.capacity - self._size
            if n > available_space:
                self.overflow_events += 1
                if self.policy == OverflowPolicy.DROP_NEWEST:
                    self.dropped_samples += n
                    return 0

                elif self.policy == OverflowPolicy.DROP_OLDEST:
                    # Advance tail by overflow amount
                    overflow = n - available_space
                    self._tail = (self._tail + overflow) % self.capacity
                    self._size -= overflow
                    self.dropped_samples += overflow

                elif self.policy == OverflowPolicy.BLOCK:
                    # Condition.wait() may wake spuriously.  wait_for repeats
                    # the predicate while preserving one overall timeout.
                    enough_space = self._not_full.wait_for(
                        lambda: self.capacity - self._size >= n,
                        timeout=timeout,
                    )
                    if not enough_space:
                        self.dropped_samples += n
                        return 0

            # Write samples with circular wrap
            part1 = min(n, self.capacity - self._head)
            part2 = n - part1

            self._buf[self._head : self._head + part1] = samples[:part1]
            if part2 > 0:
                self._buf[:part2] = samples[part1:]

            self._head = (self._head + n) % self.capacity
            self._size += n
            self.total_written += n

            self._not_empty.notify_all()
            return n

    def pull(self, n_samples: int, timeout: float | None = None) -> np.ndarray:
        """Pull up to n_samples from the ring buffer. Returns complex64 array."""
        with self._lock:
            if self._size == 0:
                if timeout is not None and timeout > 0:
                    self._not_empty.wait_for(lambda: self._size > 0, timeout=timeout)
                if self._size == 0:
                    return np.zeros(0, dtype=np.complex64)

            actual_n = min(n_samples, self._size)
            part1 = min(actual_n, self.capacity - self._tail)
            part2 = actual_n - part1

            out = np.empty(actual_n, dtype=np.complex64)
            out[:part1] = self._buf[self._tail : self._tail + part1]
            if part2 > 0:
                out[part1:] = self._buf[:part2]

            self._tail = (self._tail + actual_n) % self.capacity
            self._size -= actual_n
            self.total_read += actual_n

            self._not_full.notify_all()
            return out

    @property
    def occupancy(self) -> int:
        with self._lock:
            return self._size

    @property
    def utilization_pct(self) -> float:
        with self._lock:
            return (self._size / self.capacity) * 100.0

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "capacity": self.capacity,
                "occupancy": self._size,
                "utilization_pct": (self._size / self.capacity) * 100.0,
                "total_written": self.total_written,
                "total_read": self.total_read,
                "dropped_samples": self.dropped_samples,
                "overflow_events": self.overflow_events,
            }
