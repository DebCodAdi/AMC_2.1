"""Stateful streaming reader and ring buffer for live SDR and continuous streams."""
from __future__ import annotations
from typing import Iterator
import numpy as np

class StreamingRingBuffer:
    """Bounded circular ring buffer keeping persistent filter state across blocks."""
    def __init__(self, capacity_samples: int = 1_000_000, overlap_samples: int = 4096):
        self.capacity = capacity_samples
        self.overlap = overlap_samples
        self.buffer = np.zeros(capacity_samples, dtype=np.complex64)
        self.write_pos = 0
        self.total_written = 0

    def push(self, block: np.ndarray) -> None:
        """Append incoming block to ring buffer."""
        n = len(block)
        if n >= self.capacity:
            self.buffer[:] = block[-self.capacity:]
            self.write_pos = 0
        else:
            space = self.capacity - self.write_pos
            if n <= space:
                self.buffer[self.write_pos:self.write_pos + n] = block
                self.write_pos = (self.write_pos + n) % self.capacity
            else:
                self.buffer[self.write_pos:] = block[:space]
                self.buffer[:n - space] = block[space:]
                self.write_pos = n - space
        self.total_written += n

    def get_latest_window(self, length: int) -> np.ndarray:
        """Retrieve most recent contiguous window of samples."""
        length = min(length, self.capacity)
        if self.write_pos >= length:
            return self.buffer[self.write_pos - length:self.write_pos].copy()
        else:
            tail = length - self.write_pos
            return np.concatenate([self.buffer[-tail:], self.buffer[:self.write_pos]])
