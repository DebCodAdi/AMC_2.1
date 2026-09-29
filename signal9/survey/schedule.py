"""Priority queue scheduling and load shedding for wideband survey tasks (Phase 3).

Prevents downstream processor starvation and buffer overflows during dense
spectral activity by dynamically shedding low-priority emissions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import heapq
import time
from typing import Any

import numpy as np
from .extract import BurstExtraction


@dataclass(order=True)
class PrioritizedTask:
    priority: float
    task_id: int = field(compare=False)
    timestamp: float = field(compare=False)
    payload: Any = field(compare=False)


@dataclass
class SurveyTask:
    task_id: int
    extraction: BurstExtraction
    priority_score: float
    created_time: float


class SurveyScheduler:
    """Bounded priority scheduler with adaptive load shedding."""

    def __init__(self,
                 max_queue_size: int = 64,
                 snr_weight: float = 1.0,
                 bw_weight: float = 0.5):
        self.max_queue_size = max_queue_size
        self.snr_weight = snr_weight
        self.bw_weight = bw_weight
        self._heap: list[PrioritizedTask] = []
        self._counter = 0
        self.n_enqueued = 0
        self.n_dispatched = 0
        self.n_dropped = 0

    def compute_priority(self, extraction: BurstExtraction) -> float:
        """Higher score means higher processing priority."""
        # SNR component + normalized bandwidth component
        snr = max(0.0, extraction.snr_db)
        bw_khz = (extraction.channel_fs / 2.0) / 1e3
        score = self.snr_weight * snr + self.bw_weight * np.log10(max(1.0, bw_khz))
        return float(score)

    def enqueue(self, extraction: BurstExtraction) -> bool:
        """Enqueue extracted burst. If queue is full, sheds the lowest-priority task."""
        score = self.compute_priority(extraction)
        self._counter += 1
        self.n_enqueued += 1

        import numpy as np  # Ensure imported
        # We store negative score in min-heap so highest score has lowest numeric value
        task = PrioritizedTask(
            priority=-score,
            task_id=self._counter,
            timestamp=time.perf_counter(),
            payload=extraction,
        )

        if len(self._heap) < self.max_queue_size:
            heapq.heappush(self._heap, task)
            return True

        # Queue full: check if new task is higher priority than lowest in heap
        # Note: in negative priority, max negative number is the lowest score
        # Find item with largest priority value (lowest real score)
        worst_idx = max(range(len(self._heap)), key=lambda i: self._heap[i].priority)
        worst_task = self._heap[worst_idx]

        if task.priority < worst_task.priority:  # More negative = higher score
            # Drop the worst task
            self._heap[worst_idx] = task
            heapq.heapify(self._heap)
            self.n_dropped += 1
            return True
        else:
            # Shed the incoming task
            self.n_dropped += 1
            return False

    def dispatch_next(self) -> BurstExtraction | None:
        """Pop the highest priority burst extraction."""
        if not self._heap:
            return None
        task = heapq.heappop(self._heap)
        self.n_dispatched += 1
        return task.payload

    @property
    def queue_depth(self) -> int:
        return len(self._heap)

    def stats(self) -> dict[str, int | float]:
        return {
            "queue_depth": len(self._heap),
            "max_queue_size": self.max_queue_size,
            "n_enqueued": self.n_enqueued,
            "n_dispatched": self.n_dispatched,
            "n_dropped": self.n_dropped,
            "drop_rate": self.n_dropped / max(1, self.n_enqueued),
        }
