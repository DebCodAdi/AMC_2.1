"""Diversity-aware continual learning replay buffer (SIGNAL INFINITY §4.1)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ReplaySample:
    sample_id: str
    iq: np.ndarray
    mod: str
    snr_bin: int
    channel: str
    fec: str
    is_anchor: bool  # True for Bucket A real data


class StratifiedReplayBuffer:
    """Stratified replay buffer preserving anchor data and balanced operational strata."""
    def __init__(self, budget: int = 5000):
        self.budget = budget
        self.anchor_samples: list[ReplaySample] = []
        self.strata_samples: dict[tuple, list[ReplaySample]] = {}

    def add_sample(self, sample: ReplaySample):
        if sample.is_anchor:
            # Permanent retention for Bucket A (Rule R15)
            self.anchor_samples.append(sample)
        else:
            stratum = (sample.mod, sample.snr_bin, sample.fec)
            if stratum not in self.strata_samples:
                self.strata_samples[stratum] = []
            self.strata_samples[stratum].append(sample)

    def sample_batch(self, batch_size: int = 16, anchor_fraction: float = 0.15, rng=None) -> list[ReplaySample]:
        if rng is None:
            rng = np.random.default_rng()

        n_anchor = max(1, int(batch_size * anchor_fraction))
        n_strata = batch_size - n_anchor

        batch = []
        # Sample anchors
        if self.anchor_samples:
            indices = rng.integers(0, len(self.anchor_samples), size=n_anchor)
            batch.extend([self.anchor_samples[i] for i in indices])

        # Sample across strata
        all_strata = list(self.strata_samples.keys())
        if all_strata:
            for _ in range(n_strata):
                idx = int(rng.integers(0, len(all_strata)))
                chosen_stratum = all_strata[idx]
                pool = self.strata_samples[chosen_stratum]
                if pool:
                    sample_idx = int(rng.integers(0, len(pool)))
                    batch.append(pool[sample_idx])

        return batch
