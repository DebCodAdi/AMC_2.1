"""Learned per-frequency power baseline for long-running wideband captures.

This is intentionally a detector aid, not a classifier feature.  It learns
normal receiver/noise power per FFT bin across chunks, then highlights a new
or strengthened emission without replacing the local whitening/CFAR path.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TemporalBaseline:
    """Slowly adapting, signal-protected frequency-bin power baseline.

    ``guard_db`` prevents an active signal from being folded into the baseline;
    ``alpha`` controls adaptation of ordinary changes in receiver noise.
    """
    alpha: float = 0.05
    guard_db: float = 6.0
    baseline_db: np.ndarray | None = None
    chunks_seen: int = 0

    def update(self, power_db: np.ndarray) -> np.ndarray:
        """Return positive excess power while updating the noise baseline.

        ``power_db`` is frequency x time.  Its time median is robust to bursty
        emitters; update is additionally restricted to bins close to the
        existing baseline, preventing persistent transmissions from becoming
        the reference level.  Shape changes reset state safely.
        """
        p = np.asarray(power_db, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] == 0:
            return np.zeros_like(p, dtype=np.float32)
        summary = np.nanmedian(p, axis=1)
        summary = np.nan_to_num(summary, nan=-200.0, posinf=100.0, neginf=-200.0)

        if self.baseline_db is None or self.baseline_db.shape != summary.shape:
            self.baseline_db = summary.copy()
            self.chunks_seen = 1
            return np.zeros_like(p, dtype=np.float32)

        prior = self.baseline_db.copy()
        # A receiver-wide AGC step moves nearly every FFT bin together.  It is
        # not a spectrum event, so remove its robust common-mode shift before
        # calculating novelty.  This keeps an AGC transition from becoming a
        # false full-band CFAR detection.
        common_mode_db = float(np.median(summary - prior))
        expected = prior + common_mode_db
        # New/high-power bins are deliberately not adapted this pass.  The
        # common-mode-shifted expectation still lets ordinary floor changes
        # adapt over time.
        trusted = summary <= expected + self.guard_db
        self.baseline_db[trusted] = (
            (1.0 - self.alpha) * prior[trusted] + self.alpha * summary[trusted]
        )
        self.chunks_seen += 1
        return np.maximum(p - expected[:, None], 0.0).astype(np.float32)
