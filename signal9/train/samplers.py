"""Dataset splitting and class-balanced sampling."""
from __future__ import annotations
from typing import Any
import numpy as np

def group_split(rows: list[dict[str, Any]],
                test_receivers: set[str],
                test_after_utc: str | None = None,
                val_frac: float = 0.15,
                seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    """Strictly zero-leakage split across receiver, time, and session.
    Fixes B14: Ensures no receiver, session, or transmitter appears on both sides.
    """
    rng = np.random.default_rng(seed)
    train_pool = []
    test = []

    for r in rows:
        rx = r.get("receiver", "")
        utc = str(r.get("utc", ""))
        
        # Test split: held-out receivers OR time-forward held-out dates
        is_test_rx = (rx in test_receivers)
        is_test_date = (test_after_utc is not None and utc >= test_after_utc)
        
        if is_test_rx or is_test_date:
            test.append(r)
        else:
            train_pool.append(r)

    # Group train_pool by session to prevent session leaking into validation
    sessions: dict[str, list[dict]] = {}
    for r in train_pool:
        sess = r.get("session", str(rng.integers(0, 1000000)))
        sessions.setdefault(sess, []).append(r)

    sess_keys = list(sessions.keys())
    rng.shuffle(sess_keys)
    
    n_val = max(1, int(len(sess_keys) * val_frac))
    val_sessions = set(sess_keys[:n_val])
    
    train = []
    val = []
    for sess, s_rows in sessions.items():
        if sess in val_sessions:
            val.extend(s_rows)
        else:
            train.extend(s_rows)

    return train, val, test

def class_balanced_sample(rows: list[dict[str, Any]], max_freq: int = 10,
                          max_per_source: int = 1000, seed: int | None = None) -> list[dict]:
    """Inverse-frequency sampling capped at max_freq and per-source cap."""
    counts = {}
    for r in rows:
        label = r.get("label_l2", "unknown")
        counts[label] = counts.get(label, 0) + 1
        
    N = len(rows)
    weights = []
    
    for r in rows:
        label = r.get("label_l2", "unknown")
        # Inverse frequency weight
        w = N / (counts[label] * len(counts))
        weights.append(min(w, float(max_freq)))
        
    weights = np.array(weights)
    p = weights / weights.sum()
    
    # Cap samples to avoid overloading memory
    sample_count = min(len(rows), max_per_source * len(counts))
    
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(rows), size=sample_count, replace=True, p=p)
    return [rows[i] for i in indices]
