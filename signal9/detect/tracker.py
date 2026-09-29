"""Temporal tracker and bounding-box associator."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class BoundingBox:
    f_lo: float
    f_hi: float
    t_lo: float
    t_hi: float
    peak_db: float
    center_freq: float
    bandwidth: float
    duration: float

def track_detections(boxes: list[dict], freq_tol_hz: float = 500.0, max_gap_sec: float = 0.5) -> list[dict]:
    """Merge detections across time that belong to the same carrier / emitter track."""
    if not boxes:
        return []

    # Sort boxes by start time
    sorted_boxes = sorted(boxes, key=lambda b: b["t_lo"])
    tracks: list[dict] = []

    for b in sorted_boxes:
        matched = False
        for tr in tracks:
            # Check frequency overlap / proximity
            f_overlap = not (b["f_hi"] < tr["f_lo"] - freq_tol_hz or b["f_lo"] > tr["f_hi"] + freq_tol_hz)
            time_close = (b["t_lo"] - tr["t_hi"]) <= max_gap_sec
            if f_overlap and time_close:
                # Merge into existing track
                tr["f_lo"] = min(tr["f_lo"], b["f_lo"])
                tr["f_hi"] = max(tr["f_hi"], b["f_hi"])
                tr["t_lo"] = min(tr["t_lo"], b["t_lo"])
                tr["t_hi"] = max(tr["t_hi"], b["t_hi"])
                tr["peak_db"] = max(tr["peak_db"], b["peak_db"])
                tr["center_freq"] = 0.5 * (tr["f_lo"] + tr["f_hi"])
                tr["bandwidth"] = tr["f_hi"] - tr["f_lo"]
                tr["duration"] = tr["t_hi"] - tr["t_lo"]
                matched = True
                break
        if not matched:
            tracks.append(dict(b))

    return tracks
