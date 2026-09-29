"""Detection and tracking package for wideband signals."""
from .spectrogram import log_spectrogram
from .whiten import whiten_spectrogram
from .cfar import detect_cfar
from .tracker import track_detections, BoundingBox
from .temporal_baseline import TemporalBaseline

__all__ = [
    "log_spectrogram",
    "whiten_spectrogram",
    "detect_cfar",
    "track_detections",
    "BoundingBox",
    "TemporalBaseline",
]
