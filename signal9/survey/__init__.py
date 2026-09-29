"""Stage 0 Wideband Spectrum Survey package (Phase 3).

Orchestrates real-time wideband RF environment scanning:
  - Multi-resolution STFT bank (stft.py)
  - 2D Ordered-Statistic CFAR detector (detect_cfar.py)
  - Burst clustering & bounding-box tracking (cluster.py)
  - NCO downconversion, polyphase extraction & SigMF generation (extract.py)
  - Priority queue scheduling and load shedding (schedule.py)
"""
from __future__ import annotations

from .stft import multiresolution_stft, MultiresolutionSpectrogram
from .detect_cfar import os_cfar_2d
from .cluster import cluster_detections, SignalBurst
from .extract import extract_burst, BurstExtraction
from .schedule import SurveyScheduler, SurveyTask

__all__ = [
    "multiresolution_stft",
    "MultiresolutionSpectrogram",
    "os_cfar_2d",
    "cluster_detections",
    "SignalBurst",
    "extract_burst",
    "BurstExtraction",
    "SurveyScheduler",
    "SurveyTask",
]
