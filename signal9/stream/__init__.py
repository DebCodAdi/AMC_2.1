"""Streaming ingestion, ring-buffers, and backpressure handling (Phase 5)."""
from __future__ import annotations

from .queue import StreamingRingBuffer, StreamBlock

__all__ = ["StreamingRingBuffer", "StreamBlock"]
