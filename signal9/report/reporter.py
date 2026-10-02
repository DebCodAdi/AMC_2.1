"""Report generation: JSON summary, SigMF annotations, and plot exporter.

Phase 0 change: to_dict() now emits the full telemetry schema required by
the Failure Attribution Waterfall (spec §2.2).  Every key in
``contracts.empty_telemetry()`` is populated before output.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any
import numpy as np

from signal9.status import Status
from signal9.contracts import empty_telemetry
from signal9.decide.reconcile import reconcile_modulation


class Signal9Reporter:
    """Generates structured analysis outputs for Signal 9 detections."""
    def __init__(self, filepath: str | Path, fs: float, fc: float = 0.0):
        self.filepath = Path(filepath)
        self.fs = fs
        self.fc = fc
        self.health_info: dict[str, Any] = {}
        self.input_info: dict[str, Any] = {}
        self.detections: list[dict[str, Any]] = []

        # ── Phase 0: telemetry accumulator ───────────────────────────────────
        self._telemetry: dict[str, Any] = empty_telemetry()
        self._status: str = Status.NO_SIGNAL_DETECTED.value
        self._t_start: float = time.perf_counter()
        self._stage_timers: dict[str, float] = {}  # {stage_name: start_time}

    def set_health(self, clipping: float, impropriety_db: float | None, dc_spike: bool):
        self.health_info = {
            "clipping_fraction": float(clipping),
            "iq_impropriety_db": float(impropriety_db) if impropriety_db is not None else None,
            "dc_spike_detected": bool(dc_spike),
        }

    def set_input_info(self, info: dict[str, Any]) -> None:
        """Record where the sample rate / centre frequency came from, and the input kind."""
        self.input_info = dict(info)

    def add_detection(self, track_info: dict, predictions: dict,
                      exploitation: dict | None = None):
        det = {
            "track_id": len(self.detections) + 1,
            "t_start_sec": track_info.get("t_lo", 0.0),
            "t_end_sec": track_info.get("t_hi", 0.0),
            "duration_sec": track_info.get("duration", 0.0),
            "freq_center_relative_hz": track_info.get("center_freq", 0.0),
            "freq_center_absolute_hz": self.fc + track_info.get("center_freq", 0.0),
            "bandwidth_hz": track_info.get("bandwidth", 0.0),
            "snr_db": predictions.get("track_snr_db", 0.0),
            "classification": predictions.get("levels", {}),
        }
        if exploitation is not None:
            det["exploitation"] = exploitation
        demod = (exploitation or {}).get("demodulation", {})
        det["modulation_decision"] = reconcile_modulation(
            predictions.get("levels", {}).get("l2", {}), demod)
        detector_snr = track_info.get("detector_snr_db")
        filtered_snr = track_info.get("post_filter_snr_db")
        det["snr_measurement"] = {"definition": "in_band_signal_over_noise",
                                  "detector_db": detector_snr, "post_filter_db": filtered_snr,
                                  "interference_suspected": bool(detector_snr is not None and filtered_snr is not None
                                                                and abs(detector_snr - filtered_snr) > 3)}
        fine = demod.get("cfo_hz")
        det["carrier_offset_hz"] = (float(track_info.get("center_freq", 0)) + float(fine)
                                    if fine is not None and demod.get("modulation") != "UNRESOLVED" else None)
        det["carrier_frequency_hz"] = self.fc + det["carrier_offset_hz"] if det["carrier_offset_hz"] is not None else None
        self.detections.append(det)
        # A detection is already evidence that the detector succeeded.  Until
        # a downstream decoder promotes the result to SUCCESS_CRC_LOCKED (or
        # records a more specific failure), the truthful aggregate status is
        # the existing soft-success state rather than NO_SIGNAL_DETECTED.
        if self._status == Status.NO_SIGNAL_DETECTED.value:
            self._status = Status.DEMOD_NO_CRC.value

    def flag_overlapping_rates(self) -> None:
        """Flag near-identical clocks only when tracks overlap in time and frequency."""
        for i, a in enumerate(self.detections):
            for b in self.detections[i + 1:]:
                da = a.get("exploitation", {}).get("demodulation", {})
                db = b.get("exploitation", {}).get("demodulation", {})
                ra, rb = da.get("symbol_rate_hz", 0), db.get("symbol_rate_hz", 0)
                time_overlap = min(a["t_end_sec"], b["t_end_sec"]) > max(a["t_start_sec"], b["t_start_sec"])
                freq_overlap = abs(a["freq_center_relative_hz"] - b["freq_center_relative_hz"]) < (a["bandwidth_hz"] + b["bandwidth_hz"]) / 2
                if ra > 0 and rb > 0 and abs(ra - rb) / max(ra, rb) < 0.01 and time_overlap and freq_overlap:
                    for track, other in ((a, b), (b, a)):
                        ids = track.setdefault("duplicate_rate_track_ids", [])
                        if other["track_id"] not in ids:
                            ids.append(other["track_id"])

    # ── Telemetry setters ─────────────────────────────────────────────────

    def set_status(self, status: Status | str) -> None:
        """Update the top-level pipeline status."""
        if isinstance(status, Status):
            self._status = status.value
        else:
            self._status = str(status)

    def set_telemetry(self, **kwargs: Any) -> None:
        """Merge kwargs into the telemetry dict.  Only known keys are accepted."""
        known = set(empty_telemetry().keys())
        for k, v in kwargs.items():
            if k in known:
                self._telemetry[k] = v
            # Unknown keys: silently ignore (don't blow up production)

    def stage_start(self, name: str) -> None:
        """Mark the start of a named pipeline stage for timing."""
        self._stage_timers[name] = time.perf_counter()

    def stage_end(self, name: str) -> None:
        """Mark the end of a named pipeline stage and accumulate ms."""
        if name in self._stage_timers:
            elapsed_ms = (time.perf_counter() - self._stage_timers.pop(name)) * 1000.0
            t = self._telemetry.setdefault("t_stage_ms", {})
            t[name] = round(elapsed_ms, 2)

    # ── Output ────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        t_total = (time.perf_counter() - self._t_start) * 1000.0
        self._telemetry["t_total_ms"] = round(t_total, 2)

        result = {
            "file": str(self.filepath),
            "status": self._status,
            "sample_rate_sps": self.fs,
            "center_freq_hz": self.fc,
            "input": self.input_info,
            "health": self.health_info,
            "total_signals_detected": len(self.detections),
            "signals": self.detections,
        }
        result.update(self._telemetry)
        return result

    def save_json(self, output_path: str | Path) -> None:
        out_p = Path(output_path)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def plot_detections(self, Pw_db: np.ndarray, f: np.ndarray, t: np.ndarray, output_png: str | Path) -> None:
        """Render annotated spectrogram PNG with detection bounding boxes."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            fig, ax = plt.subplots(figsize=(10, 6))
            extent = [t[0], t[-1], f[0] / 1e3, f[-1] / 1e3]
            im = ax.imshow(Pw_db, aspect="auto", origin="lower", extent=extent, cmap="viridis")
            fig.colorbar(im, ax=ax, label="Whitened Power (dB)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (kHz relative)")
            ax.set_title(f"Signal 9 Detections: {self.filepath.name} ({len(self.detections)} emissions)")

            for d in self.detections:
                t0 = d["t_start_sec"]
                dt = d["duration_sec"]
                flo = (d["freq_center_relative_hz"] - 0.5 * d["bandwidth_hz"]) / 1e3
                df = d["bandwidth_hz"] / 1e3
                rect = patches.Rectangle((t0, flo), dt, df, linewidth=1.5, edgecolor="red", facecolor="none")
                ax.add_patch(rect)
                l2_pred = d.get("classification", {}).get("l2", {}).get("prediction", "Sig")
                ax.text(t0, flo, f" #{d['track_id']}: {l2_pred}", color="yellow", fontsize=9, weight="bold")

            plt.tight_layout()
            plt.savefig(output_png, dpi=150)
            plt.close(fig)
        except Exception:
            pass  # Non-blocking if matplotlib is headless or errors
