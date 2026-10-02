"""Signal 9 analysis GUI.

Implements the problem statement's GUI requirement: take a ``.wav`` or ``.iq``
file as input and make the analysis visible -- signal parameters, waterfall
(time-frequency), constellation, demodulated / decoded bit stream and the
frame (header / payload / CRC) structure -- with the full report exportable.

Layout (Decoding_Roadmap.md Part 15, trimmed to what the pipeline produces):

* top bar     file picker, sample-rate / centre-frequency / raw-format overrides,
              optional trained checkpoint, Analyze
* left        detected tracks
* tabs        Parameters | Waterfall & PSD | Constellation | Eye / Time |
              Bit stream | Report (JSON)

The analysis runs on a worker thread so the window stays responsive; the
pipeline itself is the same code path as the CLI (``main.py``).
"""
from __future__ import annotations

import json
import sys
import traceback
import os
from pathlib import Path
os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
os.environ["QT_API"] = "pyside6"


import numpy as np
import pyqtgraph as pg
try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    from PyQt6 import QtCore, QtGui, QtWidgets
    QtCore.Signal = QtCore.pyqtSignal
    QtCore.Slot = QtCore.pyqtSlot

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal9.pipeline import Signal9Engine  # noqa: E402

FILE_FILTER = ("RF captures (*.wav *.iq *.raw *.bin *.dat *.cf32 *.cfile *.cu8 *.cs8 *.cs16 "
               "*.cf64 *.sigmf-meta *.sigmf-data);;All files (*)")


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisWorker(QtCore.QObject):
    finished = QtCore.Signal(dict, dict)
    failed = QtCore.Signal(str)

    def __init__(self, path, fs, fc, fmt, checkpoint, max_tracks):
        super().__init__()
        self.args = (path, fs, fc, fmt, checkpoint, max_tracks)

    @QtCore.Slot()
    def run(self):
        path, fs, fc, fmt, ckpt, max_tracks = self.args
        try:
            engine = Signal9Engine(checkpoint=ckpt or None)
            artifacts: dict = {}
            report = engine.process_file(path, fs=fs, fc=fc, fmt=fmt,
                                         max_exploit_tracks=max_tracks,
                                         min_exploit_snr_db=2.0,
                                         artifacts=artifacts)
            self.finished.emit(report, artifacts)
        except Exception:
            self.failed.emit(traceback.format_exc())


def analyze_sync(path, fs=None, fc=None, fmt=None, checkpoint=None, max_tracks=8):
    """Run an analysis without the GUI event loop (used by tests and scripts)."""
    engine = Signal9Engine(checkpoint=checkpoint or None)
    artifacts: dict = {}
    report = engine.process_file(path, fs=fs, fc=fc, fmt=fmt,
                                 max_exploit_tracks=max_tracks, artifacts=artifacts)
    return report, artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v, unit="", nd=1):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}{unit}"
    return f"{v}{unit}"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal 9 — Blind RF Signal Analysis")
        self.resize(1400, 860)
        pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")
        self.report: dict = {}
        self.artifacts: dict = {}
        self._thread = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # ── input bar ────────────────────────────────────────────────────
        bar = QtWidgets.QGridLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("Select a .wav / .iq / .cf32 / .sigmf capture…")
        browse = QtWidgets.QPushButton("Open…")
        browse.clicked.connect(self.browse)
        self.fs_edit = QtWidgets.QLineEdit()
        self.fs_edit.setPlaceholderText("auto (header / SigMF / filename)")
        self.fc_edit = QtWidgets.QLineEdit()
        self.fc_edit.setPlaceholderText("optional")
        self.fmt_combo = QtWidgets.QComboBox()
        self.fmt_combo.addItems(["auto", "cf32", "cs16", "cu8", "cs8", "cf64"])
        self.ckpt_edit = QtWidgets.QLineEdit()
        self.ckpt_edit.setPlaceholderText("optional trained model (.ckpt)")
        for candidate_ckpt in [
            PROJECT_ROOT / "models" / "s9_hybrid_xcit_multilength_hardened.ckpt",
            PROJECT_ROOT / "models" / "s9_hybrid_xcit_L128_best.ckpt",
            PROJECT_ROOT / "models" / "s9_satvik_grand_corpus_best.ckpt",
            PROJECT_ROOT / "models" / "s9_satvik.ckpt",
            Path.home() / "Downloads" / "s9_hybrid_xcit_multilength_hardened(1).ckpt",
            Path.home() / "Downloads" / "s9_satvik_grand_corpus_best(1).ckpt",
            Path.home() / "Downloads" / "s9_satvik_grand_corpus_best.ckpt",
            Path.home() / "Downloads" / "s9_satvik.ckpt",
        ]:
            if candidate_ckpt.exists():
                self.ckpt_edit.setText(str(candidate_ckpt))
                break
        ckpt_btn = QtWidgets.QPushButton("…")
        ckpt_btn.clicked.connect(self.browse_ckpt)
        self.tracks_spin = QtWidgets.QSpinBox()
        self.tracks_spin.setRange(1, 64)
        self.tracks_spin.setValue(8)
        self.run_btn = QtWidgets.QPushButton("Analyze")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 6px 18px;")
        self.run_btn.clicked.connect(self.start_analysis)

        bar.addWidget(QtWidgets.QLabel("Capture"), 0, 0)
        bar.addWidget(self.path_edit, 0, 1, 1, 5)
        bar.addWidget(browse, 0, 6)
        bar.addWidget(self.run_btn, 0, 7, 2, 1)
        bar.addWidget(QtWidgets.QLabel("Sample rate (Hz)"), 1, 0)
        bar.addWidget(self.fs_edit, 1, 1)
        bar.addWidget(QtWidgets.QLabel("Centre freq (Hz)"), 1, 2)
        bar.addWidget(self.fc_edit, 1, 3)
        bar.addWidget(QtWidgets.QLabel("Raw format"), 1, 4)
        bar.addWidget(self.fmt_combo, 1, 5)
        bar.addWidget(QtWidgets.QLabel("Max tracks"), 1, 6)
        bar.addWidget(self.tracks_spin, 2, 6)
        bar.addWidget(QtWidgets.QLabel("Checkpoint"), 2, 0)
        bar.addWidget(self.ckpt_edit, 2, 1, 1, 4)
        bar.addWidget(ckpt_btn, 2, 5)
        root.addLayout(bar)

        self.banner = QtWidgets.QLabel("Open a capture and press Analyze.")
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("padding: 4px; background: #eef3f8; border-radius: 4px;")
        root.addWidget(self.banner)

        # ── body ─────────────────────────────────────────────────────────
        split = QtWidgets.QSplitter()
        root.addWidget(split, 1)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(QtWidgets.QLabel("Detected tracks"))
        self.track_list = QtWidgets.QListWidget()
        self.track_list.currentRowChanged.connect(self.show_track)
        lv.addWidget(self.track_list)
        split.addWidget(left)

        self.tabs = QtWidgets.QTabWidget()
        split.addWidget(self.tabs)
        split.setSizes([260, 1140])

        # Parameters
        self.param_table = QtWidgets.QTableWidget(0, 2)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.param_table.horizontalHeader().setStretchLastSection(True)
        self.param_table.verticalHeader().setVisible(False)
        self.param_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tabs.addTab(self.param_table, "Parameters")

        # Waterfall + PSD
        wf = pg.GraphicsLayoutWidget()
        self.wf_plot = wf.addPlot(title="Waterfall (time–frequency)")
        self.wf_plot.setLabel("bottom", "Frequency", "Hz")
        self.wf_plot.setLabel("left", "Time", "s")
        self.wf_img = pg.ImageItem()
        self.wf_plot.addItem(self.wf_img)
        self.wf_bar = pg.ColorBarItem(values=(0, 1), colorMap=pg.colormap.get("viridis"))
        self.wf_bar.setImageItem(self.wf_img, insert_in=self.wf_plot)
        wf.nextRow()
        self.psd_plot = wf.addPlot(title="Power spectral density")
        self.psd_plot.setLabel("bottom", "Frequency", "Hz")
        self.psd_plot.setLabel("left", "Power", "dB")
        self.psd_plot.setXLink(self.wf_plot)
        self.tabs.addTab(wf, "Waterfall && PSD")

        # Constellation
        self.const_plot = pg.PlotWidget(title="Constellation (recovered symbols)")
        self.const_plot.setAspectLocked(True)
        self.const_plot.showGrid(x=True, y=True, alpha=0.3)
        self.tabs.addTab(self.const_plot, "Constellation")

        # Eye / time
        et = pg.GraphicsLayoutWidget()
        self.eye_plot = et.addPlot(title="Eye diagram")
        et.nextRow()
        self.time_plot = et.addPlot(title="Channelised signal (I / Q)")
        self.tabs.addTab(et, "Eye / Time")

        # Bit stream
        bits_w = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(bits_w)
        self.frame_label = QtWidgets.QLabel()
        self.frame_label.setWordWrap(True)
        bl.addWidget(self.frame_label)
        bg = pg.GraphicsLayoutWidget()
        self.bits_plot = bg.addPlot(title="Bit matrix (one frame per row) — header | payload | CRC")
        self.bits_plot.setLabel("bottom", "Bit position in frame")
        self.bits_plot.setLabel("left", "Frame")
        self.bits_img = pg.ImageItem()
        self.bits_plot.addItem(self.bits_img)
        bg.nextRow()
        self.ent_plot = bg.addPlot(title="Per-position entropy (0 = constant field, 1 = payload)")
        self.ent_plot.setXLink(self.bits_plot)
        bl.addWidget(bg, 1)
        row = QtWidgets.QHBoxLayout()
        exp_bits = QtWidgets.QPushButton("Export decoded bits…")
        exp_bits.clicked.connect(self.export_bits)
        row.addWidget(exp_bits)
        compare_bits = QtWidgets.QPushButton("Compare reference bits…")
        compare_bits.clicked.connect(self.compare_reference_bits)
        row.addWidget(compare_bits)
        row.addStretch(1)
        bl.addLayout(row)
        self.tabs.addTab(bits_w, "Bit stream")

        # Decoded message
        msg_w = QtWidgets.QWidget()
        ml = QtWidgets.QVBoxLayout(msg_w)
        self.msg_summary = QtWidgets.QLabel("Decoded message")
        self.msg_summary.setWordWrap(True)
        ml.addWidget(self.msg_summary)
        self.msg_text = QtWidgets.QPlainTextEdit()
        self.msg_text.setReadOnly(True)
        self.msg_text.setMaximumHeight(140)
        self.msg_text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        ml.addWidget(self.msg_text)
        self.frames_table = QtWidgets.QTableWidget(0, 4)
        self.frames_table.setHorizontalHeaderLabels(["Frame", "CRC", "Payload (text)", "Frame (hex)"])
        self.frames_table.horizontalHeader().setStretchLastSection(True)
        self.frames_table.verticalHeader().setVisible(False)
        self.frames_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.frames_table.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        ml.addWidget(self.frames_table, 1)
        exp_frames = QtWidgets.QPushButton("Export decoded frames (CSV)…")
        exp_frames.clicked.connect(self.export_frames)
        ml.addWidget(exp_frames, alignment=QtCore.Qt.AlignLeft)
        self.tabs.addTab(msg_w, "Decoded message")

        # Report
        rep_w = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(rep_w)
        self.report_text = QtWidgets.QPlainTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        rl.addWidget(self.report_text)
        exp = QtWidgets.QPushButton("Export JSON report…")
        exp.clicked.connect(self.export_json)
        rl.addWidget(exp, alignment=QtCore.Qt.AlignLeft)
        self.tabs.addTab(rep_w, "Report (JSON)")

        self.statusBar().showMessage("Ready")

    # ── actions ────────────────────────────────────────────────────────
    def browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open capture", "", FILE_FILTER)
        if path:
            self.path_edit.setText(path)

    def browse_ckpt(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Model checkpoint", "",
                                                        "Checkpoints (*.ckpt *.pt *.pth);;All files (*)")
        if path:
            self.ckpt_edit.setText(path)

    def _num(self, edit):
        t = edit.text().strip()
        return float(t) if t else None

    def start_analysis(self):
        path = self.path_edit.text().strip()
        if not path or not Path(path).exists():
            QtWidgets.QMessageBox.warning(self, "Signal 9", "Select an existing capture file first.")
            return
        try:
            fs, fc = self._num(self.fs_edit), self._num(self.fc_edit)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Signal 9", "Sample rate / centre frequency must be numbers.")
            return
        fmt = None if self.fmt_combo.currentText() == "auto" else self.fmt_combo.currentText()
        self.run_btn.setEnabled(False)
        self.statusBar().showMessage("Analysing… (demodulation, FEC and frame searches can take a few seconds per track)")
        self._thread = QtCore.QThread(self)
        self._worker = AnalysisWorker(path, fs, fc, fmt, self.ckpt_edit.text().strip(),
                                      self.tracks_spin.value())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self.on_finished)
        self._worker.failed.connect(self.on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def on_failed(self, tb: str):
        self.run_btn.setEnabled(True)
        self.statusBar().showMessage("Analysis failed")
        QtWidgets.QMessageBox.critical(self, "Signal 9 — analysis failed", tb[-3000:])

    def on_finished(self, report: dict, artifacts: dict):
        self.run_btn.setEnabled(True)
        self.load_results(report, artifacts)

    # ── rendering ──────────────────────────────────────────────────────
    def load_results(self, report: dict, artifacts: dict):
        self.report, self.artifacts = report, artifacts
        inp = report.get("input", {})
        fs = report.get("sample_rate_sps", 0.0)
        warn = ("  ⚠ Sample rate ASSUMED (not in the file) — absolute frequencies are nominal; "
                "enter the real rate and re-run." if inp.get("sample_rate_assumed") else "")
        self.banner.setText(
            f"<b>{Path(report.get('file', '')).name}</b> — fs {fs:,.0f} S/s "
            f"[source: {inp.get('sample_rate_source')}], input {inp.get('input_kind')}, "
            f"{report.get('total_signals_detected', 0)} track(s) detected.{warn}")
        self.report_text.setPlainText(json.dumps(report, indent=2, default=str))

        self.track_list.clear()
        for sig in report.get("signals", []):
            ex = sig.get("exploitation") or {}
            d = ex.get("demodulation", {})
            label = (f"#{sig['track_id']}  {sig.get('freq_center_relative_hz', 0) / 1e3:+.2f} kHz  "
                     f"{d.get('modulation') or '—'}  {ex.get('status', 'not demodulated')}")
            self.track_list.addItem(label)
        self.draw_waterfall()
        if self.track_list.count():
            self.track_list.setCurrentRow(0)
        self.statusBar().showMessage("Analysis complete")

    def draw_waterfall(self):
        spec = self.artifacts.get("spectrogram")
        if not spec:
            return
        f, t, P = np.asarray(spec["f"]), np.asarray(spec["t"]), np.asarray(spec["P_db"])
        img = P.T                                        # rows = time, cols = freq
        lo, hi = np.percentile(img, 5), np.percentile(img, 99.5)
        self.wf_img.setImage(img, levels=(lo, hi))
        df = (f[-1] - f[0]) / max(1, len(f) - 1)
        dt = (t[-1] - t[0]) / max(1, len(t) - 1) if len(t) > 1 else 1.0
        self.wf_img.setRect(QtCore.QRectF(f[0] - df / 2, t[0] - dt / 2, df * len(f), dt * max(1, len(t))))
        self.wf_bar.setLevels((lo, hi))
        self.psd_plot.clear()
        self.psd_plot.plot(f, 10 * np.log10(np.mean(10 ** (P / 10), axis=1) + 1e-20), pen=pg.mkPen("#1e88e5"))
        for sig in self.report.get("signals", []):
            c, bw = sig.get("freq_center_relative_hz", 0.0), sig.get("bandwidth_hz", 0.0)
            box = pg.LinearRegionItem(values=(c - bw / 2, c + bw / 2), movable=False,
                                      brush=pg.mkBrush(255, 80, 80, 40))
            self.psd_plot.addItem(box)

    def _track_artifacts(self, track_id):
        for tr in self.artifacts.get("tracks", []):
            if tr["track_id"] == track_id:
                return tr
        return None

    def show_track(self, row: int):
        sigs = self.report.get("signals", [])
        if row < 0 or row >= len(sigs):
            return
        sig = sigs[row]
        ex = sig.get("exploitation") or {}
        self.fill_params(sig, ex)
        tr = self._track_artifacts(sig["track_id"])
        self.draw_constellation(ex, tr)
        self.draw_eye_time(ex, tr)
        self.draw_bits(ex, tr)
        self.show_message(ex)

    def show_message(self, ex: dict):
        dec = ex.get("decoded") or {}
        rows = dec.get("frames") or []
        self.frames_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            mark = {True: "OK", False: "BAD", None: "—"}[r.get("crc_ok")]
            items = [str(r["index"]), mark, r.get("payload_text", ""), r.get("hex", "")]
            for j, v in enumerate(items):
                it = QtWidgets.QTableWidgetItem(v)
                if j == 1 and mark != "—":
                    it.setForeground(QtGui.QColor("#2e7d32" if mark == "OK" else "#c62828"))
                self.frames_table.setItem(i, j, it)
        self.frames_table.resizeColumnsToContents()
        if rows:
            self.msg_summary.setText(
                f"<b>Decoded message</b> — payloads of the {dec.get('frames_crc_ok', 0)} CRC-valid "
                f"frames out of {len(rows)}, in order:")
            self.msg_text.setPlainText(dec.get("message_text", ""))
        elif dec.get("preview_text"):
            self.msg_summary.setText("<b>Decoded bytes</b> — no frame structure was identified, "
                                     "so this is a readability-selected preview, without CRC validation:")
            self.msg_text.setPlainText(dec["preview_text"] + "\n\n" + dec.get("preview_hex", ""))
        else:
            self.msg_summary.setText("No decoded data for this track.")
            self.msg_text.setPlainText("")

    def export_frames(self):
        row = self.track_list.currentRow()
        sigs = self.report.get("signals", [])
        if row < 0 or row >= len(sigs):
            return
        rows = ((sigs[row].get("exploitation") or {}).get("decoded") or {}).get("frames") or []
        if not rows:
            QtWidgets.QMessageBox.information(self, "Signal 9", "No decoded frames for this track.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save decoded frames", "frames.csv", "CSV (*.csv)")
        if not path:
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "crc_ok", "payload_text", "payload_hex", "frame_hex"])
            for r in rows:
                w.writerow([r["index"], r["crc_ok"], r["payload_text"], r["payload_hex"], r["hex"]])
        self.statusBar().showMessage(f"{len(rows)} frames saved to {path}")

    def fill_params(self, sig: dict, ex: dict):
        d = ex.get("demodulation", {})
        fsk = d.get("fsk") or {}
        fec = ex.get("fec", {})
        il = ex.get("interleaver", {})
        fr = ex.get("frames", {})
        crc = fr.get("crc") or {}
        amb = ex.get("bit_ambiguity", {})
        cls = sig.get("classification", {}).get("l2", {})
        inp = self.report.get("input", {})
        if fec.get("class") == "concatenated":
            fec_s = (f"concatenated: inner {fec['inner'].get('name')} + outer "
                     f"RS({fec['outer'].get('n')},{fec['outer'].get('k')})")
        elif fec.get("class") == "convolutional":
            fec_s = (f"convolutional K={fec.get('K')} rate {fec.get('rate')} {fec.get('polys')} "
                     f"(re-encode agreement {fec.get('reencode_agreement')})")
        elif fec.get("class") == "reed_solomon":
            fec_s = f"Reed–Solomon RS({fec.get('n')},{fec.get('k')}) {fec.get('words_ok')}/{fec.get('words_checked')} codewords"
        elif fec.get("class") == "ldpc":
            fec_s = f"LDPC ({fec.get('n')},{fec.get('k')}) {fec.get('blocks_converged')}/{fec.get('blocks_checked')} blocks"
        else:
            fec_s = fec.get("class") or "not run"
        geo = ", ".join(f"{k}={il[k]}" for k in ("rows", "cols", "slope", "branches", "delay", "size", "phase") if k in il)
        rows = [
            ("Sampling frequency", f"{self.report.get('sample_rate_sps', 0):,.0f} S/s  [{inp.get('sample_rate_source')}]"),
            ("Carrier offset (total)", _fmt(sig.get("carrier_offset_hz"), " Hz")),
            ("Occupied bandwidth", f"{sig.get('bandwidth_hz', 0):,.1f} Hz"),
            ("Track SNR (detector)", _fmt(sig.get("snr_db"), " dB")),
            ("Post-filter SNR", _fmt((sig.get("snr_measurement") or {}).get("post_filter_db"), " dB")),
            ("Modulation decision", str(sig.get("modulation_decision", {}).get("modulation", "UNRESOLVED")) +
             " [" + str(sig.get("modulation_decision", {}).get("decision_state", "UNKNOWN")) + "]"),
            ("Modulation (measured)", f"{d.get('modulation', '—')}  ({d.get('family', '—')})"),
            ("Modulation (neural)", f"{cls.get('prediction')}  [{cls.get('decision_state')}]"),
            ("Symbol rate", _fmt(d.get("symbol_rate_hz"), " Bd")),
            ("Roll-off", _fmt(d.get("rolloff"), "", 2) if d.get("family") == "linear" else "n/a"),
            ("EVM", _fmt(d.get("evm_db"), " dB") if d.get("family") == "linear" else "n/a"),
            ("FSK tones / index h", f"{fsk.get('n_tones')} / {fsk.get('modulation_index')}" if fsk else "n/a"),
            ("Bit mapping", (f"rotation {amb.get('rotation_deg')}°, mirror {amb.get('mirror')}" if "rotation_deg" in amb
                             else f"tone polarity {amb.get('tone_polarity')}") +
             f", line code {amb.get('line_code')}, differential {amb.get('differential')}" if amb else "—"),
            ("Scrambler", (ex.get("scrambler") or {}).get("type") or "not run"),
            ("Interleaving", f"{il.get('type') or 'not run'} {('(' + geo + ')') if geo else ''}"),
            ("FEC", fec_s),
            ("Frame period", _fmt(fr.get("period_bits"), " bits")),
            ("Byte preview alignment", str((ex.get("decoded") or {}).get("byte_alignment", "n/a"))),
            ("Sync word", f"{(fr.get('sync') or {}).get('length', 0)} bits at {(fr.get('sync') or {}).get('offset', '—')}"),
            ("Payload span (bits)", str((fr.get("payload") or {}).get("bits"))),
            ("Frame CRC", f"{crc.get('model')}  {fr.get('crc_frames_ok')}/{fr.get('crc_frames_checked')} frames" if crc else "none identified"),
            ("Chain status", ex.get("status", "not demodulated (below SNR floor or track limit)")),
        ]
        for note in ex.get("notes", []):
            rows.append(("Note", note))
        self.param_table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.param_table.setItem(i, 0, QtWidgets.QTableWidgetItem(k))
            self.param_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(v)))
        self.param_table.resizeColumnToContents(0)

    def draw_constellation(self, ex: dict, tr):
        self.const_plot.clear()
        if not tr:
            self.const_plot.setTitle("Constellation — select a detected track from the left panel")
            return

        symbols = tr.get("symbols")
        if symbols is not None and len(symbols) > 0:
            s = np.asarray(symbols)[:6000]
            fam = ex.get("demodulation", {}).get("family")
            if fam == "fsk":
                v = s.real
                y, x = np.histogram(v, bins=120)
                self.const_plot.setAspectLocked(False)
                self.const_plot.plot(x, y, stepMode="center", fillLevel=0, brush=(30, 136, 229, 120))
                self.const_plot.setTitle("FSK tone levels (integrate-and-dump histogram, Hz)")
                return
            self.const_plot.setAspectLocked(True)
            s = s / np.sqrt(np.mean(np.abs(s) ** 2) + 1e-12)
            self.const_plot.plot(s.real, s.imag, pen=None, symbol="o", symbolSize=4,
                                 symbolBrush=(30, 136, 229, 140), symbolPen=None)
            d = ex.get("demodulation", {})
            mod_str = d.get('modulation') or "demodulated"
            evm_str = f"  EVM {d.get('evm_db')} dB" if d.get('evm_db') is not None else ""
            self.const_plot.setTitle(f"Constellation (Equalized Symbols) — {mod_str}{evm_str}")
        elif tr.get("iq") is not None and len(tr["iq"]) > 0:
            # Fallback: display raw channelized baseband IQ samples (before/without symbol recovery)
            iq = np.asarray(tr["iq"])[:4000]
            self.const_plot.setAspectLocked(True)
            pwr = np.mean(np.abs(iq) ** 2)
            if pwr > 0:
                iq = iq / np.sqrt(pwr)
            self.const_plot.plot(iq.real, iq.imag, pen=None, symbol="o", symbolSize=3,
                                 symbolBrush=(229, 57, 53, 110), symbolPen=None)
            self.const_plot.setTitle("Constellation (Channelized Baseband IQ — Raw Scatter)")
        else:
            self.const_plot.setTitle("Constellation — no signal samples available for this track")

    def draw_eye_time(self, ex: dict, tr):
        self.eye_plot.clear()
        self.time_plot.clear()
        if not tr:
            return
        iq = np.asarray(tr.get("iq", []))
        fs = float(tr.get("fs", 1.0))
        if iq.size:
            n = min(len(iq), 4000)
            t = np.arange(n) / fs
            self.time_plot.plot(t, iq[:n].real, pen=pg.mkPen("#1e88e5"))
            self.time_plot.plot(t, iq[:n].imag, pen=pg.mkPen("#e53935"))
            self.time_plot.setLabel("bottom", "Time", "s")
        d = ex.get("demodulation", {})
        rs = float(d.get("symbol_rate_hz") or 0.0)
        if iq.size and rs > 0:
            sps = fs / rs
            x = np.angle(iq[1:] * np.conj(iq[:-1])) * fs / (2 * np.pi) if d.get("family") == "fsk" else iq.real
            seg = int(round(2 * sps))
            if seg >= 4:
                for k in range(min(200, int(len(x) / sps) - 3)):
                    s0 = int(round(k * sps))
                    self.eye_plot.plot(np.arange(seg) / sps, x[s0:s0 + seg],
                                       pen=pg.mkPen((30, 136, 229, 40)))
                self.eye_plot.setTitle(f"Eye diagram ({'frequency' if d.get('family') == 'fsk' else 'in-phase'}, 2 symbols)")

    def draw_bits(self, ex: dict, tr):
        self.bits_plot.clear()
        self.ent_plot.clear()
        self.bits_plot.addItem(self.bits_img)
        fr = ex.get("frames", {})
        bits = np.asarray(tr.get("decoded_bits", [])) if tr else np.zeros(0)
        P = int(fr.get("period_bits") or 0)
        crc = fr.get("crc") or {}
        self.frame_label.setText(
            f"Frame model: period {P or '—'} bits, sync {(fr.get('sync') or {}).get('length', 0)} bits, "
            f"payload bits {(fr.get('payload') or {}).get('bits')}, CRC "
            f"{crc.get('model', 'none')} ({fr.get('crc_frames_ok', 0)}/{fr.get('crc_frames_checked', 0)} frames ok). "
            f"Decoded bits: {len(bits)}.")
        if len(bits) == 0 or P < 8 or len(bits) < 2 * P:
            self.bits_img.clear()
            return
        # Apply the frame model's own phase so rows start at a frame boundary
        # and line up with the sync / payload / CRC spans drawn below.
        bits = bits[int(fr.get("frame_offset_bits") or 0):]
        rows = min(len(bits) // P, 200)
        m = bits[: rows * P].reshape(rows, P).astype(np.float32)
        self.bits_img.setImage(m, levels=(0, 1))
        p1 = np.clip(m.mean(axis=0), 1e-6, 1 - 1e-6)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -(p1 * np.log2(p1) + (1 - p1) * np.log2(1 - p1))
        self.ent_plot.plot(np.arange(P) + 0.5, ent, stepMode=None, pen=pg.mkPen("#43a047", width=2))
        sync = fr.get("sync") or {}
        spans = [((sync.get("offset", 0) or 0), (sync.get("offset", 0) or 0) + sync.get("length", 0), (255, 193, 7, 60))]
        pay = (fr.get("payload") or {}).get("bits")
        if pay:
            spans.append((pay[0], pay[1], (33, 150, 243, 40)))
        if crc.get("span"):
            spans.append((crc["span"][0], crc["span"][1], (229, 57, 53, 60)))
        for a, b, col in spans:
            if b > a:
                self.ent_plot.addItem(pg.LinearRegionItem(values=(a, b), movable=False, brush=pg.mkBrush(*col)))

    # ── export ─────────────────────────────────────────────────────────
    def export_json(self):
        if not self.report:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save report", "report.json", "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(self.report, indent=2, default=str), encoding="utf-8")
            self.statusBar().showMessage(f"Report saved to {path}")

    def compare_reference_bits(self):
        from ..dsp.bit_alignment import reference_ber
        row = self.track_list.currentRow()
        sigs = self.report.get("signals", [])
        if row < 0 or row >= len(sigs):
            return
        tr = self._track_artifacts(sigs[row]["track_id"])
        bits = tr.get("decoded_bits") if tr else None
        if bits is None or len(bits) == 0:
            QtWidgets.QMessageBox.information(self, "Reference BER", "No decoded bits for this track.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Reference bits (0/1 text)", "", "Text (*.txt)")
        if not path:
            return
        offset, ok = QtWidgets.QInputDialog.getInt(self, "Reference BER", "Offset in decoded bits:", 0, 0, len(bits))
        if not ok:
            return
        try:
            text = "".join(Path(path).read_text().split())
            if not text or set(text) - {"0", "1"}:
                raise ValueError("Reference must contain only 0/1 bits and whitespace.")
            result = reference_ber(bits, np.fromiter(map(int, text), dtype=np.uint8), offset)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Reference BER", str(exc))
            return
        sigs[row].setdefault("exploitation", {})["reference_ber"] = result
        self.report_text.setPlainText(json.dumps(self.report, indent=2, default=str))
        QtWidgets.QMessageBox.information(self, "Reference BER", json.dumps(result, indent=2))

    def export_bits(self):
        row = self.track_list.currentRow()
        sigs = self.report.get("signals", [])
        if row < 0 or row >= len(sigs):
            return
        tr = self._track_artifacts(sigs[row]["track_id"])
        if not tr or len(tr.get("decoded_bits", [])) == 0:
            QtWidgets.QMessageBox.information(self, "Signal 9", "No decoded bits for this track.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save decoded bits", "track_bits.bin",
                                                        "Packed bytes (*.bin);;Text 0/1 (*.txt)")
        if not path:
            return
        bits = np.asarray(tr["decoded_bits"], dtype=np.uint8)
        if path.endswith(".txt"):
            Path(path).write_text("".join(map(str, bits.tolist())), encoding="ascii")
        else:
            Path(path).write_bytes(np.packbits(bits).tobytes())
        self.statusBar().showMessage(f"{len(bits)} bits saved to {path}")


def main(argv=None):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv or sys.argv)
    w = MainWindow()
    args = (argv or sys.argv)[1:]
    if args:
        w.path_edit.setText(args[0])
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
