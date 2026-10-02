"""Production AMC engine executing the full v5 OmniSIG-class processing pipeline."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
import numpy as np

from .io.metadata import resolve_metadata
from .io.sniff import sniff_dtype
from .io.wav_io import read_wav
from .io.sigmf_io import read_sigmf
from .io.chunker import iter_raw_iq, iter_wav_iq, iter_sigmf_iq
from .health.clipping import clipping_fraction
from .health.iq_image import image_evidence_db, maybe_correct_iq_imbalance
from .health.dc_lo import detect_dc_spike, mask_dc_spike
from .health.agc_changepoint import detect_agc_ruptures
from .detect.spectrogram import log_spectrogram
from .detect.whiten import whiten_spectrogram
from .detect.temporal_baseline import TemporalBaseline
from .detect.cfar import detect_cfar
from .detect.multiscale import detect_multiscale, inband_snr
from .detect.tracker import track_detections
from .detect.yolo_adapter import YoloDetector
from .detect.merge import merge_boxes
from .channelize.channelizer import channelize
from .features.spectral import occupied_bandwidth
from .report.reporter import Signal9Reporter
from .status import Status


class Signal9Engine:
    # Scale used when a capture carries no sample rate at all; see process_file.
    NOMINAL_FS = 1_000_000.0

    def __init__(self, config_dir: str | Path | None = None, yolo_path: str | Path | None = None,
                 checkpoint: str | Path | None = None):
        """``checkpoint``: a trained ``signal9-multitask-v1`` file (from
        signal9.train.trainer).  Without one the neural classification levels
        are reported as NOT_TRAINED -- the exploitation chain's measured
        modulation is unaffected -- rather than emitting the outputs of a
        randomly initialised network as if they were classifications."""
        self.classifier = None
        if checkpoint is None:
            proj_root = Path(__file__).resolve().parent.parent
            candidates = [
                # Production-compatible Hybrid CNN/XCiT models (hardened & best)
                proj_root / "models" / "s9_hybrid_xcit_multilength_hardened.ckpt",
                proj_root / "models" / "s9_hybrid_xcit_L128_hardened.ckpt",
                proj_root / "models" / "s9_hybrid_xcit_L256_hardened.ckpt",
                proj_root / "models" / "s9_hybrid_xcit_L512_hardened.ckpt",
                proj_root / "models" / "s9_hybrid_xcit_L1024_hardened.ckpt",
                proj_root / "models" / "s9_hybrid_xcit_L128_best.ckpt",
                proj_root / "models" / "s9_satvik_grand_corpus_best.ckpt",
                proj_root / "models" / "s9_satvik.ckpt",
                Path.home() / "Downloads" / "s9_satvik_grand_corpus_best(1).ckpt",
                Path.home() / "Downloads" / "s9_satvik_grand_corpus_best.ckpt",
                Path.home() / "Downloads" / "s9_satvik.ckpt",
            ]
            checkpoint = next((p for p in candidates if p.exists()), None)

        if checkpoint:
            from .train.inference import MultiTaskClassifier
            try:
                self.classifier = MultiTaskClassifier.load(checkpoint)
            except Exception as e:
                print(f"[Signal9Engine Warning] Failed to load checkpoint {checkpoint}: {e}")
        cfg_path = Path(config_dir) if config_dir else Path(__file__).resolve().parent.parent / "configs"
        taxonomy_file = cfg_path / "taxonomy.yaml"
        self.taxonomy = {}
        if taxonomy_file.exists():
            with open(taxonomy_file, "r", encoding="utf-8") as f:
                self.taxonomy = yaml.safe_load(f)

        self.class_names = {
            "l1": self.taxonomy.get("l1", []),
            "l2": list(self.taxonomy.get("l2", {}).keys()),
            "l3": list(self.taxonomy.get("l3", {}).keys()),
        }

        # Signal9Net is an older research-only architecture.  It has no
        # compatible shipped checkpoint and must never be instantiated as a
        # fallback classifier.  All deployed predictions come exclusively
        # from ``MultiTaskClassifier`` above, or are explicitly NOT_TRAINED.
        self.yolo = YoloDetector(model_path=yolo_path)

    def process_file(self,
                     filepath: str | Path,
                     fs: float | None = None,
                     fc: float | None = None,
                     fmt: str | None = None,
                     chunk_size: int = 1_000_000,
                     exploit: bool = True,
                     max_exploit_tracks: int = 8,
                     min_exploit_snr_db: float = 6.0,
                     artifacts: dict | None = None) -> dict[str, Any]:
        """Analyse a capture end to end and return the JSON-ready report.

        Detection and neural classification run for every track.  When
        ``exploit`` is true the D0-D6 exploitation chain
        (:class:`signal9.exploit.ExploitationChain`) then runs on up to
        ``max_exploit_tracks`` tracks whose SNR clears ``min_exploit_snr_db``,
        and its parameter report -- modulation, symbol rate, roll-off, CFO,
        interleaver, FEC and frame/CRC model -- is attached to the track under
        ``"exploitation"``.  Before this, the file path stopped at
        classification, so no demodulation, de-interleaving, FEC or bitstream
        correlation ever ran on a real input file.

        ``artifacts``, when given a dict, is filled with the arrays a GUI needs
        and the JSON report cannot carry: the first chunk's spectrogram and,
        per exploited track, the channelised samples, the recovered symbols and
        the decoded bits.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        meta = resolve_metadata(filepath, cli_fs=fs, cli_fc=fc, cli_format=fmt)
        ext = filepath.suffix.lower()

        # A headerless capture with no sidecar, filename pattern or --fs has no
        # recoverable sample rate: the samples alone cannot tell 48 kS/s from
        # 2.4 MS/s (Decoding_Roadmap.md 2.3).  The pipeline's Hz-based
        # heuristics still need a scale, so a nominal rate is used -- but it is
        # reported as "assumed", together with rates normalised to fs, instead
        # of being passed off as a measurement as it used to be.
        fs_assumed = meta.sample_rate is None
        if fs_assumed:
            meta.sample_rate = self.NOMINAL_FS
            meta.sample_rate_source = "assumed"

        iq_fs = float(meta.sample_rate)
        iq_fc = float(meta.center_freq or 0.0)

        # 0.25s overlap as per plan
        overlap_samples = int(chunk_size * 0.125) if chunk_size > 4000 else 0

        # Setup iterator
        if ext == ".wav":
            data_mmap, resolved_fs, kind = read_wav(filepath)
            iq_fs = float(resolved_fs)
            meta.sample_rate = iq_fs
            # The header rate is used even when --fs was given, so that is its source.
            meta.sample_rate_source = "wav_header"
            fs_assumed = False
            meta.is_audio = (kind != "iq")
            meta.wav_kind = kind
            # `kind` must reach the chunker: it decides whether the two channels are
            # I/Q or genuine audio that needs a Hilbert transform (see iter_wav_iq).
            iterator = iter_wav_iq(data_mmap, block=chunk_size, overlap=overlap_samples, kind=kind)
        elif ext in (".sigmf-meta", ".sigmf-data"):
            data_mmap, _ = read_sigmf(filepath)
            iterator = iter_sigmf_iq(data_mmap, block=chunk_size, overlap=overlap_samples)
        else:
            use_fmt = meta.format or sniff_dtype(filepath)[0]
            iterator = iter_raw_iq(filepath, use_fmt, block=chunk_size, overlap=overlap_samples)

        reporter = Signal9Reporter(filepath, iq_fs, iq_fc)
        reporter.set_input_info({
            "sample_rate_source": meta.sample_rate_source,
            "center_freq_source": meta.center_freq_source,
            "sample_rate_assumed": bool(fs_assumed),
            "input_kind": getattr(meta, "wav_kind", None) or ("sigmf" if ext.startswith(".sigmf") else "raw_iq"),
            "sample_format": meta.format if ext not in (".wav",) and not ext.startswith(".sigmf") else None,
        })
        is_zero_if = getattr(meta, 'is_zero_if', True)

        chain = None
        if exploit:
            from .exploit import ExploitationChain
            chain = ExploitationChain()
        n_exploited = 0
        usable_chunks = 0
        # US10459020B2-style temporal learning: retain a per-frequency
        # baseline across chunks.  Local whitening remains the primary
        # detector; the temporal excess can only add evidence for an event.
        temporal_baseline = TemporalBaseline()
        if artifacts is not None:
            artifacts.setdefault("tracks", [])
            artifacts["fs"] = iq_fs

        for chunk_idx, iq_chunk in enumerate(iterator):
            if len(iq_chunk) < 64:
                continue
            usable_chunks += 1
            if len(iq_chunk) < 1024:
                iq_chunk = np.pad(iq_chunk, (0, 1024 - len(iq_chunk)), mode="constant")

            # The chunkers already deliver complex baseband: raw/SigMF readers pair
            # interleaved samples and iter_wav_iq applies the analytic conversion for
            # audio-kind WAVs, so no further format branching is needed here.

            # 2. Front-End Health Checks.  The chunkers normalise to +/-1.0 full
            # scale, so rail hits show up as |I| or |Q| >= 0.999 -- which is exactly
            # the float branch of clipping_fraction.  Measuring this matters: a
            # clipped front end produces harmonics that masquerade as extra
            # emissions, and the report has to say so rather than always print 0%.
            clip_frac = max(
                clipping_fraction(iq_chunk.real),
                clipping_fraction(iq_chunk.imag),
            )
            # An analytic signal made from real audio has no negative-frequency
            # half and no I/Q hardware behind it, so image rejection is
            # undefined for it (it used to report a meaningless ~185 dB).
            audio_input = bool(getattr(meta, "is_audio", False))
            if audio_input:
                impropriety_db = None
            else:
                ev = image_evidence_db(iq_chunk, iq_fs)
                impropriety_db = ev[0] if ev else None
            dc_spike = detect_dc_spike(iq_chunk, iq_fs)
            agc_steps = detect_agc_ruptures(iq_chunk)

            if dc_spike:
                iq_chunk = iq_chunk - np.mean(iq_chunk)

            if audio_input:
                iq_wb = iq_chunk
            else:
                iq_wb, _ = maybe_correct_iq_imbalance(iq_chunk, iq_fs, is_wideband=True, source_is_zero_if=is_zero_if)

            if chunk_idx == 0:
                reporter.set_health(clip_frac, impropriety_db, dc_spike)

            # 3. Detection
            f, t, P_db = log_spectrogram(iq_wb, iq_fs, nfft=1024)
            if dc_spike:
                P_db = mask_dc_spike(P_db)
            if artifacts is not None and "spectrogram" not in artifacts:
                artifacts["spectrogram"] = {"f": f, "t": t, "P_db": P_db}
                artifacts["iq_preview"] = np.asarray(iq_wb[: min(len(iq_wb), 1 << 18)])
            Pw_db, _ = whiten_spectrogram(P_db)
            temporal_excess_db = temporal_baseline.update(P_db)
            Pw_db = np.maximum(Pw_db, temporal_excess_db)
            if audio_input:
                # The analytic conversion leaves the negative half empty by
                # construction.  Whitening then estimates a near-zero floor
                # there, so numerical residue at -fs/2 "exceeds" it by ~100 dB
                # and CFAR reports a phantom emitter at the band edge.  Nothing
                # can exist below 0 Hz in an audio input, so it is not searched.
                Pw_db = Pw_db.copy()
                Pw_db[f <= 0.0, :] = 0.0

            # YOLO & CFAR fusion
            cfar_boxes = detect_cfar(Pw_db, f, t, thr_db=6.0)
            averaged_boxes = detect_multiscale(P_db, f, t)
            if audio_input:
                averaged_boxes = [b for b in averaged_boxes if b["f_lo"] > 0]
            cfar_boxes.extend(averaged_boxes)
            yolo_boxes = self.yolo.detect(P_db, f, t)

            merged_boxes = merge_boxes(cfar_boxes, yolo_boxes)
            tracks = track_detections(merged_boxes, freq_tol_hz=0.0)

            # Sort tracks by prominence (energy: peak_db * bandwidth) so primary emitter is always Track 1
            tracks = sorted(tracks, key=lambda tr: float(tr.get("peak_db", 0.0)) * float(tr.get("bandwidth", 1.0)), reverse=True)

            for tr in tracks:
                t0_arg = 0.0 if tr["t_lo"] <= 1024.0 / iq_fs else tr["t_lo"]
                t1_arg = None if tr["t_hi"] >= (len(iq_wb) - 1024.0) / iq_fs else tr["t_hi"]
                ch_signal, ch_fs = channelize(
                    iq_wb,
                    fs=iq_fs,
                    f_center=tr["center_freq"],
                    bw_hz=tr["bandwidth"],
                    t0=t0_arg,
                    t1=t1_arg
                )
                if len(ch_signal) < 64:
                    continue

                bw_est, _, _, snr_est = occupied_bandwidth(ch_signal, ch_fs)
                detector_snr = inband_snr(P_db, f, t, tr, tracks)
                tr["detector_snr_db"] = detector_snr
                tr["post_filter_snr_db"] = None
                if detector_snr is not None:
                    snr_est = detector_snr

                if self.classifier is not None:
                    pred_dict = self._classify_trained(ch_signal, snr_est, ch_fs=float(ch_fs))
                else:
                    pred_dict = _untrained_prediction(snr_est)

                # 5. Exploitation chain (PS requirements i-v): demodulate,
                # resolve ambiguities, descramble, de-interleave, decode FEC and
                # build the frame model.  Bounded per file because the blind
                # searches cost seconds per track.
                exploitation = None
                demod_symbols = None
                decoded_bits = None
                if (chain is not None and n_exploited < max_exploit_tracks
                        and float(snr_est) >= min_exploit_snr_db
                        and pred_dict.get("levels", {}).get("l2", {}).get("prediction") != "NOISE"):
                    l2 = pred_dict.get("levels", {}).get("l2", {})
                    # Only a confident classification is passed on as a hint;
                    # an uncertain (or untrained) network must not steer the
                    # carrier-recovery order or the FSK/linear decision.
                    hint = l2.get("prediction") if l2.get("decision_state") == "DECISION" else None
                    try:
                        ex = chain.run(ch_signal, float(ch_fs),
                                       fs_source=meta.sample_rate_source, mod_hint=hint,
                                       inband_snr_db=detector_snr, bandwidth_hz=tr["bandwidth"])
                        exploitation = ex.to_dict()
                        if ex.demod.modulation != "UNRESOLVED" and ex.demod.family == "linear" and ex.demod.symbol_rate_hz > 0:
                            from .dsp.snr_estimate import estimate_snr_m2m4
                            symbol_snr = estimate_snr_m2m4(ex.demod.symbols, ex.demod.modulation)
                            tr["post_filter_snr_db"] = float(symbol_snr - 10 * np.log10(tr["bandwidth"] / ex.demod.symbol_rate_hz))
                        demod_symbols = getattr(ex.demod, "symbols", None)
                        decoded_bits = getattr(ex, "decoded_bits", None)
                        if fs_assumed:
                            # Absolute rates are meaningless without a real fs;
                            # the ratio to the sample rate is what was measured.
                            d = exploitation["demodulation"]
                            d["symbol_rate_per_sample"] = (d["symbol_rate_hz"] / float(ch_fs)
                                                           if ch_fs else 0.0)
                            exploitation["notes"].append(
                                "Sample rate unknown: absolute frequencies assume "
                                f"{self.NOMINAL_FS:.0f} S/s; use symbol_rate_per_sample, "
                                "or rerun with --fs."
                            )
                        n_exploited += 1
                    except Exception as exc:  # never let one track kill the file
                        exploitation = {"status": "EXPLOIT_ERROR", "error": f"{type(exc).__name__}: {exc}"}

                if artifacts is not None:
                    artifacts["tracks"].append({
                        "track_id": len(reporter.detections) + 1,
                        "fs": float(ch_fs),
                        "iq": np.asarray(ch_signal[: min(len(ch_signal), 1 << 18)]),
                        "symbols": demod_symbols,
                        "decoded_bits": decoded_bits,
                        "report": exploitation,
                    })

                reporter.add_detection(tr, pred_dict, exploitation=exploitation)

        reporter.flag_overlapping_rates()

        if usable_chunks == 0:
            reporter.set_status(Status.INSUFFICIENT_SAMPLES)
        return reporter.to_dict()

    def _classify_trained(self, iq: np.ndarray, snr_est: float, ch_fs: float | None = None) -> dict[str, Any]:
        r = self.classifier.predict(iq, fs=ch_fs)
        state = "DECISION" if (r["confidence"] >= 0.60 and snr_est >= 3.0) else             ("NO_DECISION" if snr_est < 3.0 else "UNKNOWN")
        if r.get("modulation") in ("UNKNOWN", "NOISE") or r.get("open_set", {}).get("rejected"):
            state = "UNKNOWN"
        return {
            "track_snr_db": float(snr_est),
            "levels": {
                "l1": {"prediction": r["family"], "confidence": r["confidence"], "decision_state": state},
                "l2": {"prediction": r["modulation"], "confidence": r["confidence"],
                       "decision_state": state, "top3": r["top3"]},
                "l3": {"prediction": None, "confidence": 0.0, "decision_state": "NOT_TRAINED"},
                "preprocessing": r.get("preprocessing", {}),
            },
        }

    def process(self, iq: np.ndarray, fs: float = 48000.0) -> dict[str, Any]:
        """In-memory processing method for metamorphic and unit test runners."""
        iq = np.asarray(iq, dtype=np.complex64).ravel()
        if len(iq) < 32:
            return {
                "status": Status.INSUFFICIENT_SAMPLES.value,
                "payload_hex": "",
                "cfo_hz": 0.0,
                "fs_est": fs,
                "evm": 1.0,
                "crc_locked": False,
            }

        # 1. CFO Estimation & Correction
        from signal9.dsp.cfo import mth_power_cfo, correct_cfo
        cfo_hz = mth_power_cfo(iq, fs=fs, M=4)
        iq_cfo = correct_cfo(iq, fs=fs, cfo_hz=cfo_hz)

        # 2. Timing Recovery & Candidate Generation
        from signal9.dsp.timing_recovery import gardner_timing_recovery
        from signal9.decide.blackboard import BlackboardArbiter
        from signal9.config import Config

        arbiter = BlackboardArbiter(Config(early_exit_on_crc=True, enable_windowed_crc=False))

        syms_candidates = []
        if len(iq_cfo) >= 64:
            syms_candidates.append(iq_cfo[0::4])
        try:
            syms_gardner, _, _ = gardner_timing_recovery(iq_cfo, sps=4.0)
            if len(syms_gardner) > 16:
                syms_candidates.append(syms_gardner)
        except Exception:
            pass

        if not syms_candidates:
            syms_candidates = [iq_cfo]

        # 3. Fast Blackboard Demodulation & Arbitration with Early Lock.
        # Keep the best branch seen so far rather than only the first: a CRC lock
        # always wins, and among CRC-less branches the higher plausibility score
        # is the better partial result to report.
        winner = None
        best_symbols = None
        best_mod = "QPSK"

        def _better(cand, incumbent):
            if cand is None:
                return False
            if incumbent is None:
                return True
            if cand.crc_locked != incumbent.crc_locked:
                return cand.crc_locked
            return cand.plausibility > incumbent.plausibility

        for syms_1sps in syms_candidates:
            for mod in ("BPSK", "QPSK"):
                cand, _ = arbiter.arbitrate(syms_1sps, mod_type=mod)
                if _better(cand, winner):
                    winner, best_symbols, best_mod = cand, syms_1sps, mod
                if winner is not None and winner.crc_locked:
                    break
            if winner is not None and winner.crc_locked:
                break

        status_val = winner.status.value if winner else Status.ALL_BRANCHES_FAILED.value
        crc_locked = bool(winner and winner.crc_locked)
        payload_hex = winner.payload.hex() if winner and winner.payload else ""

        # Report the measured EVM of the symbol set the winning branch came from.
        # This used to be the constant 0.05, which made the field useless as a
        # demodulation-quality indicator and hid outright demod failures.
        evm_lin = 1.0
        if best_symbols is not None and len(best_symbols) >= 8:
            from signal9.dsp.snr_estimate import evm_db
            from signal9.dsp.constellation import get_constellation
            try:
                e_db = evm_db(best_symbols, get_constellation(best_mod))
                if np.isfinite(e_db):
                    evm_lin = float(10.0 ** (e_db / 20.0))
            except Exception:
                evm_lin = 1.0

        return {
            "status": status_val,
            "payload_hex": payload_hex,
            "cfo_hz": cfo_hz,
            "fs_est": fs,
            "evm": evm_lin,
            "crc_locked": crc_locked,
        }


def _untrained_prediction(snr_est: float) -> dict[str, Any]:
    """Classification levels when no trained checkpoint is loaded."""
    lvl = {"prediction": None, "confidence": 0.0, "decision_state": "NOT_TRAINED"}
    return {"track_snr_db": float(snr_est),
            "levels": {"l1": dict(lvl), "l2": dict(lvl), "l3": dict(lvl)}}
