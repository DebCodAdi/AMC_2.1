"""Load a trained multi-task checkpoint and classify IQ windows.

This is the bridge that was missing between training and analysis: the trainer
wrote checkpoints that nothing read, and the analysis pipeline instantiated a
randomly-initialised network on every run and reported its outputs, with
confidences, as classifications.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .backbone import build_backbone
from .features import expert_features, extract_hoc_dict
from .heads import SignalInfinityMultiTaskHeads
from .labels import FAMILY, MOD_VOCAB
from .trainer import CHECKPOINT_FORMAT


class MultiTaskClassifier:
    """Trained modulation classifier wrapping a ``signal9-multitask-v1`` checkpoint."""

    def __init__(self, backbone, heads, class_names, input_len, device="cpu", meta=None):
        self.backbone = backbone.eval()
        self.heads = heads.eval()
        self.class_names = list(class_names)
        self.input_len = int(input_len)
        self.device = device
        self.meta = meta or {}

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "MultiTaskClassifier":
        ck = torch.load(Path(path), map_location=device, weights_only=False)
        if ck.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path}: not a {CHECKPOINT_FORMAT} checkpoint "
                             f"(format={ck.get('format')!r}); retrain with signal9.train.trainer")
        names = ck.get("class_names") or list(MOD_VOCAB)
        architecture = ck.get("architecture", "baseline_resnet")
        expert_dim = int(ck.get("expert_dim", 32))
        backbone = build_backbone(architecture,
                                  expert_dim=expert_dim,
                                  stem_dim=ck.get("stem_dim", 128),
                                  emb_dim=ck.get("emb_dim", 512))
        heads = SignalInfinityMultiTaskHeads(emb_dim=ck.get("emb_dim", 512), n_mods=len(names),
                                             n_fecs=ck.get("n_fecs", 5),
                                             n_scramblers=ck.get("n_scramblers", 5))
        # The EMA weights are the ones validated and saved as "best".  The
        # Kaggle trainer intentionally omits the legacy, unused SNR-gating
        # modules; allow only those known missing keys.  A blanket
        # ``strict=False`` used to hide genuinely incompatible checkpoints.
        selected_backbone = ck.get("ema_backbone") or ck.get("backbone")
        if not selected_backbone:
            raise ValueError(f"{path}: checkpoint has no backbone weights")
        incompatible = backbone.load_state_dict(selected_backbone, strict=False)
        allowed_missing_prefixes = (("pre_snr_head.", "gate_mlp.")
                                    if architecture == "baseline_resnet" else ())
        unsafe_missing = [key for key in incompatible.missing_keys
                          if not key.startswith(allowed_missing_prefixes)]
        if unsafe_missing or incompatible.unexpected_keys:
            raise ValueError(
                f"{path}: incompatible backbone state; missing={unsafe_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        heads.load_state_dict(ck.get("ema_heads") or ck["heads"])
        inst = cls(backbone.to(device), heads.to(device), names, ck.get("input_len", 1024),
                   device, meta={k: ck.get(k) for k in (
                       "dataset", "metrics", "training_data", "feature_contract",
                       "alignment_preprocess", "conditioning_contract",
                       "prototype_weight", "open_set")})
        prototypes = ck.get("prototypes", None)
        inst.prototypes = prototypes.to(device) if (prototypes is not None and hasattr(prototypes, "to")) else None
        # A prototype branch is used only when its weight was selected on the
        # held-out partition.  Legacy checkpoints did not validate the blend,
        # so their safe default is the ordinary trained classification head.
        inst.prototype_weight = float(ck.get("prototype_weight", 0.0) or 0.0)
        inst.alignment_preprocess = bool(ck.get("alignment_preprocess", False))
        inst.expert_dim = expert_dim
        return inst

    @torch.no_grad()
    def predict(self, iq: np.ndarray, fs: float | None = None, max_windows: int = 8,
                apply_heuristics: bool = False) -> dict:
        """Average the class posteriors over up to ``max_windows`` windows of the input."""
        # Keep this conditioning here (rather than in one caller) so batch,
        # file and streaming inference all present the same contract to the
        # checkpoint.  It deliberately preserves a centered real carrier.
        from ..preprocess import condition_for_classifier
        x, conditioning = condition_for_classifier(iq)
        if len(x) < 32:
            return {
                "modulation": "UNKNOWN",
                "family": "UNKNOWN",
                "confidence": 0.0,
                "top3": [],
                "snr_db": 0.0,
                "presence": 0.0,
                "preprocessing": conditioning.to_dict(),
            }

        nominal_fs = float(fs) if (fs is not None and fs > 0) else 250000.0
        raw_x = x.copy()

        # 1. Physics-based Constant Envelope Detection
        mag_all = np.abs(x)
        p_all = float(np.mean(mag_all ** 2)) + 1e-12
        env_kurt = float(np.mean(mag_all ** 4)) / (p_all ** 2)
        papr_db = 10.0 * np.log10(float(np.max(mag_all ** 2)) / p_all)
        is_constant_envelope = (env_kurt < 1.15 and papr_db < 2.5)

        # 2. Check if signal looks like FSK or has large frequency variance
        from ..exploit.chain import looks_like_fsk
        from ..features.instfreq import fsk_tone_count
        is_fsk = looks_like_fsk(x, nominal_fs)

        # 3. Dominant single-carrier CFO estimation & correction
        # Only apply CFO if NOT FSK (to avoid shifting FSK tones)
        if self.alignment_preprocess and not is_fsk and len(x) >= 64:
            nfft = min(8192, len(x))
            z = x[:nfft] / np.sqrt(np.mean(np.abs(x[:nfft]) ** 2) + 1e-12)
            freqs = np.fft.fftfreq(nfft, d=1.0 / nominal_fs)

            # 4th power for 4-phase/QAM
            z4 = z ** 4
            fft_z4 = np.abs(np.fft.fft(z4))
            peak4 = float(np.max(fft_z4)) / (float(np.mean(fft_z4)) + 1e-12)
            f4 = float(freqs[np.argmax(fft_z4)] / 4.0)

            # 2nd power for 2-phase
            z2 = z ** 2
            fft_z2 = np.abs(np.fft.fft(z2))
            peak2 = float(np.max(fft_z2)) / (float(np.mean(fft_z2)) + 1e-12)
            f2 = float(freqs[np.argmax(fft_z2)] / 2.0)

            cfo_est = 0.0
            if peak2 > peak4 and peak2 > 6.0 and abs(f2) <= 2000.0:
                cfo_est = f2
            elif peak4 > 6.0 and abs(f4) <= 2000.0:
                cfo_est = f4

            if abs(cfo_est) > 5.0:
                t = np.arange(len(x)) / nominal_fs
                x = (x * np.exp(-1j * 2.0 * np.pi * cfo_est * t)).astype(np.complex64)

        # 4. Resampling for SPS alignment if symbol rate is ~50 kBd at 4 or 5 SPS
        from ..features.baud import baud_candidates
        from scipy import signal as sps_signal
        from fractions import Fraction

        if self.alignment_preprocess and fs is not None and fs > 0 and len(x) >= 512:
            cands = baud_candidates(x, nominal_fs)
            rs_est = float(cands[0][0]) if cands else 0.0
            if 10000.0 < rs_est < nominal_fs / 2.0:
                sps_est = nominal_fs / rs_est
                if 3.5 <= sps_est <= 6.5:
                    target_fs = rs_est * 8.0
                    ratio = Fraction(int(target_fs), int(nominal_fs)).limit_denominator(32)
                    x = sps_signal.resample_poly(x, ratio.numerator, ratio.denominator).astype(np.complex64)

        L = self.input_len
        if len(x) < L:
            x = np.pad(x, (0, L - len(x)))
        starts = np.linspace(0, len(x) - L, num=min(max_windows, max(1, len(x) // L)), dtype=np.int64)
        wins = []
        exps = []
        for s in starts:
            w = x[s:s + L]
            w = w / np.sqrt(np.mean(np.abs(w) ** 2) + 1e-12)
            wins.append(np.stack([w.real, w.imag]).astype(np.float32))
            legacy_papr = self.meta.get("feature_contract") == "expert32-untrimmed-v1"
            features = expert_features(w, full=(self.expert_dim > 11), legacy_papr=legacy_papr)
            if self.expert_dim > len(features):
                raise ValueError(
                    f"checkpoint requires {self.expert_dim} expert features, extractor provides {len(features)}"
                )
            exps.append(features[:self.expert_dim])
        iq_t = torch.from_numpy(np.stack(wins)).to(self.device)
        ex_t = torch.from_numpy(np.stack(exps)).to(self.device)

        emb = self.backbone(iq_t, ex_t)
        out = self.heads(emb)

        # Prototype cosine similarity blending
        if (hasattr(self, "prototypes") and self.prototypes is not None
                and self.prototype_weight > 0.0):
            import torch.nn.functional as F
            emb_norm = F.normalize(emb, p=2, dim=1)
            proto_norm = F.normalize(self.prototypes, p=2, dim=1)
            cos_sim = torch.matmul(emb_norm, proto_norm.t())
            logits = out["mod_logits"] + self.prototype_weight * cos_sim
        else:
            logits = out["mod_logits"]

        # 5. Model decision.  The legacy hard-coded overrides below were never
        # calibrated on a held-out deployment corpus.  They can turn a model
        # decision into a different, high-confidence label, so deployment
        # keeps them off until a benchmark explicitly validates them.
        open_set = self.meta.get("open_set") or {}
        temperature = max(float(open_set.get("temperature", 1.0)), 1e-3)
        aggregate_logits = logits.float().mean(dim=0)
        prob = torch.softmax(aggregate_logits / temperature, dim=0).cpu().numpy()
        energy = float(-temperature * torch.logsumexp(aggregate_logits / temperature, dim=0).cpu())
        reject_unknown = bool(
            open_set.get("enabled", False)
            and energy > float(open_set.get("energy_threshold", float("inf")))
        )
        order = np.argsort(prob)[::-1]
        base_pred = self.class_names[int(order[0])]
        top3_mods = [self.class_names[int(i)] for i in order[:3]]

        hoc = extract_hoc_dict(raw_x)
        c42 = hoc["c42"]
        c84 = hoc["c84"]
        cv = hoc["cv"]
        kurt_val = hoc["kurt"]

        p_raw = float(np.mean(np.abs(raw_x) ** 2)) + 1e-12
        xc = (raw_x - np.mean(raw_x)) / np.sqrt(p_raw)
        abs_m8 = float(np.abs(np.mean(xc ** 8)))

        from ..dsp.fsk import discriminator, estimate_fsk_symbol_rate
        freq = discriminator(raw_x, nominal_fs)
        fsk_rate, _ = estimate_fsk_symbol_rate(freq, nominal_fs)
        fsk_sps = nominal_fs / fsk_rate if fsk_rate > 0 else (nominal_fs / 50000.0)
        std_fi = float(np.std(np.angle(xc[1:] * np.conj(xc[:-1])) / (2.0 * np.pi)))
        dev_metric = float(std_fi * fsk_sps)

        final_pred = "UNKNOWN" if reject_unknown else base_pred
        if reject_unknown:
            pass
        elif apply_heuristics and is_fsk and cv < 0.10 and kurt_val < 1.05:
            # Constant envelope alone is not evidence of FSK: clean BPSK,
            # QPSK and PSK also have it.  The independent FSK detector is a
            # mandatory gate for this optional override.
            final_pred = "4FSK" if dev_metric > 0.75 else "2FSK"
        elif apply_heuristics and base_pred == "BPSK":
            final_pred = "BPSK"
        elif apply_heuristics and base_pred == "QPSK":
            if "16QAM" in top3_mods and c84 > -6.0 and cv > 0.42:
                final_pred = "16QAM"
            elif "8PSK" in top3_mods and c84 < -10.0 and abs_m8 > 0.5:
                final_pred = "8PSK"
            else:
                final_pred = "QPSK"
        elif apply_heuristics and base_pred in ("16PSK", "32PSK"):
            final_pred = "8PSK"
        elif apply_heuristics and base_pred in ("64QAM", "32QAM", "256QAM", "32APSK", "64APSK"):
            if "8PSK" in top3_mods and c84 < -8.0:
                final_pred = "8PSK"
            else:
                final_pred = "16QAM" if (c84 > -7.0 or cv > 0.40) else base_pred
        elif apply_heuristics and base_pred in ("GMSK", "MSK", "GFSK"):
            if "16QAM" in top3_mods:
                final_pred = "16QAM"
            elif "8PSK" in top3_mods:
                final_pred = "8PSK"
            elif "QPSK" in top3_mods:
                final_pred = "QPSK"
            else:
                final_pred = "16QAM" if cv > 0.35 else base_pred
        elif apply_heuristics and base_pred in ("OOK", "4ASK", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC"):
            if "16QAM" in top3_mods:
                final_pred = "16QAM"
            elif "8PSK" in top3_mods:
                final_pred = "8PSK"
            elif "BPSK" in top3_mods:
                final_pred = "BPSK"
            else:
                final_pred = "16QAM" if (c84 > -6.5 and cv > 0.40) else base_pred

        top = final_pred
        top_idx = self.class_names.index(top) if top in self.class_names else int(order[0])
        conf = float(prob[top_idx]) if top in self.class_names else 0.0
        return {
            "modulation": top,
            "family": FAMILY.get(top, "UNKNOWN"),
            "confidence": conf,
            "top3": [(self.class_names[int(i)], round(float(prob[i]), 4)) for i in order[:3]],
            "snr_db": float(out["snr_db"].mean()),
            "presence": float(torch.sigmoid(out["presence_logit"]).mean()),
            "open_set": {
                "enabled": bool(open_set.get("enabled", False)),
                "rejected": reject_unknown,
                "energy": energy,
                "energy_threshold": open_set.get("energy_threshold"),
                "temperature": temperature,
            },
            "preprocessing": conditioning.to_dict(),
        }
