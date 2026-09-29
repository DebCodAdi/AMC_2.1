"""Open-set recognition and out-of-distribution (OOD) waveform rejection (Track F-5).

Rejects unknown, unmodeled, or severely corrupted waveforms rather than
forcing a closed-set misclassification.

Spec reference: Track F-5 (PS26147 / SIGNAL10).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from signal9.status import Status
from signal9.dsp.constellation import get_constellation


@dataclass(frozen=True)
class OpenSetDecision:
    is_known: bool
    predicted_class: str
    confidence: float
    unknown_probability: float
    min_evm: float
    status: Status


class OpenSetClassifier:
    """Open-set decision engine combining distance-to-centroid and constellation EVM."""

    def __init__(self,
                 evm_reject_threshold: float = 0.38,
                 unknown_prob_threshold: float = 0.5):
        self.evm_reject_threshold = evm_reject_threshold
        self.unknown_prob_threshold = unknown_prob_threshold

        # Class reference centroids for common modulations in (papr, env_var, kurtosis) space
        self.class_centroids = {
            "BPSK": np.array([1.0, 0.05, 1.0]),
            "QPSK": np.array([1.0, 0.05, 1.0]),
            "8PSK": np.array([1.0, 0.05, 1.0]),
            "16QAM": np.array([1.5, 0.35, 1.35]),
            "64QAM": np.array([1.8, 0.55, 1.55]),
            "FSK2": np.array([1.0, 0.02, 1.0]),
            "OFDM": np.array([3.2, 0.85, 2.0]),
        }

    def compute_evm(self, symbols: np.ndarray, mod_type: str) -> float:
        """Compute minimum Error Vector Magnitude against reference constellation."""
        symbols = np.asarray(symbols, dtype=np.complex64).ravel()
        if len(symbols) == 0:
            return 1.0

        const = get_constellation(mod_type)
        # Normalize symbol power
        pwr = np.mean(np.abs(symbols) ** 2)
        if pwr < 1e-12:
            return 1.0
        norm_syms = symbols / np.sqrt(pwr)

        # Distance to nearest constellation point
        # diff: (len(symbols), M)
        dists2 = np.abs(norm_syms[:, None] - const[None, :]) ** 2
        min_err2 = np.min(dists2, axis=1)
        evm = float(np.sqrt(np.mean(min_err2)))
        return evm

    def classify(self,
                 symbols: np.ndarray,
                 closed_set_logits: dict[str, float] | None = None,
                 candidate_mods: list[str] | None = None) -> OpenSetDecision:
        """Classify symbols with open-set rejection option.

        Parameters
        ----------
        symbols : 1D complex array
        closed_set_logits : dict of {class_name: logit}, optional
        candidate_mods : list of modulations to check, optional

        Returns
        -------
        OpenSetDecision
        """
        symbols = np.asarray(symbols, dtype=np.complex64).ravel()
        if len(symbols) < 32:
            return OpenSetDecision(
                is_known=False,
                predicted_class="UNKNOWN",
                confidence=0.0,
                unknown_probability=1.0,
                min_evm=1.0,
                status=Status.INSUFFICIENT_SAMPLES,
            )

        # Physical statistical features
        mag2 = np.abs(symbols) ** 2
        m2 = float(np.mean(mag2)) + 1e-12
        m4 = float(np.mean(mag2 ** 2))
        s2 = float(np.abs(np.mean(symbols ** 2)) ** 2)
        c42 = (m4 - 2.0 * (m2 ** 2) - s2) / (m2 ** 2)

        env_var = float(np.var(np.sqrt(mag2) / np.sqrt(m2)))

        # Chirp detection via frequency slope
        if len(symbols) >= 64:
            dphi = np.angle(symbols[1:] * np.conj(symbols[:-1]))
            t_idx = np.arange(len(dphi), dtype=float)
            corr_chirp = abs(float(np.corrcoef(t_idx, dphi)[0, 1])) if np.std(dphi) > 1e-6 else 0.0
        else:
            corr_chirp = 0.0

        # Physical rejection flags
        is_gaussian_noise = abs(c42) < 0.12 and env_var > 0.05
        is_chirp_waveform = corr_chirp > 0.65 and env_var < 0.05

        mod_evm_limits = {
            "BPSK": 0.28,
            "QPSK": 0.28,
            "8PSK": 0.22,
            "16QAM": 0.16,
        }

        candidates = candidate_mods or ["BPSK", "QPSK", "8PSK", "16QAM"]
        evm_scores = {}
        for mod in candidates:
            evm_val = self.compute_evm(symbols, mod)
            # 16QAM requires multi-ring envelope variance
            if mod == "16QAM" and env_var < 0.04:
                evm_val += 0.5  # Penalize constant envelope on 16QAM
            evm_scores[mod] = evm_val

        best_mod = min(evm_scores, key=evm_scores.get)
        min_evm = evm_scores[best_mod]
        limit = mod_evm_limits.get(best_mod, self.evm_reject_threshold)

        # Open-set unknown probability
        if is_gaussian_noise or is_chirp_waveform or min_evm > limit:
            p_unknown = 0.95
            is_known = False
        else:
            p_unknown = float(np.clip(min_evm / limit * 0.35, 0.0, 0.35))
            is_known = True

        if is_known:
            conf = 1.0 - p_unknown
            pred = best_mod
            status = Status.SUCCESS_CRC_LOCKED if conf > 0.85 else Status.DEMOD_NO_CRC
        else:
            conf = 1.0 - p_unknown
            pred = "UNKNOWN"
            status = Status.UNKNOWN_MODULATION

        return OpenSetDecision(
            is_known=is_known,
            predicted_class=pred,
            confidence=conf,
            unknown_probability=p_unknown,
            min_evm=min_evm,
            status=status,
        )
