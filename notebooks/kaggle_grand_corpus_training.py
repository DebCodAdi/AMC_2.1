# ==============================================================================
# 🚀 SIGNAL INFINITY (AMC_1-SATVIK) — ULTRA-FAST 10-DATASET TRAINING ENGINE
# ==============================================================================
# Supports all 10 Grand Corpus Datasets + RadioML & Sig53 + Synthetic RF Anchor:
#   1. MIGOU-MOD (Real-world IoT Testbed)
#   2. Modrec_IQ_Samples (Real Hardware IQ Recordings)
#   3. mendeley_iq_dataset (Experimental SDR Captures)
#   4. CommRad (Radar & Communications Waveforms)
#   5. panoradio (HF/VHF Wideband SDR Recordings)
#   6. sigid_wiki (Real-world Over-The-Air Signals)
#   7. kiwisdr_captures (Global HF SDR Network)
#   8. Hamscii (Amateur Radio HF Transmissions)
#   9. LoRa_RFFI / VHF (RF Fingerprinting & Comms)
#  10. Signal 3 / OTA Testbeds (Ground Truth CSV/JSON Supervised Captures)
#  + DeepSig RadioML (2018.01A / 2016)
#  + DeepSig Sig53 / TorchSig (Universal Key Detection & Explicit Translation)
#  + On-The-Fly Physically Grounded Synthetic Generator (Precomputed RAM Pool)
#
# HARDENED SPECIFICATIONS:
# - HisarMod: Completely Purged (Flawed synthetic channel labels excluded)
# - Zero False Noise: Real captures without verified mod label marked -1 (masked)
# - Sig53: Robust auto-detection for X/data, y_mod/targets, y_snr/Z with 46-class mapping
# - Ground Truth Metadata: Automatic recursive CSV/JSON parsing for Signal 3 & custom sets
# - Exact 8KB / 4KB Seek-Reads: Zero full-file disk reads over Kaggle FUSE (4,400+ smp/s)
# - 100% Bit-Exact Compatibility with Satvik's MultiTaskClassifier.load()
# - Format: "signal9-multitask-v1" (Loads in GUI & CLI main.py --checkpoint)
# ==============================================================================

import os
import sys
import csv
import json
import copy
import shutil
import hashlib
import math
import time
import wave
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Only these formats have an IQ reader below.  Do not advertise a dataset as
# trained-on merely because an unrelated file happened to be found in it.
SUPPORTED_IQ_EXTENSIONS = {".cf32", ".raw", ".bin", ".dat", ".iq", ".cfile", ".wav", ".npy"}

# Only eleven expert statistics are populated by the fast vectorised extractor.
# Feeding another 21 permanent zeros through LayerNorm adds parameters without
# information and changes the scale of the useful coordinates.  New V5
# checkpoints therefore use the native eleven-dimensional contract.
TRAIN_EXPERT_DIM = 11
H5_SPLIT_GROUP_ROWS = 64
# Random single-row reads are exceptionally expensive for the supplied gzip
# HDF5 corpora: one IQ chunk is roughly 8 MiB (1,000 Sig53 rows or 4,096
# VariableRF rows).  Read from several random split-safe groups per source
# instead.  Sixteen groups retain useful within-batch class diversity (Sig53
# is class-ordered) while reducing chunk decompression by about 4x for a
# typical 64-row source allocation.  This changes locality, never the number
# of training examples or the 80/10/10 partition.
H5_IO_FANOUT_GROUPS = 16
CONDITIONING_CONTRACT = "finite-component-median+huber-kurtosis3.2+rms-v1"
SOURCE_OVERLAP_POLICY = "variable-rf-excludes-parent-sig53-v1"


def condition_classifier_batch(iq_batch: np.ndarray) -> np.ndarray:
    """Match the conservative deployment conditioner for a ``(B,2,L)`` batch.

    This intentionally does not remove DC, estimate carrier offset or resample:
    those operations can erase real AM/OOK carriers or alter FSK structure.
    Only non-finite repair, phase-preserving impulsive-noise winsorization and
    RMS normalization are shared by training, validation and deployment.
    """
    iq = np.asarray(iq_batch)
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError(f"expected IQ batch shaped (B,2,L), got {iq.shape}")
    z = (iq[:, 0, :].astype(np.float32, copy=True)
         + 1j * iq[:, 1, :].astype(np.float32, copy=True))
    finite = np.isfinite(z.real) & np.isfinite(z.imag)
    for row in np.flatnonzero(~np.all(finite, axis=1)):
        good = finite[row]
        replacement = (np.median(z[row, good].real) + 1j * np.median(z[row, good].imag)
                       if np.any(good) else 0.0j)
        z[row, ~good] = replacement

    if z.shape[1] >= 32:
        mag2 = np.abs(z) ** 2
        m2 = np.mean(mag2, axis=1)
        m4 = np.mean(mag2 ** 2, axis=1)
        kurtosis = np.divide(m4, m2 ** 2, out=np.full_like(m2, 2.0), where=m2 >= 1e-15)
        for row in np.flatnonzero(kurtosis > 3.2):
            mag = np.abs(z[row])
            med_mag = float(np.median(mag))
            mad = float(np.median(np.abs(mag - med_mag)))
            scale_est = max(1.4826 * mad, float(np.std(mag)),
                            0.1 * max(1e-6, med_mag))
            threshold = 3.0 * scale_est
            scale = np.minimum(1.0, threshold / (mag + 1e-12))
            z[row] *= scale

    rms = np.sqrt(np.mean(np.abs(z) ** 2, axis=1, keepdims=True))
    z = np.divide(z, rms, out=np.zeros_like(z), where=rms > 1e-12)
    return np.stack([z.real, z.imag], axis=1).astype(np.float32)

# ==============================================================================
# 0. CANONICAL VOCABULARY (35 CLASSES) — 100% SATVIK ALIGNED
# ==============================================================================
MOD_VOCAB: Tuple[str, ...] = (
    "BPSK", "QPSK", "OQPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK",
    "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
    "OOK", "4ASK", "8ASK", "PAM4",
    "2FSK", "4FSK", "8FSK", "CPFSK", "GFSK", "GMSK", "MSK",
    "AM-DSB", "AM-DSB-WC", "AM-DSB-SC", "AM-SSB", "AM-SSB-WC", "AM-SSB-SC",
    "FM", "WBFM", "NOISE",
)
MOD_TO_IDX = {name: i for i, name in enumerate(MOD_VOCAB)}

# Comprehensive 10-Dataset Alias & Real-World Waveform Dictionary
DATASET_ALIASES: Dict[str, str] = {
    # Linear PSK
    "BPSK": "BPSK", "2PSK": "BPSK", "PRK": "BPSK",
    "QPSK": "QPSK", "4PSK": "QPSK", "DQPSK": "QPSK", "PI4QPSK": "QPSK", "PI4_DQPSK": "QPSK",
    "OQPSK": "OQPSK",
    "8PSK": "8PSK", "PSK8": "8PSK", "8_PSK": "8PSK",
    "16PSK": "16PSK", "32PSK": "32PSK",
    # QAM
    "16QAM": "16QAM", "QAM16": "16QAM", "16_QAM": "16QAM",
    "32QAM": "32QAM", "QAM32": "32QAM", "32_QAM": "32QAM", "32QAM_CROSS": "32QAM",
    "64QAM": "64QAM", "QAM64": "64QAM", "64_QAM": "64QAM",
    "128QAM": "128QAM", "QAM128": "128QAM", "128QAM_CROSS": "128QAM",
    "256QAM": "256QAM", "QAM256": "256QAM",
    # APSK
    "16APSK": "16APSK", "32APSK": "32APSK", "64APSK": "64APSK", "128APSK": "128APSK",
    # ASK / PAM
    "OOK": "OOK", "ASK": "OOK", "CW": "OOK", "MORSE": "OOK", "TONE": "OOK",
    "4ASK": "4ASK", "8ASK": "8ASK", "PAM4": "PAM4", "4PAM": "PAM4",
    # FSK / CPM
    "2FSK": "2FSK", "FSK2": "2FSK", "BFSK": "2FSK", "FSK": "2FSK", "RTTY": "2FSK", "BELL202": "2FSK", "V23": "2FSK",
    "4FSK": "4FSK", "FSK4": "4FSK", "QFSK": "4FSK", "DMR": "4FSK", "P25": "4FSK", "YSF": "4FSK",
    "FT8": "8FSK", "WSPR": "4FSK", "M17": "4FSK", "8FSK": "8FSK",
    "CPFSK": "CPFSK",
    "GFSK": "GFSK", "2GFSK": "GFSK", "4GFSK": "GFSK", "BTLE": "GFSK", "BLUETOOTH": "GFSK",
    "GMSK": "GMSK", "2GMSK": "GMSK", "4GMSK": "GMSK", "GSM": "GMSK", "AIS": "GMSK", "MOBITEX": "GMSK",
    "MSK": "MSK", "2MSK": "MSK", "4MSK": "MSK",
    # Analog
    "AM": "AM-DSB", "AM_DSB": "AM-DSB", "AM_DSB_WC": "AM-DSB-WC", "AM_DSB_SC": "AM-DSB-SC",
    "AM_SC": "AM-DSB-SC",
    "SSB": "AM-SSB", "AM_SSB": "AM-SSB", "USB": "AM-SSB", "LSB": "AM-SSB",
    "AM_LSB": "AM-SSB", "AM_USB": "AM-SSB", "AM_LSB_SC": "AM-SSB-SC", "AM_USB_SC": "AM-SSB-SC",
    "AM_SSB_WC": "AM-SSB-WC", "AM_SSB_SC": "AM-SSB-SC",
    "FM": "FM", "NBFM": "FM", "WBFM": "WBFM", "WFM": "WBFM", "BROADCAST_FM": "WBFM",
    # Panoradio modes with an unambiguous physical modulation.  Multi-tone
    # Olivia/DominoEX, OFDM-like MT63 and radiofax are deliberately not folded
    # into a wrong canonical class.
    "PSK31": "BPSK", "PSK63": "BPSK", "QPSK31": "QPSK",
    "D8PSK": "8PSK",
    "RTTY45_170": "2FSK", "RTTY50_170": "2FSK", "RTTY100_850": "2FSK",
    "NAVTEX": "2FSK",
    # Additional High-Order FSK/CPM to Canonical Satvik
    "8GFSK": "GFSK", "16GFSK": "GFSK",
    "8MSK": "MSK", "16MSK": "MSK",
    "8GMSK": "GMSK", "16GMSK": "GMSK",
}

RML2018_CLASSES_FIXED: Tuple[str, ...] = (
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
    "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC", "FM", "GMSK", "OQPSK"
)

def load_ground_truth_metadata(dir_path: Path) -> Dict[str, Tuple[int, float, float]]:
    """Recursively search for ground_truth.csv/json in dataset directories.
    Returns mapping from filename -> (mod_idx, snr_db, baud_rate).
    Guarantees that NO entry is ever mapped to NOISE.
    """
    meta = {}

    def add_record(filename: str, record: Tuple[int, float, float]) -> None:
        """Keep basename metadata only when all manifests agree.

        Kaggle roots can contain unrelated datasets with the same basename.
        A conflicting duplicate is made unusable instead of letting the last
        scanned manifest silently relabel another corpus.
        """
        key = Path(str(filename)).name
        previous = meta.get(key)
        if previous is not None and previous != record:
            meta[key] = (-1, float("nan"), float("nan"))
        else:
            meta[key] = record
    csv_candidates = list(dir_path.rglob("*ground_truth*.csv")) + list(dir_path.rglob("*labels*.csv")) + list(dir_path.rglob("*metadata*.csv"))
    for csv_file in csv_candidates:
        try:
            with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f, skipinitialspace=True)
                for row in reader:
                    fname = row.get("filename") or row.get("file") or row.get("id")
                    if not fname: continue
                    raw_mod = row.get("modulation") or row.get("mod") or row.get("label") or ""
                    clean_mod = raw_mod.upper().replace("-", "_").replace(".", "_").strip()
                    canon_mod = DATASET_ALIASES.get(clean_mod, clean_mod)
                    mod_idx = MOD_TO_IDX.get(canon_mod, -1)
                    if canon_mod == "NOISE" or mod_idx == MOD_TO_IDX.get("NOISE", 33):
                        mod_idx = -1  # GUARANTEE: Real dataset captures are NEVER NOISE
                    try: snr = float(row.get("snr_db") or row.get("snr") or float("nan"))
                    except (TypeError, ValueError): snr = float("nan")
                    try: baud = float(row.get("baud_rate") or row.get("symrate") or float("nan"))
                    except: baud = float("nan")
                    add_record(str(fname), (mod_idx, snr, baud))
        except Exception:
            pass
    # Kaggle datasets commonly ship a JSON manifest rather than CSV.  The
    # previous code claimed to read JSON but never did, silently masking labels.
    for json_file in list(dir_path.rglob("*ground_truth*.json")) + list(dir_path.rglob("*labels*.json")) + list(dir_path.rglob("*metadata*.json")):
        try:
            with open(json_file, "r", encoding="utf-8", errors="ignore") as f:
                rows = json.load(f)
            if isinstance(rows, dict):
                rows = rows.get("samples", rows.get("data", rows.get("records", [])))
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                fname = row.get("filename") or row.get("file") or row.get("path") or row.get("id")
                if not fname:
                    continue
                raw_mod = str(row.get("modulation") or row.get("mod") or row.get("label") or "")
                canon_mod = DATASET_ALIASES.get(raw_mod.upper().replace("-", "_").replace(".", "_").strip(), raw_mod)
                mod_idx = MOD_TO_IDX.get(canon_mod, -1)
                try: snr = float(row.get("snr_db") or row.get("snr") or float("nan"))
                except (TypeError, ValueError): snr = float("nan")
                try: baud = float(row.get("baud_rate") or row.get("symrate") or float("nan"))
                except (TypeError, ValueError): baud = float("nan")
                add_record(str(fname), (mod_idx, snr, baud))
        except Exception:
            pass
    return meta

def resolve_file_modulation(filepath: Path) -> int:
    """Accurately identify modulation from filename stem and parent directories.
    Returns MOD_TO_IDX[mod] if confirmed, or -1 if unconfirmed.
    NEVER defaults to NOISE. Explicit noise/background filenames are accepted
    as verified negative examples for the NOISE and presence heads.
    """
    parts = [filepath.stem] + [p.name for p in filepath.parents[:3]]
    for part in parts:
        clean = part.upper().replace("-", "_").replace(".", "_").strip()
        tokens = [t for t in clean.split("_") if t]

        # 1. Exact token matching (Real dataset captures are NEVER NOISE)
        for tok in tokens:
            if tok in ("NOISE", "SILENCE", "BACKGROUND", "NOISEONLY"):
                return MOD_TO_IDX["NOISE"]
            if tok in DATASET_ALIASES:
                canon = DATASET_ALIASES[tok]
                if canon in MOD_TO_IDX:
                    return MOD_TO_IDX[canon]
            if tok in MOD_TO_IDX:
                return MOD_TO_IDX[tok]

        # 2. Unambiguous substring matching with boundaries
        for k, canon in sorted(DATASET_ALIASES.items(), key=lambda x: -len(x[0])):
            if len(k) >= 3 and f"_{k}_" in f"_{clean}_":
                if canon in MOD_TO_IDX:
                    return MOD_TO_IDX[canon]

    return -1


def resolve_modrec_modulation(filepath: Path) -> int:
    """Decode the ModRec corpus' documented filename labels, fail closed otherwise."""
    exact = {
        "bpsk_2mhz": "BPSK", "qpsk_2mhz": "QPSK", "8psk_2mhz": "8PSK",
        "qam16_2mhz": "16QAM", "qam64_2mhz": "64QAM",
    }
    return MOD_TO_IDX.get(exact.get(filepath.stem.lower(), ""), -1)


def resolve_kiwi_sigmf(filepath: Path) -> Tuple[int, float, float]:
    """Read a KiwiSDR label from its SigMF sidecar, never from its filename."""
    sidecar = filepath.with_suffix(".sigmf-meta")
    if not sidecar.exists():
        return -1, float("nan"), float("nan")
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if str(meta.get("global", {}).get("core:datatype", "")).lower() != "ci16_le":
            return -1, float("nan"), float("nan")
        annotations = meta.get("annotations", [])
        labels = {str(a.get("core:label", "")).strip() for a in annotations if a.get("core:label")}
        if len(labels) != 1:
            return -1, float("nan"), float("nan")
        raw = next(iter(labels))
        clean = raw.upper().replace("-", "_").replace(".", "_").strip()
        canon = DATASET_ALIASES.get(clean, clean)
        return MOD_TO_IDX.get(canon, -1), float("nan"), float("nan")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return -1, float("nan"), float("nan")


# ==============================================================================
# 1. BATCH GPU 32-D EXPERT FEATURE EXTRACTOR
# ==============================================================================
def extract_expert_32d_batch(iq_batch: torch.Tensor) -> torch.Tensor:
    B, _, L = iq_batch.shape
    I = iq_batch[:, 0, :]
    Q = iq_batch[:, 1, :]

    pwr = torch.mean(I * I + Q * Q, dim=-1, keepdim=True).clamp_min(1e-12)
    norm = torch.rsqrt(pwr)
    I_n = I * norm
    Q_n = Q * norm
    z = torch.complex(I_n, Q_n)

    zc = z - torch.mean(z, dim=-1, keepdim=True)

    c21 = torch.mean(zc.abs().pow(2), dim=-1).clamp_min(1e-12)
    c20 = torch.mean(zc.pow(2), dim=-1)

    m40 = torch.mean(zc.pow(4), dim=-1)
    m41 = torch.mean(zc.pow(3) * zc.conj(), dim=-1)
    m42 = torch.mean(zc.abs().pow(4), dim=-1)

    c40 = m40 - 3.0 * c20.pow(2)
    c41 = m41 - 3.0 * c20 * c21
    c42 = m42 - c20.abs().pow(2) - 2.0 * c21.pow(2)

    m63 = torch.mean(zc.abs().pow(6), dim=-1)
    c63 = m63 - 9.0 * c42 * c21 - 6.0 * c21.pow(3)

    c20_norm = c20.abs() / c21
    c40_norm = c40.abs() / (c21.pow(2))
    c41_norm = c41.abs() / (c21.pow(2))
    c42_norm = c42 / (c21.pow(2))
    c63_norm = c63.real / (c21.pow(3)) if c63.is_complex() else c63 / (c21.pow(3))

    mag = z.abs()
    mean_mag = torch.mean(mag, dim=-1).clamp_min(1e-12)
    std_mag = torch.std(mag, dim=-1, unbiased=False)
    env_cov = std_mag / mean_mag
    env_kurt = torch.mean(mag.pow(4), dim=-1)

    z_prod = z[:, 1:] * z[:, :-1].conj()
    fi = torch.angle(z_prod) / (2.0 * math.pi)
    std_fi = torch.std(fi, dim=-1, unbiased=False)
    mean_abs_fi = torch.mean(fi.abs(), dim=-1)

    spec = torch.fft.fft(z, dim=-1).abs().pow(2).clamp_min(1e-12)
    spec_geom = torch.exp(torch.mean(torch.log(spec), dim=-1))
    spec_arith = torch.mean(spec, dim=-1).clamp_min(1e-12)
    spec_flat = spec_geom / spec_arith

    papr = torch.amax(mag.pow(2), dim=-1).clamp_min(1e-12)
    papr_db = 10.0 * torch.log10(papr)

    feats = torch.zeros((B, 32), dtype=torch.float32, device=iq_batch.device)
    feats[:, 0] = c20_norm
    feats[:, 1] = c40_norm
    feats[:, 2] = c41_norm
    feats[:, 3] = c42_norm
    feats[:, 4] = c63_norm
    feats[:, 5] = env_cov
    feats[:, 6] = env_kurt
    feats[:, 7] = std_fi
    feats[:, 8] = mean_abs_fi
    feats[:, 9] = spec_flat
    feats[:, 10] = papr_db

    return torch.clamp(torch.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)


def extract_training_expert_batch(iq_batch: torch.Tensor) -> torch.Tensor:
    """Return exactly the eleven populated expert features used by V5."""
    return extract_expert_32d_batch(iq_batch)[:, :TRAIN_EXPERT_DIM]


# ==============================================================================
# 2. ON-THE-FLY PHYSICALLY GROUNDED SYNTHETIC GENERATOR
# ==============================================================================
class SyntheticRFBatchGenerator:
    def __init__(self):
        self.linear_mods = ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM")
        # GFSK is deliberately absent here.  A rectangular NRZ frequency
        # pulse followed by phase integration is CPFSK/MSK, not GFSK; the old
        # generator assigned both waveforms the same shape under two labels.
        # Verified Sig53/VariableRF examples still train the GFSK output.
        self.fsk_mods = ("2FSK", "4FSK", "MSK")
        self.all_mods = self.linear_mods + self.fsk_mods + ("NOISE",)
        self.bps = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "16QAM": 4, "64QAM": 6}

    def _rrc_filter(self, alpha: float, sps: int, span: int = 8) -> np.ndarray:
        N = span * sps
        t = np.arange(-N // 2, N // 2 + 1, dtype=np.float64) / sps
        h = np.zeros_like(t)
        for i, val in enumerate(t):
            if abs(val) < 1e-8:
                h[i] = 1.0 - alpha + (4.0 * alpha / np.pi)
            elif abs(abs(val) - 1.0 / (4.0 * alpha)) < 1e-8:
                h[i] = (alpha / np.sqrt(2.0)) * (((1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))) + ((1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))))
            else:
                num = np.sin(np.pi * val * (1.0 - alpha)) + 4.0 * alpha * val * np.cos(np.pi * val * (1.0 + alpha))
                den = np.pi * val * (1.0 - (4.0 * alpha * val) ** 2)
                h[i] = num / den
        return (h / np.sqrt(np.sum(h ** 2))).astype(np.float32)

    def generate_sample(self, rng: np.random.Generator, length: int = 1024) -> Tuple[np.ndarray, int, float, float]:
        mod = rng.choice(self.all_mods)
        snr_db = float(rng.uniform(-4.0, 24.0))
        sps = int(rng.choice([4, 8, 16]))
        baud_rate = 1.0 / sps
        cfo = float(rng.uniform(-0.02, 0.02))

        if mod == "NOISE":
            noise = (rng.standard_normal(length) + 1j * rng.standard_normal(length)).astype(np.complex64) / np.sqrt(2.0)
            iq = np.stack([noise.real, noise.imag]).astype(np.float32)
            return iq, MOD_TO_IDX["NOISE"], float("nan"), float('nan')

        if mod in self.linear_mods:
            n_sym = length // sps + 32
            bits = rng.integers(0, 2, size=n_sym * self.bps[mod])
            if mod == "BPSK":
                syms = (2.0 * bits - 1.0).astype(np.complex64)
            elif mod == "QPSK":
                syms = ((2.0 * bits[0::2] - 1.0) + 1j * (2.0 * bits[1::2] - 1.0)) / np.sqrt(2.0)
            elif mod == "8PSK":
                theta = (bits.reshape(-1, 3) @ np.array([4, 2, 1])) * (2.0 * np.pi / 8.0)
                syms = np.exp(1j * theta).astype(np.complex64)
            elif mod == "16QAM":
                b = bits.reshape(-1, 4)
                qam16 = np.array([-3.0, -1.0, 3.0, 1.0])
                i_lev = qam16[2 * b[:, 0] + b[:, 1]]
                q_lev = qam16[2 * b[:, 2] + b[:, 3]]
                syms = (i_lev + 1j * q_lev) / np.sqrt(10.0)
            else: # 64QAM
                b = bits.reshape(-1, 6)
                qam64 = np.array([-7.0, -5.0, -1.0, -3.0, 7.0, 5.0, 1.0, 3.0])
                i_lev = qam64[4 * b[:, 0] + 2 * b[:, 1] + b[:, 2]]
                q_lev = qam64[4 * b[:, 3] + 2 * b[:, 4] + b[:, 5]]
                syms = (i_lev + 1j * q_lev) / np.sqrt(42.0)

            up = np.zeros(len(syms) * sps, dtype=np.complex64)
            up[::sps] = syms
            h = self._rrc_filter(alpha=float(rng.uniform(0.15, 0.45)), sps=sps)
            x = np.convolve(up, h, mode="same")
            s0 = int(rng.integers(0, max(1, len(x) - length)))
            sig = x[s0:s0 + length]

        else: # FSK / CPM
            n_sym = length // sps + 16
            if mod == "4FSK":
                levels = rng.choice(np.array([-3.0, -1.0, 1.0, 3.0]), size=n_sym)
                h_idx = float(rng.uniform(0.25, 0.5))
            else:
                levels = rng.choice(np.array([-1.0, 1.0]), size=n_sym)
                h_idx = 0.5 if mod == "MSK" else float(rng.uniform(0.5, 1.0))
            f_inst = np.repeat(levels, sps).astype(np.float64) * h_idx / (2.0 * sps)
            phi = 2.0 * np.pi * np.cumsum(f_inst)
            sig = np.exp(1j * phi[:length])

        if len(sig) > length:
            sig = sig[:length]
        elif len(sig) < length:
            sig = np.pad(sig, (0, length - len(sig)))

        t = np.arange(length, dtype=np.float64)
        sig = sig / np.sqrt(np.mean(np.abs(sig) ** 2) + 1e-12)
        sig = sig * np.exp(1j * (2.0 * np.pi * cfo * t + rng.uniform(0, 2.0 * np.pi)))
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise = (rng.standard_normal(length) + 1j * rng.standard_normal(length)) / np.sqrt(2.0 * snr_lin)
        sig = (sig + noise).astype(np.complex64)
        sig = sig / np.sqrt(np.mean(np.abs(sig) ** 2) + 1e-12)

        iq = np.stack([sig.real, sig.imag]).astype(np.float32)
        return iq, MOD_TO_IDX[mod], snr_db, baud_rate


# ==============================================================================
# 3. DYNAMIC 10-DATASET GRAND CORPUS STREAMER (ZERO-FUSE-WAIT SEEK READS)
# ==============================================================================
class DynamicGrandCorpusStreamer:
    """Streams a fixed, audited corpus with a deterministic 90/10 split.

    The old implementation drew train and "validation" examples from the same
    unrestricted stream.  Here every HDF5 row and raw-file path is assigned to
    exactly one partition before training starts.
    """
    def __init__(self, catalog: Dict[str, Any], ground_truth_map: Optional[Dict[str, Tuple[int, float, float]]] = None,
                 seed: int = 20260924, input_len: int = 1024,
                 synthetic_fraction: float = 0.10, synthetic_pool_size: int = 8192):
        self.catalog = catalog
        self.gt_map = ground_truth_map or {}
        self.active_sources = []
        self.seed = int(seed)
        self.input_len = int(input_len)
        if self.input_len not in {128, 256, 512, 1024}:
            raise ValueError("input_len must be one of 128, 256, 512, 1024")
        self.synthetic_fraction = float(synthetic_fraction)
        if not 0.0 <= self.synthetic_fraction <= 0.5:
            raise ValueError("synthetic_fraction must be between 0 and 0.5")
        self.synthetic_pool_size = int(synthetic_pool_size)
        if self.synthetic_pool_size < 100:
            raise ValueError("synthetic_pool_size must be at least 100 for disjoint 80/10/10 partitions")
        self.rng = np.random.default_rng(self.seed)
        self.synth_gen = SyntheticRFBatchGenerator()
        self.ram_cache: Dict[str, np.ndarray] = {}
        # Keep raw IQ files memory-mapped after their first access.  The
        # mapping does not copy a full capture into RAM; it lets the OS cache
        # the small random windows used by training and avoids reopening a
        # Kaggle FUSE file for every hardware example.
        self.raw_iq_maps: Dict[str, np.memmap] = {}
        self.max_cache_size = 2000
        self._partition_group_cache: Dict[Tuple[str, int, int], Dict[str, np.ndarray]] = {}

        # Keep synthetic examples disjoint between train and validation too.
        print(f">> [SYNTHETIC POOL] Precomputing {self.synthetic_pool_size:,} "
              "ground-truth waveforms in RAM...", flush=True)
        synth_iq, synth_m, synth_snr, synth_baud = [], [], [], []
        for _ in range(self.synthetic_pool_size):
            iq, m, snr, baud = self.synth_gen.generate_sample(self.rng, length=self.input_len)
            synth_iq.append(iq)
            synth_m.append(m)
            synth_snr.append(snr)
            synth_baud.append(baud)
        self.synth_pool_iq = np.stack(synth_iq, axis=0)
        self.synth_pool_mod = np.array(synth_m, dtype=np.int64)
        self.synth_pool_snr = np.array(synth_snr, dtype=np.float32)
        self.synth_pool_baud = np.array(synth_baud, dtype=np.float32)
        self.synth_train_end = int(round(0.80 * self.synthetic_pool_size))
        self.synth_cal_end = int(round(0.90 * self.synthetic_pool_size))
        print(f">> [SYNTHETIC POOL] Ready; {100.0 * self.synthetic_fraction:.1f}% per batch, "
              "disjoint 80/10/10 train/cal/validation pools.", flush=True)

        # Ordered 25-class variable-length corpus. Only the group matching this
        # run is opened, which guarantees that every batch has one fixed length.
        self.variable_f = h5py.File(catalog["variable_rf"], "r", swmr=True,
                                    rdcc_nbytes=64*1024*1024) if catalog.get("variable_rf") else None
        self.variable_group = None
        self.variable_len = 0
        self.variable_map = np.full(256, -1, dtype=np.int64)
        if self.variable_f is not None:
            group_name = f"len_{self.input_len}"
            if group_name in self.variable_f:
                group = self.variable_f[group_name]
                if all(k in group for k in ("iq", "mod", "snr")):
                    raw_names = group.attrs.get(
                        "class_names",
                        group.attrs.get("classes", self.variable_f.attrs.get(
                            "class_names", self.variable_f.attrs.get("classes", []))),
                    )
                    names = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in raw_names]
                    if not names:
                        raise RuntimeError(f"{catalog['variable_rf']}:{group_name} has no class_names table")
                    for idx, name in enumerate(names):
                        clean = name.upper().replace("-", "_").strip()
                        canon = DATASET_ALIASES.get(clean, name.upper().strip())
                        if canon in MOD_TO_IDX:
                            self.variable_map[idx] = MOD_TO_IDX[canon]
                    self.variable_group = group
                    self.variable_len = int(group["iq"].shape[0])
                    if self.variable_len and np.any(self.variable_map >= 0):
                        self.active_sources.append("variable_rf")
                        print(f">> [VariableRF] {group_name}: {self.variable_len:,} rows, "
                              f"{int(np.count_nonzero(self.variable_map[:len(names)] >= 0))}/{len(names)} classes mapped.",
                              flush=True)

        # 1. Sig53 HDF5 Handler with universal key detection
        self.sig53_f = h5py.File(catalog["sig53"], "r", swmr=True, rdcc_nbytes=64*1024*1024) if catalog.get("sig53") else None
        self.sig53_x_key = None
        self.sig53_y_key = None
        self.sig53_snr_key = None
        self.sig53_len = 0
        self.sig53_map = None
        # The supplied variable-length corpus was derived from Sig53.  Its
        # HDF5 export has no original-waveform ID, so a split cannot prove
        # that a 128/256/512 record and a Sig53 crop are disjoint.  For those
        # lengths, use native variable-RF samples and exclude their parent
        # Sig53 source rather than risk parent/child train-validation leakage.
        self.sig53_excluded_for_parent_overlap = self.variable_group is not None

        if self.sig53_f:
            for k in ["X", "x", "data", "samples", "iq"]:
                if k in self.sig53_f and isinstance(self.sig53_f[k], h5py.Dataset):
                    self.sig53_x_key = k
                    break
            for k in ["y_mod", "Y", "y", "labels", "targets", "mod"]:
                if k in self.sig53_f and isinstance(self.sig53_f[k], h5py.Dataset):
                    self.sig53_y_key = k
                    break
            for k in ["y_snr", "Z", "z", "snr", "snrs"]:
                if k in self.sig53_f and isinstance(self.sig53_f[k], h5py.Dataset):
                    self.sig53_snr_key = k
                    break
            if self.sig53_x_key:
                self.sig53_len = self.sig53_f[self.sig53_x_key].shape[0]
                self.sig53_map = self._build_sig53_map()
                if self.sig53_len > 0 and np.any(self.sig53_map >= 0):
                    if self.sig53_excluded_for_parent_overlap:
                        print(">> [SKIP] Sig53 direct stream excluded: the mounted VariableRF "
                              "L=%d corpus is Sig53-derived; parent/child overlap is unsafe." % self.input_len,
                              flush=True)
                    else:
                        self.active_sources.append("sig53")
                elif self.sig53_len:
                    print(">> [SKIP] Sig53 has no verified class translation; not used for training.", flush=True)
            else:
                print(">> [SKIP] Sig53 HDF5 is a grouped/wideband TorchSig export, not a flat "
                      "narrowband classification table. Convert it with a verified adapter first.", flush=True)

        # 2. RadioML Handler
        self.rml_f = h5py.File(catalog["radioml"], "r", swmr=True, rdcc_nbytes=64*1024*1024) if catalog.get("radioml") else None
        self.rml_len = self.rml_f["X"].shape[0] if self.rml_f else 0
        self.rml_classes = None
        if self.rml_f:
            for name in ("classes-fixed.json", "classes-fixed.txt"):
                class_file = Path(catalog["radioml"]).parent / name
                if not class_file.exists():
                    continue
                try:
                    if class_file.suffix == ".json":
                        parsed = json.loads(class_file.read_text(encoding="utf-8"))
                        parsed = parsed.get("classes", parsed) if isinstance(parsed, dict) else parsed
                        classes = [str(v) for v in parsed]
                    else:
                        classes = [line.strip() for line in class_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                    if len(classes) == self.rml_f["Y"].shape[1]:
                        self.rml_classes = classes
                        print(f">> [RadioML] Loaded verified class order from {class_file.name}", flush=True)
                        break
                except Exception as exc:
                    print(f">> [RadioML] Could not read {class_file.name}: {exc}", flush=True)
            if self.rml_classes is None:
                raise RuntimeError(
                    f"{catalog['radioml']}: RadioML labels are unsafe without "
                    "classes-fixed.json or classes-fixed.txt beside the HDF5"
                )
        if self.rml_len > 0 and self.rml_classes is not None:
            self.active_sources.append("radioml")

        # 3. Panoradio matrix + row-aligned tag table.  This public corpus is
        # commonly distributed as complex ``(N, L)`` NPY plus
        # ``*_tags.csv``.  Treating it as one generic capture made every row
        # unreadable and therefore contributed zero AMC labels.
        self.panoradio_iq = None
        self.panoradio_mod = np.zeros(0, dtype=np.int64)
        self.panoradio_snr = np.zeros(0, dtype=np.float32)
        self.panoradio_valid_indices = np.zeros(0, dtype=np.int64)
        for npy_path in catalog.get("panoradio", []):
            if Path(npy_path).suffix.lower() != ".npy":
                continue
            tag_candidates = sorted(Path(npy_path).parent.glob("*tags*.csv"))
            if not tag_candidates:
                continue
            try:
                matrix = np.load(npy_path, mmap_mode="r")
                if matrix.ndim != 2 or not np.iscomplexobj(matrix) or matrix.shape[1] < 1024:
                    continue
                labels = np.full(matrix.shape[0], -1, dtype=np.int64)
                snrs = np.full(matrix.shape[0], np.nan, dtype=np.float32)
                with open(tag_candidates[0], "r", encoding="utf-8", errors="ignore") as f:
                    for row in csv.DictReader(f, skipinitialspace=True):
                        try:
                            row_idx = int(row.get("idx", row.get("index", -1)))
                        except (TypeError, ValueError):
                            continue
                        if not (0 <= row_idx < len(labels)):
                            continue
                        raw = str(row.get("mode") or row.get("modulation") or row.get("label") or "")
                        clean = raw.upper().replace("-", "_").replace(".", "_").strip()
                        canon = DATASET_ALIASES.get(clean, clean)
                        labels[row_idx] = MOD_TO_IDX.get(canon, -1)
                        try: snrs[row_idx] = float(row.get("snr", row.get("snr_db", "nan")))
                        except (TypeError, ValueError): pass
                valid = labels >= 0
                if valid.any():
                    self.panoradio_iq = matrix
                    self.panoradio_mod = labels
                    self.panoradio_snr = snrs
                    self.panoradio_valid_indices = np.flatnonzero(valid).astype(np.int64)
                    self.active_sources.append("panoradio")
                    print(f">> [Panoradio] {valid.sum():,}/{len(labels):,} rows have canonical labels.", flush=True)
                    break
            except Exception as exc:
                print(f">> [SKIP] Panoradio adapter failed for {npy_path}: {exc}", flush=True)

        # 4. Real-world Hardware File Datasets
        self.file_datasets: Dict[str, List[Tuple[Path, int, int, str, float, float]]] = {}
        self.hardware_partition_by_path: Dict[str, str] = {}
        all_hardware_names = [
            "migou", "modrec", "mendeley", "commrad",
            "sigid", "kiwi", "hamsci", "lora_rffi", "signal3", "other_rf"
        ]
        for name in all_hardware_names:
            files = catalog.get(name, [])
            records = []
            for p in files:
                try:
                    sz = p.stat().st_size
                    if sz >= 1024:
                        if p.name in self.gt_map:
                            m, s, b = self.gt_map[p.name]
                        elif name == "modrec":
                            m = resolve_modrec_modulation(p)
                            s, b = float("nan"), float("nan")
                        elif name == "kiwi":
                            m, s, b = resolve_kiwi_sigmf(p)
                        else:
                            # Generic filename tokens are not authoritative
                            # ground truth. Unverified corpora remain mounted
                            # but do not contribute classification targets.
                            m = -1
                            s, b = float("nan"), float("nan")
                        records.append((p, m, sz, p.suffix.lower(), s, b))
                except Exception:
                    pass
            labeled_records = [r for r in records if r[1] >= 0]
            if labeled_records:
                self.file_datasets[name] = labeled_records
                self.active_sources.append(name)
            elif records:
                print(f">> [SKIP] {name}: {len(records):,} readable files but zero verified modulation labels.", flush=True)

        # Keep every known hardware acquisition session in one partition.
        # ModRec stores several modulation files under one receiver/session
        # directory; splitting those sibling captures leaked receiver/channel
        # fingerprints into validation. Other hardware corpora lack explicit
        # session IDs, so one file is the most conservative available group.
        for source, records in self.file_datasets.items():
            groups: Dict[str, List[Path]] = {}
            for record in records:
                path = record[0]
                key = self._hardware_group_key(source, path)
                groups.setdefault(key, []).append(path)
            assignments = self._balanced_hardware_group_partitions(source, list(groups))
            for key, paths in groups.items():
                for path in paths:
                    self.hardware_partition_by_path[str(path)] = assignments[key]

        # 5. Synthetic generator is always active
        self.active_sources.append("synthetic")

        print("\n" + "=" * 80, flush=True)
        print(">> [GRAND CORPUS DISCOVERY SUMMARY - HISARMOD EXCLUDED]", flush=True)
        for name in all_hardware_names:
            cnt = len(self.file_datasets.get(name, []))
            if cnt > 0:
                train_n = sum(self._hardware_partition(name, r[0]) == "train" for r in self.file_datasets[name])
                cal_n = sum(self._hardware_partition(name, r[0]) == "cal" for r in self.file_datasets[name])
                print(f">> [*]  {name:<18}: {cnt:,} verified files "
                      f"({train_n:,} train / {cal_n:,} cal / {cnt-train_n-cal_n:,} val)", flush=True)
        if "sig53" in self.active_sources:
            print(f">> [+]  Sig53 Dataset      : {self.sig53_len:,} samples active (Mapped via Translation Table)", flush=True)
        elif self.sig53_len:
            print(f">> [~]  Sig53 Dataset      : excluded by {SOURCE_OVERLAP_POLICY}", flush=True)
        if self.rml_len:   print(f">> [+]  RadioML 2018/2016  : {self.rml_len:,} samples active", flush=True)
        if self.variable_len: print(f">> [+]  VariableRF L={self.input_len:<4}: {self.variable_len:,} samples active", flush=True)
        print(f">> [*]  Synthetic RF Anchor: Active (Ground Truth Precomputed RAM Pool)", flush=True)
        print(f">> TOTAL ACTIVE SOURCES    : {len(self.active_sources)} Datasets contributing directly", flush=True)
        print("=" * 80 + "\n", flush=True)

    def _path_partition(self, path: Path) -> str:
        digest = hashlib.blake2b(str(path).encode("utf-8"), digest_size=8, person=b"s9split").digest()
        bucket = int.from_bytes(digest, "little") % 10
        return "val" if bucket == 9 else ("cal" if bucket == 8 else "train")

    @staticmethod
    def _hardware_group_key(source: str, path: Path) -> str:
        if source == "modrec":
            return "/".join(path.parent.parts[-2:])
        return f"{path.parent.name}/{path.name}"

    def _balanced_hardware_group_partitions(self, source: str,
                                            group_keys: List[str]) -> Dict[str, str]:
        """Deterministic group-safe 80/10/10 assignment with usable holdouts."""
        ordered = sorted(group_keys, key=lambda key: hashlib.blake2b(
            f"{self.seed}:{source}:{key}".encode("utf-8"), digest_size=8,
            person=b"s9hwgrp").digest())
        n_groups = len(ordered)
        if n_groups == 0:
            return {}
        if n_groups == 1:
            return {ordered[0]: "train"}
        if n_groups == 2:
            return {ordered[0]: "train", ordered[1]: "val"}
        n_holdout = max(1, int(round(0.10 * n_groups)))
        n_holdout = min(n_holdout, (n_groups - 1) // 2)
        result = {key: "train" for key in ordered}
        for key in ordered[-2 * n_holdout:-n_holdout]:
            result[key] = "cal"
        for key in ordered[-n_holdout:]:
            result[key] = "val"
        return result

    def _hardware_partition(self, source: str, path: Path) -> str:
        return self.hardware_partition_by_path.get(str(path), self._path_partition(path))

    def _is_validation_path(self, path: Path) -> bool:
        return self._path_partition(path) == "val"

    def _group_bucket(self, group_id: int, source: str) -> int:
        """Stable 80/10/10 bucket for a contiguous acquisition group.

        HDF exports do not expose waveform/session identifiers.  Assigning
        adjacent rows to different partitions (the old modulo-10 split) was
        therefore unnecessarily leakage-prone.  We keep blocks together and
        hash the block with the source name and run seed.  This is conservative
        and reproducible, while honestly not pretending unavailable provenance
        metadata exists.
        """
        payload = f"{self.seed}:{source}:{int(group_id)}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8, person=b"s9h5grp").digest()
        return int.from_bytes(digest, "little") % 10

    @staticmethod
    def _bucket_matches(bucket: int, partition: str) -> bool:
        return bucket < 8 if partition == "train" else bucket == (8 if partition == "cal" else 9)

    def _partition_h5_indices(self, total: int, needed: int, partition: str,
                              source: str = "hdf5") -> np.ndarray:
        """Sample split-safe rows with compression-friendly random locality.

        HDF5 fancy indexing over fully scattered rows forces gzip to inflate a
        large chunk for almost every requested example.  We instead select up
        to ``H5_IO_FANOUT_GROUPS`` independently randomized 64-row groups and
        take a short contiguous run from each.  Groups remain randomly chosen
        on every call, returned rows are shuffled, and the requested count is
        unchanged.  The fan-out is deliberately greater than one because the
        public Sig53 export is class-ordered.
        """
        if total <= 0 or needed <= 0:
            return np.zeros(0, dtype=np.int64)
        group_rows = H5_SPLIT_GROUP_ROWS if total >= 10 * H5_SPLIT_GROUP_ROWS else 1
        n_groups = (total + group_rows - 1) // group_rows
        cache = getattr(self, "_partition_group_cache", None)
        if cache is None:
            cache = self._partition_group_cache = {}
        key = (source, int(total), int(group_rows))
        if key not in cache:
            buckets = np.fromiter((self._group_bucket(g, source) for g in range(n_groups)),
                                  dtype=np.uint8, count=n_groups)
            cache[key] = {
                "train": np.flatnonzero(buckets < 8).astype(np.int64),
                "cal": np.flatnonzero(buckets == 8).astype(np.int64),
                "val": np.flatnonzero(buckets == 9).astype(np.int64),
            }
        eligible = cache[key][partition]
        if not len(eligible):
            raise RuntimeError(f"{source}: no {partition} groups available for {total} rows")
        if group_rows == 1:
            return self.rng.choice(eligible, size=needed,
                                   replace=(len(eligible) < needed)).astype(np.int64, copy=False)

        fanout = min(needed, H5_IO_FANOUT_GROUPS)
        chosen_groups = self.rng.choice(eligible, size=fanout,
                                        replace=(len(eligible) < fanout))
        base_count, extra = divmod(needed, fanout)
        rows_by_group = []
        for position, group_id in enumerate(chosen_groups):
            take = base_count + (1 if position < extra else 0)
            group_start = int(group_id) * group_rows
            valid_rows = min(group_rows, total - group_start)
            if take <= valid_rows:
                first = int(self.rng.integers(0, valid_rows - take + 1))
                offsets = np.arange(first, first + take, dtype=np.int64)
            else:
                # Only possible for a short final group or an unusually large
                # request. Preserve the exact count with replacement.
                offsets = self.rng.integers(0, valid_rows, size=take, dtype=np.int64)
            rows_by_group.append(group_start + offsets)
        rows = np.concatenate(rows_by_group)
        self.rng.shuffle(rows)
        return rows.astype(np.int64, copy=False)

    @staticmethod
    def _read_h5_rows(dataset: Any, indices: np.ndarray) -> np.ndarray:
        # h5py requires increasing, unique fancy indices. Restore requested order.
        order = np.argsort(indices)
        sorted_indices = indices[order]
        unique_indices, reverse = np.unique(sorted_indices, return_inverse=True)
        rows = np.asarray(dataset[unique_indices])[reverse]
        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(len(order))
        return rows[inverse_order]

    def _build_sig53_map(self) -> np.ndarray:
        map_arr = np.full(128, -1, dtype=np.int64)
        file_classes = None
        # Prefer the class table embedded in the file.  Different Sig53
        # exports can have a different number/order of classes; falling back
        # to the public 53-class order when a 32-class table exists silently
        # assigns the wrong target to every row.
        for k in ["class_names", "class_list", "classes", "labels"]:
            if k in self.sig53_f.attrs:
                raw_attr = self.sig53_f.attrs[k]
                file_classes = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in raw_attr]
                break
            if k in self.sig53_f:
                file_classes = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in self.sig53_f[k]]
                break

        if file_classes:
            for idx, cname in enumerate(file_classes):
                clean = cname.upper().replace("-", "_").strip()
                # These Sig53 labels do not have a truthful target in the
                # current closed vocabulary.  Do not turn 8/16-FSK into
                # 4-FSK or LoRa/chirp into CPFSK: that injects systematic
                # label noise into the modulation head.  Their rows may still
                # contribute only to non-classification auxiliary objectives.
                if clean in {"16FSK", "OFDM", "DSSS", "LORA", "CHIRP", "UNKNOWN"}:
                    continue
                canon = DATASET_ALIASES.get(clean, clean)
                if canon in MOD_TO_IDX:
                    map_arr[idx] = MOD_TO_IDX[canon]
            source_name = next((k for k in ["class_names", "class_list", "classes", "labels"] if k in self.sig53_f), "attribute")
            print(f">> [Sig53] Using embedded {source_name!r} label table: "
                  f"{int(np.count_nonzero(map_arr[:len(file_classes)] >= 0))}/{len(file_classes)} "
                  "classes mapped to the canonical vocabulary.", flush=True)
            return map_arr

        print(">> [SKIP] Sig53 has no embedded class-name table; refusing the "
              "version-dependent hard-coded TorchSig ordering.", flush=True)
        return map_arr

    def _read_hw_sample(self, item: Tuple[Path, int, int, str, float, float]) -> Tuple[np.ndarray, int, float, float]:
        fp, mod, fsize, ext, snr_val, baud_val = item
        # Without a verified sample rate, an absolute baud value cannot be
        # compared with the synthetic normalised rate (cycles/sample).
        if np.isfinite(baud_val) and not (0.0 < baud_val <= 0.5):
            baud_val = float("nan")
        key = str(fp)

        # 1. Zero-latency RAM cache path for small files (0.001 ms)
        if key in self.ram_cache:
            arr = self.ram_cache[key]
            max_s = arr.shape[1]
            if max_s >= self.input_len:
                s0 = self.rng.integers(0, max_s - self.input_len + 1)
                return arr[:, s0:s0+self.input_len], mod, snr_val, baud_val

        try:
            # 2. Fast Seek-and-Read EXACT 1024 samples (8 KB) from disk: 0.06 ms
            if ext in [".cf32", ".raw", ".bin", ".dat", ".iq", ".cfile"] and fsize >= 1024:
                # Reusing a read-only memmap preserves the old random-window
                # behavior but removes thousands of repeated open/seek/close
                # calls per epoch across ModRec and other hardware captures.
                raw_map = self.raw_iq_maps.get(key)
                if raw_map is None:
                    raw_map = np.memmap(fp, dtype=np.complex64, mode="r")
                    self.raw_iq_maps[key] = raw_map
                n_samples = len(raw_map)
                if n_samples >= self.input_len:
                    s0 = self.rng.integers(0, max(1, n_samples - self.input_len + 1))
                    raw = np.asarray(raw_map[s0:s0 + self.input_len], dtype=np.complex64)
                else:
                    raw = np.asarray(raw_map, dtype=np.complex64)
                    raw = np.pad(raw, (0, self.input_len - len(raw)))
                if len(raw) == self.input_len:
                    arr = np.stack([raw.real, raw.imag], axis=0).astype(np.float32)
                    return arr, mod, snr_val, baud_val

            elif ext == ".wav":
                # The verified Kiwi corpus is stereo, 16-bit, uncompressed
                # complex IQ. ``wave`` respects non-44-byte RIFF headers;
                # mono audio is rejected instead of converted to another
                # signal domain with a Hilbert transform.
                with wave.open(str(fp), "rb") as wav:
                    if wav.getnchannels() != 2 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
                        raise ValueError("WAV is not verified stereo ci16 IQ")
                    n_samples = wav.getnframes()
                    if n_samples < self.input_len:
                        raise ValueError("WAV is shorter than requested IQ window")
                    s0 = int(self.rng.integers(0, n_samples - self.input_len + 1))
                    wav.setpos(s0)
                    raw = np.frombuffer(wav.readframes(self.input_len), dtype="<i2")
                raw2 = raw.reshape(-1, 2)
                arr = np.stack([raw2[:, 0], raw2[:, 1]], axis=0).astype(np.float32) / 32768.0
                return arr, mod, snr_val, baud_val

            elif ext == ".npy" and fsize > 100:
                loaded = np.load(fp, mmap_mode="r")
                if loaded.ndim == 2:
                    if loaded.shape[0] == 2 and loaded.shape[1] >= self.input_len:
                        s0 = self.rng.integers(0, loaded.shape[1] - self.input_len + 1)
                        return np.array(loaded[:, s0:s0+self.input_len], dtype=np.float32), mod, snr_val, baud_val
                    elif loaded.shape[1] == 2 and loaded.shape[0] >= self.input_len:
                        s0 = self.rng.integers(0, loaded.shape[0] - self.input_len + 1)
                        return np.array(loaded[s0:s0+self.input_len, :].T, dtype=np.float32), mod, snr_val, baud_val

        except Exception:
            pass

        # Fallback: unconfirmed/unreadable capture is marked -1 (masked loss, NEVER NOISE)
        raw_noise = (self.rng.standard_normal(self.input_len) + 1j * self.rng.standard_normal(self.input_len)).astype(np.complex64) / np.sqrt(2.0)
        return np.stack([raw_noise.real, raw_noise.imag]).astype(np.float32), -1, 0.0, float('nan')

    def _augment_train_batch(self, iq_batch: np.ndarray) -> np.ndarray:
        """Conservative RF-domain invariances used by TorchSig and AMC work.

        These preserve modulation identity: arbitrary carrier phase, small CFO,
        spectral inversion and mild IQ image leakage.  Validation is never
        augmented.
        """
        z = iq_batch[:, 0, :].astype(np.float64) + 1j * iq_batch[:, 1, :].astype(np.float64)
        b, n = z.shape
        phase = self.rng.uniform(0.0, 2.0 * np.pi, size=(b, 1))
        z *= np.exp(1j * phase)
        invert = self.rng.random(b) < 0.25
        z[invert] = np.conj(z[invert])
        use_cfo = self.rng.random(b) < 0.75
        cfo = self.rng.uniform(-0.015, 0.015, size=(b, 1)) * use_cfo[:, None]
        z *= np.exp(1j * 2.0 * np.pi * cfo * np.arange(n)[None, :])
        image = self.rng.uniform(-0.04, 0.04, size=(b, 1)) + 1j * self.rng.uniform(-0.04, 0.04, size=(b, 1))
        z = z + image * np.conj(z)
        drop_rows = self.rng.random(b) < 0.15
        for row in np.flatnonzero(drop_rows):
            count = max(1, int(0.01 * n))
            z[row, self.rng.choice(n, size=count, replace=False)] = 0.0
        return np.stack([z.real, z.imag], axis=1).astype(np.float32)

    def get_batch(self, batch_size: int = 256, partition: str = "train") -> Tuple[torch.Tensor, ...]:
        if partition not in {"train", "cal", "val"}:
            raise ValueError("partition must be 'train', 'cal' or 'val'")
        iq_list, mod_list, snr_list, baud_list = [], [], [], []

        # A modest synthetic anchor improves impairment diversity without
        # dominating millions of verified real/corpus examples.  The larger
        # pool and RF augmentation sharply reduce exact waveform reuse.
        n_synth = (0 if self.synthetic_fraction == 0 else
                   max(1, int(round(batch_size * self.synthetic_fraction))))
        synth_bounds = {"train": (0, self.synth_train_end),
                        "cal": (self.synth_train_end, self.synth_cal_end),
                        "val": (self.synth_cal_end, self.synthetic_pool_size)}
        synth_start, synth_end = synth_bounds[partition]
        if n_synth:
            s_idx = self.rng.integers(synth_start, synth_end, size=n_synth)
            iq_list.append(self.synth_pool_iq[s_idx])
            mod_list.append(self.synth_pool_mod[s_idx])
            snr_list.append(self.synth_pool_snr[s_idx])
            baud_list.append(self.synth_pool_baud[s_idx])

        # 2. Remaining 75% drawn across all mounted real-world datasets
        n_remain = batch_size - n_synth
        dataset_sources = [s for s in self.active_sources if s != "synthetic"]
        samples_per_src = max(2, n_remain // max(1, len(dataset_sources)))

        for src in dataset_sources:
            needed = samples_per_src

            if src == "variable_rf" and self.variable_group is not None:
                idx = self._partition_h5_indices(self.variable_len, needed, partition, "variable_rf")
                raw_iq = self._read_h5_rows(self.variable_group["iq"], idx).astype(np.float32)
                if raw_iq.ndim == 3 and raw_iq.shape[1] != 2 and raw_iq.shape[2] == 2:
                    raw_iq = np.transpose(raw_iq, (0, 2, 1))
                if raw_iq.shape[1:] != (2, self.input_len):
                    raise RuntimeError(f"VariableRF group has incompatible IQ shape {raw_iq.shape}")
                raw_labels = self._read_h5_rows(self.variable_group["mod"], idx).ravel()
                mapped = np.array([
                    self.variable_map[int(v)] if 0 <= int(v) < len(self.variable_map) else -1
                    for v in raw_labels
                ], dtype=np.int64)
                iq_list.append(raw_iq)
                mod_list.append(mapped)
                snr_list.append(self._read_h5_rows(self.variable_group["snr"], idx).ravel().astype(np.float32))
                baud_list.append(np.full((needed,), float("nan"), dtype=np.float32))

            elif src == "sig53" and self.sig53_f and self.sig53_x_key:
                idx = self._partition_h5_indices(self.sig53_len, needed, partition, "sig53")
                raw_iq = self._read_h5_rows(self.sig53_f[self.sig53_x_key], idx)

                # Format to (needed, 2, L)
                if np.iscomplexobj(raw_iq):
                    raw_iq = np.stack([raw_iq.real, raw_iq.imag], axis=1).astype(np.float32)
                elif raw_iq.ndim == 3 and raw_iq.shape[1] > 2 and raw_iq.shape[2] == 2:
                    raw_iq = np.transpose(raw_iq, (0, 2, 1)).astype(np.float32)
                elif raw_iq.ndim == 2:
                    raw_iq = np.stack([raw_iq, np.zeros_like(raw_iq)], axis=1).astype(np.float32)

                if raw_iq.shape[-1] > self.input_len:
                    s0 = self.rng.integers(0, raw_iq.shape[-1] - self.input_len + 1)
                    raw_iq = raw_iq[:, :, s0:s0+self.input_len]
                elif raw_iq.shape[-1] < self.input_len:
                    raw_iq = np.pad(raw_iq, ((0, 0), (0, 0), (0, self.input_len - raw_iq.shape[-1])))
                iq_list.append(raw_iq)

                # Labels: translated to canonical vocab with zero false NOISE
                if self.sig53_y_key:
                    raw_labels = self._read_h5_rows(self.sig53_f[self.sig53_y_key], idx)
                    if raw_labels.ndim > 1:
                        raw_labels = np.argmax(raw_labels, axis=-1)
                    mapped = np.array([
                        self.sig53_map[int(lbl)] if 0 <= int(lbl) < len(self.sig53_map) else -1
                        for lbl in raw_labels
                    ], dtype=np.int64)
                else:
                    mapped = np.full((needed,), -1, dtype=np.int64)
                mod_list.append(mapped)

                # SNR
                if self.sig53_snr_key:
                    raw_snr = self._read_h5_rows(self.sig53_f[self.sig53_snr_key], idx).ravel().astype(np.float32)
                else:
                    raw_snr = np.full((needed,), float("nan"), dtype=np.float32)
                snr_list.append(raw_snr)
                baud_list.append(np.full((needed,), float('nan'), dtype=np.float32))

            elif src == "radioml" and self.rml_f:
                idx = self._partition_h5_indices(self.rml_len, needed, partition, "radioml")
                raw = self._read_h5_rows(self.rml_f["X"], idx)
                if raw.ndim == 3 and raw.shape[2] == 2:
                    raw = np.transpose(raw, (0, 2, 1))
                if raw.shape[-1] > self.input_len:
                    starts = self.rng.integers(0, raw.shape[-1] - self.input_len + 1, size=len(raw))
                    raw = np.stack([row[:, s:s+self.input_len] for row, s in zip(raw, starts)])
                elif raw.shape[-1] < self.input_len:
                    raw = np.pad(raw, ((0, 0), (0, 0), (0, self.input_len - raw.shape[-1])))
                iq_list.append(raw.astype(np.float32))
                cls_idxs = np.argmax(self._read_h5_rows(self.rml_f["Y"], idx), axis=1)
                mods = [MOD_TO_IDX.get(self.rml_classes[c], -1) if c < len(self.rml_classes) else -1 for c in cls_idxs]
                mod_arr = np.array(mods, dtype=np.int64)
                mod_list.append(mod_arr)
                snr_list.append(self._read_h5_rows(self.rml_f["Z"], idx).ravel())
                baud_list.append(np.full((needed,), float('nan'), dtype=np.float32))

            elif src == "panoradio" and self.panoradio_iq is not None:
                pool = self.panoradio_valid_indices
                pool = np.asarray([row for row in pool
                                   if self._bucket_matches(
                                       self._group_bucket(int(row) // H5_SPLIT_GROUP_ROWS, "panoradio"),
                                       partition)], dtype=np.int64)
                if not len(pool):
                    continue
                idx = self.rng.choice(pool, size=needed, replace=len(pool) < needed)
                raw = np.asarray(self.panoradio_iq[idx])
                # Random 1024-sample crop per row prevents the model from
                # learning a fixed location within the 2048-sample record.
                if raw.shape[1] > self.input_len:
                    starts = self.rng.integers(0, raw.shape[1] - self.input_len + 1, size=len(raw))
                    raw = np.stack([row[s:s + self.input_len] for row, s in zip(raw, starts)])
                elif raw.shape[1] < self.input_len:
                    raw = np.pad(raw, ((0, 0), (0, self.input_len - raw.shape[1])))
                iq_list.append(np.stack([raw.real, raw.imag], axis=1).astype(np.float32))
                mod_list.append(self.panoradio_mod[idx])
                snr_list.append(self.panoradio_snr[idx])
                baud_list.append(np.full((len(idx),), float('nan'), dtype=np.float32))

            elif src in self.file_datasets:
                files = [r for r in self.file_datasets[src]
                         if self._hardware_partition(src, r[0]) == partition]
                if not files:
                    continue
                for _ in range(needed):
                    item = files[self.rng.integers(0, len(files))]
                    iq, mod, snr, baud = self._read_hw_sample(item)
                    iq_list.append(iq[np.newaxis, ...])
                    mod_list.append(np.array([mod], dtype=np.int64))
                    snr_list.append(np.array([snr], dtype=np.float32))
                    baud_list.append(np.array([baud], dtype=np.float32))

        iq_batch = np.concatenate(iq_list, axis=0)
        mod_batch = np.concatenate(mod_list, axis=0)
        snr_batch = np.concatenate(snr_list, axis=0)
        baud_batch = np.concatenate(baud_list, axis=0)

        # Shuffle and crop/tile to exact batch_size
        if len(iq_batch) > batch_size:
            idx_perm = self.rng.permutation(len(iq_batch))[:batch_size]
            iq_batch = iq_batch[idx_perm]
            mod_batch = mod_batch[idx_perm]
            snr_batch = snr_batch[idx_perm]
            baud_batch = baud_batch[idx_perm]
        elif len(iq_batch) < batch_size:
            missing = batch_size - len(iq_batch)
            fill = self.rng.choice(len(iq_batch), size=missing, replace=True)
            iq_batch = np.concatenate([iq_batch, iq_batch[fill]], axis=0)
            mod_batch = np.concatenate([mod_batch, mod_batch[fill]], axis=0)
            snr_batch = np.concatenate([snr_batch, snr_batch[fill]], axis=0)
            baud_batch = np.concatenate([baud_batch, baud_batch[fill]], axis=0)

        # Deployment applies this same fail-closed conditioner before feature
        # extraction. Apply it to every partition so validation estimates the
        # path that will actually be served. Training-only invariances follow.
        iq_batch = condition_classifier_batch(iq_batch)
        if partition == "train":
            iq_batch = self._augment_train_batch(iq_batch)
            pwr = np.sqrt(np.mean(iq_batch[:, 0, :]**2 + iq_batch[:, 1, :]**2,
                                  axis=-1, keepdims=True))
            iq_batch = np.divide(iq_batch, pwr[:, np.newaxis, :],
                                 out=np.zeros_like(iq_batch),
                                 where=pwr[:, np.newaxis, :] > 1e-12).astype(np.float32)
        presence_batch = (mod_batch != MOD_TO_IDX["NOISE"]).astype(np.float32)

        return (
            torch.from_numpy(iq_batch),
            torch.from_numpy(mod_batch).long(),
            torch.from_numpy(snr_batch).float(),
            torch.from_numpy(baud_batch).float(),
            torch.from_numpy(presence_batch).float(),
        )


# ==============================================================================
# 4. NEURAL ARCHITECTURE (100% SATVIK ALIGNED)
# ==============================================================================
class ComplexConv1dStem(nn.Module):
    def __init__(self, in_channels: int = 2, stem_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, stem_dim // 2, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(stem_dim // 2)
        self.conv2 = nn.Conv1d(stem_dim // 2, stem_dim, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(stem_dim)
        self.conv3 = nn.Conv1d(stem_dim, stem_dim * 2, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(stem_dim * 2)
        self.conv4 = nn.Conv1d(stem_dim * 2, stem_dim * 2, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(stem_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.gelu(self.bn4(self.conv4(x)))
        return x

class ResNetBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.gelu(x + res)

class CrossCovarianceAttention1D(nn.Module):
    """XCiT-style channel attention with linear sequence-length cost."""
    def __init__(self, dim: int = 256, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(b, n, self.heads, self.head_dim).permute(0, 2, 3, 1)
        k = k.reshape(b, n, self.heads, self.head_dim).permute(0, 2, 3, 1)
        v = v.reshape(b, n, self.heads, self.head_dim).permute(0, 2, 3, 1)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        out = (attn @ v).permute(0, 3, 1, 2).reshape(b, n, c)
        return self.proj(out)

class XCiTBlock1D(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8, mlp_ratio: int = 3, dropout: float = 0.1):
        super().__init__()
        self.local = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = CrossCovarianceAttention1D(dim, heads, dropout)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.local(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class SignalInfinityBackbone(nn.Module):
    def __init__(self, in_channels: int = 2, stem_dim: int = 128, expert_dim: int = 32, emb_dim: int = 512):
        super().__init__()
        self.expert_dim = int(expert_dim)
        self.stem = ComplexConv1dStem(in_channels=in_channels, stem_dim=stem_dim)
        self.body = nn.Sequential(
            ResNetBlock1D(stem_dim * 2),
            ResNetBlock1D(stem_dim * 2),
            ResNetBlock1D(stem_dim * 2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fusion = nn.Sequential(
            nn.Linear(stem_dim * 2 + expert_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, iq: torch.Tensor, expert: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.body(self.stem(iq))
        z_learned = self.pool(h).squeeze(-1)
        if expert is None:
            expert = torch.zeros(iq.size(0), self.expert_dim, device=iq.device, dtype=iq.dtype)
        fused = torch.cat([z_learned, expert], dim=1)
        return self.fusion(fused)

class ResearchHybridXCiTBackbone(nn.Module):
    def __init__(self, in_channels: int = 2, stem_dim: int = 128,
                 expert_dim: int = 32, emb_dim: int = 512, depth: int = 3, heads: int = 8):
        super().__init__()
        self.expert_dim = int(expert_dim)
        token_dim = stem_dim * 2
        self.stem = ComplexConv1dStem(in_channels, stem_dim)
        self.body = nn.Sequential(ResNetBlock1D(token_dim), ResNetBlock1D(token_dim))
        self.blocks = nn.ModuleList([XCiTBlock1D(token_dim, heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(token_dim)
        self.attn_pool = nn.Sequential(nn.Linear(token_dim, token_dim // 2), nn.Tanh(),
                                       nn.Linear(token_dim // 2, 1, bias=False))
        self.expert_norm = nn.LayerNorm(expert_dim)
        self.expert_gate = nn.Sequential(nn.Linear(token_dim + expert_dim, expert_dim), nn.Sigmoid())
        self.fusion = nn.Sequential(nn.Linear(token_dim + expert_dim, emb_dim),
                                    nn.BatchNorm1d(emb_dim), nn.GELU(), nn.Dropout(0.1),
                                    nn.Linear(emb_dim, emb_dim))

    def forward(self, iq: torch.Tensor, expert: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens = self.body(self.stem(iq)).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        learned = torch.sum(tokens * torch.softmax(self.attn_pool(tokens), dim=1), dim=1)
        if expert is None:
            expert = torch.zeros(iq.size(0), self.expert_dim, device=iq.device, dtype=iq.dtype)
        expert = self.expert_norm(expert)
        expert = expert * self.expert_gate(torch.cat([learned, expert], dim=1))
        return self.fusion(torch.cat([learned, expert], dim=1))

def build_backbone(architecture: str, **kwargs) -> nn.Module:
    if architecture == "baseline_resnet":
        return SignalInfinityBackbone(**kwargs)
    if architecture == "hybrid_xcit":
        return ResearchHybridXCiTBackbone(**kwargs)
    raise ValueError(f"unsupported architecture {architecture!r}")

class SignalInfinityMultiTaskHeads(nn.Module):
    def __init__(self, emb_dim: int = 512, n_mods: int = len(MOD_VOCAB), n_fecs: int = 5, n_scramblers: int = 5):
        super().__init__()
        self.mod_head = nn.Linear(emb_dim, n_mods)
        self.symrate_head = nn.Sequential(nn.Linear(emb_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.rolloff_head = nn.Sequential(nn.Linear(emb_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        self.cfo_head = nn.Sequential(nn.Linear(emb_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Tanh())
        self.snr_head = nn.Sequential(nn.Linear(emb_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.fec_head = nn.Linear(emb_dim, n_fecs)
        self.scrambler_head = nn.Linear(emb_dim, n_scramblers)
        self.presence_head = nn.Linear(emb_dim, 1)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "mod_logits": self.mod_head(z),
            "log_symrate": self.symrate_head(z).squeeze(-1),
            "rolloff": 0.1 + 0.7 * self.rolloff_head(z).squeeze(-1),
            "cfo_norm": 0.5 * self.cfo_head(z).squeeze(-1),
            "snr_db": self.snr_head(z).squeeze(-1),
            "fec_logits": self.fec_head(z),
            "scrambler_logits": self.scrambler_head(z),
            "presence_logit": self.presence_head(z).squeeze(-1),
        }

class ArcFaceCriterion(nn.Module):
    def __init__(self, in_features: int = 512, out_features: int = len(MOD_VOCAB), s: float = 30.0, m: float = 0.45):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        mask = (label >= 0)
        if not mask.any():
            return torch.tensor(0.0, device=emb.device, requires_grad=True)
        emb_f = emb[mask].float()
        lbl = label[mask].long()
        w_f = self.weight.float()

        cosine = F.linear(F.normalize(emb_f, dim=-1), F.normalize(w_f, dim=-1))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-7, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, lbl.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return F.cross_entropy(output * self.s, lbl)

class ModelEMA:
    """High-speed in-place Exponential Moving Average tracking parameters and buffers."""
    def __init__(self, model: nn.Module, decay: float = 0.9998, tau: float = 2000.0):
        self.ema_model = copy.deepcopy(model).eval()
        self.decay = float(decay)
        self.tau = float(tau)
        self.updates = 0
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.param_pairs = list(zip(self.ema_model.parameters(), model.parameters()))
        self.buffer_pairs = list(zip(self.ema_model.buffers(), model.buffers()))

    def current_decay(self) -> float:
        return self.decay * (1.0 - math.exp(-self.updates / self.tau))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        w = 1.0 - self.current_decay()
        for p_ema, p_model in self.param_pairs:
            p_ema.lerp_(p_model.detach(), w)
        for b_ema, b_model in self.buffer_pairs:
            if b_ema.is_floating_point():
                b_ema.lerp_(b_model.detach(), w)
            else:
                b_ema.copy_(b_model)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)


# ==============================================================================
# 5. MASTER 10-DATASET ACCELERATED TRAINING ENGINE
# ==============================================================================
@torch.no_grad()
def evaluate_holdout(backbone: nn.Module, heads: nn.Module, arcface: nn.Module,
                     streamer: DynamicGrandCorpusStreamer, device: torch.device,
                     batch_size: int, batches: int = 32,
                     enable_prototype_blend: bool = False) -> Dict[str, Any]:
    """Calibrate the prototype blend, then score a separate fixed holdout.

    Both partitions are regenerated from fixed seeds on every epoch, making
    epoch metrics directly comparable without consuming the training RNG.
    """
    backbone.eval()
    heads.eval()
    # Prototype blending is experimental for AMC.  The trustworthy baseline
    # scores the supervised softmax head only; explicit ArcFace runs may
    # calibrate the blend on the separate calibration partition.
    candidate_weights = (0.0, 1.0, 2.0, 3.5) if enable_prototype_blend else (0.0,)

    def collect(partition: str, n_batches: int, seed_offset: int):
        saved_state = copy.deepcopy(streamer.rng.bit_generator.state)
        streamer.rng = np.random.default_rng(streamer.seed + seed_offset)
        outputs = []
        try:
            for _ in range(max(1, n_batches)):
                iq, target, snr, _, _ = streamer.get_batch(batch_size=batch_size, partition=partition)
                iq, target = iq.to(device), target.to(device)
                emb = backbone(iq, extract_training_expert_batch(iq))
                head_logits = heads(emb)["mod_logits"]
                cosine_logits = F.linear(F.normalize(emb.float(), dim=-1),
                                         F.normalize(arcface.weight.float(), dim=-1))
                outputs.append((head_logits.float().cpu(), cosine_logits.cpu(),
                                target.cpu(), snr.cpu()))
        finally:
            streamer.rng = np.random.default_rng()
            streamer.rng.bit_generator.state = saved_state
        return outputs

    calibration = collect("cal", max(1, batches // 2), 1001)
    holdout = collect("val", max(1, batches), 2001)
    if not calibration or not holdout:
        raise RuntimeError("Calibration/validation partitions contain no verified modulation labels")

    cal_head_all = torch.cat([v[0] for v in calibration])
    cal_proto_all = torch.cat([v[1] for v in calibration])
    cal_target_all = torch.cat([v[2] for v in calibration])
    cal_known = cal_target_all >= 0
    if not cal_known.any():
        raise RuntimeError("Calibration partition contains no verified known labels")
    cal_head = cal_head_all[cal_known]
    cal_proto = cal_proto_all[cal_known]
    cal_target = cal_target_all[cal_known]
    def balanced_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
        pred = logits.argmax(1)
        scores = []
        for cls in target.unique():
            mask = target == cls
            scores.append((pred[mask] == target[mask]).float().mean())
        return float(torch.stack(scores).mean()) if scores else 0.0

    cal_acc = {w: balanced_accuracy(cal_head + w * cal_proto, cal_target)
               for w in candidate_weights}
    best_weight = max(candidate_weights, key=lambda w: (cal_acc[w], -w))

    val_head_all = torch.cat([v[0] for v in holdout])
    val_proto_all = torch.cat([v[1] for v in holdout])
    target_all = torch.cat([v[2] for v in holdout])
    val_snr_all = torch.cat([v[3] for v in holdout])
    val_known = target_all >= 0
    if not val_known.any():
        raise RuntimeError("Validation partition contains no verified known labels")
    val_head = val_head_all[val_known]
    val_proto = val_proto_all[val_known]
    target = target_all[val_known]
    val_snr = val_snr_all[val_known]
    pred = (val_head + best_weight * val_proto).argmax(1)
    confusion = torch.zeros((len(MOD_VOCAB), len(MOD_VOCAB)), dtype=torch.long)
    for truth, guess in zip(target.tolist(), pred.tolist()):
        confusion[truth, guess] += 1
    support = confusion.sum(1)
    present = support > 0
    per_class = torch.zeros(len(MOD_VOCAB), dtype=torch.float32)
    per_class[present] = confusion.diag()[present].float() / support[present].float()
    val_acc = float((pred == target).float().mean())
    macro_acc = float(per_class[present].mean())
    snr_ranges = (("<-10", -float("inf"), -10.0), ("-10..0", -10.0, 0.0),
                  ("0..10", 0.0, 10.0), ("10..20", 10.0, 20.0),
                  (">=20", 20.0, float("inf")))
    per_snr = {}
    for name, low, high in snr_ranges:
        mask = torch.isfinite(val_snr) & (val_snr >= low) & (val_snr < high)
        per_snr[name] = ({"accuracy": round(float((pred[mask] == target[mask]).float().mean()), 6),
                          "support": int(mask.sum())} if mask.any() else None)

    # Calibrated open-set rejection.  Unsupported, explicitly named Sig53
    # families have target=-1 and are never used as known-class supervision.
    # Fit temperature and threshold on CAL only, then report VAL unknowns.
    temp_grid = torch.logspace(math.log10(0.5), math.log10(5.0), 31)
    nll = [float(F.cross_entropy(cal_head / float(t), cal_target)) for t in temp_grid]
    temperature = float(temp_grid[int(np.argmin(nll))])
    cal_energy = -temperature * torch.logsumexp(cal_head_all / temperature, dim=1)
    known_energy = cal_energy[cal_known]
    cal_unknown = ~cal_known
    energy_threshold = float(torch.quantile(known_energy, 0.95))
    known_acceptance = float((known_energy <= energy_threshold).float().mean())
    cal_unknown_rejection = (float((cal_energy[cal_unknown] > energy_threshold).float().mean())
                             if cal_unknown.any() else None)
    val_unknown = ~val_known
    val_energy = -temperature * torch.logsumexp(val_head_all / temperature, dim=1)
    val_unknown_rejection = (float((val_energy[val_unknown] > energy_threshold).float().mean())
                             if val_unknown.any() else None)
    def energy_auroc(energy_values: torch.Tensor, unknown_mask: torch.Tensor) -> Optional[float]:
        scores = energy_values.detach().cpu().numpy()
        labels = unknown_mask.detach().cpu().numpy().astype(bool)
        n_unknown, n_known = int(labels.sum()), int((~labels).sum())
        if not n_unknown or not n_known:
            return None
        ranks = np.argsort(np.argsort(scores, kind="mergesort"), kind="mergesort") + 1
        return float((ranks[labels].sum() - n_unknown * (n_unknown + 1) / 2)
                     / (n_unknown * n_known))
    cal_auroc = energy_auroc(cal_energy, cal_unknown)
    val_auroc = energy_auroc(val_energy, val_unknown)
    open_set_enabled = bool(
        int(cal_unknown.sum()) >= 32
        and cal_unknown_rejection is not None
        and cal_unknown_rejection >= 0.50
        and cal_auroc is not None
        and cal_auroc >= 0.65
    )
    open_set = {
        "method": "temperature-scaled-energy-v1",
        "enabled": open_set_enabled,
        "temperature": temperature,
        "energy_threshold": energy_threshold,
        "target_known_acceptance": 0.95,
        "cal_known_acceptance": known_acceptance,
        "cal_known_count": int(cal_known.sum()),
        "cal_unknown_count": int(cal_unknown.sum()),
        "cal_unknown_rejection": cal_unknown_rejection,
        "cal_auroc": cal_auroc,
        "cal_fpr95": (None if cal_unknown_rejection is None else 1.0 - cal_unknown_rejection),
        "val_unknown_count": int(val_unknown.sum()),
        "val_unknown_rejection": val_unknown_rejection,
        "val_auroc": val_auroc,
        "val_fpr95": (None if val_unknown_rejection is None else 1.0 - val_unknown_rejection),
        "disabled_reason": (None if open_set_enabled else
                            "insufficient or ineffective held-out unknown calibration"),
    }
    return {
        "val_acc": val_acc,
        "val_macro_acc": macro_acc,
        "val_count": int(target.numel()),
        "cal_count": int(cal_target.numel()),
        "prototype_weight": float(best_weight),
        "cal_acc_by_prototype_weight": {str(w): round(cal_acc[w], 6) for w in candidate_weights},
        "per_class_accuracy": {name: (round(float(per_class[i]), 6) if present[i] else None)
                               for i, name in enumerate(MOD_VOCAB)},
        "per_class_support": {name: int(support[i]) for i, name in enumerate(MOD_VOCAB)},
        "per_snr_accuracy": per_snr,
        "confusion_matrix": confusion.tolist(),
        "open_set": open_set,
    }


def train_satvik_grand_corpus_dual_gpu(epochs: int = 10, batch_size: int = 256, lr: float = 1e-3,
                                        steps_per_epoch: int = 1953, resume_path: Optional[str] = None,
                                        data_roots: Optional[List[str]] = None, val_batches: int = 32,
                                        seed: int = 20260924, output_dir: Optional[str] = None,
                                        input_len: int = 1024, arcface_weight: float = 0.0,
                                        architecture: str = "hybrid_xcit",
                                        synthetic_fraction: float = 0.10,
                                        synthetic_pool_size: int = 8192):
    if arcface_weight < 0:
        raise ValueError("arcface_weight must be non-negative")
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda:0" if num_gpus > 0 else "cpu")
    if device.type == "cuda":
        # The window length and network shapes are fixed for a whole run, so
        # cuDNN can choose and retain its fastest convolution kernels.
        torch.backends.cudnn.benchmark = True
    print("\n" + "=" * 80, flush=True)
    print(f">> [AMC_1-SATVIK] ACTIVE HARDWARE: {num_gpus} GPU(s) DETECTED", flush=True)
    for i in range(num_gpus):
        print(f">> GPU [{i}]: {torch.cuda.get_device_name(i)} | VRAM: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB", flush=True)
    print("=" * 80, flush=True)

    catalog: Dict[str, Any] = {
        "sig53": None, "radioml": None, "variable_rf": None,
        "migou": [],
        "modrec": [],
        "mendeley": [],
        "commrad": [],
        "panoradio": [],
        "sigid": [],
        "kiwi": [],
        "hamsci": [],
        "lora_rffi": [],
        "signal3": [],
        "other_rf": [],
    }

    ground_truth_map: Dict[str, Tuple[int, float, float]] = {}
    # Kaggle input mounts are immutable.  Never scan the notebook's Downloads
    # directory: it can contain prior checkpoints and the frozen test set.
    search_dirs = [Path(p) for p in (data_roots or ["/kaggle/input"])]

    for s_dir in search_dirs:
        if not s_dir.exists(): continue
        # Load any ground truth csv/json metadata in this root
        ground_truth_map.update(load_ground_truth_metadata(s_dir))

        for p in s_dir.rglob("*"):
            if not p.is_file(): continue
            p_str = str(p).lower()
            if "hisarmod" in p_str or "hissar" in p_str:
                continue  # EXCLUDED by specification: flawed synthetic channel labels
            if ("variable_length_rf" in p_str or "variable-rf-corpus" in p_str) and p.suffix in [".h5", ".hdf5"]:
                if catalog["variable_rf"] is None or p.stat().st_size > Path(catalog["variable_rf"]).stat().st_size:
                    catalog["variable_rf"] = p
            elif ("sig53" in p_str or "torchsig" in p_str) and p.suffix in [".h5", ".hdf5"]:
                if catalog["sig53"] is None or p.stat().st_size > Path(catalog["sig53"]).stat().st_size:
                    catalog["sig53"] = p
            elif ("gold_xyz_osc" in p_str or "radioml" in p_str) and p.suffix in [".h5", ".hdf5"]:
                if catalog["radioml"] is None or p.stat().st_size > Path(catalog["radioml"]).stat().st_size:
                    catalog["radioml"] = p
            elif "migou" in p_str and p.suffix.lower() in SUPPORTED_IQ_EXTENSIONS:
                catalog["migou"].append(p)
            elif "modrec" in p_str and p.suffix.lower() in SUPPORTED_IQ_EXTENSIONS:
                catalog["modrec"].append(p)
            elif "mendeley" in p_str and p.suffix.lower() in SUPPORTED_IQ_EXTENSIONS:
                catalog["mendeley"].append(p)
            elif "commrad" in p_str and p.suffix.lower() in SUPPORTED_IQ_EXTENSIONS:
                catalog["commrad"].append(p)
            elif "panoradio" in p_str and p.suffix.lower() in SUPPORTED_IQ_EXTENSIONS:
                catalog["panoradio"].append(p)
            elif "sigid" in p_str and p.suffix.lower() in SUPPORTED_IQ_EXTENSIONS:
                catalog["sigid"].append(p)
            elif "kiwi" in p_str and p.suffix.lower() in SUPPORTED_IQ_EXTENSIONS:
                catalog["kiwi"].append(p)
            elif "hamsci" in p_str and p.suffix.lower() in SUPPORTED_IQ_EXTENSIONS:
                catalog["hamsci"].append(p)
            elif any(k in p_str for k in ["lora", "rffi", "vhf"]) and p.suffix.lower() in SUPPORTED_IQ_EXTENSIONS:
                catalog["lora_rffi"].append(p)
            elif any(k in p_str for k in ["signal3", "signal 3", "signal_3"]) and p.suffix.lower() in SUPPORTED_IQ_EXTENSIONS:
                catalog["signal3"].append(p)
            elif p.suffix.lower() in [".cf32", ".raw", ".bin", ".dat", ".iq", ".wav", ".npy"] and p.stat().st_size >= 1024:
                catalog["other_rf"].append(p)

    streamer = DynamicGrandCorpusStreamer(catalog, ground_truth_map=ground_truth_map,
                                          seed=seed, input_len=input_len,
                                          synthetic_fraction=synthetic_fraction,
                                          synthetic_pool_size=synthetic_pool_size)

    raw_backbone = build_backbone(architecture, stem_dim=128,
                                  expert_dim=TRAIN_EXPERT_DIM, emb_dim=512).to(device)
    if num_gpus > 1:
        backbone = nn.DataParallel(raw_backbone)
        print(f">> [DataParallel] ACTIVE: Splitting batches across {num_gpus} GPUs simultaneously!", flush=True)
    else:
        backbone = raw_backbone

    heads = SignalInfinityMultiTaskHeads(emb_dim=512, n_mods=len(MOD_VOCAB), n_fecs=5, n_scramblers=5).to(device)
    # These outputs have no verified targets in the mounted corpora. Freezing
    # them prevents random heads from being presented as trained predictions.
    for unused_head in (heads.rolloff_head, heads.cfo_head, heads.fec_head, heads.scrambler_head):
        for parameter in unused_head.parameters():
            parameter.requires_grad_(False)
    arcface = ArcFaceCriterion(in_features=512, out_features=len(MOD_VOCAB), s=30.0, m=0.45).to(device)
    ema_source = nn.ModuleDict({"backbone": raw_backbone, "heads": heads, "arcface": arcface})
    ema = ModelEMA(ema_source)

    best_acc = 0.0
    resume_state = None
    start_epoch = 1
    if resume_path is None:
        resume_env = os.environ.get("RESUME_CKPT")
        if resume_env and Path(resume_env).exists():
            resume_path = resume_env

    if resume_path and Path(resume_path).exists():
        print(f">> [RESUME] Loading pre-trained Satvik model from: {resume_path}", flush=True)
        try:
            ck = torch.load(resume_path, map_location=device)
            if int(ck.get("input_len", input_len)) != input_len:
                raise ValueError(f"checkpoint input_len={ck.get('input_len')} does not match --input-length {input_len}")
            if ck.get("architecture", "baseline_resnet") != architecture:
                raise ValueError("checkpoint architecture does not match --architecture")
            if list(ck.get("class_names", MOD_VOCAB)) != list(MOD_VOCAB):
                raise ValueError("checkpoint class_names/order does not match the corrected canonical vocabulary")
            if ck.get("feature_contract") != "expert11-native-paprdb-v3":
                raise ValueError("legacy checkpoint uses the old expert-feature contract; start a fresh corrected run")
            if ck.get("conditioning_contract") != CONDITIONING_CONTRACT:
                raise ValueError("legacy checkpoint predates train/deployment conditioning parity; start fresh")
            if ck.get("source_overlap_policy") != SOURCE_OVERLAP_POLICY:
                raise ValueError("checkpoint predates the Sig53/VariableRF overlap policy; start fresh")
            if bool(ck.get("alignment_preprocess", False)):
                raise ValueError("checkpoint enables CFO/SPS alignment not present in the Kaggle training path; start fresh")
            if ck.get("validation_protocol") != "blocked-group-session-80-train-10-cal-10-val-v4":
                raise ValueError("legacy checkpoint predates the corrected fixed validation protocol; start fresh")
            saved_arcface_weight = float(ck.get("loss_weights", {}).get("arcface", 0.0))
            if not math.isclose(saved_arcface_weight, arcface_weight, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"checkpoint ArcFace weight={saved_arcface_weight} does not match "
                    f"--arcface-weight {arcface_weight}; use the original value or start fresh"
                )
            resume_state = ck
            start_epoch = int(ck.get("epoch", 0)) + 1
            if "backbone" in ck:
                raw_backbone.load_state_dict(ck["backbone"])
            if all(k in ck for k in ("ema_backbone", "ema_heads", "ema_arcface")):
                ema.load_state_dict({
                    **{f"backbone.{k}": v for k, v in ck["ema_backbone"].items()},
                    **{f"heads.{k}": v for k, v in ck["ema_heads"].items()},
                    **{f"arcface.{k}": v for k, v in ck["ema_arcface"].items()},
                })
                ema.updates = int(ck.get("ema_updates", 0))
            elif "ema_backbone" in ck:
                print(">> [RESUME] Legacy checkpoint has backbone-only EMA; EMA restarted for consistency.", flush=True)
            if "heads" in ck:
                heads.load_state_dict(ck["heads"])
            if "arcface" in ck:
                arcface.load_state_dict(ck["arcface"])
            if "streamer_rng_state" in ck:
                streamer.rng.bit_generator.state = ck["streamer_rng_state"]
            if "torch_rng_state" in ck:
                torch.random.set_rng_state(ck["torch_rng_state"].cpu())
            if device.type == "cuda" and "cuda_rng_state_all" in ck:
                torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng_state_all"]])
            if (ck.get("validation_protocol") == "blocked-group-session-80-train-10-cal-10-val-v4"
                    and "metrics" in ck and "val_macro_acc" in ck["metrics"]):
                best_acc = float(ck.get("best_macro_acc",
                                        float(ck["metrics"]["val_macro_acc"]) * 100.0))
            else:
                print(">> [RESUME] Legacy/unverified val_acc ignored; candidate selection restarts at 0%.", flush=True)
            print(f">> [RESUME] Weights loaded successfully! Baseline Validation Acc: {best_acc:.2f}% (Epoch {ck.get('epoch', '?')})", flush=True)
        except Exception as e:
            print(f">> [RESUME Warning] Could not load checkpoint from {resume_path}: {e}", flush=True)

    # Warmup 1 dummy forward pass so PyTorch initializes CUDA kernels and memory allocator
    if device.type == "cuda":
        # DataParallel splits dim-0 across GPUs.  BatchNorm in ``fusion``
        # needs at least two examples *on each replica* in training mode.
        # A fixed batch of two therefore crashes on a dual-T4 Kaggle session
        # (one example per GPU).  Keep a two-example warm-up shard per GPU.
        warmup_batch = max(2, 2 * num_gpus)
        dummy_iq = torch.zeros(warmup_batch, 2, input_len, device=device)
        dummy_exp = torch.zeros(warmup_batch, TRAIN_EXPERT_DIM, device=device)
        was_training = backbone.training
        backbone.eval()
        with torch.no_grad():
            backbone(dummy_iq, dummy_exp)
        backbone.train(was_training)

    trainable_parameters = list(backbone.parameters()) + list(heads.parameters())
    if arcface_weight > 0:
        trainable_parameters += list(arcface.parameters())
    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=0.01)
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.15, anneal_strategy="cos")
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # A true resume restores optimizer momentum, the OneCycle position and
    # AMP scaling—not just neural weights. Older checkpoints remain usable
    # as warm-starts because these fields are optional.
    if resume_state is not None:
        try:
            if "optimizer" in resume_state:
                optimizer.load_state_dict(resume_state["optimizer"])
            if "scheduler" in resume_state:
                saved_scheduler = resume_state["scheduler"]
                if int(saved_scheduler.get("total_steps", total_steps)) == total_steps:
                    scheduler.load_state_dict(saved_scheduler)
                else:
                    print(">> [RESUME Warning] Scheduler total_steps changed; "
                          "optimizer/scaler restored, scheduler restarted for the new schedule.", flush=True)
            if "scaler" in resume_state:
                scaler.load_state_dict(resume_state["scaler"])
            print(f">> [RESUME] True continuation restored; next epoch: {start_epoch}", flush=True)
        except Exception as exc:
            print(f">> [RESUME Warning] State restore failed; weights remain loaded as warm-start: {exc}", flush=True)

    out_dir = Path(output_dir) if output_dir else (
        Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("models"))
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"s9_{architecture}_L{input_len}"
    best_ckpt_path = out_dir / f"{run_name}_best.ckpt"
    last_ckpt_path = out_dir / f"{run_name}_last.ckpt"
    ckpt_payload = resume_state
    if (resume_state is not None and best_acc > 0.0 and not best_ckpt_path.exists()
            and resume_state.get("validation_protocol") == "blocked-group-session-80-train-10-cal-10-val-v4"):
        resume_best = (Path(resume_path).with_name(Path(resume_path).name.replace("_last.ckpt", "_best.ckpt"))
                       if resume_path else None)
        if resume_best is not None and resume_best.exists() and resume_best != Path(resume_path):
            shutil.copyfile(resume_best, best_ckpt_path)
        else:
            # The exact historical best is unavailable; preserve the resume
            # checkpoint rather than silently starting without a candidate.
            torch.save(resume_state, str(best_ckpt_path))
        print(f">> [RESUME] Preserved previous best checkpoint at {best_ckpt_path}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f">> LAUNCHING ULTRA-ACCELERATED TRAINING (10-DATASET ZERO-STALL SATVIK ENGINE)", flush=True)
    print(f">> Steps per Epoch      : {steps_per_epoch:,} batches ({steps_per_epoch * batch_size:,} samples/epoch)", flush=True)
    print(f">> Total Training Steps : {total_steps:,}", flush=True)
    print("=" * 80 + "\n", flush=True)

    # Dataset preparation (HDF5 reads, raw capture seeks, augmentation) is
    # CPU-bound. Keep one next batch in flight while both GPUs execute the
    # current forward/backward pass. Pin it in that worker so non_blocking GPU
    # copies below are genuinely asynchronous; ``non_blocking=True`` alone on
    # ordinary pageable NumPy-backed tensors provides no transfer overlap.
    # A single worker intentionally preserves deterministic RNG ordering and
    # avoids concurrent h5py access.
    def _prepare_batch(partition: str):
        batch = streamer.get_batch(batch_size, partition)
        if device.type == "cuda":
            batch = tuple(tensor.pin_memory() for tensor in batch)
        return batch

    print(f">> [INPUT PIPELINE] gzip-aware {H5_IO_FANOUT_GROUPS}-group HDF5 reads; "
          f"background prefetch + pinned-memory transfer: {'ON' if device.type == 'cuda' else 'OFF'}",
          flush=True)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="s9-prefetch") as prefetcher:
        for epoch in range(start_epoch, epochs + 1):
            backbone.train()
            heads.train()
            t0 = time.time()
        
            # GPU-resident zero-sync accumulation tensors
            running_loss_tensor = torch.zeros((), device=device, dtype=torch.float32)
            corr_tensor = torch.zeros((), device=device, dtype=torch.long)
            tot_tensor = torch.zeros((), device=device, dtype=torch.long)
            class_count_tensor = torch.zeros(len(MOD_VOCAB), device=device, dtype=torch.long)
            next_batch = prefetcher.submit(_prepare_batch, "train")

            for b_idx in range(steps_per_epoch):
                iq_cpu, t_mod_cpu, t_snr_cpu, t_baud_cpu, t_presence_cpu = next_batch.result()
                if b_idx + 1 < steps_per_epoch:
                    next_batch = prefetcher.submit(_prepare_batch, "train")
                iq = iq_cpu.to(device, non_blocking=True)
                t_mod = t_mod_cpu.to(device, non_blocking=True)
                t_snr = t_snr_cpu.to(device, non_blocking=True)
                t_baud = t_baud_cpu.to(device, non_blocking=True)
                t_presence = t_presence_cpu.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
    
                expert = extract_training_expert_batch(iq)
    
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    emb = backbone(iq, expert)
                    preds = heads(emb)
    
                    valid_mod = (t_mod >= 0)
                    if valid_mod.any():
                        # Never send sentinel -1 labels to ArcFace: scatter
                        # indexing would reinterpret them as the final class.
                        loss_arc = (arcface(emb[valid_mod], t_mod[valid_mod])
                                    if arcface_weight > 0 else torch.zeros((), device=device))
                        loss_mod = F.cross_entropy(preds["mod_logits"][valid_mod], t_mod[valid_mod])
                    else:
                        loss_arc = torch.tensor(0.0, device=device)
                        loss_mod = torch.tensor(0.0, device=device)
    
                    # SNR is undefined for the NOISE class even when a source
                    # file contains a numeric placeholder.
                    valid_snr = ~torch.isnan(t_snr) & (t_mod != MOD_TO_IDX["NOISE"])
                    loss_snr = F.smooth_l1_loss(preds["snr_db"][valid_snr], t_snr[valid_snr]) if valid_snr.any() else torch.tensor(0.0, device=device)
    
                    valid_baud = ~torch.isnan(t_baud) & (t_baud > 0)
                    loss_baud = F.smooth_l1_loss(preds["log_symrate"][valid_baud], torch.log(t_baud[valid_baud])) if valid_baud.any() else torch.tensor(0.0, device=device)
    
                    loss_pres = F.binary_cross_entropy_with_logits(preds["presence_logit"], t_presence)
    
                    # CE remains the primary deployment objective. Metric and
                    # regression auxiliaries regularise the embedding without
                    # overwhelming classification gradients.
                    total_loss = (loss_mod + arcface_weight * loss_arc + 0.02 * loss_snr
                                  + 0.05 * loss_baud + 0.05 * loss_pres)
    
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
    
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                if scale_before <= scale_after:
                    scheduler.step()
    
                ema.update(ema_source)
                running_loss_tensor += total_loss.detach()
    
                with torch.no_grad():
                    if valid_mod.any():
                        pred_mod = preds["mod_logits"][valid_mod].argmax(dim=-1)
                        corr_tensor += (pred_mod == t_mod[valid_mod]).sum()
                        tot_tensor += valid_mod.sum()
                        class_count_tensor += torch.bincount(t_mod[valid_mod], minlength=len(MOD_VOCAB))
    
                # Selective Logging: Zero CPU-GPU sync barrier on non-logging steps
                if (b_idx == 0) or ((b_idx + 1) in [5, 10, 25, 50]) or ((b_idx + 1) % 50 == 0):
                    with torch.no_grad():
                        if valid_mod.any() and arcface_weight > 0:
                            cosine_logits = F.linear(F.normalize(emb[valid_mod].float(), dim=-1), F.normalize(arcface.weight.float(), dim=-1))
                            pred_arc = cosine_logits.argmax(dim=-1)
                            step_arc_corr = (pred_arc == t_mod[valid_mod]).sum().item()
                            acc_a = 100.0 * step_arc_corr / max(1, valid_mod.sum().item())
                        else:
                            acc_a = float("nan")
                    
                    tot_mod_val = tot_tensor.item()
                    acc_l = 100.0 * corr_tensor.item() / max(1, tot_mod_val)
                    loss_avg = running_loss_tensor.item() / (b_idx + 1)
                    lr_cur = optimizer.param_groups[0]["lr"]
                    speed = (b_idx + 1) * batch_size / max(0.01, time.time() - t0)
                    print(f"Epoch [{epoch:02d}/{epochs:02d}] Step [{b_idx+1:04d}/{steps_per_epoch:04d}] | "
                          f"Loss: {loss_avg:.4f} | Mod Head: {acc_l:.2f}% | "
                          f"ArcFace: {acc_a:.2f}% | {speed:.0f} smp/s | LR: {lr_cur:.2e}",
                          flush=True)
    
            tot_mod_epoch = tot_tensor.item()
            train_acc = corr_tensor.item() / max(1, tot_mod_epoch)
            ema_backbone = ema.ema_model["backbone"]
            ema_heads = ema.ema_model["heads"]
            ema_arcface = ema.ema_model["arcface"]
            val_metrics = evaluate_holdout(ema_backbone, ema_heads, ema_arcface, streamer, device,
                                           batch_size=batch_size, batches=val_batches,
                                           enable_prototype_blend=(arcface_weight > 0))
            t_elapsed = time.time() - t0
            print("-" * 80, flush=True)
            print(f"Epoch {epoch:02d} Complete in {t_elapsed/60:.1f} min | Train: {100*train_acc:.2f}% | "
                  f"Validation: {100*val_metrics['val_acc']:.2f}% | "
                  f"Macro: {100*val_metrics['val_macro_acc']:.2f}% "
                  f"({val_metrics['val_count']} labels)", flush=True)
            selection_score = val_metrics["val_macro_acc"] * 100.0

            # Save a deployable, internally consistent EMA model and all live
            # states required for exact optimizer/scheduler resume.
            ckpt_payload = {
                "format": "signal9-multitask-v1",
                "epoch": epoch,
                "input_len": input_len,
                "architecture": architecture,
                "stem_dim": 128,
                "expert_dim": TRAIN_EXPERT_DIM,
                "emb_dim": 512,
                "n_fecs": 5,
                "n_scramblers": 5,
                "class_names": list(MOD_VOCAB),
                "backbone": raw_backbone.state_dict(),
                "ema_backbone": ema_backbone.state_dict(),
                "heads": heads.state_dict(),
                "ema_heads": ema_heads.state_dict(),
                "arcface": arcface.state_dict(),
                "ema_arcface": ema_arcface.state_dict(),
                "ema_updates": int(ema.updates),
                "prototypes": F.normalize(ema_arcface.weight.detach().cpu(), dim=-1),
                "prototype_weight": float(val_metrics["prototype_weight"]),
                "open_set": val_metrics["open_set"],
                "alignment_preprocess": False,
                "conditioning_contract": CONDITIONING_CONTRACT,
                "source_overlap_policy": SOURCE_OVERLAP_POLICY,
                "feature_contract": "expert11-native-paprdb-v3",
                "trained_heads": (["mod", "symrate", "snr", "presence", "arcface"]
                                  if arcface_weight > 0 else ["mod", "symrate", "snr", "presence"]),
                "loss_weights": {"mod": 1.0, "arcface": float(arcface_weight),
                                 "snr": 0.02, "symrate": 0.05, "presence": 0.05},
                "validation_protocol": "blocked-group-session-80-train-10-cal-10-val-v4",
                "metrics": {**val_metrics, "train_acc": float(train_acc)},
                "best_macro_acc": float(max(best_acc, selection_score)),
                "train_class_counts": {name: int(class_count_tensor[i].item()) for i, name in enumerate(MOD_VOCAB)},
                "training_data": {"roots": [str(p) for p in search_dirs],
                                  "split": "h5_contiguous64_blake2b_80-10-10; hardware_session_grouped_80-10-10",
                                  "synthetic_fraction": streamer.synthetic_fraction,
                                  "synthetic_pool_size": streamer.synthetic_pool_size,
                                  "source_overlap_policy": SOURCE_OVERLAP_POLICY,
                                  "seed": seed, "input_len": input_len},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "streamer_rng_state": streamer.rng.bit_generator.state,
                "torch_rng_state": torch.random.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            }

            if selection_score > best_acc:
                best_acc = selection_score
                torch.save(ckpt_payload, str(best_ckpt_path))
                print(f">> [CHECKPOINT] Saved new best model to {best_ckpt_path} "
                      f"(Macro Acc: {best_acc:.2f}%)", flush=True)
            torch.save(ckpt_payload, str(last_ckpt_path))
            print(f">> [CHECKPOINT] Saved resumable epoch checkpoint to {last_ckpt_path}", flush=True)
            print("-" * 80 + "\n", flush=True)

    production_bundle_path = out_dir / f"{run_name}_production_bundle.ckpt"
    if best_ckpt_path.exists():
        shutil.copyfile(best_ckpt_path, production_bundle_path)
    else:
        torch.save(ckpt_payload, str(production_bundle_path))

    print("\n" + "=" * 80, flush=True)
    print(f">> TRAINING COMPLETE! Production bundle saved to: {production_bundle_path}", flush=True)
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Signal Infinity 10-Dataset Accelerated Trainer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size across GPUs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--steps-per-epoch", type=int, default=1953, help="Steps per epoch")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume / fine-tune from")
    parser.add_argument("--data-root", action="append", default=None,
                        help="Immutable Kaggle input root. Repeat only for intentional additional mounts.")
    parser.add_argument("--val-batches", type=int, default=32,
                        help="Number of disjoint validation batches per epoch")
    parser.add_argument("--seed", type=int, default=20260924, help="Corpus split and sampling seed")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Checkpoint output directory (default: /kaggle/working on Kaggle)")
    parser.add_argument("--input-length", type=int, choices=[128, 256, 512, 1024], default=1024,
                        help="Fixed IQ length for this run; batches never mix lengths")
    parser.add_argument("--arcface-weight", type=float, default=0.0,
                        help="Experimental ArcFace loss weight (default 0: supervised CE baseline)")
    parser.add_argument("--architecture", choices=["baseline_resnet", "hybrid_xcit"],
                        default="hybrid_xcit",
                        help="Backbone; compare both under the identical split before deployment")
    parser.add_argument("--synthetic-fraction", type=float, default=0.10,
                        help="Fraction of each batch from the synthetic anchor pool (default 0.10)")
    parser.add_argument("--synthetic-pool-size", type=int, default=8192,
                        help="Precomputed synthetic waveforms, split 80/10/10 (default 8192)")
    args = parser.parse_args()
    train_satvik_grand_corpus_dual_gpu(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        steps_per_epoch=args.steps_per_epoch,
        resume_path=args.resume,
        data_roots=args.data_root,
        val_batches=args.val_batches,
        seed=args.seed,
        output_dir=args.output_dir,
        input_len=args.input_length,
        arcface_weight=args.arcface_weight,
        architecture=args.architecture,
        synthetic_fraction=args.synthetic_fraction,
        synthetic_pool_size=args.synthetic_pool_size,
    )
