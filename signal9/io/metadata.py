"""Metadata extraction and inference chain for IQ captures."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import yaml

@dataclass
class SignalMetadata:
    sample_rate: float | None = None
    center_freq: float | None = None
    format: str | None = None
    source: str = "unknown"
    is_audio: bool = False
    # Where each value came from: "cli", "wav_header", "sigmf", "filename" or
    # "unknown".  Reported next to the value itself, because for a headerless
    # .iq file the sample rate cannot be measured from the samples at all
    # (Decoding_Roadmap.md 2.3) and a reader must be able to tell a fact from
    # an assumption.
    sample_rate_source: str = "unknown"
    center_freq_source: str = "unknown"

def resolve_metadata(path: str | Path,
                     patterns_yaml: str | Path | None = None,
                     cli_fs: float | None = None,
                     cli_fc: float | None = None,
                     cli_format: str | None = None) -> SignalMetadata:
    """Resolve sample_rate, center_freq, and format through the priority chain:
    1. CLI flags (override everything)
    2. SigMF metadata (if present)
    3. Filename patterns
    4. Heuristic format sniffer (for format only)
    """
    path = Path(path)
    meta = SignalMetadata(
        sample_rate=cli_fs,
        center_freq=cli_fc,
        format=cli_format,
        source="cli" if cli_fs and cli_fc else "unknown",
        # Provenance is tracked per field.  The single `source` flag was only
        # set to "cli" when *both* --fs and --fc were given, so a bare --fs was
        # reported as "unknown" and then overwritten by "filename" even when
        # the filename supplied nothing but the centre frequency.
        sample_rate_source="cli" if cli_fs else "unknown",
        center_freq_source="cli" if cli_fc else "unknown",
    )

    # Check for SigMF companion
    meta_sigmf = path.with_suffix(".sigmf-meta")
    if meta_sigmf.exists():
        try:
            import json
            with open(meta_sigmf, "r", encoding="utf-8") as f:
                s_meta = json.load(f)
            if meta.sample_rate is None:
                meta.sample_rate = float(s_meta.get("global", {}).get("core:sample_rate", 0)) or None
                if meta.sample_rate:
                    meta.sample_rate_source = "sigmf"
            if meta.center_freq is None and s_meta.get("captures"):
                meta.center_freq = float(s_meta["captures"][0].get("core:frequency", 0)) or None
                if meta.center_freq:
                    meta.center_freq_source = "sigmf"
            meta.source = "sigmf"
        except Exception:
            pass

    # Try regex patterns from filename if still missing
    if meta.sample_rate is None or meta.center_freq is None:
        filename = path.name
        # Fallback built-in regexes
        builtin_regexes = [
            (re.compile(r"gqrx_\d{8}_\d{6}_(?P<fc>\d+)_(?P<fs>\d+)_fc\.raw$", re.I), "cf32"),
            (re.compile(r"SDRSharp_\d{8}_\d{6}Z_(?P<fc>\d+)Hz_IQ\.wav$", re.I), None),
            (re.compile(r"(?P<fc>\d+)_(?P<fs>\d+)(?:sps|Sps)?.*\.(?P<ext>cu8|cs8|cs16|cf32)$", re.I), None),
            (re.compile(r"\D*(?P<fc>\d{6,10})_\D*(?P<fs>\d{4,8})\D*", re.I), None),
        ]
        for reg, def_fmt in builtin_regexes:
            m = reg.search(filename)
            if m:
                gd = m.groupdict()
                if meta.center_freq is None and gd.get("fc"):
                    meta.center_freq = float(gd["fc"])
                    meta.center_freq_source = "filename"
                if meta.sample_rate is None and gd.get("fs"):
                    meta.sample_rate = float(gd["fs"])
                    meta.sample_rate_source = "filename"
                if meta.format is None:
                    meta.format = gd.get("ext") or def_fmt
                meta.source = "filename"
                break

    return meta
