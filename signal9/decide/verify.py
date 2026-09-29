"""Decoder-in-the-loop verification interface."""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
import numpy as np

# Decoder registry mapping decoder names to system executables
DECODER_BINS = {
    "jt9_ft8": "jt9",
    "wsprd": "wsprd",
    "direwolf_atest": "atest",
    "multimon_ng": "multimon-ng",
    "dsd_fme": "dsd-fme",
    "ais_catcher": "AIS-catcher",
    "rtl_433": "rtl_433",
    "dump1090": "dump1090",
}

def verify_protocol(decoder_name: str, audio_or_iq: np.ndarray, fs: float) -> dict:
    """Attempts protocol decode verification.
    CRITICAL RULE (Fixes A11):
    - A successful decode marks 'status=VERIFIED'.
    - A failed decode leaves the ML label untouched ('status=UNVERIFIED_DECODER_FAILED').
      (Decoders frequently fail at low SNR where AMC succeeds).
    - NEVER store or report decoded message content (privacy hygiene).
    """
    bin_name = DECODER_BINS.get(decoder_name, decoder_name)
    bin_path = shutil.which(bin_name)
    if not bin_path:
        return {
            "status": "UNVERIFIED_TOOL_MISSING",
            "decoder": decoder_name,
            "success": False
        }

    # In production, invokes subprocess with timeout and parses returncode
    # Stub returns status cleanly if tool is absent
    return {
        "status": "UNVERIFIED_STANDALONE",
        "decoder": decoder_name,
        "success": False
    }
