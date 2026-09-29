"""Differential test for blind scrambler identification (Track F-4)."""
from __future__ import annotations
import pytest
import numpy as np

from signal9.sim.framing import (
    g3ruh_scramble,
    v35_scramble,
    ccsds_scramble,
    frame_ax25,
)
from signal9.label.scrambler_id import identify_scrambler


def test_identify_g3ruh():
    # AX.25 packet has callsigns and HDLC flags
    payload = b"WEATHER REPORT FROM STATION ALPHA LAT 45.2 LON 73.1"
    ax_frame = frame_ax25("CQ", "WX1AW", payload)
    raw_bits = np.unpackbits(np.frombuffer(ax_frame, dtype=np.uint8))
    # Scramble with G3RUH
    scrambled = g3ruh_scramble(raw_bits)

    name, conf = identify_scrambler(scrambled)
    assert name == "G3RUH", f"Expected G3RUH, got {name} (conf={conf})"
    assert conf > 0.3


def test_identify_v35():
    payload = b"TESTING V.35 PROTOCOL STREAM DATA " * 8
    raw_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    scrambled = v35_scramble(raw_bits)

    name, conf = identify_scrambler(scrambled)
    assert name == "V35", f"Expected V35, got {name} (conf={conf})"
    assert conf > 0.3


def test_identify_ccsds():
    payload = b"CCSDS TELEMETRY PACKET VERSION 1 APID 100 SEQUENCE 42" * 4
    scrambled_bytes = ccsds_scramble(payload)
    scrambled_bits = np.unpackbits(np.frombuffer(scrambled_bytes, dtype=np.uint8))

    name, conf = identify_scrambler(scrambled_bits)
    assert name == "CCSDS", f"Expected CCSDS, got {name} (conf={conf})"
    assert conf > 0.3


def test_identify_unscrambled_or_random():
    rng = np.random.default_rng(42)
    pure_random_bits = rng.integers(0, 2, 1024, dtype=np.uint8)

    name, conf = identify_scrambler(pure_random_bits)
    assert name == "none"
