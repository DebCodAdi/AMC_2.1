"""Regression tests for standards-shaped AX.25 radio-wire handling."""
from __future__ import annotations

import numpy as np

from signal9.exploit.chain import ExploitationChain
from signal9.protocol.ax25 import deframe_ax25
from signal9.sim.framing import frame_ax25_bits


def test_ax25_wire_bits_round_trip_with_stuffing_and_x25_fcs():
    # A run of 1-bits in the payload exercises HDLC stuffing, not just the
    # convenient text-only path used by the user-facing demo.
    payload = b"WIRE-ORDER-TEST" + bytes([0xFF, 0xF8]) + b"-OK"
    bits = frame_ax25_bits("APRS", "K1ABC", payload, src_ssid=3)
    frames = deframe_ax25(bits)

    assert len(frames) == 1
    assert frames[0].crc_valid
    assert frames[0].dest_call == "APRS"
    assert frames[0].src_call == "K1ABC"
    assert frames[0].src_ssid == 3
    assert frames[0].payload == payload


def test_wire_valid_ax25_is_checked_before_blind_transforms():
    bits = frame_ax25_bits("DEST", "SRC", b"CRC-gated packet")
    out = ExploitationChain()._run_bit_stages({
        "bits": bits,
        "llr": np.ones(bits.size, dtype=np.float32),
        "score": 0.0,
    })

    assert out["crc_hit"]
    assert out["frames"]["protocol"] == "AX.25"
    assert out["frames"]["crc_frames_ok"] == 1
    assert out["frames"]["ax25_frames"][0]["payload_text"] == "CRC-gated packet"
    assert out["scrambler"]["type"] == "none"


def test_ax25_rejects_an_abort_instead_of_silently_unstuffing_it():
    bits = frame_ax25_bits("DEST", "SRC", b"ABORT-CHECK")
    # The first inserted stuffed zero becomes a one: a run of six ones is an
    # invalid HDLC payload, therefore no CRC-valid AX.25 frame may emerge.
    corrupted = bits.copy()
    mutated = False
    # Skip the two HDLC flags.  The only zero immediately following five ones
    # in the encoded payload is a stuff bit, because the transmitter inserts
    # one even when the original next bit was already zero.
    for candidate in range(8, corrupted.size - 8):
        if corrupted[candidate] == 0 and np.all(corrupted[candidate - 5:candidate] == 1):
            corrupted[candidate] = 1
            mutated = True
            break
    assert mutated
    assert not any(frame.crc_valid for frame in deframe_ax25(corrupted))
