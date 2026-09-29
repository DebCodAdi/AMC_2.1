"""Protocol deframing oracle test suite (Phase 4).

Spec reference: Rule R9, Phase 4.
"""
from __future__ import annotations
import numpy as np

from signal9.protocol.ax25 import deframe_ax25
from signal9.protocol.ccsds import deframe_ccsds_tm
from signal9.protocol.ism import deframe_ism
from signal9.sim.framing import frame_ax25, frame_ccsds, crc16_ccitt, CCSDS_ASM_BYTES


def test_ax25_oracle_roundtrip():
    payload = b"TELEMETRY VOLTAGE=3.85V CURRENT=120MA TEMP=22.4C"
    frame = frame_ax25("APRS", "K1ABC", payload, dest_ssid=0, src_ssid=3)

    # Prepend and append noise bytes to simulate continuous channel
    stream = b"\x00\xff\xaa" + frame + b"\x55\x00"
    extracted = deframe_ax25(stream)

    assert len(extracted) == 1
    f = extracted[0]
    assert f.dest_call == "APRS"
    assert f.src_call == "K1ABC"
    assert f.src_ssid == 3
    assert f.payload == payload
    assert f.crc_valid


def test_ccsds_oracle_roundtrip():
    # Build a valid CCSDS TM frame: 6B header + 120B payload + 2B CRC = 128B
    h0 = 0x0000  # version 0, SCID 0, VCID 0
    h2 = 0x01    # MCFC 1
    h3 = 0x02    # VCFC 2
    h4 = 0x0000  # FHP 0
    header = bytes([h0 >> 8, h0 & 0xFF, h2, h3, h4 >> 8, h4 & 0xFF])
    payload = b"CCSDS ORACLE PAYLOAD" + (b"X" * 100)
    content = header + payload
    crc = crc16_ccitt(content)
    raw_frame = content + bytes([crc >> 8, crc & 0xFF])

    # Scramble and prepend ASM
    framed = frame_ccsds(raw_frame, apply_scrambler=True)
    stream = b"\xaa\xbb" + framed + b"\xcc\xdd"

    extracted = deframe_ccsds_tm(stream, frame_length=128, has_fecf=True)
    assert len(extracted) == 1
    f = extracted[0]
    assert f.version == 0
    assert f.mcfc == 1
    assert f.vcfc == 2
    assert f.payload == payload
    assert f.crc_valid


def test_ism_oracle_roundtrip():
    payload = b"SENSOR_ID_99_TEMP_72_HUM_45"
    sync = 0xD391
    sync_b = bytes([sync >> 8, sync & 0xFF])
    length_b = bytes([len(payload)])
    content = length_b + payload
    crc = crc16_ccitt(content)
    packet = sync_b + content + bytes([crc >> 8, crc & 0xFF])

    stream = b"\xaa\xaa\xaa\xaa" + packet + b"\x00\x00"
    extracted = deframe_ism(stream)

    assert len(extracted) == 1
    f = extracted[0]
    assert f.sync_word == 0xD391
    assert f.length == len(payload)
    assert f.payload == payload
    assert f.crc_valid
