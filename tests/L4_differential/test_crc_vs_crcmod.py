"""Differential testing of CRC computation vs crcmod (SIGNAL10 §G.1 / Rule R9)."""
from __future__ import annotations
import pytest
import struct
crcmod = pytest.importorskip("crcmod")
import crcmod.predefined
from signal9.decide.crc_search import crc32_ieee, find_trailing_crc32, windowed_crc32_search


@pytest.mark.parametrize("payload", [
    b"123456789",
    b"THE_QUICK_BROWN_FOX_JUMPS_OVER_A_LAZY_DOG",
    b"\x00" * 32,
    b"\xff" * 64,
    b"SIGNAL10_DIFFERENTIAL_ORACLE_CHECK",
])
def test_crc32_matches_crcmod(payload):
    """crc32_ieee must match crcmod predefined 'crc-32'."""
    crc32_func = crcmod.predefined.mkCrcFun("crc-32")
    expected = crc32_func(payload)
    actual = crc32_ieee(payload)
    assert actual == expected, f"CRC-32 mismatch on payload {payload!r}: got {hex(actual)}, want {hex(expected)}"


def test_windowed_crc_finds_embedded_packet():
    """find_windowed_crc32 must find CRC embedded inside prefix headers and suffix padding."""
    crc32_func = crcmod.predefined.mkCrcFun("crc-32")
    payload = b"VALID_TRANSMISSION_PAYLOAD_BODY"
    crc = crc32_func(payload)

    # Wrap payload with header and trailer noise
    packet = payload + struct.pack(">I", crc)
    wrapped = b"PREAMBLE_HEADER_NOISE_1234" + packet + b"TRAILER_PADDING_BYTES_5678"

    res = windowed_crc32_search(wrapped)
    assert res.found is True
    assert res.payload == payload
    assert res.crc_found == crc
