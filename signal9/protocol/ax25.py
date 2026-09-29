"""AX.25 UI-frame deframer with HDLC wire-bit support.

Real AX.25/HDLC is not an MSB-first byte stream: each octet is serialized
least-significant bit first, payload bits are zero-stuffed, and the FCS is
CRC-16/X-25.  This module accepts that radio-wire representation while still
accepting the project's older byte-container fixtures for compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from signal9.sim.framing import crc16_ccitt, crc16_x25


@dataclass(frozen=True)
class AX25Frame:
    dest_call: str
    dest_ssid: int
    src_call: str
    src_ssid: int
    control: int
    pid: int
    payload: bytes
    crc_valid: bool


def _segments_from_wire_bits(bits: np.ndarray, *, bitorder: str) -> list[bytes]:
    """Extract HDLC payload octets without ever unstuffing a delimiter.

    ``bitorder='little'`` is the AX.25 on-air convention.  The MSB-first
    fallback exists only for old project fixtures; CRC-valid wire frames always
    take priority in :func:`deframe_ax25`.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel() & 1
    flag = np.unpackbits(np.array([0x7E], dtype=np.uint8), bitorder=bitorder)
    if bits.size < flag.size:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(bits, flag.size)
    positions = np.flatnonzero(np.all(windows == flag, axis=1))
    segments: list[bytes] = []
    for left, right in zip(positions[:-1], positions[1:]):
        payload = bits[left + flag.size:right]
        if payload.size == 0:
            continue
        unstuffed = _hdlc_unstuff_strict(payload)
        if unstuffed is None:
            # Six consecutive 1s in a payload is an HDLC abort/error, not a
            # stuffed zero to discard.  Never turn a malformed frame into a
            # different byte sequence and then try to parse it as AX.25.
            continue
        byte_count = unstuffed.size // 8
        if byte_count:
            segments.append(bytes(np.packbits(unstuffed[: byte_count * 8], bitorder=bitorder)))
    return segments


def _hdlc_unstuff_strict(payload: np.ndarray) -> np.ndarray | None:
    """Undo legal zero stuffing, rejecting malformed/aborted payload bits."""
    source = np.asarray(payload, dtype=np.uint8).ravel() & 1
    out: list[int] = []
    ones = 0
    index = 0
    while index < source.size:
        bit = int(source[index])
        out.append(bit)
        index += 1
        if bit:
            ones += 1
            if ones == 5:
                if index >= source.size or int(source[index]) != 0:
                    return None
                # Consume the only legal stuffed bit; it is not frame data.
                index += 1
                ones = 0
        else:
            ones = 0
    return np.asarray(out, dtype=np.uint8)


def _parse_segments(segments: list[bytes], crc_function) -> list[AX25Frame]:
    frames: list[AX25Frame] = []
    for seg in segments:
        # destination + source + control + PID + FCS
        if len(seg) < 18:
            continue
        content = seg[:-2]
        # AX.25 sends FCS little-endian, as does the legacy local container.
        received_crc = seg[-2] | (seg[-1] << 8)
        crc_ok = received_crc == crc_function(content)

        def parse_call(address: bytes) -> tuple[str, int]:
            call = "".join(chr(byte >> 1) for byte in address[:6] if (byte >> 1) > 32).strip()
            return call, (address[6] >> 1) & 0x0F

        try:
            # AX.25 permits zero or more repeater address fields.  The
            # extension bit of each SSID octet, not a fixed 14-byte offset,
            # marks the control field.  We retain destination/source in the
            # compact public record while correctly locating payloads in
            # digipeated captures as well.
            addresses: list[tuple[str, int]] = []
            offset = 0
            while offset + 7 <= len(content) and len(addresses) < 16:
                address = content[offset:offset + 7]
                addresses.append(parse_call(address))
                offset += 7
                if address[6] & 0x01:
                    break
            if len(addresses) < 2 or offset + 2 > len(content):
                continue
            dest_call, dest_ssid = addresses[0]
            src_call, src_ssid = addresses[1]
            frames.append(AX25Frame(
                dest_call=dest_call,
                dest_ssid=dest_ssid,
                src_call=src_call,
                src_ssid=src_ssid,
                control=content[offset],
                pid=content[offset + 1],
                payload=content[offset + 2:],
                crc_valid=crc_ok,
            ))
        except (IndexError, ValueError, UnicodeError):
            continue
    return frames


def deframe_ax25(data: bytes | np.ndarray) -> list[AX25Frame]:
    """Find AX.25 UI frames and independently validate every FCS.

    Bit-array input is interpreted as an on-air HDLC stream first (LSB-first
    octets, bit stuffing, CRC-16/X-25).  MSB-first/legacy parsing is only a
    compatibility fallback and must not be used to claim a CRC-locked radio
    decode when the wire parser found no valid AX.25 frame.
    """
    if isinstance(data, (bytes, bytearray)):
        # Existing non-radio tests and file-oriented callers use this local
        # compatibility container with the old CCITT helper.
        return _parse_segments(bytes(data).split(b"\x7e"), crc16_ccitt)

    bits = np.asarray(data, dtype=np.uint8).ravel() & 1
    wire_frames = _parse_segments(_segments_from_wire_bits(bits, bitorder="little"), crc16_x25)
    if any(frame.crc_valid for frame in wire_frames):
        return wire_frames

    # Do not remove this compatibility path until all historic fixtures are
    # migrated.  Its frames are never preferred over a verified wire result.
    legacy_frames = _parse_segments(_segments_from_wire_bits(bits, bitorder="big"), crc16_ccitt)
    return legacy_frames if legacy_frames else wire_frames
