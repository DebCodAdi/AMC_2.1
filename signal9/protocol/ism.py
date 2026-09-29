"""Generic ISM device protocol deframer (Phase 4).

Parses variable-length packets with preamble, 16-bit sync word (0xD391 / 0x2D4D),
length byte, payload, and CRC-16.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from signal9.sim.framing import crc16_ccitt


@dataclass(frozen=True)
class ISMFrame:
    sync_word: int
    length: int
    payload: bytes
    crc_valid: bool


SYNC_WORDS = [0xD391, 0x2D4D]


def deframe_ism(data: bytes | np.ndarray) -> list[ISMFrame]:
    """Search and extract ISM variable-length packets.

    Format: [Sync 2B] [Length 1B] [Payload LB] [CRC 2B]
    """
    if isinstance(data, (bytes, bytearray)):
        byte_streams = [bytes(data)]
    else:
        bits = np.asarray(data, dtype=np.uint8).ravel() & 1
        byte_streams = []
        for offset in range(min(8, max(1, len(bits)))):
            usable = ((len(bits) - offset) // 8) * 8
            if usable:
                byte_streams.append(bytes(np.packbits(bits[offset:offset + usable])))

    frames: list[ISMFrame] = []

    for raw_bytes in byte_streams:
      for sync in SYNC_WORDS:
        sync_bytes = bytes([sync >> 8, sync & 0xFF])
        start = 0
        while True:
            pos = raw_bytes.find(sync_bytes, start)
            if pos == -1:
                break

            # Need at least length byte + 2 bytes CRC = 3 bytes after sync
            if pos + 2 + 3 <= len(raw_bytes):
                length = raw_bytes[pos + 2]
                frame_total = 2 + 1 + length + 2
                if pos + frame_total <= len(raw_bytes):
                    content = raw_bytes[pos + 2 : pos + 3 + length]
                    rx_crc = (raw_bytes[pos + 3 + length] << 8) | raw_bytes[pos + 4 + length]
                    calc_crc = crc16_ccitt(content)
                    payload = raw_bytes[pos + 3 : pos + 3 + length]

                    frame = ISMFrame(
                        sync_word=sync,
                        length=length,
                        payload=payload,
                        crc_valid=(rx_crc == calc_crc),
                    )
                    if frame not in frames:
                        frames.append(frame)

            start = pos + 1

    return frames
