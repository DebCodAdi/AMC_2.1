"""CCSDS Telemetry Transfer Frame protocol deframer (Phase 4).

Reference: CCSDS 132.0-B-2, 'TM Space Data Link Protocol'.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from signal9.sim.framing import ccsds_descramble, CCSDS_ASM_BYTES, crc16_ccitt


@dataclass(frozen=True)
class CCSDSFrame:
    version: int
    scid: int
    vcid: int
    mcfc: int
    vcfc: int
    first_header_pointer: int
    payload: bytes
    crc_valid: bool


def deframe_ccsds_tm(data: bytes | np.ndarray,
                     frame_length: int = 128,
                     has_fecf: bool = True) -> list[CCSDSFrame]:
    """Search, descramble, and parse CCSDS Telemetry Transfer Frames.

    Parameters
    ----------
    data : bytes or bit array
    frame_length : int, default 128 bytes
        Total transfer frame length excluding ASM.
    has_fecf : bool, default True
        Whether frame includes a trailing 16-bit CRC FECF.

    Returns
    -------
    frames : list of CCSDSFrame
    """
    if isinstance(data, (bytes, bytearray)):
        byte_streams = [bytes(data)]
    else:
        bits = np.asarray(data, dtype=np.uint8).ravel() & 1
        # A blind demodulator does not promise byte alignment.  Try every bit
        # phase; the 32-bit ASM and optional FECF provide strong validation.
        byte_streams = []
        for offset in range(min(8, max(1, len(bits)))):
            usable = ((len(bits) - offset) // 8) * 8
            if usable:
                byte_streams.append(bytes(np.packbits(bits[offset:offset + usable])))

    frames: list[CCSDSFrame] = []
    asm = CCSDS_ASM_BYTES

    # Search for ASM occurrences at all candidate bit phases.
    for raw_bytes in byte_streams:
      start = 0
      while True:
        pos = raw_bytes.find(asm, start)
        if pos == -1:
            break

        frame_start = pos + len(asm)
        frame_end = frame_start + frame_length

        if frame_end <= len(raw_bytes):
            scrambled = raw_bytes[frame_start:frame_end]
            # Descramble
            descrambled = ccsds_descramble(scrambled)

            # Header is 6 bytes
            if len(descrambled) >= 6:
                h = descrambled[:6]
                w0 = (h[0] << 8) | h[1]
                version = (w0 >> 14) & 0x03
                scid = (w0 >> 4) & 0x03FF
                vcid = (w0 >> 1) & 0x07

                mcfc = h[2]
                vcfc = h[3]
                w4 = (h[4] << 8) | h[5]
                fhp = w4 & 0x07FF

                if has_fecf and len(descrambled) >= 8:
                    content = descrambled[:-2]
                    rx_crc = (descrambled[-2] << 8) | descrambled[-1]
                    calc_crc = crc16_ccitt(content)
                    crc_ok = (rx_crc == calc_crc)
                    payload = descrambled[6:-2]
                else:
                    crc_ok = True
                    payload = descrambled[6:]

                frame = CCSDSFrame(
                    version=version,
                    scid=scid,
                    vcid=vcid,
                    mcfc=mcfc,
                    vcfc=vcfc,
                    first_header_pointer=fhp,
                    payload=payload,
                    crc_valid=crc_ok,
                )
                if frame not in frames:
                    frames.append(frame)

        start = pos + 1

    return frames
