"""Standard framing, sync markers, and scramblers (Phase 1).

Implements:
  - CCSDS 32-bit Attached Sync Marker (ASM): 0x1ACFFC1D
  - CCSDS Additive Scrambler (polynomial x^8 + x^7 + x^5 + x^3 + 1, period 255)
  - G3RUH Multiplicative Scrambler (1 + x^12 + x^17) for 9600-baud amateur packet
  - V.35 Multiplicative Scrambler (1 + x^18 + x^23)
  - HDLC bit-stuffing and framing (0x7E delimiter)
  - AX.25 UI-frame assembly with on-air HDLC bit order and CRC-16/X-25
"""
from __future__ import annotations
import numpy as np

# CCSDS 32-bit ASM
CCSDS_ASM_BYTES = bytes.fromhex("1acffc1d")
CCSDS_ASM_BITS = np.unpackbits(np.frombuffer(CCSDS_ASM_BYTES, dtype=np.uint8))

# CCSDS standard 255-bit pseudorandom sequence (Table 10-1)
def _generate_ccsds_pn_sequence() -> np.ndarray:
    """Generate the standard 255-bit CCSDS additive scrambler sequence."""
    # Register initialized to all 1s (8 bits)
    reg = 0xFF
    seq = []
    for _ in range(255):
        out_bit = (reg >> 7) & 1
        # Polynomial: x^8 + x^7 + x^5 + x^3 + 1 (CCSDS 131.0-B-4 Table 10-1)
        # feedback = b7 ^ b4 ^ b2 ^ b0 under left-shift representation
        fb = ((reg >> 7) ^ (reg >> 4) ^ (reg >> 2) ^ reg) & 1
        reg = ((reg << 1) | fb) & 0xFF
        seq.append(out_bit)
    return np.array(seq, dtype=np.uint8)

_CCSDS_PN255 = _generate_ccsds_pn_sequence()


def ccsds_scramble(data_bytes: bytes) -> bytes:
    """XOR data with CCSDS 255-bit additive pseudorandom sequence."""
    if not data_bytes:
        return b""
    bits = np.unpackbits(np.frombuffer(data_bytes, dtype=np.uint8))
    pn_repeated = np.tile(_CCSDS_PN255, int(np.ceil(len(bits) / 255)))[:len(bits)]
    scrambled_bits = bits ^ pn_repeated
    return bytes(np.packbits(scrambled_bits))


def ccsds_descramble(scrambled_bytes: bytes) -> bytes:
    """Additive descrambling is symmetric with scrambling."""
    return ccsds_scramble(scrambled_bytes)


def g3ruh_scramble(bits: np.ndarray | bytes) -> np.ndarray:
    """G3RUH multiplicative scrambler: y[n] = x[n] ^ y[n-12] ^ y[n-17]."""
    if isinstance(bits, (bytes, bytearray)):
        b = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))
    else:
        b = np.asarray(bits, dtype=np.uint8)

    y = np.empty_like(b)
    reg = 0  # 17-bit shift register

    for n in range(len(b)):
        # y[n] = x[n] ^ y[n-12] ^ y[n-17]
        bit_12 = (reg >> 11) & 1
        bit_17 = (reg >> 16) & 1
        out = int(b[n]) ^ bit_12 ^ bit_17
        y[n] = out
        reg = ((reg << 1) | out) & 0x1FFFF

    return y


def g3ruh_descramble(bits: np.ndarray) -> np.ndarray:
    """G3RUH multiplicative descrambler: x[n] = y[n] ^ y[n-12] ^ y[n-17]."""
    y = np.asarray(bits, dtype=np.uint8)
    x = np.empty_like(y)
    reg = 0

    for n in range(len(y)):
        curr_y = int(y[n])
        bit_12 = (reg >> 11) & 1
        bit_17 = (reg >> 16) & 1
        x[n] = curr_y ^ bit_12 ^ bit_17
        reg = ((reg << 1) | curr_y) & 0x1FFFF

    return x


def v35_scramble(bits: np.ndarray | bytes) -> np.ndarray:
    """V.35 multiplicative scrambler: y[n] = x[n] ^ y[n-18] ^ y[n-23]."""
    if isinstance(bits, (bytes, bytearray)):
        b = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))
    else:
        b = np.asarray(bits, dtype=np.uint8)

    y = np.empty_like(b)
    reg = 0  # 23-bit shift register

    for n in range(len(b)):
        bit_18 = (reg >> 17) & 1
        bit_23 = (reg >> 22) & 1
        out = int(b[n]) ^ bit_18 ^ bit_23
        y[n] = out
        reg = ((reg << 1) | out) & 0x7FFFFF

    return y


def v35_descramble(bits: np.ndarray) -> np.ndarray:
    """V.35 multiplicative descrambler: x[n] = y[n] ^ y[n-18] ^ y[n-23]."""
    y = np.asarray(bits, dtype=np.uint8)
    x = np.empty_like(y)
    reg = 0

    for n in range(len(y)):
        curr_y = int(y[n])
        bit_18 = (reg >> 17) & 1
        bit_23 = (reg >> 22) & 1
        x[n] = curr_y ^ bit_18 ^ bit_23
        reg = ((reg << 1) | curr_y) & 0x7FFFFF

    return x


def hdlc_bit_stuff(bits: np.ndarray) -> np.ndarray:
    """Apply HDLC zero-bit-stuffing: insert 0 after five consecutive 1s."""
    out = []
    consec_ones = 0
    for b in bits:
        out.append(b)
        if b == 1:
            consec_ones += 1
            if consec_ones == 5:
                out.append(0)
                consec_ones = 0
        else:
            consec_ones = 0
    return np.array(out, dtype=np.uint8)


def hdlc_bit_unstuff(bits: np.ndarray) -> np.ndarray:
    """Remove HDLC zero-bit-stuffing."""
    out = []
    consec_ones = 0
    skip_next = False
    for b in bits:
        if skip_next:
            skip_next = False
            continue
        out.append(b)
        if b == 1:
            consec_ones += 1
            if consec_ones == 5:
                skip_next = True
                consec_ones = 0
        else:
            consec_ones = 0
    return np.array(out, dtype=np.uint8)


def crc16_ccitt(data: bytes) -> int:
    """Compute standard CRC-16-CCITT (polynomial 0x1021, init 0xFFFF, inverted)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return (~crc) & 0xFFFF


def crc16_x25(data: bytes) -> int:
    """Return the reflected CRC-16/X-25 FCS used by AX.25/HDLC.

    ``crc16_ccitt`` above is retained for the project's existing CCSDS/ISM
    examples.  AX.25 uses the reflected HDLC/X-25 convention (initial value
    and final complement both ``0xffff``), so it must not share that helper.
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= int(byte)
        for _ in range(8):
            crc = (crc >> 1) ^ (0x8408 if (crc & 1) else 0)
    return crc ^ 0xFFFF


def _ax25_ui_content(dest_call: str, src_call: str, payload: bytes,
                     dest_ssid: int = 0, src_ssid: int = 0) -> bytes:
    """Build an AX.25 UI frame body, without HDLC flags or FCS."""
    def _format_addr(call: str, ssid: int, is_last: bool = False) -> bytes:
        c = call.upper().ljust(6)[:6]
        addr_bytes = bytearray([(ord(ch) << 1) for ch in c])
        ssid_byte = 0x60 | ((ssid & 0x0F) << 1) | (1 if is_last else 0)
        addr_bytes.append(ssid_byte)
        return bytes(addr_bytes)

    return (_format_addr(dest_call, dest_ssid, is_last=False)
            + _format_addr(src_call, src_ssid, is_last=True)
            + b"\x03\xf0" + bytes(payload))


def _bytes_to_lsb_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")


def frame_ax25_bits(dest_call: str,
                    src_call: str,
                    payload: bytes,
                    dest_ssid: int = 0,
                    src_ssid: int = 0) -> np.ndarray:
    """Build a wire-order AX.25 UI frame as an HDLC bit stream.

    The body octets are emitted least-significant bit first, the body plus
    reflected CRC-16/X-25 FCS are bit-stuffed, and 0x7e flags bound the frame.
    This is the representation that must be passed to a symbol mapper; it is
    deliberately separate from :func:`frame_ax25`, whose byte container is
    preserved for legacy non-radio tests and file-oriented callers.
    """
    content = _ax25_ui_content(dest_call, src_call, payload, dest_ssid, src_ssid)
    fcs = crc16_x25(content)
    # HDLC transmits the FCS low byte first.  Each octet is serialized LSB-first.
    protected = content + fcs.to_bytes(2, byteorder="little")
    flag = _bytes_to_lsb_bits(b"\x7e")
    return np.concatenate((flag, hdlc_bit_stuff(_bytes_to_lsb_bits(protected)), flag)).astype(np.uint8)


def frame_ccsds(payload: bytes, apply_scrambler: bool = True) -> bytes:
    """Encapsulate payload with CCSDS 32-bit ASM and optional additive scrambler."""
    scrambled = ccsds_scramble(payload) if apply_scrambler else payload
    return CCSDS_ASM_BYTES + scrambled


def frame_ax25(dest_call: str,
               src_call: str,
               payload: bytes,
               dest_ssid: int = 0,
               src_ssid: int = 0) -> bytes:
    """Return the legacy byte-container form used by older local tests.

    For an on-air/over-symbols AX.25 signal use :func:`frame_ax25_bits`.
    Keeping this compatibility container avoids silently changing unrelated
    CCSDS/ISM test fixtures while preventing it from being mistaken for an
    HDLC wire stream.
    """
    content = _ax25_ui_content(dest_call, src_call, payload, dest_ssid, src_ssid)
    crc = crc16_ccitt(content)
    return b"\x7e" + content + bytes([crc & 0xFF, (crc >> 8) & 0xFF]) + b"\x7e"
