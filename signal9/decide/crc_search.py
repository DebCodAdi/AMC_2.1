"""Windowed CRC search across candidate frame boundaries (DEF-022).

DEF-022: The naive arbiter assumes the CRC is exactly the trailing 4 bytes of
the decoded stream. If the format includes a header, framing markers, or trailing
padding, this assumption fails.

This module:
  - Searches valid (start, end) frame boundaries for CRC-32 matches.
  - identify_crc_bytes(): multi-width (8/16/32) CRC catalogue search on raw
    byte frames (integrates the Plan §10.2 catalogue; avoids duplicating
    d6_frames.CRC_CATALOGUE by re-using it when available).
  - windowed_crc_search(): original DEF-022 sliding-window CRC-32 search.
  - windowed_crc_search_multiwidth(): extends the window search to 8 and 16-bit
    CRCs in addition to CRC-32 (plan improvement).

Plan reference: §10.2 (D6.4), Blind_Exploitation_Chain_Plan.md.
Spec reference: DEF-022 (PS26147).
"""
from __future__ import annotations
import struct
import zlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CRCResult:
    """Result of a windowed CRC search."""
    found: bool
    start: int
    end: int
    payload: bytes
    crc_expected: int
    crc_found: int


def crc32_ieee(data: bytes) -> int:
    """Standard IEEE 802.3 CRC-32 (reflected polynomial 0xEDB88320)."""
    return zlib.crc32(data) & 0xFFFFFFFF


def find_trailing_crc32(data: bytes) -> CRCResult:
    """Check standard trailing 4-byte CRC-32 on data."""
    if len(data) < 8:
        return CRCResult(found=False, start=0, end=len(data), payload=data, crc_expected=0, crc_found=0)

    payload = data[:-4]
    expected = struct.unpack(">I", data[-4:])[0]
    computed = crc32_ieee(payload)

    if computed == expected:
        return CRCResult(
            found=True,
            start=0,
            end=len(data),
            payload=payload,
            crc_expected=expected,
            crc_found=computed
        )

    # Also check little-endian CRC-32
    expected_le = struct.unpack("<I", data[-4:])[0]
    if computed == expected_le:
        return CRCResult(
            found=True,
            start=0,
            end=len(data),
            payload=payload,
            crc_expected=expected_le,
            crc_found=computed
        )

    return CRCResult(found=False, start=0, end=len(data), payload=payload, crc_expected=expected, crc_found=computed)


def windowed_crc32_search(data: bytes,
                          min_payload_len: int = 4,
                          max_candidates: int = 10000) -> CRCResult:
    """Perform sliding window search for CRC-32 matches (DEF-022).

    Parameters
    ----------
    data : bytes
        Decoded bitstream or byte buffer.
    min_payload_len : int, default 4 (minimum payload size in bytes)
    max_candidates : int, default 10000

    Returns
    -------
    CRCResult
    """
    N = len(data)
    if N < min_payload_len + 4:
        return CRCResult(found=False, start=0, end=N, payload=b"", crc_expected=0, crc_found=0)

    # 1. Fast check: full buffer trailing CRC (covers 95% of standard cases)
    res_full = find_trailing_crc32(data)
    if res_full.found:
        return res_full

    # 2. Window search across candidate start offsets
    if N <= 512:
        valid_starts = list(range(N - (min_payload_len + 4) + 1))
    else:
        # For large buffers, check fine starts up to 128 then step by 4
        valid_starts = list(range(min(128, N - (min_payload_len + 4) + 1))) + [
            s for s in range(128, N - (min_payload_len + 4) + 1, 4)
        ]

    # Search each candidate start.
    #
    # The payload for a given (start, end) is data[start : end-4], and it grows
    # by exactly one byte each time `end` advances, so its CRC can be carried
    # forward incrementally.  An earlier revision maintained a `running_crc`
    # that it never read, and recomputed crc32 over the whole payload from
    # scratch on every iteration -- O(N^2) byte hashing per start offset on a
    # buffer that can be megabytes.  Carrying the register makes it O(N).
    candidates_tested = 0
    for s in valid_starts:
        # CRC of data[s : end-4], carried forward as `end` advances.
        payload_crc = 0
        for end in range(s + 1, N + 1):
            payload_len = end - s - 4
            if payload_len >= 1:
                # data[end-5] is the byte that just joined the payload.
                payload_crc = zlib.crc32(data[end - 5 : end - 4], payload_crc) & 0xFFFFFFFF
            if payload_len >= min_payload_len:
                # CRC is at data[end-4:end]
                # Check big-endian and little-endian
                crc_bytes = data[end - 4 : end]
                crc_be = struct.unpack(">I", crc_bytes)[0]
                crc_le = struct.unpack("<I", crc_bytes)[0]

                payload = data[s : end - 4]
                p_crc = payload_crc

                if p_crc == crc_be:
                    return CRCResult(
                        found=True,
                        start=s,
                        end=end,
                        payload=payload,
                        crc_expected=crc_be,
                        crc_found=p_crc
                    )
                if p_crc == crc_le:
                    return CRCResult(
                        found=True,
                        start=s,
                        end=end,
                        payload=payload,
                        crc_expected=crc_le,
                        crc_found=p_crc
                    )

                candidates_tested += 1
                if candidates_tested >= max_candidates:
                    break
        if candidates_tested >= max_candidates:
            break

    return CRCResult(found=False, start=0, end=N, payload=b"", crc_expected=0, crc_found=0)


def identify_crc_bytes(frames: list[tuple[bytes, int]],
                        width: int | None = None) -> list[str]:
    """Multi-width CRC catalogue search on (payload, crc) byte pairs (Plan §10.2).

    Re-uses the catalogue from signal9.label.d6_frames when available,
    falling back to the built-in CRC-32/ISO-HDLC and CRC-16/X-25 set.

    Parameters
    ----------
    frames : list of (payload_bytes, observed_crc_int) pairs.
    width  : int or None — restrict to this CRC width.

    Returns
    -------
    list of matching CRC model names.
    """
    try:
        from signal9.label.d6_frames import identify_crc as _d6_identify_crc
        return _d6_identify_crc(frames, width=width)
    except ImportError:
        pass

    # Minimal fallback catalogue if d6_frames is unavailable
    def _crc_core(data: bytes, w: int, poly: int, init: int,
                  ri: bool, ro: bool, xo: int) -> int:
        def rev(v: int, n: int) -> int:
            return int(format(v, f"0{n}b")[::-1], 2)
        top = 1 << (w - 1)
        mask = (1 << w) - 1
        crc = init
        for byte in data:
            b = rev(byte, 8) if ri else byte
            crc ^= (b << (w - 8)) & mask if w >= 8 else b >> (8 - w)
            for _ in range(8):
                crc = ((crc << 1) ^ poly) & mask if crc & top else (crc << 1) & mask
        if ro:
            crc = rev(crc, w)
        return crc ^ xo

    catalogue = [
        ("CRC-16/X-25",      16, 0x1021,      0xFFFF,     True,  True,  0xFFFF),
        ("CRC-16/IBM-3740",  16, 0x1021,      0xFFFF,     False, False, 0x0000),
        ("CRC-32/ISO-HDLC",  32, 0x04C11DB7,  0xFFFFFFFF, True,  True,  0xFFFFFFFF),
    ]
    hits = []
    for name, w, poly, init, ri, ro, xo in catalogue:
        if width is not None and w != width:
            continue
        if all(_crc_core(d, w, poly, init, ri, ro, xo) == c for d, c in frames):
            hits.append(name)
    return hits


# ─────────────────────────────────────────────────────────────────────────────
# Bit-alignment aware search
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BitstreamCRCResult:
    """A CRC-32 hit located in a bit stream of unknown byte alignment."""
    found: bool
    bit_offset: int
    start_byte: int
    end_byte: int
    payload: bytes
    crc: int


def _crc32_table() -> np.ndarray:
    """256-entry table for reflected CRC-32/ISO-HDLC (polynomial 0xEDB88320)."""
    table = np.zeros(256, dtype=np.uint32)
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ 0xEDB88320 if c & 1 else c >> 1
        table[i] = c
    return table


_CRC32_TABLE = _crc32_table()


def _crc32_sweep_all_starts(data: np.ndarray,
                            min_payload_len: int,
                            max_payload_len: int,
                            row_end: np.ndarray | None = None) -> tuple[int, int] | None:
    """Find a CRC-32 framed window starting at *any* byte offset of ``data``.

    One CRC register is kept per start offset and all of them are advanced
    together, one payload byte per step, with a vectorised table lookup.  After
    ``L`` steps register ``s`` holds ``crc32(data[s:s+L])``, which is compared
    against the four bytes that follow it, read both big- and little-endian.

    Cost is ``max_payload_len`` vectorised steps over ``N`` registers, versus
    ``N * max_payload_len`` interpreted zlib calls for a per-start loop -- and
    every start offset in the buffer is covered, not just the first few.

    ``row_end`` lets several independent buffers be concatenated and swept in a
    single pass: register ``s`` is retired as soon as its window plus CRC field
    would run past ``row_end[s]``, so no window ever straddles two buffers.

    Returns ``(start, payload_len)`` of the hit with the shortest payload (then
    the lowest start), or None.  Results match ``zlib.crc32`` exactly.
    """
    N = len(data)
    if N < min_payload_len + 4:
        return None
    d = data.astype(np.uint32)
    starts = np.arange(N, dtype=np.int64)
    ends = np.full(N, N, dtype=np.int64) if row_end is None else np.asarray(row_end, dtype=np.int64)
    reg = np.full(N, 0xFFFFFFFF, dtype=np.uint32)
    for L in range(1, int(max_payload_len) + 1):
        alive = starts + L + 4 <= ends
        if not alive.any():
            break
        if not alive.all():
            starts, ends, reg = starts[alive], ends[alive], reg[alive]
        byte = d[starts + L - 1]
        reg = _CRC32_TABLE[(reg ^ byte) & 0xFF] ^ (reg >> np.uint32(8))
        if L < min_payload_len:
            continue
        crc = reg ^ np.uint32(0xFFFFFFFF)
        q = starts + L
        be = (d[q] << 24) | (d[q + 1] << 16) | (d[q + 2] << 8) | d[q + 3]
        le = d[q] | (d[q + 1] << 8) | (d[q + 2] << 16) | (d[q + 3] << 24)
        hit = np.flatnonzero((crc == be) | (crc == le))
        if hit.size:
            return int(starts[hit[0]]), L
    return None


def crc32_search_bitstream(bits: np.ndarray,
                           min_payload_len: int = 4,
                           max_payload_len: int = 512,
                           max_start_bytes: int | None = None) -> BitstreamCRCResult:
    """Search a raw bit stream for a CRC-32 framed payload at any bit alignment.

    Why this exists: a demodulated bit stream does not begin on a byte boundary.
    Timing recovery discards leading samples, the matched filter and the
    equaliser each add their own group delay, and the capture itself started at
    an arbitrary point in the transmission -- so the first recovered bit sits at
    an unknown offset within the transmitted byte.

    ``windowed_crc32_search`` sweeps *byte* offsets, which cannot recover from
    that: packing a stream misaligned by 1-7 bits produces bytes whose contents
    are a mixture of two transmitted bytes, and no byte-level offset fixes it
    (measured: it finds the frame at 1 of the 8 possible alignments).

    All 8 bit alignments are packed, concatenated, and swept in one vectorised
    pass by :func:`_crc32_sweep_all_starts`, which covers every start offset of
    every alignment.  ``max_start_bytes`` optionally limits each alignment to a
    leading prefix, for callers that need a strict time bound.

    Returns the hit with the shortest payload, deterministically.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if len(bits) < 8 * (min_payload_len + 4):
        return BitstreamCRCResult(False, 0, 0, 0, b"", 0)

    rows: list[np.ndarray] = []
    for bit_offset in range(8):
        tail = bits[bit_offset:]
        n_bytes = len(tail) // 8
        row = np.packbits(tail[: n_bytes * 8]) if n_bytes else np.zeros(0, dtype=np.uint8)
        if max_start_bytes is not None:
            row = row[: int(max_start_bytes) + int(max_payload_len) + 4]
        rows.append(row)

    lengths = np.array([len(r) for r in rows], dtype=np.int64)
    if lengths.max() < min_payload_len + 4:
        return BitstreamCRCResult(False, 0, 0, 0, b"", 0)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    data = np.concatenate(rows)
    row_end = np.repeat(offsets + lengths, lengths)

    hit = _crc32_sweep_all_starts(data, min_payload_len, max_payload_len, row_end=row_end)
    if hit is None:
        return BitstreamCRCResult(False, 0, 0, 0, b"", 0)
    g_start, L = hit
    bit_offset = int(np.searchsorted(offsets, g_start, side="right") - 1)
    s0 = int(g_start - offsets[bit_offset])
    payload = rows[bit_offset][s0:s0 + L].tobytes()
    return BitstreamCRCResult(
        found=True,
        bit_offset=bit_offset,
        start_byte=s0,
        end_byte=s0 + L + 4,
        payload=payload,
        crc=zlib.crc32(payload) & 0xFFFFFFFF,
    )
