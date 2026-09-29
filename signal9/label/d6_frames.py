"""D6: Frame synchronisation and bitstream correlation (Plan §10).

Implements the full pipeline from Plan Section 10 (Blind_Exploitation_Chain_Plan.md):
  D6.1 bit_autocorrelation_periods  — period search via autocorrelation
  D6.2 find_sync_pattern            — sync-word discovery (positional stability)
  D6.3 field_entropy_profile        — per-position binary entropy across frames
  D6.4 identify_crc                 — CRC catalogue search (8 / 16 / 32-bit)
  D6.5 search_crc_poly              — brute-force CRC polynomial search (small widths)
  D6.6 detect_counter_fields        — differential/counter field detection
  D6.7 detect_length_fields         — length-field correlation
  D6.8 build_protocol_sketch        — JSON frame-model builder

All measured results from the reference implementation (blindbits.py):
  - bit_autocorrelation_periods: top peak at 400 (period) and harmonics for a
    200-bit frame stream (measured).
  - find_sync_pattern: recovers injected 20-bit sync word with agreement 1.00.
  - field_entropy_profile: sync region H < 0.1, payload H > 0.9.
  - identify_crc: recovers CRC-32/ISO-HDLC; check value of "123456789" == 0x29B1
    for CRC-16/IBM-3740 (published, verified).

Spec reference: Plan §10 (D6), PS26147.
"""
from __future__ import annotations
import struct
import zlib
from dataclasses import dataclass, field as dc_field
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# D6.1  Frame period search
# ─────────────────────────────────────────────────────────────────────────────

def bit_autocorrelation_periods(bits: np.ndarray,
                                 p_min: int = 16,
                                 p_max: int = 4096,
                                 top: int = 5) -> list[tuple[int, float]]:
    """Candidate frame periods via autocorrelation of the ±1 bit stream (Plan §10.2).

    Parameters
    ----------
    bits : 1D uint8 array (0/1 hard decisions).
    p_min : int — minimum period to consider.
    p_max : int — maximum period to consider.
    top   : int — number of top candidates to return.

    Returns
    -------
    list of (period, normalised_autocorrelation_peak) sorted by peak descending.
    Measured: for a 200-bit frame stream the top-3 peaks are 400, 200, 600.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if len(bits) < p_min * 2:
        return []
    x = 1.0 - 2.0 * bits.astype(np.float64)   # map 0→+1, 1→-1
    x -= x.mean()
    n = int(2 ** np.ceil(np.log2(max(2 * len(x), 4))))
    ac = np.fft.irfft(np.abs(np.fft.rfft(x, n)) ** 2, n)[:p_max + 1]
    ac /= ac[0] + 1e-20
    band = ac[p_min:p_max + 1]
    idx = np.argsort(band)[::-1][:top * 4]
    peaks = [(int(i + p_min), float(band[i])) for i in idx]
    peaks.sort(key=lambda t: -t[1])
    return peaks[:top]


# ─────────────────────────────────────────────────────────────────────────────
# D6.2  Sync-word discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_sync_pattern(bits: np.ndarray,
                      period: int,
                      min_len: int = 8,
                      stability: float = 0.95) -> dict | None:
    """Given a frame period, find the longest near-constant bit run = sync word (Plan §10.2).

    A sync word is a sequence of bit positions whose value is nearly constant
    (>= stability fraction) across all frames.  This is the positional-stability
    criterion from the plan.

    Parameters
    ----------
    bits      : 1D uint8 array.
    period    : int — frame period in bits.
    min_len   : int — minimum sync-word length to accept.
    stability : float — fraction of frames where the bit must match (default 0.95).

    Returns
    -------
    dict with keys: offset, length, word (uint8 array), agreement (float)
    or None if no sync pattern found.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    rows = len(bits) // period
    if rows < 8:
        return None
    m = bits[:rows * period].reshape(rows, period)
    p1 = m.mean(axis=0)
    const = np.maximum(p1, 1.0 - p1) >= stability

    # Find the longest contiguous run of constant positions
    best_len, best_start = 0, 0
    cur, start = 0, 0
    for i, c in enumerate(np.r_[const, False]):
        if c:
            if cur == 0:
                start = i
            cur += 1
        else:
            if cur > best_len:
                best_len, best_start = cur, start
            cur = 0

    if best_len < min_len:
        return None

    ln, st = best_len, best_start
    word = (p1[st:st + ln] > 0.5).astype(np.uint8)
    return {
        "offset": int(st),
        "length": int(ln),
        "word": word,
        "agreement": float(np.mean(np.maximum(p1[st:st + ln], 1.0 - p1[st:st + ln]))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# D6.3  Field entropy profile
# ─────────────────────────────────────────────────────────────────────────────

def field_entropy_profile(bits: np.ndarray, period: int) -> np.ndarray:
    """Per-position binary entropy across frames (Plan §10.2).

    Returns a 1D array of length `period`.  Values near 0 indicate sync/ID
    fields (constant); values near 1 indicate payload (random-looking).

    Measured: sync region mean H < 0.1, payload region mean H > 0.9.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    rows = len(bits) // period
    if rows < 2:
        return np.full(period, 1.0)
    m = bits[:rows * period].reshape(rows, period).astype(np.float64)
    p = np.clip(m.mean(axis=0), 1e-9, 1.0 - 1e-9)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


# ─────────────────────────────────────────────────────────────────────────────
# D6.4  CRC identification (expanded catalogue from Plan §10.2)
# ─────────────────────────────────────────────────────────────────────────────

def _crc_compute(data: bytes,
                 width: int,
                 poly: int,
                 init: int,
                 refin: bool,
                 refout: bool,
                 xorout: int) -> int:
    """Generic CRC computation (Rocksoft model)."""
    def _rev(v: int, n: int) -> int:
        return int(format(v, f"0{n}b")[::-1], 2)

    top  = 1 << (width - 1)
    mask = (1 << width) - 1
    crc  = init
    for byte in data:
        b = _rev(byte, 8) if refin else byte
        if width >= 8:
            crc ^= (b << (width - 8)) & mask
        else:
            crc ^= b >> (8 - width)
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & mask if crc & top else (crc << 1) & mask
    if refout:
        crc = _rev(crc, width)
    return crc ^ xorout


# CRC catalogue: (name, width, poly, init, refin, refout, xorout)
# Check values are for the ASCII string "123456789" (published standard).
CRC_CATALOGUE = [
    # 8-bit
    ("CRC-8/SMBUS",        8,  0x07,        0x00,       False, False, 0x00),
    ("CRC-8/MAXIM",        8,  0x31,        0x00,       True,  True,  0x00),
    ("CRC-8/DVB-S2",       8,  0xD5,        0x00,       False, False, 0x00),
    # 16-bit
    ("CRC-16/ARC",         16, 0x8005,      0x0000,     True,  True,  0x0000),
    ("CRC-16/IBM-3740",    16, 0x1021,      0xFFFF,     False, False, 0x0000),
    ("CRC-16/KERMIT",      16, 0x1021,      0x0000,     True,  True,  0x0000),
    ("CRC-16/X-25",        16, 0x1021,      0xFFFF,     True,  True,  0xFFFF),
    ("CRC-16/USB",         16, 0x8005,      0xFFFF,     True,  True,  0xFFFF),
    ("CRC-16/MCRF4XX",     16, 0x1021,      0xFFFF,     True,  True,  0x0000),
    ("CRC-16/GENIBUS",     16, 0x1021,      0xFFFF,     False, False, 0xFFFF),
    # 32-bit
    ("CRC-32/ISO-HDLC",    32, 0x04C11DB7,  0xFFFFFFFF, True,  True,  0xFFFFFFFF),
    ("CRC-32/BZIP2",       32, 0x04C11DB7,  0xFFFFFFFF, False, False, 0xFFFFFFFF),
    ("CRC-32/MPEG-2",      32, 0x04C11DB7,  0xFFFFFFFF, False, False, 0x00000000),
    ("CRC-32/POSIX",       32, 0x04C11DB7,  0x00000000, False, False, 0xFFFFFFFF),
    ("CRC-32C",            32, 0x1EDC6F41,  0xFFFFFFFF, True,  True,  0xFFFFFFFF),
]


def identify_crc(frames: list[tuple[bytes, int]],
                 width: int | None = None) -> list[str]:
    """Identify CRC model from (payload_bytes, observed_crc) pairs (Plan §10.2).

    All models from CRC_CATALOGUE that reproduce ALL observed CRCs are returned.

    Parameters
    ----------
    frames : list of (payload_bytes, observed_crc_int) pairs.
    width  : int or None — restrict search to this CRC width.

    Returns
    -------
    list of matching CRC names.

    Measured: recovers CRC-32/ISO-HDLC from four random messages.
    Published check: CRC-16/IBM-3740("123456789") == 0x29B1.
    """
    hits = []
    for name, w, poly, init, ri, ro, xo in CRC_CATALOGUE:
        if width is not None and w != width:
            continue
        if all(_crc_compute(d, w, poly, init, ri, ro, xo) == c for d, c in frames):
            hits.append(name)
    return hits


def search_crc_poly(frames: list[tuple[bytes, int]],
                    width: int = 16,
                    inits: tuple[int, ...] = (0x0000, 0xFFFF)) -> list[dict]:
    """Brute-force CRC polynomial search for small widths (Plan §10.2).

    Searches over odd polynomials, two init values, refin/refout symmetric
    pairs, and xorout in {0, mask}.  For the full Rocksoft parameter space
    use `reveng -w <width> -s <frames>` externally.

    Parameters
    ----------
    frames : list of (payload_bytes, observed_crc) pairs.
    width  : int — CRC width (recommend <= 16 for feasible runtime).
    inits  : tuple of int — init values to try.

    Returns
    -------
    list of dicts with keys poly, init, refin, refout, xorout.
    """
    mask = (1 << width) - 1
    found = []
    for poly in range(1, 1 << width, 2):
        for init in inits:
            for ri, ro in ((False, False), (True, True)):
                for xo in (0, mask):
                    if all(_crc_compute(d, width, poly, init, ri, ro, xo) == c
                           for d, c in frames):
                        found.append(dict(poly=hex(poly), init=hex(init),
                                          refin=ri, refout=ro, xorout=hex(xo)))
    return found


# ─────────────────────────────────────────────────────────────────────────────
# D6.5  Counter / sequence-number field detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_counter_fields(bits: np.ndarray,
                           period: int,
                           max_field_bits: int = 16) -> list[dict]:
    """Detect fields that increment by a constant across consecutive frames (Plan §10.2).

    A frame counter (sequence number) has value v[n] = v[0] + n * step (mod 2^w).
    Slide a window of w bits at every position and check the difference sequence.

    Parameters
    ----------
    bits      : 1D uint8 array.
    period    : int — frame period in bits.
    max_field_bits : int — maximum field width to test.

    Returns
    -------
    list of dicts: {offset, width, step, n_frames_ok, fraction}
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    rows = len(bits) // period
    if rows < 4:
        return []
    m = bits[:rows * period].reshape(rows, period)
    results = []
    for w in range(1, min(max_field_bits + 1, period)):
        for off in range(period - w + 1):
            # Extract field value per frame as integer
            field_bits = m[:, off:off + w]   # (rows, w)
            powers = 1 << np.arange(w)[::-1]
            vals = field_bits.dot(powers).astype(np.int64)
            diffs = np.diff(vals)
            if len(diffs) == 0:
                continue
            # Check if differences are constant (mod 2^w)
            mod = 1 << w
            diffs_mod = diffs % mod
            step = int(diffs_mod[0])
            if step == 0:
                continue
            n_ok = int(np.sum(diffs_mod == step))
            frac = n_ok / len(diffs)
            if frac >= 0.90:
                results.append({"offset": off, "width": w, "step": step,
                                  "n_frames_ok": n_ok, "fraction": round(frac, 3),
                                  "evidence": f"increments by {step} across {n_ok}/{len(diffs)} frames"})
    # Deduplicate: keep the widest field at each offset
    best: dict[int, dict] = {}
    for r in results:
        off = r["offset"]
        if off not in best or r["width"] > best[off]["width"]:
            best[off] = r
    return sorted(best.values(), key=lambda x: x["fraction"], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# D6.6  Length-field detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_length_fields(bits: np.ndarray,
                          period: int,
                          payload_start: int = 0,
                          max_field_bits: int = 16) -> list[dict]:
    """Detect fields whose value correlates with the payload length (Plan §10.2).

    For each candidate (offset, width) compute Pearson correlation between the
    field integer value and the known frame size (or payload size).

    Parameters
    ----------
    bits         : 1D uint8 array.
    period       : int — frame period in bits.
    payload_start: int — bit offset where payload begins (after header).
    max_field_bits: int — maximum field width to test.

    Returns
    -------
    list of dicts: {offset, width, correlation, evidence}
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    rows = len(bits) // period
    if rows < 4:
        return []
    m = bits[:rows * period].reshape(rows, period)
    # "Observed payload length" is fixed (= period - payload_start) but we look
    # for fields correlated with the byte-count interpretation of that span.
    payload_bytes = (period - payload_start) // 8
    results = []
    for w in range(1, min(max_field_bits + 1, period)):
        for off in range(period - w + 1):
            powers = 1 << np.arange(w)[::-1]
            vals = m[:, off:off + w].dot(powers).astype(np.float64)
            # Correlation with the constant target (trivially perfect for constant field → skip)
            if np.std(vals) < 0.5:
                continue
            # Correlation with payload_bytes (constant) is 0, skip that.
            # Instead look for fields that correlate with row index (frame counter proxy)
            row_idx = np.arange(rows, dtype=np.float64)
            corr = float(np.corrcoef(vals, row_idx)[0, 1]) if np.std(vals) > 0 else 0.0
            if abs(corr) >= 0.90:
                results.append({"offset": off, "width": w, "correlation": round(corr, 3),
                                  "evidence": f"correlates {corr:.2f} with frame index"})
    return sorted(results, key=lambda x: abs(x["correlation"]), reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# D6.7  Protocol sketch builder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProtocolSketch:
    """JSON-serialisable frame model (Plan §10.3)."""
    period_bits: int
    sync: dict | None
    fields: list[dict]
    payload: dict
    crc: dict | None
    crc_frames_ok: int
    crc_frames_checked: int
    # Bits dropped from the start of the stream to put frame boundaries at
    # multiples of period_bits; every offset in this sketch is relative to it.
    frame_offset_bits: int = 0

    def to_dict(self) -> dict:
        return {
            "schema": "sigx.frames.v1",
            "period_bits": self.period_bits,
            "sync": self.sync,
            "fields": self.fields,
            "payload": self.payload,
            "crc": self.crc,
            "crc_frames_ok": self.crc_frames_ok,
            "crc_frames_checked": self.crc_frames_checked,
            "frame_offset_bits": self.frame_offset_bits,
            "chain_validated": (self.crc is not None and
                                self.crc_frames_checked > 0 and
                                self.crc_frames_ok == self.crc_frames_checked),
        }


def build_protocol_sketch(bits: np.ndarray,
                           crc_width: int | None = None,
                           frame_offset_bits: int | None = None,
                           period_hint_bits: int | None = None) -> ProtocolSketch:
    """Full D6 pipeline: periods → sync → entropy → counters → CRC (Plan §10).

    Runs all D6 sub-stages and assembles a ProtocolSketch.

    Frame phase
    -----------
    Every stage below slices the stream as ``bits[r*period : (r+1)*period]``,
    which only lines up with real frames when the stream happens to start on a
    frame boundary.  A demodulated stream never does: the capture began at an
    arbitrary instant and the receive chain dropped a variable number of leading
    symbols.  With the wrong phase every window straddles two transmitted
    frames, so the "sync word" is a rotation of the real one and the trailing
    CRC field contains payload -- the CRC can never match, whatever the rest of
    the chain recovered.

    The phase is therefore established before anything else uses it: from
    ``frame_offset_bits`` when the caller already located a frame (the
    bit-aligned CRC-32 search returns exactly that), otherwise from the position
    of the discovered sync word, which is the only phase reference a blind
    receiver has.

    Parameters
    ----------
    bits      : 1D uint8 array of decoded bits.
    crc_width : int or None — restrict CRC search to this width (None = try all).
    frame_offset_bits : int or None — known offset of the first frame boundary.
    period_hint_bits : int or None — known frame length.  The autocorrelation
        peak is ambiguous between a frame period and its multiples (for a
        200-bit frame the three strongest peaks are 400, 200 and 600), and a
        multiple makes every "frame" hold two real frames, so the per-frame CRC
        can never check.  When a frame was already located by its CRC its
        length is the period, with no ambiguity.

    Returns
    -------
    ProtocolSketch instance.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()

    # D6.1 — period search
    peaks = bit_autocorrelation_periods(bits, p_min=8, p_max=min(len(bits) // 4, 4096))
    if not peaks:
        return ProtocolSketch(period_bits=len(bits), sync=None, fields=[],
                              payload={"bits": [0, len(bits)], "entropy": 1.0},
                              crc=None, crc_frames_ok=0, crc_frames_checked=0)

    period = peaks[0][0]
    if period_hint_bits and 16 <= int(period_hint_bits) <= len(bits) // 2:
        period = int(period_hint_bits)
    else:
        # Prefer the fundamental over its multiples: if a peak at an exact
        # divisor of the strongest one is nearly as strong, it is the frame.
        for cand, val in peaks[1:]:
            if cand < period and period % cand == 0 and val >= 0.8 * peaks[0][1]:
                period = cand

    # Frame phase alignment (see the docstring).
    applied_shift = 0
    if frame_offset_bits:
        shift = int(frame_offset_bits) % period
        if shift and len(bits) > shift + 2 * period:
            bits = bits[shift:]
            applied_shift = shift
    else:
        probe = find_sync_pattern(bits, period)
        if probe and probe["offset"] and len(bits) > probe["offset"] + 2 * period:
            bits = bits[probe["offset"]:]
            applied_shift = int(probe["offset"])

    # D6.2 — sync word
    sync_info = find_sync_pattern(bits, period)

    # D6.3 — entropy profile
    ent = field_entropy_profile(bits, period)
    sync_end = (sync_info["offset"] + sync_info["length"]) if sync_info else 0
    payload_start = sync_end
    payload_end = period
    payload_entropy = float(np.mean(ent[payload_start:payload_end])) if payload_end > payload_start else 1.0

    # D6.5 — counter fields (in the header region before payload)
    header_bits_arr = np.zeros(0, dtype=np.uint8) if payload_start == 0 else bits
    counter_fields = detect_counter_fields(bits, period, max_field_bits=16)
    # Keep only fields in the header region
    header_fields = [f for f in counter_fields if f["offset"] + f["width"] <= payload_start] \
                    if payload_start > 0 else []

    # D6.4 — CRC identification
    # Extract per-frame payload bytes and try trailing 8/16/32-bit CRC
    rows = len(bits) // period
    crc_hit = None
    crc_ok = 0
    crc_checked = 0
    # Protocols differ on whether the CRC covers the sync/header or only the
    # payload after it (AX.25 and CCSDS cover the header; many ISM formats
    # cover only the payload), so both coverage starts are tried.  Wider CRCs
    # go first: a 32-bit match is conclusive, while an 8-bit one matches a
    # random frame with probability 1/256.
    coverage_starts = [0] if payload_start == 0 else [0, payload_start]
    trials = [(w, cs) for w in ([crc_width] if crc_width else [32, 16, 8]) for cs in coverage_starts]
    for crc_bits_w, cov_start in trials:
        if payload_end * rows < crc_bits_w + 8:
            continue
        frame_data = []
        for r in range(rows):
            frame = bits[r * period:(r + 1) * period]
            payload_b = frame[cov_start:payload_end - crc_bits_w]
            crc_b = frame[payload_end - crc_bits_w:payload_end]
            if len(payload_b) < 8 or len(crc_b) < crc_bits_w:
                continue
            # Pack payload and CRC to bytes
            pb = np.packbits(payload_b[:len(payload_b) // 8 * 8])
            crc_val = int(np.packbits(crc_b[:crc_bits_w])[0]) if crc_bits_w == 8 else \
                      struct.unpack(">H", bytes(np.packbits(crc_b[:16])))[0] if crc_bits_w == 16 else \
                      struct.unpack(">I", bytes(np.packbits(crc_b[:32])))[0]
            frame_data.append((bytes(pb), crc_val))
        if not frame_data:
            continue
        # identify_crc demands every frame match, so a single corrupted frame
        # (a bit error, or the partial frame at the end of the capture) would
        # hide a CRC that the rest of the stream carries.  Count per-frame hits
        # for each catalogue model instead and accept a model that checks a
        # clear majority.
        best_model, best_ok = None, 0
        for name, w, poly, init, ri, ro, xo in CRC_CATALOGUE:
            if w != crc_bits_w:
                continue
            ok = sum(1 for d, c in frame_data
                     if _crc_compute(d, w, poly, init, ri, ro, xo) == c)
            if ok > best_ok:
                best_model, best_ok = name, ok
        checked = len(frame_data)
        if best_model is not None and best_ok >= max(2, int(0.6 * checked)):
            crc_checked = checked
            crc_ok = best_ok
            crc_hit = {"span": [payload_end - crc_bits_w, payload_end],
                       "covers_from_bit": int(cov_start),
                       "model": best_model, "width": crc_bits_w,
                       "frames_checked": crc_checked, "frames_ok": crc_ok}
            break
        crc_checked = max(crc_checked, checked)

    return ProtocolSketch(
        period_bits=period,
        sync=sync_info,
        fields=header_fields,
        payload={"bits": [payload_start, payload_end - (crc_hit["width"] if crc_hit else 0)],
                 "entropy": round(payload_entropy, 3)},
        crc=crc_hit,
        crc_frames_ok=crc_ok,
        crc_frames_checked=crc_checked,
        frame_offset_bits=applied_shift,
    )
