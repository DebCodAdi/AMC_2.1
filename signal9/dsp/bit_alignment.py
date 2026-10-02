"""Readable byte hypotheses and explicit, reference-based BER measurement."""
from __future__ import annotations

import numpy as np


def align_bytes(bits: np.ndarray, max_bytes: int = 4096) -> dict:
    """Search byte phase, polarity and bit order; readability is not CRC evidence."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    best = None
    scores = []
    for offset in range(8):
        tail = bits[offset:offset + max_bytes * 8]
        tail = tail[:len(tail) // 8 * 8]
        if not len(tail):
            continue
        for inverted in (False, True):
            for bitorder in ("big", "little"):
                raw = np.packbits(tail ^ int(inverted), bitorder=bitorder).tobytes()
                text = raw.decode("utf-8", errors="replace")
                score = sum(c != "\ufffd" and (c.isprintable() or c in "\r\n\t")
                            for c in text) / max(1, len(text))
                candidate = dict(bit_offset=offset, inverted=inverted, bitorder=bitorder,
                                 printable_fraction=float(score), bytes_scored=len(raw))
                scores.append(score)
                if best is None or score > best["printable_fraction"]:
                    best = {**candidate, "raw": raw}
    if best is None:
        return {"bit_offset": 0, "inverted": False, "bitorder": "big", "raw": b"",
                "printable_fraction": 0.0, "bytes_scored": 0, "ambiguous": True}
    best["ambiguous"] = (best["bytes_scored"] < 16 or best["printable_fraction"] < 0.85
                         or sum(s >= best["printable_fraction"] - 0.02 for s in scores) > 1)
    return best


def reference_ber(bits: np.ndarray, reference: np.ndarray, offset: int = 0) -> dict:
    """Compare at an explicit alignment; never search for an optimistically low BER."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    reference = np.asarray(reference, dtype=np.uint8).ravel()
    if offset < 0 or np.any(bits > 1) or np.any(reference > 1):
        raise ValueError("Expected binary bits and a nonnegative offset")
    observed = bits[offset:offset + len(reference)]
    n = len(observed)
    errors = int(np.count_nonzero(observed != reference[:n]))
    return {"bits_compared": n, "bit_errors": errors, "ber": errors / n if n else None,
            "reference_bits": len(reference), "complete": n == len(reference) and n > 0,
            "bit_offset": offset}
