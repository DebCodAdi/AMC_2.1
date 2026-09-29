"""blindbits.py — reference implementations for the demod→bits→FEC→frames chain.

NumPy/SciPy only, so every function runs inside a DataLoader worker or a GUI thread.
Each function is unit-tested in test_blindbits.py. Nothing here assumes the transmitter
is cooperative: every estimator returns (value, quality) so a search engine can score it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from scipy import signal as sps

# =============================================================== 1. symbol-rate / timing

def estimate_symbol_rate(x: np.ndarray, fs: float, f_min: float = 0.0,
                         f_max: float | None = None, nfft: int | None = None):
    """Cyclostationary symbol-rate estimate: spectral line of |x|^2 (and of |x|^4 for CPM).

    Returns (rate_hz, quality) where quality = peak-to-median of the line spectrum.
    Works for linear modulations with excess bandwidth > 0 and for FSK/CPM (|x|^4 branch).
    """
    f_max = f_max or fs / 2
    out = []
    for p, sig in ((2, np.abs(x) ** 2), (4, np.abs(x) ** 4)):
        s = sig - sig.mean()
        n = nfft or int(2 ** np.ceil(np.log2(len(s))))
        S = np.abs(np.fft.rfft(s * np.hanning(len(s)), n=n))
        f = np.fft.rfftfreq(n, 1 / fs)
        band = (f > max(f_min, 4 * fs / n)) & (f < f_max)
        if not band.any():
            continue
        Sb, fb = S[band], f[band]
        i = int(np.argmax(Sb))
        q = float(Sb[i] / (np.median(Sb) + 1e-20))
        out.append((float(fb[i]), q, p))
    if not out:
        return 0.0, 0.0
    rate, q, _ = max(out, key=lambda t: t[1])
    return rate, q


def gardner_timing_recovery(x: np.ndarray, sps_in: float, loop_bw: float = 0.002,
                            damping: float = 1.0):
    """Gardner TED + 2nd-order loop, cubic-Lagrange interpolation.

    x is matched-filtered at sps_in >= 2 samples/symbol; returns 1 sample/symbol.
    Gardner's error is independent of carrier phase, so this runs BEFORE carrier recovery.
    Two details that decide whether it works: the error is normalised by the signal power
    (otherwise the loop gain scales with AGC state) and the sign convention below is the one
    that pulls toward the peak of the matched-filter output -- with the sign flipped the loop
    locks onto the zero crossings and the EVM is ~18 dB worse (measured).
    """
    theta = 2 * np.pi * loop_bw
    k1 = (4 * damping * theta) / (1 + 2 * damping * theta + theta ** 2)
    k2 = (4 * theta ** 2) / (1 + 2 * damping * theta + theta ** 2)
    pw = float(np.mean(np.abs(x) ** 2)) + 1e-12

    def interp(pos: float) -> complex:
        i = int(np.floor(pos))
        mu = pos - i
        if i < 1 or i > len(x) - 3:
            return 0j
        p = x[i - 1:i + 3]
        c0 = -mu * (mu - 1) * (mu - 2) / 6.0
        c1 = (mu + 1) * (mu - 1) * (mu - 2) / 2.0
        c2 = -(mu + 1) * mu * (mu - 2) / 2.0
        c3 = (mu + 1) * mu * (mu - 1) / 6.0
        return c0 * p[0] + c1 * p[1] + c2 * p[2] + c3 * p[3]

    pos, period, freq = 2.0 * sps_in, float(sps_in), 0.0
    prev = interp(float(sps_in))
    out = []
    while pos + period + 3 < len(x) - 3:
        mid = interp(pos - period / 2.0)
        cur = interp(pos)
        e = -(np.real(cur - prev) * np.real(mid) + np.imag(cur - prev) * np.imag(mid)) / pw
        e = float(np.clip(e, -1.0, 1.0))
        freq += k2 * e
        period = float(np.clip(sps_in + freq + k1 * e, 0.9 * sps_in, 1.1 * sps_in))
        out.append(cur)
        prev = cur
        pos += period
    return np.asarray(out, dtype=np.complex64)


# =============================================================== 2. carrier / equaliser

def cfo_mth_power(x: np.ndarray, fs: float, M: int = 4):
    """Carrier offset for M-PSK via the M-th power spectral line. Returns (hz, quality)."""
    y = x ** M
    n = int(2 ** np.ceil(np.log2(len(y))))
    S = np.abs(np.fft.fft(y, n=n))
    f = np.fft.fftfreq(n, 1 / fs)
    i = int(np.argmax(S))
    q = float(S[i] / (np.median(S) + 1e-20))
    return float(f[i] / M), q


def costas_loop(x: np.ndarray, order: int = 4, loop_bw: float = 0.01):
    """Decision-directed Costas loop for BPSK(2)/QPSK(4)/8PSK(8). Returns phase-locked symbols."""
    zeta, theta = 1.0, 2 * np.pi * loop_bw
    k1 = (4 * zeta * theta) / (1 + 2 * zeta * theta + theta ** 2)
    k2 = (4 * theta ** 2) / (1 + 2 * zeta * theta + theta ** 2)
    phase = freq = 0.0
    out = np.empty(len(x), dtype=np.complex64)
    for n, s in enumerate(x):
        y = s * np.exp(-1j * phase)
        out[n] = y
        if order == 2:
            err = np.real(y) * np.imag(y)
        elif order == 4:
            err = (np.sign(np.real(y)) * np.imag(y) - np.sign(np.imag(y)) * np.real(y))
        else:
            err = np.angle(y ** order) / order
        err = float(np.clip(err, -1.0, 1.0))
        freq += k2 * err
        phase += freq + k1 * err
    return out


def cma_equalize(x: np.ndarray, n_taps: int = 11, mu: float = 1e-3, R2: float = 1.0):
    """Constant-modulus blind equaliser (Godard p=2). Good for PSK/FSK; for QAM switch to
    MMA/decision-directed after the CMA has opened the eye."""
    w = np.zeros(n_taps, dtype=np.complex64)
    w[n_taps // 2] = 1.0
    out = np.empty(len(x) - n_taps, dtype=np.complex64)
    for n in range(len(out)):
        u = x[n:n + n_taps][::-1]
        y = np.vdot(w.conj(), u)
        out[n] = y
        w += mu * (R2 - abs(y) ** 2) * y * np.conj(u)
    return out


# =============================================================== 3. symbols → LLRs

CONSTELLATIONS = {
    "BPSK": (np.array([-1, 1], dtype=complex), ["0", "1"]),
    "QPSK": (np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2),
             ["00", "01", "11", "10"]),                       # Gray
    "8PSK": (np.exp(1j * 2 * np.pi * np.arange(8) / 8),
             ["000", "001", "011", "010", "110", "111", "101", "100"]),
}


def qam_constellation(m: int):
    """Square M-QAM with Gray labelling on I and Q, unit average power."""
    k = int(np.log2(m))
    assert k % 2 == 0, "square QAM only"
    side = int(np.sqrt(m))
    g = [i ^ (i >> 1) for i in range(side)]                   # Gray code
    levels = np.arange(side) * 2 - (side - 1)
    pts, labels = [], []
    for qi, q in enumerate(levels):
        for ii, i in enumerate(levels):
            pts.append(i + 1j * q)
            labels.append(format(g[ii], f"0{k//2}b") + format(g[qi], f"0{k//2}b"))
    pts = np.asarray(pts)
    pts = pts / np.sqrt(np.mean(np.abs(pts) ** 2))
    return pts, labels


def symbols_to_llr(sym: np.ndarray, points: np.ndarray, labels: list[str],
                   noise_var: float | None = None) -> np.ndarray:
    """Max-log LLRs, bit 0 first within each symbol. Positive LLR == bit 0 more likely.

    noise_var is estimated from the residual to the nearest constellation point when not given
    (this is the EVM-based estimate; it is biased at very low SNR but adequate for decoding).
    """
    sym = sym / np.sqrt(np.mean(np.abs(sym) ** 2) + 1e-20)
    d2 = np.abs(sym[:, None] - points[None, :]) ** 2
    if noise_var is None:
        noise_var = max(float(np.mean(np.min(d2, axis=1))) / 2.0, 1e-6)
    nbits = len(labels[0])
    bits0 = [np.array([i for i, L in enumerate(labels) if L[b] == "0"]) for b in range(nbits)]
    bits1 = [np.array([i for i, L in enumerate(labels) if L[b] == "1"]) for b in range(nbits)]
    llr = np.empty((len(sym), nbits), dtype=np.float32)
    for b in range(nbits):
        llr[:, b] = (d2[:, bits1[b]].min(axis=1) - d2[:, bits0[b]].min(axis=1)) / noise_var
    return llr.reshape(-1)


def evm_db(sym: np.ndarray, points: np.ndarray) -> float:
    sym = sym / np.sqrt(np.mean(np.abs(sym) ** 2) + 1e-20)
    d2 = np.min(np.abs(sym[:, None] - points[None, :]) ** 2, axis=1)
    return float(10 * np.log10(np.mean(d2) + 1e-20))


# =============================================================== 4. bit-level normalisation

def llr_to_bits(llr: np.ndarray) -> np.ndarray:
    return (llr < 0).astype(np.uint8)          # LLR>0 ⇒ bit 0


def differential_decode(bits: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(bits[1:], bits[:-1])


def nrzi_decode(bits: np.ndarray, invert: bool = False) -> np.ndarray:
    d = np.bitwise_xor(bits[1:], bits[:-1])
    return d ^ 1 if not invert else d          # NRZI: transition == 0 in most standards


def manchester_decode(bits: np.ndarray, ieee: bool = True):
    """Returns (bits, valid_fraction). IEEE 802.3: 01 -> 1, 10 -> 0."""
    b = bits[: len(bits) // 2 * 2].reshape(-1, 2)
    valid = (b[:, 0] != b[:, 1])
    out = b[:, 1] if ieee else b[:, 0]
    return out.astype(np.uint8), float(valid.mean())


# =============================================================== 5. GF(2) linear algebra

def gf2_rank(mat: np.ndarray) -> int:
    """Rank over GF(2) of a 0/1 uint8 matrix (Gaussian elimination, bit-packed rows)."""
    m = np.packbits(mat.astype(np.uint8), axis=1)
    rows, rank = list(m), 0
    ncols = mat.shape[1]
    for col in range(ncols):
        byte, bit = col >> 3, 7 - (col & 7)
        piv = None
        for i in range(rank, len(rows)):
            if rows[i][byte] >> bit & 1:
                piv = i
                break
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for i in range(len(rows)):
            if i != rank and (rows[i][byte] >> bit & 1):
                rows[i] = rows[i] ^ rows[rank]
        rank += 1
        if rank == len(rows):
            break
    return rank


def rank_profile(bits: np.ndarray, n_min: int, n_max: int, n_rows: int = 200,
                 offset: int = 0) -> dict[int, float]:
    """Rank ratio rank(M_n)/n for each candidate row length n (Barbier/Cluzeau rank criterion).

    A codeword length n (or a multiple) shows a rank ratio ≈ k/n < 1; uncoded or wrong n gives ≈ 1.
    """
    prof = {}
    for n in range(n_min, n_max + 1):
        rows = min(n_rows, (len(bits) - offset) // n)
        if rows < n + 5:                                   # need more rows than columns
            continue
        m = bits[offset:offset + rows * n].reshape(rows, n)
        prof[n] = gf2_rank(m) / n
    return prof


def detect_code_length(bits: np.ndarray, n_min: int = 4, n_max: int = 64, n_rows: int = 300):
    """Returns (n_hat, k_hat, offset_hat, rank_ratio) or None. Scans offsets 0..n-1."""
    best = None
    for n in range(n_min, n_max + 1):
        rows = min(n_rows, len(bits) // n - 1)
        if rows < n + 5:
            continue
        for off in range(n):
            m = bits[off:off + rows * n].reshape(rows, n)
            r = gf2_rank(m)
            ratio = r / n
            if ratio < 0.98 and (best is None or ratio < best[3]):
                best = (n, r, off, ratio)
    return best


# =============================================================== 6. interleaver detection

def block_interleave(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Write rows-first, read columns-first (classic block interleaver)."""
    n = len(bits) // (rows * cols) * (rows * cols)
    b = bits[:n].reshape(-1, rows, cols)
    return b.transpose(0, 2, 1).reshape(-1)


def block_deinterleave(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    n = len(bits) // (rows * cols) * (rows * cols)
    b = bits[:n].reshape(-1, cols, rows)
    return b.transpose(0, 2, 1).reshape(-1)


def detect_block_interleaver(bits: np.ndarray, code_n: int, depth_max: int = 32,
                             n_rows: int = 300):
    """Depth search for a block interleaver in front of an (n,k) block code.

    For the true depth D the de-interleaved stream regains the code's rank deficiency, so we
    de-interleave under each hypothesis and score with the rank criterion (lower is better).
    """
    scores = {}
    for d in range(2, depth_max + 1):
        de = block_deinterleave(bits, d, code_n)
        rows = min(n_rows, len(de) // code_n)
        if rows < code_n + 5:
            continue
        m = de[:rows * code_n].reshape(rows, code_n)
        scores[d] = gf2_rank(m) / code_n
    if not scores:
        return None, {}
    best = min(scores, key=scores.get)
    return best, scores


# =============================================================== 7. scrambler / LFSR

def berlekamp_massey(seq: np.ndarray):
    """LFSR synthesis over GF(2). Returns (L, connection_poly) with poly[0]=1."""
    s = seq.astype(np.uint8)
    n = len(s)
    C = np.zeros(n, dtype=np.uint8); C[0] = 1
    B = C.copy()
    L, m = 0, 1
    for N in range(n):
        d = s[N]
        for i in range(1, L + 1):
            d ^= C[i] & s[N - i]
        if d == 0:
            m += 1
        elif 2 * L <= N:
            T = C.copy()
            C[m:] ^= B[: n - m]
            L, B, m = N + 1 - L, T, 1
        else:
            C[m:] ^= B[: n - m]
            m += 1
    return L, C[: L + 1]


def linear_complexity_profile(bits: np.ndarray, window: int = 512, step: int = 256):
    """Windowed linear complexity. L ≈ window/2 ⇒ random/coded; L ≪ window/2 ⇒ LFSR fill
    (idle pattern, PRBS test sequence, or an unscrambled constant under an additive scrambler)."""
    out = []
    for s in range(0, max(len(bits) - window, 0) + 1, step):
        L, _ = berlekamp_massey(bits[s:s + window])
        out.append(L)
    return np.asarray(out)


def lfsr_sequence(poly: np.ndarray, state: np.ndarray, n: int) -> np.ndarray:
    """Sequence of the LFSR whose connection polynomial is `poly` (poly[0]=1, taps at poly[i]=1).

    Convention matches berlekamp_massey(): s[k] = XOR_{i in taps} s[k-i], seeded with `state`
    (L = len(state) initial bits). Use this to regenerate and strip an additive scrambler.
    """
    taps = np.nonzero(np.asarray(poly)[1:])[0] + 1
    L = len(state)
    out = np.zeros(n, dtype=np.uint8)
    m = min(L, n)
    out[:m] = np.asarray(state, dtype=np.uint8)[:m]
    for k in range(L, n):
        acc = 0
        for t in taps:
            acc ^= int(out[k - t])
        out[k] = acc
    return out


def try_descramble_additive(bits: np.ndarray, poly: np.ndarray, state: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(bits, lfsr_sequence(poly, state, len(bits)))


def self_sync_descramble(bits: np.ndarray, taps: list[int]) -> np.ndarray:
    """Multiplicative (self-synchronising) descrambler: y[n] = x[n] xor sum x[n-t].
    taps are the feedback positions, e.g. V.34 / DVB uses [14, 15] and [15, 17] style pairs."""
    y = bits.copy()
    for n in range(len(bits)):
        acc = bits[n]
        for t in taps:
            if n - t >= 0:
                acc ^= bits[n - t]
        y[n] = acc
    return y


# =============================================================== 8. frame / field analysis

def bit_autocorrelation_periods(bits: np.ndarray, p_min: int = 16, p_max: int = 4096,
                                top: int = 5):
    """Candidate frame periods from the autocorrelation of the ±1 bit stream."""
    x = 1.0 - 2.0 * bits.astype(np.float64)
    x -= x.mean()
    n = int(2 ** np.ceil(np.log2(2 * len(x))))
    ac = np.fft.irfft(np.abs(np.fft.rfft(x, n)) ** 2, n)[: p_max + 1]
    ac /= ac[0] + 1e-20
    band = ac[p_min:p_max + 1]
    idx = np.argsort(band)[::-1][: top * 4]
    peaks = [(int(i + p_min), float(band[i])) for i in idx]
    peaks.sort(key=lambda t: -t[1])
    return peaks[:top]


def find_sync_pattern(bits: np.ndarray, period: int, min_len: int = 8, stability: float = 0.95):
    """Given a frame period, find the longest run of near-constant bit positions = sync word."""
    rows = len(bits) // period
    if rows < 8:
        return None
    m = bits[: rows * period].reshape(rows, period)
    p1 = m.mean(axis=0)
    const = np.maximum(p1, 1 - p1) >= stability
    best, cur, start = (0, 0), 0, 0
    for i, c in enumerate(np.r_[const, False]):
        if c:
            if cur == 0:
                start = i
            cur += 1
        else:
            if cur > best[0]:
                best = (cur, start)
            cur = 0
    if best[0] < min_len:
        return None
    ln, st = best
    word = (p1[st:st + ln] > 0.5).astype(np.uint8)
    return dict(offset=int(st), length=int(ln), word=word,
                agreement=float(np.mean(np.maximum(p1[st:st + ln], 1 - p1[st:st + ln]))))


def field_entropy_profile(bits: np.ndarray, period: int) -> np.ndarray:
    """Per-position binary entropy across frames: ~0 = constant (sync/ID), ~1 = payload."""
    rows = len(bits) // period
    m = bits[: rows * period].reshape(rows, period).astype(np.float64)
    p = np.clip(m.mean(axis=0), 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


# =============================================================== 9. CRC identification

def _crc(data: bytes, width: int, poly: int, init: int, refin: bool, refout: bool, xorout: int):
    def rev(v, n):
        return int(format(v, f"0{n}b")[::-1], 2)
    top, mask = 1 << (width - 1), (1 << width) - 1
    crc = init
    for byte in data:
        b = rev(byte, 8) if refin else byte
        crc ^= (b << (width - 8)) & mask if width >= 8 else (b >> (8 - width))
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & mask if crc & top else (crc << 1) & mask
    if refout:
        crc = rev(crc, width)
    return crc ^ xorout


CRC_CATALOGUE = [   # (name, width, poly, init, refin, refout, xorout) — check value of "123456789"
    ("CRC-8/SMBUS", 8, 0x07, 0x00, False, False, 0x00),
    ("CRC-8/MAXIM", 8, 0x31, 0x00, True, True, 0x00),
    ("CRC-16/ARC", 16, 0x8005, 0x0000, True, True, 0x0000),
    ("CRC-16/IBM-3740", 16, 0x1021, 0xFFFF, False, False, 0x0000),
    ("CRC-16/KERMIT", 16, 0x1021, 0x0000, True, True, 0x0000),
    ("CRC-16/X-25", 16, 0x1021, 0xFFFF, True, True, 0xFFFF),
    ("CRC-16/USB", 16, 0x8005, 0xFFFF, True, True, 0xFFFF),
    ("CRC-32/ISO-HDLC", 32, 0x04C11DB7, 0xFFFFFFFF, True, True, 0xFFFFFFFF),
    ("CRC-32/BZIP2", 32, 0x04C11DB7, 0xFFFFFFFF, False, False, 0xFFFFFFFF),
]


def identify_crc(frames: list[tuple[bytes, int]], width: int | None = None):
    """frames = [(payload_bytes, observed_crc_int), ...] → catalogue models that reproduce ALL."""
    hits = []
    for name, w, poly, init, ri, ro, xo in CRC_CATALOGUE:
        if width and w != width:
            continue
        if all(_crc(d, w, poly, init, ri, ro, xo) == c for d, c in frames):
            hits.append(name)
    return hits


def search_crc_poly(frames: list[tuple[bytes, int]], width: int = 16, inits=(0x0000, 0xFFFF)):
    """Brute-force (poly, init, refin/refout, xorout∈{0,mask}) — the reveng search, small widths.
    Use CRC RevEng (`reveng -w 16 -s ...`) for the full search space."""
    mask = (1 << width) - 1
    found = []
    for poly in range(1, 1 << width, 2):
        for init in inits:
            for ri, ro in ((False, False), (True, True)):
                for xo in (0, mask):
                    if all(_crc(d, width, poly, init, ri, ro, xo) == c for d, c in frames):
                        found.append(dict(poly=poly, init=init, refin=ri, refout=ro, xorout=xo))
    return found


# =============================================================== 10. convolutional codes

def conv_encode(bits: np.ndarray, polys=(0o171, 0o133), K: int = 7) -> np.ndarray:
    """Non-recursive rate-1/n convolutional encoder, MSB-first shift register, zero state."""
    n = len(polys)
    reg = 0
    out = np.empty(len(bits) * n, dtype=np.uint8)
    for i, b in enumerate(bits):
        reg = ((reg << 1) | int(b)) & ((1 << K) - 1)
        for j, p in enumerate(polys):
            out[i * n + j] = bin(reg & p).count("1") & 1
    return out


def viterbi_decode(llr: np.ndarray, polys=(0o171, 0o133), K: int = 7,
                   terminated: bool = False) -> np.ndarray:
    """Soft-decision Viterbi for rate-1/n codes. llr>0 ⇒ bit 0 (same convention as symbols_to_llr).

    Returns the decoded information bits. Complexity 2^(K-1) states × steps.
    """
    n = len(polys)
    steps = len(llr) // n
    S = 1 << (K - 1)
    out_tab = np.zeros((S, 2, n), dtype=np.int8)
    nxt = np.zeros((S, 2), dtype=np.int32)
    for s in range(S):
        for b in (0, 1):
            reg = ((s << 1) | b) & ((1 << K) - 1)
            for j, p in enumerate(polys):
                out_tab[s, b, j] = bin(reg & p).count("1") & 1
            nxt[s, b] = reg & (S - 1)
    INF = 1e18
    pm = np.full(S, INF)
    pm[0] = 0.0
    trace = np.zeros((steps, S), dtype=np.uint8)
    prev = np.zeros((steps, S), dtype=np.int32)
    for t in range(steps):
        y = llr[t * n:(t + 1) * n]
        new = np.full(S, INF)
        for s in range(S):
            if pm[s] >= INF:
                continue
            for b in (0, 1):
                # cost = sum over outputs of |llr| when the hypothesis disagrees with the sign
                c = 0.0
                for j in range(n):
                    bit_hyp = out_tab[s, b, j]
                    c += abs(y[j]) if (y[j] < 0) != (bit_hyp == 1) else 0.0
                ns = nxt[s, b]
                m = pm[s] + c
                if m < new[ns]:
                    new[ns] = m
                    trace[t, ns] = b
                    prev[t, ns] = s
        pm = new
    s = int(np.argmin(pm)) if not terminated else 0
    bits = np.zeros(steps, dtype=np.uint8)
    for t in range(steps - 1, -1, -1):
        bits[t] = trace[t, s]
        s = prev[t, s]
    return bits


def conv_code_score(llr: np.ndarray, polys, K: int, n_check: int = 2000) -> float:
    """Re-encode-and-compare score: fraction of channel bits that agree with the re-encoded
    decoder output. ≈1 ⇒ this (polys, K) hypothesis explains the stream; ≈0.5 ⇒ it does not."""
    m = min(len(llr), n_check * len(polys))
    dec = viterbi_decode(llr[:m], polys, K)
    re = conv_encode(dec, polys, K)
    hard = llr_to_bits(llr[:len(re)])
    return float(np.mean(re[:len(hard)] == hard))
