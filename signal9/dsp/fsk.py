"""Non-coherent FSK demodulation: 2-FSK, 4-FSK, MSK and GFSK (Roadmap §5.4).

The problem statement requires FSK demodulation alongside PSK and QAM.  The
linear demodulator in :mod:`signal9.exploit.chain` cannot do it -- FSK carries
information in *frequency*, has a constant envelope, and has no constellation
to recover a phase reference for -- so this module implements the
discriminator receiver the plan specifies:

    inst_freq = angle(x[n] * conj(x[n-1]))           (FM discriminator)
      -> symbol rate from the cyclostationary line of |d/dt inst_freq|
      -> integrate-and-dump over each symbol, with the timing phase tracked
         blockwise so a sampling-clock offset cannot walk the dump window
         off the symbol
      -> slice at the (M-1) decision thresholds found from the data
      -> Gray-mapped bits, and per-bit soft values

The discriminator path is 1-2 dB worse than a trellis receiver for CPM with
memory (GMSK), which the plan accepts: it is adequate for identification and it
is what ``dsd-fme``-class tools use.  MSK and GFSK are handled as 2-level FSK.

Why the symbol rate has to be estimated differently here: the linear-mod
estimator looks for a spectral line in ``|x|^2``, which is produced by the
envelope ripple of pulse shaping.  An FSK signal has a constant envelope, so
that line does not exist; the information is in the frequency waveform, which
is an NRZ-like real signal whose squared derivative does carry a line at Rs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import resample_poly

# 4-FSK level -> dibit.  Outer levels map to 01 / 11 and inner to 00 / 10, which
# is the DMR/P25/C4FM convention (+3 -> 01, +1 -> 00, -1 -> 10, -3 -> 11); it is
# Gray, so the commonest error -- one level off -- costs a single bit.  The
# mapping itself is a D2 ambiguity and the chain also tries its polarity flip.
_GRAY4 = {0: (1, 1), 1: (1, 0), 2: (0, 0), 3: (0, 1)}   # index 0 = lowest tone


@dataclass
class FSKResult:
    """Outcome of FSK demodulation."""
    modulation: str = "UNRESOLVED"
    n_tones: int = 0
    symbol_rate_hz: float = 0.0
    tone_freqs_hz: list[float] = field(default_factory=list)
    tone_spacing_hz: float = 0.0
    modulation_index: float = 0.0
    eye_opening: float = 0.0
    n_symbols: int = 0
    symbol_values: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    hard_bits: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint8), repr=False)
    soft: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32), repr=False)

    def to_dict(self) -> dict:
        return {
            "modulation": self.modulation,
            "n_tones": int(self.n_tones),
            "symbol_rate_hz": round(float(self.symbol_rate_hz), 3),
            "tone_freqs_hz": [round(float(f), 2) for f in self.tone_freqs_hz],
            "tone_spacing_hz": round(float(self.tone_spacing_hz), 2),
            "modulation_index": round(float(self.modulation_index), 3),
            "eye_opening": round(float(self.eye_opening), 3),
            "n_symbols": int(self.n_symbols),
        }


def discriminator(x: np.ndarray, fs: float) -> np.ndarray:
    """Instantaneous frequency in Hz, one sample shorter than ``x``."""
    x = np.asarray(x, dtype=np.complex128).ravel()
    if len(x) < 2:
        return np.zeros(0)
    return np.angle(x[1:] * np.conj(x[:-1])) * (fs / (2.0 * np.pi))


def envelope_cv(x: np.ndarray) -> float:
    """Coefficient of variation of the envelope: ~0 for FSK, 0.3-0.5 for RRC PSK/QAM."""
    a = np.abs(np.asarray(x).ravel())
    m = float(np.mean(a))
    return float(np.std(a) / m) if m > 0 else 0.0


def _levels(v: np.ndarray) -> np.ndarray:
    """Tone levels of a frequency waveform: 4 if equally spaced and separated, else 2."""
    c2 = _kmeans_1d(v, 2)
    c4 = _kmeans_1d(v, 4)
    lab = np.argmin(np.abs(v[:, None] - c4[None, :]), axis=1)
    spread = float(np.sqrt(np.mean((v - c4[lab]) ** 2)))
    gaps = np.diff(c4)
    if gaps.min() / max(gaps.max(), 1e-12) > 0.5 and gaps.min() > 3.0 * spread:
        return c4
    return c2


def _transition_times(freq: np.ndarray, smooth: int) -> np.ndarray:
    """Symbol transitions of an FSK frequency waveform, via hysteresis quantisation.

    Each sample is assigned to a tone level only when it lies clearly inside
    that level's region -- within ``0.25 * gap`` of the level, i.e. well away
    from every decision threshold.  Samples in between inherit the previous
    level, and an event is the midpoint between leaving one level and settling
    in the next.

    That gives exactly one event per change of tone, whatever the number of
    levels -- a midline test misses 4-FSK changes between two levels on the
    same side of the centre -- while noise hovering near a threshold can never
    trigger one.  A derivative-peak detector, tried first, fired roughly twenty
    times per real transition on a 20 dB GFSK signal, because the
    discriminator's noise derivative dwarfs a Gaussian-smoothed transition.
    """
    f = np.asarray(freq, dtype=np.float64)
    if smooth > 1:
        f = np.convolve(f, np.ones(smooth) / smooth, mode="same")
    if len(f) < 32:
        return np.zeros(0)
    c = _levels(f[:: max(1, len(f) // 20000)])
    if len(c) < 2:
        return np.zeros(0)
    gap = float(np.min(np.diff(c)))
    lab = np.argmin(np.abs(f[:, None] - c[None, :]), axis=1)
    strong = np.abs(f - c[lab]) < 0.25 * gap
    idx = np.flatnonzero(strong)
    if idx.size < 2:
        return np.zeros(0)
    lv = lab[idx]
    change = np.flatnonzero(lv[1:] != lv[:-1])
    # Midpoint of the gap between the last confident sample of the old level
    # and the first confident sample of the new one.
    return 0.5 * (idx[change] + idx[change + 1]).astype(np.float64)


def _event_fundamental(idx: np.ndarray, fs: float, min_rate: float) -> tuple[float, float]:
    """Fundamental of an event train via its periodogram (see caller)."""
    if len(idx) < 16:
        return 0.0, 0.0
    # The event periodogram on a uniform grid is exactly the power spectrum of
    # an impulse train at the event times, so one zero-padded FFT evaluates it
    # instead of a direct sum costing (grid x events) complex exponentials.
    L = int(idx[-1]) + 1
    n = 1 << int(np.ceil(np.log2(max(4 * L, 1024))))
    train = np.zeros(n)
    train[idx.astype(np.int64)] = 1.0
    P = np.abs(np.fft.rfft(train)) ** 2 / len(idx) ** 2
    grid = np.fft.rfftfreq(n, 1.0 / fs)
    duration = L / fs
    band = (grid >= max(min_rate, 8.0 / duration)) & (grid <= fs / 4.0)
    if not band.any():
        return 0.0, 0.0
    Pb, gb = P[band], grid[band]
    peak = float(Pb.max())
    if peak <= 0:
        return 0.0, 0.0

    def at(freq_hz: float) -> float:
        k = int(round((freq_hz - gb[0]) / (gb[1] - gb[0]))) if len(gb) > 1 else 0
        lo, hi = max(0, k - 2), min(len(Pb), k + 3)
        return float(Pb[lo:hi].max()) if hi > lo else 0.0

    # An impulse train at k*T has lines of similar height at every m/T, so the
    # fundamental is the lowest strong lobe *that also has a line at twice its
    # frequency* -- which rejects low-frequency lobes produced by data
    # patterns rather than by the symbol clock.
    df = float(gb[1] - gb[0]) if len(gb) > 1 else 1.0

    def contrast(j: int) -> float:
        """Peak height over the median of its +/-20 % neighbourhood.

        A symbol-clock line is narrow -- about 1/duration wide -- while the
        low-frequency power that data patterns put into the event train is a
        broad continuum.  Both can clear an absolute threshold; only a line
        stands out from its own surroundings.
        """
        f = gb[j]
        lo = np.searchsorted(gb, 0.8 * f)
        hi = np.searchsorted(gb, 1.2 * f)
        guard = max(3, int(round(4.0 / (duration * df))))
        mask = np.ones(hi - lo, dtype=bool)
        c = j - lo
        mask[max(0, c - guard):c + guard + 1] = False
        ref = Pb[lo:hi][mask]
        return float(Pb[j] / (np.median(ref) + 1e-12)) if ref.size else 0.0

    f0 = 0.0
    for i in np.flatnonzero(Pb >= 0.4 * peak):
        j = int(i)
        while j + 1 < len(Pb) and Pb[j + 1] >= Pb[j]:
            j += 1
        cand = float(gb[j])
        if contrast(j) < 5.0:
            continue
        if 2 * cand > fs / 4.0 or at(2 * cand) >= 0.25 * peak:
            f0 = cand
            break
    if f0 <= 0.0:
        return 0.0, 0.0

    t = idx / fs
    step = fs / n
    fine = np.linspace(max(gb[0], f0 - 2 * step), f0 + 2 * step, 161)
    Pf = np.abs(np.exp(2j * np.pi * np.outer(fine, t)).sum(axis=1)) ** 2 / len(t) ** 2
    j = int(np.argmax(Pf))
    return float(fine[j]), float(Pf[j])


def estimate_fsk_symbol_rate(freq: np.ndarray, fs: float,
                             min_rate: float = 10.0) -> tuple[float, float]:
    """Symbol rate from an event-time periodogram of the frequency transitions.

    Frequency transitions of an FSK signal happen only at symbol boundaries
    ``k*T + tau``, so ``P(f) = |sum_k exp(2j*pi*f*t_k)|^2 / N^2`` has peaks at
    every multiple ``m/T`` -- and not at ``1/(2T)``, where the terms for odd and
    even k cancel.  The fundamental is the lowest strong peak.

    This replaces a spectral-line search on the squared frequency derivative:
    transitions are impulse-like, so that spectrum has lines of equal height at
    Rs, 2Rs, 4Rs..., and taking the maximum returned a harmonic (measured 2400
    and 4800 Hz for a 1200 Bd 2-FSK signal), which made every later stage fail.

    Noise adds spurious events and too much smoothing erases real ones, so the
    estimate is taken at several smoothing lengths and the most coherent event
    train (the highest normalised periodogram peak) wins.

    Returns ``(rate_hz, quality)``: quality is 1.0 for a perfectly periodic
    event train and falls as spurious events are mixed in.
    """
    f = np.asarray(freq, dtype=np.float64)
    if len(f) < 64:
        return 0.0, 0.0
    best = (0.0, 0.0)
    for smooth in (1, 2, 4, 8, 16, 32):
        if smooth * 8 > len(f):
            break
        rate, q = _event_fundamental(_transition_times(f, smooth), fs, min_rate)
        # A symbol must span several samples after smoothing, or the smoothing
        # itself has merged adjacent symbols and the estimate is meaningless.
        if rate > 0 and fs / rate >= 2 * smooth and q > best[1]:
            best = (rate, q)
    return best


def _kmeans_1d(v: np.ndarray, k: int, iters: int = 30) -> np.ndarray:
    """Deterministic 1-D k-means, initialised at the data quantiles."""
    c = np.quantile(v, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        lab = np.argmin(np.abs(v[:, None] - c[None, :]), axis=1)
        new = np.array([v[lab == j].mean() if np.any(lab == j) else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return np.sort(c)


def _dump(freq: np.ndarray, sps: int, block_syms: int = 256) -> np.ndarray:
    """Integrate-and-dump at the best timing phase, re-chosen every block.

    Within each block the phase maximising the spread of the dumped values is
    kept -- at the correct phase every window lies inside one symbol, so the
    values sit at the tone levels; a misaligned window averages two symbols and
    pulls toward the middle.  Re-choosing per block tracks a clock offset.
    """
    n_sym = len(freq) // sps
    out: list[np.ndarray] = []
    pos = 0
    while pos + sps * 4 <= len(freq):
        seg = freq[pos:pos + sps * (block_syms + 1)]
        best, best_score = None, -1.0
        for ph in range(sps):
            m = (len(seg) - ph) // sps
            if m < 2:
                continue
            vals = seg[ph:ph + m * sps].reshape(m, sps).mean(axis=1)
            score = float(np.mean(np.abs(vals - np.median(vals))))
            if score > best_score:
                best, best_score = vals, score
        if best is None:
            break
        take = best[:block_syms]
        out.append(take)
        pos += sps * len(take)
        if len(take) < block_syms:
            break
    return np.concatenate(out) if out else np.zeros(0)


def fsk_demodulate(x: np.ndarray,
                   fs: float,
                   n_tones: int | None = None,
                   symbol_rate: float | None = None,
                   target_sps: int = 8) -> FSKResult:
    """Demodulate 2-FSK / 4-FSK (and MSK / GFSK as 2-level) to bits.

    ``n_tones`` and ``symbol_rate`` are estimated when not supplied.  The tone
    count is chosen by comparing how well 2 and 4 levels explain the dumped
    symbol values, with a parsimony margin so 2-FSK is not split into four.
    """
    res = FSKResult()
    x = np.asarray(x, dtype=np.complex64).ravel()
    if len(x) < 256 or fs <= 0:
        return res

    freq = discriminator(x, fs)
    rate = float(symbol_rate) if symbol_rate else estimate_fsk_symbol_rate(freq, fs)[0]
    if rate <= 0 or rate >= fs / 2:
        return res
    res.symbol_rate_hz = rate

    # Resample the frequency waveform to an integer number of samples per
    # symbol so each dump window is exactly one symbol long.
    up = target_sps * rate
    from math import gcd
    ratio = up / fs
    best_p, best_q, best_err = 1, 1, float("inf")
    for q in range(1, 513):
        p = int(round(ratio * q))
        if 1 <= p <= 512 and abs(p / q - ratio) < best_err:
            best_p, best_q, best_err = p, q, abs(p / q - ratio)
    g = gcd(best_p, best_q)
    fr = resample_poly(freq, best_p // g, best_q // g) if best_p != best_q else freq

    vals = _dump(fr, target_sps)
    if len(vals) < 32:
        return res

    # Tone count: 2 vs 4 levels, whichever explains the values with less
    # residual -- 4 must win by a clear factor, since more clusters always fit
    # a little better.
    fits = {}
    for k in ((int(n_tones),) if n_tones else (2, 4)):
        c = _kmeans_1d(vals, k)
        resid = float(np.mean(np.min(np.abs(vals[:, None] - c[None, :]), axis=1) ** 2))
        fits[k] = (resid, c)
    if n_tones:
        k = int(n_tones)
    else:
        # 4-FSK levels are equally spaced (+/-1, +/-3), so a genuine 4-level fit
        # has three similar gaps.  Clean 2-FSK split into four clusters instead
        # gives two tight pairs -- a small, a large and a small gap.  A residual
        # ratio cannot tell these apart, because with little noise both
        # residuals are tiny and the extra clusters always "fit better".
        c4 = fits[4][1]
        gaps = np.diff(c4)
        spread4 = float(np.sqrt(fits[4][0]))
        equal = gaps.min() / max(gaps.max(), 1e-12) > 0.5
        separated = gaps.min() > 3.0 * spread4
        k = 4 if (equal and separated) else 2
    resid, centres = fits[k]

    spacing = float(np.mean(np.diff(centres))) if k > 1 else 0.0
    spread = float(np.var(vals)) or 1e-12
    res.n_tones = k
    res.tone_freqs_hz = [float(c) for c in centres]
    res.tone_spacing_hz = spacing
    # Deviation index h = tone spacing / symbol rate; MSK is h = 0.5.
    res.modulation_index = spacing / rate if rate > 0 else 0.0
    # Eye opening: 1 - residual/variance.  ~1 for clean FSK, ~0 when the dumped
    # values do not cluster at all (wrong family, or noise).
    res.eye_opening = float(np.clip(1.0 - resid / spread, 0.0, 1.0))
    res.symbol_values = vals
    res.n_symbols = int(len(vals))

    if k == 2:
        thr = float(np.mean(centres))
        res.hard_bits = (vals > thr).astype(np.uint8)
        # Soft value in the decoders' convention (positive => bit 1), scaled by
        # the half-spacing so it reads in units of "distance to the threshold".
        res.soft = ((vals - thr) / (0.5 * max(spacing, 1e-9))).astype(np.float32)
        res.modulation = "MSK" if abs(res.modulation_index - 0.5) < 0.08 else "2FSK"
    else:
        lab = np.argmin(np.abs(vals[:, None] - centres[None, :]), axis=1)
        bits = np.array([_GRAY4[int(i)] for i in lab], dtype=np.uint8).reshape(-1)
        res.hard_bits = bits
        # Per-bit soft values from the level distance: for each dibit bit, the
        # signed distance to the nearest threshold that flips it.
        half = 0.5 * max(spacing, 1e-9)
        t_lo, t_mid, t_hi = (centres[0] + centres[1]) / 2, (centres[1] + centres[2]) / 2, (centres[2] + centres[3]) / 2
        # First bit is 1 below t_mid (levels 0,1), second bit is 1 on the outer levels.
        b0 = (t_mid - vals) / half
        b1 = np.where(vals < t_mid, t_lo - vals, vals - t_hi) / half
        res.soft = np.stack([b0, b1], axis=1).reshape(-1).astype(np.float32)
        res.modulation = "4FSK"
    return res


def looks_like_fsk(x: np.ndarray, fs: float) -> bool:
    """Cheap family test: constant envelope *and* a multi-level frequency waveform.

    A constant envelope alone is not enough -- unfiltered PSK is constant
    envelope too -- but PSK's instantaneous frequency is ~0 between transition
    spikes, whereas FSK's sits at two or more distinct non-zero levels.
    """
    if len(x) < 256:
        return False
    # Pulse-shaped PSK/QAM has an envelope CV of roughly 0.3-0.5 even without
    # noise; a constant-envelope FSK signal reaches ~0.22 only from the noise
    # itself at 10 dB SNR.  0.28 separates the two, and the frequency-waveform
    # test below is the real discriminator.
    if envelope_cv(x) > 0.28:
        return False
    fr = discriminator(x, fs)
    lo, hi = np.quantile(fr, [0.1, 0.9])
    spread = hi - lo
    if spread <= 0.0:
        return False

    # A rectangular-pulse PSK capture is still constant-envelope.  Its
    # discriminator is a large pile of *zero* phase increments punctuated by
    # transition impulses, which used to satisfy the two tests below and made
    # a clean QPSK demo take the FSK path.  FSK has plateaux at two or more
    # tones; it must not be dominated by one carrier-like bin.  This is a
    # deliberately conservative rejection gate: an uncertain FSK waveform
    # should remain unresolved rather than corrupt a linear-modulation decode.
    core = fr[(fr >= lo) & (fr <= hi)]
    if core.size < 64:
        return False
    hist, _ = np.histogram(core, bins=32, range=(lo, hi))
    dominant_fraction = float(hist.max() / core.size) if hist.size else 1.0
    if dominant_fraction > 0.60:
        return False

    # FSK spends most of its time away from the midpoint between its outer
    # tones; PSK spends it at the carrier except during phase transitions.
    # Do not use the sample median here: for a balanced 2-FSK waveform NumPy
    # may place the median on one of the two tones, incorrectly making half the
    # samples look like centre-frequency energy.
    centre = 0.5 * (lo + hi)
    near = float(np.mean(np.abs(fr - centre) < 0.15 * spread)) if spread > 0 else 1.0
    # RRC-shaped PSK can approach 0.35 at high transition density.  A 0.30
    # boundary retains a margin while accepting clean 2/4-FSK tone plateaus.
    return near < 0.30
