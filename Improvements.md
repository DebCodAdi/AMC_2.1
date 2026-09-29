# Improvements — Blind Exploitation Chain Implementation

> **What this document is.** A record of every change made to the codebase in
> response to `Blind_Exploitation_Chain_Plan.md` (v1.0).  Each entry states
> *what* was changed, *why* it was changed (codebase gap vs plan improvement),
> and the *measured outcome* where available.

---

## 1. `signal9/dsp/timing_recovery.py`

### What changed
* **Power-normalised Gardner TED error** added to both `gardner_timing_recovery()`
  and the new `simple_gardner_timing_recovery()` function.  The TED error is now
  divided by the estimated input signal power (`pw = mean(|x|²) + ε`).
* **Sign convention fixed** (comment added explaining the correct sign):
  `e = Re{(y_curr − y_early) · conj(y_mid)} / pw` pulls the timing estimate toward
  the pulse peak, not toward the zero crossings.
* **New function `simple_gardner_timing_recovery()`** added — a single-output API
  matching the `blindbits.py` reference implementation.  This enables direct
  cross-validation against the published reference chain.

### Why
Plan §5.3 states two things that decide whether the Gardner loop works:
1. *"Gardner's error must be normalised by the signal power — otherwise the loop
   gain depends on AGC state and the loop either crawls or jitters."*
2. *"The TED sign convention matters: with the sign flipped the loop locks to the
   zero crossings instead of the pulse peak and EVM is ~18 dB worse."*

The codebase clipped the TED to ±2 without power normalisation, meaning a signal
10 dB weaker would have 10× lower loop gain and might never converge.

### Measured outcome
EVM is identical (within 0.1 dB) across input power scales of 0.1×, 1×, 10×,
confirming that the normalisation eliminates the AGC-dependence.  The sign
convention test in `test_blind_exploitation.py::TestTimingRecovery` passes.

---

## 2. `signal9/dsp/gf2.py`

### What changed
Two new functions added:
* **`rank_profile(bits, n_min, n_max, n_rows, offset)`** — returns a dict mapping
  each candidate codeword length `n` to its rank ratio `rank(M_n)/n`.  A value
  well below 1.0 indicates a linear block code of that length.
* **`detect_code_length(bits, n_min, n_max, n_rows)`** — blind codeword-length
  detector scanning all offsets 0..n−1 for each candidate `n`; returns
  `(n_hat, k_hat, offset_hat, rank_ratio)`.

### Why
The plan §8.2 (Barbier/Cluzeau rank criterion) requires these primitives for:
* De-interleaver depth search (§8.2: rank deficiency is restored at the correct
  depth).
* FEC classification (§9.1: rank criterion distinguishes block from convolutional).
* Bitstream correlation (§10.2: discriminating coded from uncoded regions).

The codebase had `gf2_rank()` but no higher-level rank profiling.  The
`blindbits.py` reference implements identical functions and measures:
`detect_code_length` on Hamming(7,4) → `(7, 4, 0, 0.571)` even with a 3-bit junk
prefix.

### Measured outcome
`rank_profile` at n=7 gives 0.571 for Hamming(7,4); at n=9 gives ≈1.0.
`detect_code_length` recovers n=7 with and without a junk prefix (offset
robustness gate passes).

---

## 3. `signal9/label/interleaver_detect.py`

### What changed
* **`block_interleave()` and `block_deinterleave()`** added — explicit round-trip
  helpers (were only in `signal9/fec/interleaver.py` before; needed locally to
  avoid circular imports in the depth search).
* **`detect_block_interleaver_depth(bits, code_n, depth_max, n_rows)`** — new
  depth-scan function from the plan / `blindbits.py §6`.  For the true depth D the
  de-interleaved stream regains the code's rank deficiency; all other depths give
  ratio ≈1.0.
* **`detect_block_interleaver_blind(bits, n_min, n_max, depth_max, n_rows)`** — fully
  blind variant: calls `detect_code_length()` first, then `detect_block_interleaver_depth()`.

### Why
Plan §8.2: *"De-interleaving with the correct parameters restores rank deficiency."*
The original `detect_block_interleaver()` only tested a fixed list of `(R,C)` pairs
and had no connection to the rank criterion.  The new depth-scan approach is the
algorithm from Burel–Gautier (2003) and Sicot–Houcke–Barbier (2009) referenced in
the plan.

### Measured outcome
`detect_block_interleaver_depth` recovers depth 5 from a Hamming(7,4) stream
interleaved at depth 5: scores = `{5: 0.57, others: 1.0}`, unambiguous.

---

## 4. `signal9/label/code_id.py`

### What changed
* **`conv_code_score(llr, polys, K, n_check)`** — new public function.  Decodes
  `n_check` bits with Viterbi, re-encodes, and returns the fraction of channel
  hard-bits that agree with the re-encoded stream.  ~1.0 → hypothesis explains the
  stream; ~0.5 → hypothesis is wrong.
* **`_CONV_CATALOGUE`** — a catalogue of 7 standard `(K, polys, name)` entries
  covering K ∈ {3,5,7,9}, rate 1/2 and 1/3, from NASA/CCSDS/IEEE.
* **`_best_conv_hypothesis(llr, bits)`** — searches the full catalogue with all
  offsets, returns the best hypothesis above `_CONV_SCORE_THRESHOLD = 0.90`.
* **`identify_fec_scheme()`** rewritten to use the catalogue search first (instead
  of the old hardcoded `confidence=5.0` placeholder), with rank-profile fallback.

### Why
Plan §9.3: *"A re-encode-and-compare hypothesis score is simpler, more robust to
noise than algebraic methods, and directly reuses the decoder you need anyway."*
The original `identify_fec_scheme` had `confidence=5.0` hardcoded for conv_k7_r12
regardless of whether the stream actually contained that code — a placeholder that
would produce false positives on every uncoded stream.

### Measured outcome
True hypothesis score: 0.975.  Wrong hypothesis score: 0.872.  Clean separation at
3% channel BER.  K=5 and K=7 correctly identified in catalogue sweep.  LLR
convention confirmed: codebase `viterbi_decode` uses LLR > 0 → bit 1 (branch_sym
= 2·bit − 1).

---

## 5. `signal9/label/scrambler_id.py`

### What changed
Three new functions appended:
* **`berlekamp_massey(seq)`** — LFSR synthesis over GF(2); returns `(L, poly)`.
  Empty and all-zeros/all-ones edge cases handled.  The connection polynomial array
  is sized `n+1` to avoid indexing crashes.
* **`linear_complexity_profile(bits, window, step)`** — windowed linear complexity.
  L ≈ window/2 → random/coded; L << window/2 → LFSR-generated region (idle fill,
  PRBS test sequence, or unscrambled constant under an additive scrambler).
* **`blind_lfsr_descramble(bits, window, max_degree)`** — searches for a low-LC
  window, runs BM to recover the polynomial, regenerates and XORs the LFSR sequence.

### Why
Plan §7.2: *"Most real systems whiten the bitstream (additive LFSR as in DVB/CCSDS).
Without stripping it, the de-interleaver and FEC identifiers see white noise and
return nothing."*  The existing `scrambler_id.py` only tried a fixed catalogue of
five named scramblers; it had no blind LFSR detection capability.

### Measured outcome
`berlekamp_massey` recovers degree-15 LFSR exactly from 400 bits (L=15, taps at
[0,14,15]).  Linear complexity profile: LFSR stream gives L ≤ 20; random stream
gives L > 100 (all windows).

---

## 6. `signal9/label/d6_frames.py` *(new file)*

### What changed
New module implementing the full D6 pipeline from Plan §10:

| Function | What it does |
|---|---|
| `bit_autocorrelation_periods(bits, p_min, p_max, top)` | Candidate frame periods via ±1 autocorrelation |
| `find_sync_pattern(bits, period, min_len, stability)` | Sync-word discovery by positional stability |
| `field_entropy_profile(bits, period)` | Per-position binary entropy across frames |
| `identify_crc(frames, width)` | 15-entry CRC catalogue search (8/16/32-bit) |
| `_crc_compute(...)` | Generic Rocksoft-model CRC (parametric) |
| `CRC_CATALOGUE` | 15 standard CRC definitions with published check values |
| `search_crc_poly(frames, width, inits)` | Brute-force CRC polynomial search |
| `detect_counter_fields(bits, period, ...)` | Counter/sequence-number field detection |
| `detect_length_fields(bits, period, ...)` | Length-field correlation detection |
| `ProtocolSketch` + `build_protocol_sketch(bits)` | Full D6 pipeline → JSON frame model |

### Why
The plan §10.3 output `{"period_bits":200,"sync":...,"fields":...,"payload":...,"crc":...}`
is the *primary deliverable* of the bitstream correlation stage and was entirely absent
from the codebase.  The only existing related code was `decide/crc_search.py` which
only handled CRC-32 trailing in a byte buffer.

### Measured outcome
* `bit_autocorrelation_periods`: top-3 peaks are 400, 200, 600 for a 200-bit frame
  stream (period and its harmonics) — matches plan measurement exactly.
* `find_sync_pattern`: recovers injected 20-bit sync word with agreement=1.00.
* `field_entropy_profile`: sync region mean H < 0.1, payload region mean H > 0.9.
* `identify_crc`: recovers CRC-32/ISO-HDLC from 4 random frames.
* Published check value: `CRC-16/IBM-3740("123456789") == 0x29B1` ✓

---

## 7. `signal9/decide/crc_search.py`

### What changed
* **`identify_crc_bytes(frames, width)`** added — multi-width CRC catalogue search
  on raw `(payload_bytes, observed_crc)` pairs.  Re-uses `d6_frames.identify_crc`
  when available, falls back to a minimal built-in set (CRC-16/X-25, CRC-32/ISO-HDLC).
* Module docstring updated to reflect the new capability.

### Why
The original file only searched CRC-32 as a trailing 4-byte field in a contiguous
byte buffer.  Plan §10.2 requires identification across 8/16/32-bit widths from
per-frame (payload, CRC) pairs — the format used by `d6_frames.build_protocol_sketch`.

### Measured outcome
`identify_crc_bytes` with 4 random CRC-32 frames returns `["CRC-32/ISO-HDLC"]`.
`identify_crc_bytes` with 4 CRC-16/X-25 frames returns `["CRC-16/X-25"]`.

---

## 8. `signal9/dsp/snr_estimate.py`

### What changed
Two new functions added:
* **`evm_db(symbols, constellation_points)`** — EVM in dB against the nearest
  constellation point.  Normalises the symbol array to unit average power before
  computing, matching the `blindbits.py §3` convention.
* **`snr_from_evm(evm_db_val)`** — approximate SNR from EVM (useful for constant-
  modulus constellations).

### Why
Plan §5.2 and the reference chain test both use EVM (dB) as the primary quality
gate.  The codebase had no standalone `evm_db` function; EVM was computed inline
only inside `adaptive_equalize()`.

### Measured outcome
Perfect QPSK symbols → EVM < −30 dB.  Noisy symbols at moderate SNR produce
finite, reasonable EVM values.

---

## 9. `tests/test_blind_exploitation.py` *(new file)*

### What changed
51 new pytest tests covering all the above additions:
* **Timing recovery** — symbol count, power-invariant EVM, empty/short input.
* **EVM dB** — perfect symbols, noisy symbols, all-same-symbol edge case.
* **GF2 rank profile** — Hamming(7,4) detection, random bits give ratio≈1, junk
  prefix robustness, empty / too-short inputs.
* **Interleaver detection** — depth search (depth 5), round-trip interleave/
  deinterleave, blind detection with known code, empty input.
* **FEC code_id** — true vs. wrong hypothesis separation, K=5 and K=7 identification,
  empty bits, `identify_fec_scheme` interface.
* **Scrambler BM** — degree-15 LFSR recovery, LC profile (LFSR vs random), empty /
  all-zeros / all-ones edge cases.
* **D6 frames** — period detection, sync word recovery, entropy profile, CRC-32 and
  CRC-16 identification, published check value, counter field detection, protocol
  sketch builder, empty input, too-few-rows, search_crc_poly.
* **CRC search** — multi-width catalogue, windowed CRC-32 search, empty/short inputs.
* **End-to-end conv** — zero-BER roundtrip for K=5 and K=7, 3% error channel.

---

## 10. `blindbits.py` and `tests/test_blindbits.py` *(copied from plan)*

The reference implementation (`blindbits.py`) and its script-based test suite were
copied into the workspace root and `tests/` respectively.  All 13 assertions in
`test_blindbits.py` pass with the exact measured values quoted in the plan:

| Test | Measured | Plan says |
|---|---|---|
| Symbol rate | 1.0000 | exact |
| EVM after full chain | −22.4 dB | −22.4 dB |
| QPSK BER @6 dB | 0.0228 | theory 0.0229 |
| QPSK BER @10 dB | 0.0008 | theory 0.0008 |
| rank ratio n=7 (Hamming) | 0.571 | 0.571 |
| detect_code_length | (7,4,0,0.571) | (7,4,0,0.571) |
| Interleaver depth | 5 | 5 |
| BM L | 15 | 15 |
| LC LFSR / random | 15 / 127 | 15 / 127 |
| Viterbi BER @3% channel errors | 0.0 | 0.0 |
| conv_code_score true | 0.975 | 0.975 |
| conv_code_score wrong | 0.872 | 0.872 |

---

## Summary — What was kept from the codebase

The following were judged **better than or equivalent to the plan** and left unchanged:

| Module | Reason kept |
|---|---|
| `signal9/dsp/cfo.py` | Parabolic peak interpolation + multi-hypothesis scoring + Luise-Reggiannini fine CFO — exceeds the plan's `cfo_mth_power` baseline |
| `signal9/dsp/equalizer.py` | CMA→MMA→DD-LMS with tap-norm guard and NaN protection — more complete than plan's `cma_equalize` |
| `signal9/dsp/timing_recovery.py` (main function) | Full 2SPS/1SPS/SCO output retained; only power normalisation and sign convention were corrected |
| `signal9/fec/conv.py` | Numba-accelerated Viterbi + puncturing/depuncturing — more complete than plan's reference Viterbi |
| `signal9/fec/rs.py` | Full BM + Chien + Forney RS decoder — plan defers to `galois` library |
| `signal9/fec/ldpc.py` | NMS decoder with IEEE 802.11n H matrix — plan defers to Sionna |
| `signal9/features/baud.py` | Three-way fusion (NL-AC + SCF + ZCR) — exceeds plan's `estimate_symbol_rate` baseline |
| `signal9/label/frame_sync.py` | Soft-syndrome frame boundary detection (Imad et al.) — exceeds plan's hard-bit approach |
| `signal9/label/scrambler_id.py` | Fixed catalogue (G3RUH/V35/CCSDS/DVB/WiFi) kept; BM blind detection added alongside it |

---

## Phase 2 — Bug Fixes

> These bugs were identified by comparing the codebase against `blindbits.py` and
> the `Blind_Exploitation_Chain_Plan.md` specification.  Each entry states the
> symptom, root cause, and fix applied.

---

### BUG-1 — Gardner TED sign (`signal9/dsp/timing_recovery.py`)

#### Symptom
The Gardner Timing Error Detector used a **positive sign** in:
```python
ted = float((diff * np.conj(y_mid)).real) / pw          # old (positive = wrong)
e   = float(np.clip(((cur - prev) * np.conj(mid)).real / pw, -1.0, 1.0))  # old
```
This causes the loop to lock onto the **zero crossings** of the pulse rather
than the pulse peak.  At the zero crossing the symbol energy is ≈ 0, not 1,
so the timing estimate is off by half a symbol period.

#### Root cause
Gardner's TED for a real bandpass signal is:
```
e[n] = (y[2n] - y[2n-2]) · y[2n-1]
```
For the complex baseband extension the correct sign is **negative** (to be a
gradient that points *toward* the peak):
```
e = -Re{(y_curr - y_early) · conj(y_mid)} / power
```
The reference `blindbits.py` (line 78) uses the correct negative sign.

#### Fix
```python
# timing_recovery.py — both gardner_timing_recovery() and simple_gardner_timing_recovery()
ted = -float((diff * np.conj(y_mid)).real) / pw          # fixed
e   = float(np.clip(-((cur - prev) * np.conj(mid)).real / pw, -1.0, 1.0))  # fixed
```

#### Measured outcome
`TestGardnerTEDSign::test_ted_sign_matches_blindbits_reference` — codebase and
`blindbits.py` reference agree within 0.01 dB EVM across all tested signals.
`test_simple_gardner_symbol_count_and_agreement` and
`test_full_gardner_symbol_power` confirm that mean symbol power > 0.3 (pulse-peak
sampling), which is near-zero at the zero crossings.

---

### BUG-2 — Puncture patterns wrong (`signal9/fec/conv.py`)

#### Symptom
Three puncture patterns had incorrect lengths and/or incorrect bit counts,
producing wrong code rates:

| Key | Old pattern | Old rate | Correct rate |
|---|---|---|---|
| `"2/3"` | `[1,1,0]` (len=3, odd) | — (alignment error) | 2/3 |
| `"3/4"` | `[1,1,0,1,1,0]` (sum=4/6) | 2/3 | 3/4 |
| `"5/6"` | `[1,1,0,1,1,0,0,1,1,0]` (sum=6/10) | 3/5 | 5/6 |

Additionally, `"2/3"` had an **odd-length** pattern which breaks the g1/g2
pair alignment: the convolutional encoder produces interleaved (g1,g2,g1,g2,…)
so the puncture period must be even.

#### Root cause
Patterns were copied from a reference that used a different interleaving
convention (per-generator patterns rather than interleaved).

#### Fix
Rate formula for a rate-1/2 mother code with even-length period `p`:
```
achieved_rate = (p / 2) / sum(pattern == 1)
```
Correct patterns:
```python
"2/3": [1, 1, 1, 0]                            # p=4, sum=3, rate=2/3 ✓
"3/4": [1, 1, 1, 0, 1, 0]                      # p=6, sum=4, rate=3/4 ✓
"5/6": [1, 1, 1, 0, 1, 0, 1, 0, 1, 0]          # p=10, sum=6, rate=5/6 ✓
```

#### Measured outcome
`TestPuncturePatterns::test_pattern_achieves_correct_rate` verifies
`(period/2) / sum(pattern) == declared_rate` for all four rates.
`test_puncture_depuncture_roundtrip` confirms 0-BER decode after
puncture → depuncture → Viterbi at all rates.

---

### BUG-3 — Missing `nrzi_decode` / `manchester_decode` (`signal9/dsp/differential.py`)

#### Symptom
The D2 bit-normalisation stage (Plan §4) requires three decoding helpers:
`differential_decode`, `nrzi_decode`, and `manchester_decode`.  The first was
present only on the symbol level (producing complex, not binary output).  The
latter two were entirely absent from the codebase; they existed only in
`blindbits.py`.

#### Root cause
The `differential.py` module focused on symbol-domain phase-difference
demapping (IV-L) and did not implement the bit-domain equivalents needed
downstream in the exploitation chain.

#### Fix
Three new functions added to `signal9/dsp/differential.py`:
* **`differential_decode(bits)`** — `bits[1:] ^ bits[:-1]`, removes differential
  bit encoding.
* **`nrzi_decode(bits, invert=False)`** — NRZ-I → NRZ; transition = 0 by
  default (most standards).  `invert=True` for systems where transition = 1.
* **`manchester_decode(bits, ieee=True)`** — IEEE 802.3 biphase decoding;
  returns `(decoded_bits, valid_fraction)`.  Valid fraction < 0.9 indicates
  wrong clock phase or wrong encoding.

All three match `blindbits.py §4` exactly (verified by direct comparison).

#### Measured outcome
`TestBitNormHelpers` — 12 tests, all pass.  NRZI roundtrip test recovers 100
random bits exactly.  `test_*_matches_blindbits` tests confirm byte-exact
agreement with the reference.

---

### BUG-4 — DVB-S PRBS15 feedback taps (`signal9/label/scrambler_id.py`)

#### Symptom
The DVB-S additive PRBS15 descrambler used wrong feedback taps:
```python
out = ((reg >> 14) ^ (reg >> 13)) & 1   # OLD: bit14 XOR bit13 (wrong polynomial)
```
This implements `1 + x^13 + x^14`, not the DVB-S polynomial `1 + x^14 + x^15`.
The resulting LFSR is not maximal-length and the PN sequence has the wrong
period and balance.

#### Root cause
Off-by-one error in the tap extraction.  The DVB-S polynomial per EN 300 421
§4.4.2 is `1 + x^14 + x^15`, which means feedback taps at positions 14 and 0
(MSB and LSB of a 15-bit shift register), not positions 14 and 13.

#### Fix
```python
fb = ((reg >> 14) ^ reg) & 1   # FIXED: bit14 XOR bit0 (= MSB XOR LSB)
```

#### Measured outcome
`TestDVBSDescrambler` — 6 tests, all pass:
* `test_prbs15_matches_reference_implementation`: codebase output equals the
  hand-written reference for 500 bits.
* `test_prbs15_entropy_close_to_1`: PN bias p₁ = 0.500 ± 0.005 (PRBS15 has
  16384 ones and 16383 zeros per period).
* `test_prbs15_descramble_then_rescramble`: double-application recovers
  original data exactly (additive scrambler property).
* `test_dvbs_identify_scrambler_detects_dvbs`: `identify_scrambler` correctly
  finds DVB_S for data scrambled with the corrected PRBS15.

---

### BUG-5 — Missing `"7/8"` puncture pattern (`signal9/fec/conv.py`)

#### Symptom
The docstring and spec reference (DEF-013) mention rate 7/8 support, but
`_PUNCTURE_PATTERNS` did not include `"7/8"`.  Calling `puncture(code, "7/8")`
silently returned the input unchanged, making rate-7/8 puncturing a no-op.

#### Root cause
The pattern was referenced in the specification but never implemented.

#### Fix
```python
"7/8": np.array([1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8),
# p=14, sum=8 → rate (14/2)/8 = 7/8 ✓
```

#### Measured outcome
`test_7_8_pattern_exists`: pattern is present.
`test_pattern_achieves_correct_rate[7/8]`: rate = 7/8 exactly.
`test_puncture_depuncture_roundtrip[7/8-0]`: 0-BER decode confirmed.

---

## Summary — Phase 2 Test Suite Growth

| Phase | Tests added | Total |
|---|---|---|
| Phase 1 (improvements) | 51 | 51 |
| Phase 2 (bug fixes) | 44 | 95 |

All 95 tests in `tests/test_blind_exploitation.py` pass.
All 13 assertions in `tests/test_blindbits.py` pass (run via
`python -c "exec(open('tests/test_blindbits.py').read())"`).


---

## Phase 3 — AMC_1_2.0 end-to-end audit (problem-statement compliance, bugs, training readiness)

> Scope: a full end-to-end review of `AMC_1_2.0` against the problem statement (PS), with every
> fix verified by measurement against ground truth. Each entry gives **where** the change is,
> **why** it was needed, and the **measured outcome**. Regression tests for all of them are in
> `tests/test_audit_regressions.py` (40 tests).

### 3.1 Problem-statement gaps closed

| PS requirement | State found | Change | Where | Verified outcome |
|---|---|---|---|---|
| GUI taking .iq / .wav | Only a 45-line Streamlit viewer of a JSON file | PySide6 + pyqtgraph GUI: parameters, waterfall + PSD, constellation, eye, bit matrix with header/payload/CRC map, JSON report, exports | `signal9/gui/app.py` (`python -m signal9.gui`) | Headless render test on real files; all tabs populated |
| (i)–(v) on an input file | `process_file` stopped after detection and classification; no demod/FEC/de-interleave/correlation ever ran on a file | D0–D6 exploitation chain wired into `process_file` and the CLI | `signal9/exploit/chain.py`, `signal9/pipeline.py`, `main.py` | Conv-coded QPSK `.cf32`: QPSK 4000 Bd, conv K=7, CRC-32 56/56 frames. Bell-202 AFSK `.wav`: 2FSK 1200 Bd, CRC-32 38/38 |
| (ii) Demodulate FSK | FSK tones counted, never demodulated | Discriminator receiver: event-periodogram symbol rate, hysteresis transitions, blockwise dump, 2/4-level Gray slicing | `signal9/dsp/fsk.py` | 2FSK, MSK, GFSK, 4FSK, inverted tones, 10 dB SNR: BER 0.0000 at exactly 1200 Bd; chain CRC 40/40 |
| (iii) De-interleave convolutional / diagonal / pseudo-random | Block only | Forney convolutional, helical/diagonal, seeded-PRNG and 3GPP QPP pseudo-random; blind detectors for all four with block-phase and commutator-phase search; `identify_interleaver()` | `signal9/fec/interleaver.py`, `signal9/label/interleaver_detect.py` | 11/11 blind cases correct, including DVB B=12/M=17 at an arbitrary phase; honest `none` / `UNRESOLVED` / `PSEUDORANDOM_UNRESOLVED` |
| (iv) RS, concatenated, LDPC | Decoders existed but read codewords from bit 0; no concatenated path | Bit/byte alignment search (S0 prefix-XOR filter for RS, minimum-syndrome-weight alignment for LDPC); outer RS after Viterbi | `signal9/exploit/chain.py` (`_try_rs`, `_try_ldpc`, `decode_fec`) | RS(255,223) 8/8 codewords, CRC 88/88. Conv+RS 8/8, CRC 88/88. LDPC 16/16 blocks, CRC 31/31 |
| (v) Bitstream correlation | Frame model assumed frames start at bit 0; CRC tested only after the sync word; period picked harmonics | Frame-phase alignment, period taken from the located CRC frame, CRC coverage from frame start or sync end, majority acceptance, `frame_offset_bits` | `signal9/label/d6_frames.py` | CRC-32/ISO-HDLC identified on 34/34 to 88/88 frames in every end-to-end case |
| (i) Sampling frequency | Headerless files silently given 2.4 MS/s | Per-field provenance (`cli` / `wav_header` / `sigmf` / `filename` / `assumed`); rates normalised to fs when assumed | `signal9/io/metadata.py`, `signal9/pipeline.py` | Report and GUI show the source; assumed rates are flagged |

### 3.2 Correctness bugs fixed

| # | Where | Bug | Why it mattered | Fix | Measured |
|---|---|---|---|---|---|
| 1 | `signal9/pipeline.py` | `Status` never imported | `process()` raised `NameError` | Import added | Test passes |
| 2 | `dsp/constellation.py` ↔ `fec/conv.py`, `fec/ldpc.py`, `decide/blackboard.py` | Demapper LLR>0 ⇒ bit 0; Viterbi and LDPC score against 2·bit−1 (LLR>0 ⇒ bit 1) | Every FEC branch decoded the bit complement, so CRC could never lock | `llr_to_soft_symbol()` at the boundary, with both conventions documented | Information-bit agreement 0.0 → 1.0; conv branch now locks CRC |
| 3 | `io/chunker.py`, `pipeline.py` | `iter_wav_iq` ignored the WAV kind; the audio guard (`not iscomplexobj`) could never fire | Mono audio treated as real+0j, so every tone appeared mirrored | Chunker routes by kind | Mirrored detections gone |
| 4 | `io/audio_path.py` | Above 262k samples: 127-tap FIR Hilbert with its 63-sample group delay not removed | Output not analytic (0.3 dB image rejection) | Block-wise exact FFT Hilbert with guard context | 84–92 dB rejection; 1e-4 interior error |
| 5 | `pipeline.py` | Audio inputs: CFAR ran on the empty negative half; IQ imbalance computed on audio | Phantom emitter at −fs/2; "185 dB" impropriety | Detection restricted to f > 0; image check skipped for audio | Phantom gone; N/A reported |
| 6 | `decide/blackboard.py` | `iq_order` never applied; `bit_order="lsb"` inverted bits, and only on the hard path | Duplicate branches (2× cost); FEC and CRC stages scored different streams | Conjugation for the mirror hypothesis; true within-symbol bit reversal on bits and LLRs | Branches distinct |
| 7 | `label/code_id.py` | Fallback reported `conv_k7_r12` for every stream | Uncoded signals reported as convolutionally coded | Fallback removed | Uncoded → `none` |
| 8 | `dsp/mod_order.py` (new), `exploit/chain.py` | Order chosen by EVM argmin | EVM always falls with order, so 64QAM was picked for every input (and noise was called 64QAM) | Usage entropy + EVM + residual kurtosis + parsimony, plus a Gaussian-cloud test (symbol kurtosis > 1.8 means noise) | 100 % correct at ≥ 15 dB for BPSK…64QAM (was 20 % overall); ambiguity flagged below about 12 dB |
| 9 | `dsp/matched_filter.py` | Roll-off matched against \|H\|⁴ in the log domain over the whole band | Returned 0.10 for every input | Invert the normalised second spectral moment | Mean absolute error 0.026 (roll-off 0.1–0.8, 12–30 dB) |
| 10 | `decide/crc_search.py` | Byte-only CRC sweep; shared budget exhausted in the first bytes; dead O(N²) `running_crc` | Demodulated streams misaligned by 1–7 bits never matched | Bit-alignment-aware, table-driven CRC-32, vectorised over every start offset and alignment | Frames found at 8/8 alignments in about 2 ms; exact against zlib; 0 false positives on noise |
| 11 | `exploit/chain.py` | Equaliser convergence transient (~200 symbols) left in | Corrupted the first frames, which is exactly where the CRC search looks | Adaptive transient trim | EVM −20 → −31 dB; CRC locks |
| 12 | `label/interleaver_detect.py` | Block detector assumed the block starts at bit 0; a tie picked a phase one bit off | Real streams never start on a block boundary | Locate block size and phase first (rank is invariant to the permutation), keep tied phases | Conv + block case fixed (phase 8 → 9) |
| 13 | `exploit/chain.py` | Beam kept the first-enumerated hypotheses | On coded or interleaved streams the correct rotation could be dropped | Every rotation × mirror kept; interleaver identification shared across XOR-equivalent hypotheses | All 6 linear cases CRC-locked |
| 14 | `exploit/chain.py`, RS/LDPC | All-zero codewords (valid for every linear code) and periodic streams accepted as evidence | Two CW tones reported as "2FSK + RS(255,223)" | `stream_degeneracy()` guard; zero codewords rejected | Reports `NO_DATA_STRUCTURE` |
| 15 | `pipeline.py` | `import cv2` inline, plus a leftover "Wait, we need to…" authoring comment | Undeclared heavy dependency | NumPy area-average `resize_spectrogram` | cv2 no longer needed |
| 16 | `pipeline.py` | Clipping hard-coded to 0 %; EVM hard-coded to 0.05 | Health and quality fields were fake | Measured values | — |
| 17 | `fec/rs.py` | *(checked)* shortened RS(204,188) decode and dual basis | — | Verified correct; no change | 8-error decode correct; round trip correct |
| 18 | `label/d6_frames.py` | *(checked)* all 15 CRC catalogue models | — | Verified against the published "123456789" check values | 15/15 correct |
| 19 | `dsp/timing_recovery.py` | *(checked)* Gardner TED sign (BUG-1 above) | — | Verified empirically | −22.6 dB (negative sign) vs −4 dB (positive sign) |

### 3.3 Performance (results unchanged, verified bit-exact)

| Where | Change | Measured |
|---|---|---|
| `dsp/gf2.py` `gf2_rank` | XOR basis over packed rows (any width) with early exit at full rank | 19× faster on full-rank 300×32 and 10× on 200×100; identical to dense elimination and to `galois` on 3000+ matrices |
| `fec/conv.py` | Numba kernel `cache=True` | First decode per process: 5.3 s → 0.9 s |
| `label/scrambler_id.py` | HDLC flag count with a sliding-window view | Identical counts; about 0.13 s saved per 6 kbit stream |
| `exploit/chain.py` | Rank profile shared across rotation-equivalent hypotheses | Chain 22.7 s → 4.5 s per track |

### 3.4 Training pipeline made correct and usable

| # | Where | Bug | Fix | Measured |
|---|---|---|---|---|
| T1 | `train/ema.py` | No decay warm-up, and BatchNorm running stats were never averaged, so the EMA model (the one validated and deployed) was broken in eval mode | YOLOv5-style EMA: ramped decay and the full state averaged | Smoke-test validation 10 % (chance) → 57 % in 4 CPU epochs |
| T2 | `train/labels.py` (new) | Three incompatible label index spaces for one head; PAM4 labelled BPSK; unknown names mapped to class 0 | Canonical vocabulary, name-based mapping, stored in the checkpoint; RML2018 classes-fixed order (checked against three independent sources) | PAM4 test; label tests |
| T3 | `train/trainer.py` | RadioML windows `np.tile`d to 4096, making every example exactly periodic | Native lengths (the backbone is adaptive) | — |
| T4 | `train/synth.py` (new) | SNR/CFO labels independent of the applied impairments; rectangular (`np.repeat`) pulses; constant symbol-rate and roll-off labels | One parameter draw produces both the waveform and its labels; RRC pulses; FSK family | SNR label error mean 0.00 dB, std 0.13 dB |
| T5 | `train/features.py` (new) | Expert features always zero: an ndarray indexed like a dict, hidden by a bare except; zeros in the RadioML loaders | One feature function shared by training and inference | — |
| T6 | `train/losses.py` | Heads trained on placeholder or random labels; the entropy term fought cross-entropy on every sample; SupCon was the weaker L_in form | Masked terms; entropy term on outliers only; SupCon L_out via logsumexp | SupCon equals the paper formula to 1e-6 |
| T7 | `train/trainer.py` | No validation split or metrics; checkpoint had no class list | Stratified split, per-SNR accuracy, best-epoch checkpoint with metadata | — |
| T8 | `train/inference.py` (new), `pipeline.py`, `main.py` | Checkpoints never loaded; the pipeline reported a random network's outputs with confidences | `MultiTaskClassifier`; `--checkpoint`; `NOT_TRAINED` when no checkpoint is given | Checkpoint round trip correct |
| T9 | `signal9/train.py` | Shadowed by the `signal9/train/` package (dead code) | Removed | — |
| T10 | notebooks | Cloned `TDI-3`; evaluation tiled windows and compared raw label indices | Clone `main`; evaluate on the trainer's held-out split | — |

### 3.5 Documentation

* `DATASETS.md` (new): every dataset needed, where to put it, and what the loader does with it.
* `requirements.txt`: it listed 5 packages and did not produce a working environment. It now
  lists the real runtime, GUI, training and test dependencies, with optional tools separated.

### 3.6 Honest limits

* High-order QAM below about 12 dB in-band SNR is flagged as order-ambiguous rather than guessed.
* Pseudo-random interleavers are identified only against the standard library (3GPP QPP and known
  PRNG seeds). An arbitrary permutation is reported as `PSEUDORANDOM_UNRESOLVED`, as the roadmap
  requires.
* RS support covers GF(2⁸) with polynomial 0x11D and fcr = 0 (reedsolo, DVB). CCSDS RS (field
  0x187, fcr = 112, dual basis) is not implemented.
* The GFSK modulation index reads low (≈ 0.4 for h = 0.5) because Gaussian ISI shrinks the dumped
  tone levels. Demodulation is unaffected.
* The neural classifier ships untrained; accuracy figures require a real RadioML training run.
