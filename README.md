# NTRO SIH 26147 — Automated Signal Analysis, Parameter Extraction & Demodulation Engine

> **Problem Statement ID**: `SIH26147` / `PSC26147`  
> **Title**: *Automated model for analysis of .IQ and .wav files along with signal parameter extraction*  
> **Sponsoring Organization**: National Technical Research Organisation (NTRO)  
> **Theme**: Defense & Intelligence / Space Technology / RF Signal Intelligence  

---

## 1. Executive Summary

This repository is a research and engineering prototype for **NTRO Problem Statement SIH 26147**. It accepts headerless **`.IQ`** streams (for example `cf32`, `cu8`, `cs16`, and SigMF) and **`.wav`** recordings, then attempts parameter extraction, modulation classification, demodulation and protocol recovery. A reported modulation, FEC, interleaver, or payload is useful only when its stated evidence gate is passed; unknown captures and unsupported protocols must remain **UNRESOLVED**, not be guessed.

The project features both:
1. **Interactive Full-Stack GUI (`gui.py`)**: Hardware-accelerated PyQt/PySide6 application with real-time time-frequency waterfall, PSD, I/Q constellation diagrams, eye diagrams, and frame/bit stream inspection.
2. **Automated CLI Engine (`main.py`)**: Scriptable, high-throughput pipeline outputting structured, sensor-ready JSON metadata reports for designated downstream processing centers.

---

## 2. Open-Source Repositories & Architectural References

The projects below were checked as real, relevant upstream projects. They are **algorithmic references or differential-test oracles**, not a claim that this project copied their source code, is API-compatible with them, or inherits their validation. `VERIFIED_SOURCES_AND_LIMITS.md` records the exact verification and current limitations.

| Repository & Link | Core Capability Adopted | Implementation in this Codebase |
| :--- | :--- | :--- |
| **[jopohl/urh (Universal Radio Hacker)](https://github.com/jopohl/urh)** | Valid reference tool for wireless-protocol investigation | UX/workflow inspiration only; no code-equivalence claim |
| **[BatchDrake/SigDigger](https://github.com/BatchDrake/SigDigger)** | Valid real-time unknown-signal analyser | UX/DSP workflow reference; individual estimators are independently tested here |
| **[SigMF specification](https://github.com/sigmf/SigMF)** / [IQEngine signal library](https://github.com/spectriclabs/IQEngine-signal-library) | SigMF is the metadata standard; IQEngine is a recording visualisation tool | `signal9/io/sigmf_io.py`, `signal9/detect/spectrogram.py` |
| **[miek/inspectrum](https://github.com/miek/inspectrum)** | Valid capture inspection, visual measurement and manual symbol extraction tool | File-format and visual-analysis reference; not an automatic tracker implementation |
| **[MrZoller/sdr-playground](https://github.com/MrZoller/sdr-playground)** | Educational NumPy SDR/DSP reference | Supplementary educational reference, not a production oracle |
| **[TorchDSP/torchsig](https://github.com/TorchDSP/torchsig)** | RF impairment/data-generation toolkit | Optional synthetic-data generation and impairment reference; current TorchSig no longer distributes legacy Sig53 generation directly |
| **[veeresht/CommPy](https://github.com/veeresht/CommPy)** | RRC, modulation and coding reference implementation | Differential-test oracle for RRC and convolutional coding |
| **[daniestevez/gr-satellites](https://github.com/daniestevez/gr-satellites)** | Satellite telemetry and CCSDS/AX.25 decoding reference | Protocol/FEC reference; a protocol must still be identified and CRC-validated locally |
| **[mhostetter/galois](https://github.com/mhostetter/galois) / [reedsolo](https://github.com/tomerfiliba-org/reedsolomon)** | GF arithmetic and Reed-Solomon implementation references | Differential-test oracles for local GF/RS functions |
| **[hichamjanati/pyldpc](https://github.com/hichamjanati/pyldpc)** | LDPC belief-propagation reference | LDPC algorithm reference. The previous `orion-sdr` attribution is removed: it was not verifiable. |

---

## 3. End-to-End System Architecture

```
 Raw RF Input (.IQ, .wav, .cf32, .cu8, .sigmf)
                     │
                     ▼
  ┌────────────────────────────────────────────────────────┐
  │  Stage 0: Ingestion & Front-End Health Checks          │
  │  - Chunker & Auto-Sniffing (CF32, CS16, CU8, WAV, etc.) │
  │  - DC / LO Leakage Detection & Centered Notch Masking   │
  │  - I/Q Impropriety & Imbalance Correction (Gram-Schmidt)│
  │  - Adaptive Threshold Clipping Mitigation              │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │  Stage 1: Channelization & Parameter Extraction        │
  │  - CFAR (Constant False Alarm Rate) Spectral Detector  │
  │  - Sampling Frequency (Fs) Estimation & Provenance Tag  │
  │  - Baud Rate Estimation via Cyclic Autocorrelation     │
  │  - CFO (Carrier Frequency Offset) Estimation via M-th  │
  │  - SNR Estimation via M2M4 Higher-Order Cumulants      │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │  Stage 2: Automatic Modulation Classification (AMC)    │
  │  - 11-feature expert-vector checkpoint input (HOC-derived)│
  │  - Hybrid 1D-ResNet + XCiT Attention Neural Backbone   │
  │  - 35-Class Hierarchical Taxonomy (L1/L2/L3)           │
  │  - Open-set rejection only with a calibrated checkpoint│
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │  Stage 3: Adaptive Demodulation & Equalization         │
  │  - Linear (BPSK, QPSK, 8PSK, 16QAM, 64QAM):            │
  │    * Interpolation to 4 SPS -> Adaptive RRC Filtering  │
  │    * Gardner Timing Error Detector (TED) Clock Recovery│
  │    * CMA / Decision-Directed Adaptive Equalizer        │
  │  - Continuous Phase (2FSK, 4FSK, GFSK, MSK):           │
  │    * Delay-and-Multiply Instantaneous Frequency Discrim│
  │  - Soft Log-Likelihood Ratio (LLR) Generation          │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │  Stage 4: Blind De-Interleaving & De-Scrambling        │
  │  - Ambiguity Traversal (Rotations, Inversions, Line)   │
  │  - Candidate interleaver geometry scoring via GF(2)     │
  │  - Matrix / Block De-interleaver (M x N)               │
  │  - Convolutional De-interleaver (Ramsey / Forney Banks)│
  │  - Diagonal De-interleaver                             │
  │  - De-Scramblers: CCSDS PN255, G3RUH (x17), V.35 (x23) │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │  Stage 5: Forward Error Correction (FEC)               │
  │  - Soft-decision Viterbi Decoder (K=7, Rate 1/2)       │
  │  - Reed-Solomon candidate decoders: CCSDS/DVB families  │
  │  - Concatenated Codes (Viterbi Inner + RS Outer)       │
  │  - Log-SPA Belief Propagation LDPC Decoder             │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │  Stage 6: Bitstream Correlation & Protocol Deframing   │
  │  - 0..7 Bit-Slip Synchronization Search                │
  │  - Sync Word Correlation (CCSDS ASM 0x1ACFFC1D, AX.25) │
  │  - CRC Verification & Identification (CRC-16, CRC-32)  │
  │  - Header/payload extraction only after evidence gates  │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
               ┌───────────────────────────┐
               │ Visual GUI / JSON Output  │
               └───────────────────────────┘
```

---

## 4. Deep-Dive: Addressing the 5 Problem Statement Deliverables

### Deliverable I: Identify Signal Parameters

The listed estimators are measured hypotheses, not universally recoverable facts. In particular, a headerless raw capture cannot reveal its absolute sample rate from samples alone; pass `--fs` or use SigMF/WAV metadata. Every returned value includes provenance.
1. **Sampling Frequency ($F_s$)**: 
   * Reads metadata headers for `.wav`, `.sigmf`, or SDR recordings. 
   * For headerless raw `.iq` captures, runs an iterative multi-rate cyclostationary search evaluating spectral roll-off corners and occupied bandwidth (OBW 99% energy integration).
2. **Carrier Frequency Offset (CFO)**:
   * Uses $M$-th power non-linear phase extraction: $\hat{f}_{\text{cfo}} = \frac{1}{2\pi M T} \arg \left( \sum_{n} (r[n])^M \cdot (r[n-1])^{-M} \right)$ to strip PSK modulation without knowing the payload.
3. **Baud Rate ($R_s$)**:
   * Evaluates the cyclic autocorrelation function (CAF) and instantaneous frequency zero-crossing intervals. Verified via Gardner TED convergence.
4. **Signal-to-Noise Ratio (SNR) & EVM**:
   * Evaluates $M_2 = E[|r|^2]$ and $M_4 = E[|r|^4]$ moments. Constellation Error Vector Magnitude (EVM in dB) is continuously tracked to ensure synchronization lock.

### Deliverable II: Demodulate Signals (FSK, QAM, PSK, ASK)
1. **Linear Modulation (PSK / QAM)**:
   * Matched filtering using a parameterized Root Raised Cosine (RRC) filter with roll-off factor estimation $\alpha \in [0.10, 0.70]$.
   * Symbol timing recovery using a 2-sample/symbol Gardner TED loop with a proportional-integral loop filter.
   * Adaptive equalization using Constant Modulus Algorithm (CMA) for blind channel inversion, switching to Decision-Directed Least Mean Squares (DD-LMS) when EVM drops below $-12\text{ dB}$.
   * Emits soft Log-Likelihood Ratios (LLRs) preserving confidence for the downstream decoders.
2. **Frequency Modulation (2FSK, 4FSK, GFSK, MSK)**:
   * Delay-and-multiply frequency discriminator: $d[n] = \arg(r[n] \cdot r^*[n-1])$.
   * Automated tone center clustering and symbol decision threshold estimation.

### Deliverable III: Carry Out De-Interleaving

Blind interleaver identification is an underdetermined statistical search. Results are hypotheses until independently validated by re-encoding, a protocol check, or a CRC. The pipeline must report `UNRESOLVED` when no validation exists.
1. **Block / Matrix Interleaver**:
   * Reads data into an $M \times N$ matrix column-wise and unrolls row-wise.
2. **Convolutional Interleaver**:
   * Simulates a bank of $B$ shift registers with delay increments $M \cdot (i-1)$ (Ramsey Type III / Forney standard).
3. **Diagonal Interleaver**:
   * Matrix traversal reading across diagonal indices $(i+j) \pmod N$ to disperse two-dimensional burst fading.
4. **Pseudo-Random De-interleaver / De-scrambler**:
   * Synchronous additive PN255 (CCSDS).
   * Self-synchronizing multiplicative polynomial descramblers: G3RUH ($1 + x^{12} + x^{17}$) and ITU V.35 ($1 + x^{18} + x^{23}$).
5. **Blind Detection**:
   * Uses a GF(2) Gaussian elimination rank test across sliding window lengths: structured code blocks exhibit rank deficiency when the correct interleaver geometry is applied.

### Deliverable IV: Forward Error Correction (FEC)

FEC decoding requires a known or strongly evidenced code family, rate, framing and bit convention. The available decoders are candidate tools; they do not identify arbitrary FEC schemes from arbitrary raw RF alone.
1. **Convolutional Code with Viterbi Decoding**:
   * Constraint length $K=7$, rate $r=1/2$, generator polynomials $G_1 = 171_8$ (`0x6D`) and $G_2 = 133_8$ (`0x4F`). Supports both soft-decision LLR and hard-decision branch metrics.
2. **Reed-Solomon (RS) Block Codes**:
   * Full Galois Field $GF(2^8)$ Berlekamp-Massey algorithm with Chien search and Forney error value calculation:
     * **CCSDS Standard**: $RS(255, 223)$ capable of correcting up to $t=16$ byte errors.
     * **DVB Standard**: $RS(204, 188)$ capable of correcting up to $t=8$ byte errors.
3. **Concatenated Codes**:
   * Pipeline connecting inner soft Viterbi decoding directly to the de-interleaver and outer Reed-Solomon decoder.
4. **LDPC (Low-Density Parity-Check)**:
   * Belief Propagation algorithm operating in the log-domain (Log-SPA) to prevent numerical underflow over sparse parity-check matrices ($H$).

### Deliverable V: Bit Stream Correlation & Header/Payload Identification
1. **Bit-Slip Ambiguity Elimination**:
   * Baseband demodulation cannot guarantee whether the first recovered bit matches a byte boundary. The pipeline tests all 8 phase offsets ($\text{slip} \in \{0, 1, \dots, 7\}$ bits).
2. **Constellation Phase Ambiguity Resolution**:
   * For $M$-PSK and QAM, the constellation locks at an arbitrary multiple of $2\pi/M$. The pipeline evaluates all 4 or 8 candidate phase rotations ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) and spectral inversions ($I \leftrightarrow Q$).
3. **Sync-Word Correlation**:
   * Fast bitwise cross-correlation with known synchronization preambles:
     * **CCSDS Telemetry**: `0x1ACFFC1D` (32-bit Attached Sync Marker).
     * **AX.25 Packet Radio**: `0x7E` HDLC flags, LSB-first octets, strict zero-bit unstuffing, and CRC-16/X-25 FCS validation.
     * **ISM Band**: `0xD391` sync word.
4. **CRC Verification**:
   * Computes CRC-16-CCITT (polynomial `0x1021`), CRC-16-IBM (`0x8005`), and CRC-32-IEEE (`0x04C11DB7`) to validate recovered packet integrity.

---

## 5. Software Structure

```
NTRO_SIH26147_SignalAnalysis/
├── main.py                     # CLI Entry Point
├── gui.py                      # GUI Entry Point (PySide6 / pyqtgraph)
├── run_demo.py                 # Self-contained End-to-End Demo Script
├── requirements.txt            # Python dependencies
├── README.md                   # This detailed engineering report
├── configs/
│   └── taxonomy.yaml           # Hierarchical L1/L2/L3 modulation taxonomy
├── models/
│   ├── s9_hybrid_xcit_L128_best.ckpt  # Trained Hybrid ResNet-XCiT Checkpoint
│   └── s9_hybrid_xcit_L128_last.ckpt  # Epoch Checkpoint
├── signal9/                    # Core Architecture Package
│   ├── pipeline.py             # Signal9Engine master orchestrator
│   ├── preprocess.py           # Finite repair, impulsive winsorization, RMS normalization
│   ├── dsp/                    # Signal Processing Algorithms
│   │   ├── cfo.py              # Carrier frequency offset estimation & correction
│   │   ├── constellation.py    # Symbol-to-bit mapping & constellation plotting
│   │   ├── equalizer.py        # CMA & Decision-Directed adaptive equalizers
│   │   ├── fsk.py              # FSK frequency discriminator demodulator
│   │   ├── gf2.py              # Galois Field GF(2) linear algebra & rank tests
│   │   ├── matched_filter.py   # RRC filtering and alpha roll-off estimation
│   │   ├── snr_estimate.py     # M2M4 cumulant SNR estimation
│   │   └── timing_recovery.py  # Gardner TED symbol clock synchronizer
│   ├── fec/                    # Forward Error Correction Decoders
│   │   ├── conv.py             # Soft/Hard Viterbi convolutional decoder (K=7, r=1/2)
│   │   ├── interleave.py       # Block, Convolutional, Diagonal de-interleavers
│   │   ├── ldpc.py             # Log-SPA Belief Propagation LDPC decoder
│   │   └── rs.py               # Reed-Solomon (255,223) & (204,188) decoder
│   ├── exploit/
│   │   └── chain.py            # D0-D6 Blind Exploitation Pipeline
│   ├── gui/
│   │   └── app.py              # PySide6 Multi-threaded GUI application
│   ├── io/                     # Chunker, WAV, and SigMF binary readers
│   ├── protocol/               # AX.25, CCSDS, and ISM deframers
│   └── train/                  # Neural network backbones and training routines
│       ├── backbone.py         # ResNet1D & Hybrid XCiT models
│       ├── features.py         # Expert feature extraction (shipped checkpoint uses 11)
│       ├── heads.py            # Eight multi-task head definitions
│       └── inference.py        # MultiTaskClassifier runtime wrapper
└── tests/                      # 465+ Comprehensive Unit and Integration Tests
```

---

## 6. Installation & Quick Start

### Receiver review improvements

The file-analysis path now supplements burst CFAR with linear-power time
averaging and frequency filters from 2 to 64 bins, using an iteratively masked
noise floor. It grows frequency support from the averaged spectrum and merges
overlapping detections. No detection now remains `NO_SIGNAL_DETECTED`; the
pipeline no longer creates an unconditional fallback track. The thresholds
are empirical, so a CFAR name does not imply a calibrated false-alarm rate.

Channels are filtered to their detected bandwidth before baud estimation.
Independent in-band SNR is measured using clean spectrum on both sides with
other detected bands excluded. That estimate is converted to symbol Es/N0
for EVM consistency and noise-corrected fourth-cumulant checks. Low-SNR order
decisions abstain, retaining PSK/QAM family evidence when available. Neural/DSP
disagreements are explicit in `modulation_decision`. Missing or clipped
roll-off estimates are null; `carrier_offset_hz` combines coarse and residual
offsets. Reports flag overlapping tracks sharing a near-identical baud rate
and detector/post-filter SNR differences exceeding 3 dB. These flags indicate
possible interference, not proof of it.

Byte previews search all eight offsets, both polarities, and both byte bit
orders. The chosen alignment and ambiguity are reported; readable text alone
does not validate a payload. Whole-frame repetition cannot count as a sync
word. To measure BER against a known bit sequence at an explicit alignment:

```bash
python main.py capture.cf32 --fs 256000 --reference-bits expected.txt --reference-offset 3 --json
```

The GUI offers **Compare reference bits…** in the Bit stream tab. Reference
files contain `0`/`1` text with optional whitespace. BER reports include the
number of compared bits and whether the reference was fully covered.

Run the reproducible development benchmark and regression cases with:

```bash
python tools/benchmark_receiver_review.py --trials 20 --output receiver-review.json
pytest tests/test_feedback_improvements.py -q
```

The benchmark compares burst CFAR with the supplemented detector and measures
order decisions with known timing. It records per-SNR confusion matrices,
abstention, accuracy on decided cases, detection probability, capture-level
false-alarm probability, Wilson intervals, and a SHA256 commitment to its
generation specification. Its disjoint `--split holdout` seed range should
remain unused during tuning; this is not an access-controlled sealed corpus.
It covers AWGN, not the entire multipath/overlap/real-SDR operating envelope.

These changes do not retrain or recalibrate neural checkpoint weights.
Successive interference cancellation, deployment-set temperature fitting,
and validation against new real SDR recordings remain future work. Existing
blind equalization remains in the receiver. The report's family evidence is
statistical evidence, not a calibrated confidence probability.

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Key packages: `numpy`, `scipy`, `torch`, `PySide6`, `pyqtgraph`, `reedsolo`, `pyyaml`)*

### 3. Run the Demonstration Script
A deterministic diagnostic script generates a QPSK/AX.25 capture and runs the pipeline. It prints **PASS** only if it actually finds a CRC-valid frame; otherwise it prints **NOT VALIDATED** and writes the diagnostic JSON. It must not be used as a performance claim:
```bash
python run_demo.py
```

---

## 7. User Guide: GUI & CLI

### Option A: Graphical User Interface (`gui.py`)

Launch the visual workspace:
```bash
python gui.py
```

#### GUI Highlights
1. **Top Bar**: Select input `.wav` or `.iq` file, set sample rate / center frequency overrides, choose model checkpoint, and hit **Analyze**.
2. **Left Panel**: Displays all detected signal tracks, bandwidths, center offsets, and SNRs.
3. **Tab 1: Parameters**: Clean tabular view of:
   * Sampling frequency ($F_s$), Center Frequency ($F_c$), SNR, EVM.
   * Modulation classification (L1 Family, L2 Specific Class).
   * Demodulator settings: Baud rate, estimated roll-off ($\alpha$).
   * Interleaver geometry and Scrambler identity.
   * FEC code identity (Convolutional, Reed-Solomon, LDPC).
   * CRC status and packet containment.
4. **Tab 2: Waterfall & PSD**: Interactive 2D time-frequency spectrogram showing signal activity over time.
5. **Tab 3: Constellation**: Live In-phase vs Quadrature (I/Q) scatter plot after timing lock and equalization.
6. **Tab 4: Eye / Time Domain**: Time-domain I/Q traces and eye diagrams verifying eye opening.
7. **Tab 5: Bit Stream**: Hex and ASCII payload viewers showing recovered telemetry/data.
8. **Tab 6: Report**: Complete machine-readable JSON summary with copy/export functionality.

---

### Option B: Command Line Interface (`main.py`)

The CLI is optimized for automated sensor processing:

1. **Analyze a `.wav` file**:
   ```bash
   python main.py capture.wav
   ```

2. **Analyze a raw `.iq` / `.cf32` capture with explicit sample rate**:
   ```bash
   python main.py capture.cf32 --fs 48000 --format cf32
   ```

3. **Analyze an 8-bit unsigned RTL-SDR capture (`cu8`) at 2.4 MHz**:
   ```bash
   python main.py capture.iq --fs 2400000 --format cu8
   ```

4. **Export structured JSON for downstream sensor networks**:
   ```bash
   python main.py capture.cf32 --fs 48000 --format cf32 --save-json report.json
   ```

---

## 8. Verification & Test Suite

The codebase includes an extensive, multi-tier test gauntlet validating mathematical exactness:
* **L0 Contracts**: Schema, metadata, and invariant tests.
* **L1 Golden**: Known-Answer Tests (KAT) for scramblers, framing, and codecs.
* **L4 Differential**: Differential testing against reference libraries:
  * Reed-Solomon vs `reedsolo`
  * Convolutional codes vs `CommPy`
  * CRC routines vs `crcmod`
  * GF(2) linear algebra vs `galois`
* **L10 Integration**: CLI and end-to-end pipeline execution smoke tests.

To run the integration test suite:
```bash
pytest tests/L10_integration/test_cli.py
```

---

## 9. Conclusion & Competition Readiness

This system supplies a serious starting point for every **SIH 26147** workstream, but it must be evaluated as a research system:

* ✅ Ingests supported raw I/Q, SigMF and WAV formats with health/provenance reporting.
* ✅ Provides trained 35-class AMC, parameter-estimation and DSP demodulation paths.
* ✅ Provides candidate de-interleaving, FEC, framing and CRC-analysis paths.
* ✅ Provides GUI visualisation and a structured JSON report.
* ⚠️ Absolute sample rate cannot be inferred from headerless raw bytes; supply `--fs`.
* ⚠️ The shipped checkpoint has no calibrated OOD/open-set gate and is not evidence of arbitrary-signal generalisation.
* ⚠️ FEC, interleaver and payload results are accepted only after independent validation such as a CRC or re-encoding check.

The intended next milestone is a held-out, source-disjoint benchmark that reports per-class AMC metrics, bit error rate, frame success rate and false-lock rate before any operational claim is made.
