# Source verification and evidence limits

Verified 2026-09-26. The entries below are provenance for algorithms and test
oracles. They do not establish that this repository has the same accuracy,
coverage, license, API, or deployment readiness as the referenced project.

| Reference | Verified use in this repository | Evidence limit |
|---|---|---|
| [SigMF specification](https://github.com/sigmf/SigMF) | Metadata parsing follows the `.sigmf-meta` / `.sigmf-data` model. | SigMF describes recordings and metadata; it does not label or demodulate a capture. |
| [URH](https://github.com/jopohl/urh) | Workflow/UX reference for protocol inspection. | URH is a separate application; no local claim of code reuse or equivalent blind-protocol performance. |
| [SigDigger](https://github.com/BatchDrake/SigDigger) | Unknown-signal analysis workflow reference. | The cited repository does not validate this project's CFO, timing, or classifier implementations. |
| [inspectrum](https://github.com/miek/inspectrum) | File-format and manual visual measurement reference. | It is an inspection tool, not evidence that automatic channel tracking works here. |
| [IQEngine signal library](https://github.com/spectriclabs/IQEngine-signal-library) | Spectrogram/recording exploration reference. | It is not the SigMF specification and is not used as an automatic detector oracle. |
| [TorchSig](https://github.com/TorchDSP/torchsig) | Optional impairment and synthetic-data generation reference. | Latest upstream treats legacy Sig53 generation as legacy; dataset class order must be read from local metadata, never assumed from TorchSig. |
| [CommPy](https://github.com/veeresht/CommPy) | Differential tests compare local RRC and convolutional-code behavior. | Test agreement on synthetic vectors does not prove blind carrier/timing recovery on OTA captures. |
| [galois](https://github.com/mhostetter/galois) and [reedsolo](https://github.com/tomerfiliba-org/reedsolomon) | Differential tests compare local GF(2) and RS operations. | Agreement covers the tested RS parameters/conventions only. |
| [gr-satellites](https://github.com/daniestevez/gr-satellites) | Protocol and FEC reference for known satellite formats. | A capture needs a correct waveform, framing, code and protocol hypothesis before these tools can recover payload. |
| [pyldpc](https://github.com/hichamjanati/pyldpc) | LDPC BP algorithm reference. | It is not a blind LDPC-family identifier. |

## Model truthfulness

The shipped `s9_hybrid_xcit_L128_best.ckpt` is a Hybrid CNN/XCiT modulation
classifier with an **11-feature** expert vector and a 128-sample I/Q window. It
does **not** train FEC, scrambler, roll-off, or CFO heads in the shipped Kaggle
training loss. Those values may be estimated by DSP stages, but must not be
represented as learned predictions.

The shipped checkpoint has open-set rejection disabled because it was not
calibrated with an unknown/OOD validation partition. It therefore cannot claim
reliable unknown-signal rejection. Calibrate the energy threshold on a held-out
known-vs-unknown dataset before enabling it.

## Acceptance rules

1. A modulation result is accepted only after an independent held-out,
   source-disjoint evaluation reports its confusion matrix and per-class metrics.
2. A demodulation result is accepted only after known payload/bit error testing.
3. A FEC or interleaver hypothesis is accepted only after re-encoding agreement
   or a protocol-level CRC/frame check.
4. A payload is accepted only with a valid CRC, checksum, or independently
   specified frame format.

`UNRESOLVED` is the correct outcome when these acceptance rules are not met.
