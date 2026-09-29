# Final AMC logic, labeling, and research audit

Date: 2026-09-26

## Verdict

The repaired trainer is a defensible **supervised residual-CNN baseline**, not
proof of the universally best AMC architecture. It is safe to start a fresh
Kaggle run after the checks below. Do not resume the old
`s9_satvik_grand_corpus_best(3).ckpt`: it predates the corrected 35-class,
feature, validation, EMA, and label contracts.

## What was verified locally

- All repository tests: **448 passed, 4 skipped**.
- End-to-end CPU smoke train: one full forward/backward/evaluation/checkpoint
  cycle with a 64-example batch and mounted real HDF5 data.
- `variable_length_rf_corpus.h5`: all **15,000,000** labels scanned; 25 embedded
  class names in the expected order; 128/256/512 IQ shapes correct; every
  class populated; class-count ratio below 1.05; distributed IQ/SNR probes
  finite.
- `sig53_master_5million.h5`: all **5,000,000** labels scanned; 32 embedded
  class names match the file's contract; shape `(N,2,1024)`; every class
  populated and balanced; distributed IQ/SNR probes finite.
- Training and inference feature extraction match under regression tests.
- Inference now rejects incomplete/incompatible backbone weights instead of
  hiding arbitrary differences with unrestricted `strict=False` loading.

These checks prove structural integrity and internal label alignment. They do
not independently prove the physical truth of every generated waveform; that
requires the original generator recipe or a blinded OTA benchmark.

## Label policy (fail closed)

| Source | Accepted labeling evidence | Action |
|---|---|---|
| Variable-RF | HDF5 embedded class table | Accept exact 0..24 labels only |
| Sig53 export | HDF5 embedded `class_names` | Map exact/declared aliases; skip unsupported labels |
| RadioML 2018 | `classes-fixed.json` or `classes-fixed.txt` beside HDF5 | Refuse startup if corrected sidecar is absent |
| ModRec | Exact documented stems such as `qam16_2mhz.dat` | Accept exact stems only; no substring guessing |
| Kiwi | Matching SigMF sidecar, `ci16_le`, one unambiguous annotation | Accept stereo ci16 IQ; reject mono audio |
| Panoradio | Unambiguous tag translation only | Exclude ambiguous multi-tone/protocol labels |
| Other captures | Ground-truth CSV/JSON record | Mask from modulation loss if provenance is missing/conflicting |

Physically defensible equivalences remain: `DQPSK -> QPSK`, `D8PSK -> 8PSK`,
and `pi/4-QPSK -> QPSK`. `16FSK`, OFDM, DSSS, LoRa, chirp, and unknown are
not forced into unrelated closed-set classes. Their rows can be ignored by the
modulation loss rather than poisoning a class.

## Repaired training faults

1. RadioML could use the known ambiguous `classes.txt`; it now requires a
   corrected class-order sidecar.
2. Sig53 could fall back to a version-dependent hard-coded TorchSig order; it
   now requires the class table embedded in the actual file.
3. Generic filenames could silently invent labels; unverified files are now
   masked from classification.
4. Kiwi WAV ingestion assumed a 44-byte header and converted mono audio into
   analytic IQ. It now parses RIFF properly and accepts only documented stereo
   ci16 IQ with matching SigMF metadata.
5. Mixed valid/invalid targets were passed wholesale to ArcFace; `-1` could be
   interpreted as the final class. ArcFace now receives valid rows only.
6. ArcFace was treated as a default despite lacking controlled evidence on
   this corpus. Default is now CE-only (`--arcface-weight 0`); ArcFace is an
   explicit ablation.
7. SNR regression trained on NOISE rows although signal-to-noise ratio is
   undefined for noise-only examples; those targets are now masked.
8. CUDA warm-up changed restored BatchNorm statistics; warm-up now runs in
   evaluation mode.
9. Prototype weight was selected by overall accuracy; optional prototype
   calibration now uses balanced accuracy and a disjoint calibration split.
10. Small smoke batches could be 100% synthetic; the fraction is now truly
    25%, so smoke tests exercise mounted data.
11. Synthetic GFSK duplicated rectangular-pulse CPM/MSK physics; synthetic
    GFSK was removed, while verified real-file GFSK remains usable.
12. HDF label mapping and inference checkpoint loading now fail closed on
    out-of-range labels or unexpected network keys.

## Research comparison

- The current 1-D residual CNN with supervised softmax is supported as a solid
  baseline by published residual AMC work, but not established as globally
  optimal.
- The Sig53/TorchSig paper reports that XCiT-style Transformers outperform its
  tested ConvNets on Sig53 and that RF-domain augmentations help. XCiT is the
  highest-priority architecture comparison for the 1024-sample corpus.
- MCDformer is a credible Transformer comparison for RadioML 2016 at length
  128. Its reported task and taxonomy differ, so its paper accuracy cannot be
  copied as an expected result for this 35-class mixed corpus.
- MAMC/Mamba specifically targets longer sequences and efficiency. It is a
  useful 512/1024 candidate, but must be tested on the same split and class map.
- Contrastive AMC pretraining is promising for noise robustness, but it changes
  the training protocol and is a phase-two experiment after a clean supervised
  baseline.
- ArcFace's primary paper is about face recognition and explicitly discusses
  sensitivity to label noise. There is no basis to call it the best default
  loss for this heterogeneous AMC corpus.
- Cross-dataset AMC studies show that good in-dataset validation can collapse
  on independently generated signals. An untouched RadioML 2016/OTA corpus is
  therefore a required final test, not optional polish.
- The two cited patents concern spectrum detection/management and temporal
  spectrum features. They support a detector/channelizer before AMC; they do
  not establish a replacement modulation-classification architecture.

## Recommended experiment order

1. Train the repaired CE-only residual baseline independently for lengths
   128, 256, 512, and 1024 using the fixed split and 1,953 steps per epoch.
2. Evaluate per class and per SNR on the fixed internal holdout and on one
   completely external dataset/OTA capture that was never sampled in training.
3. On the exact same saved split, compare XCiT at 1024 and MCDformer at 128;
   compare MAMC at 512/1024 if compute permits.
4. Only then test ArcFace or contrastive pretraining as ablations. Adopt a
   change only when macro accuracy, difficult-class confusion, calibration,
   and external-dataset accuracy improve—not because training accuracy rises.

No research paper can guarantee that a checkpoint will classify the user's
real receiver correctly. That claim requires receiver-matched, blinded OTA
testing with documented sample rate, bandwidth, front-end, SNR, and capture
provenance.

## Primary sources and author implementations

- TorchSig/Sig53 paper: https://arxiv.org/abs/2207.09918
- Official TorchSig implementation: https://github.com/TorchDSP/torchsig
- Independent-dataset AMC generalization study:
  https://arxiv.org/abs/2307.02450
- Residual AMC paper implementation: https://github.com/drexelwireless/RadioML
- MCDformer author implementation: https://github.com/JHL-HUST/MCDformer
- MAMC/Mamba author implementation: https://github.com/ZhangYezhuo/MAMC
- Contrastive AMC author implementation:
  https://github.com/dumingyang20/CSSL-AMC-Pytorch
- ArcFace primary paper: https://arxiv.org/abs/1801.07698
- US10459020B2: https://patents.google.com/patent/US10459020B2/en
- US11076308B2: https://patents.google.com/patent/US11076308B2/en
