# AMC Research V5: evidence, design, and acceptance contract

Date: 2026-09-26

## Scope and non-negotiable meaning of “unknown”

No finite classifier can name every modulation it has never observed. This
project follows the open-set-recognition definition: classify supported known
classes and reject sufficiently novel inputs as `UNKNOWN`. Fine-grained naming
of a new unknown class requires labeled examples, clustering followed by human
verification, or incremental learning.

This is a systematic review of primary papers and author implementations
relevant to this corpus. It is not a claim to have read literally every RF
publication or patent ever issued, and paper-reported accuracies are not copied
across incompatible datasets, taxonomies, SNR ranges, or train/test splits.

## Evidence-to-code matrix

| Problem | Selected evidence | V5 implementation |
|---|---|---|
| Strong closed-set baseline | Published residual AMC and author code | `baseline_resnet`, supervised CE |
| Large Sig53 / longer context | TorchSig Sig53 reports XCiT above tested ConvNets | `hybrid_xcit`: residual RF stem, XCA blocks, attentive pooling |
| Variable lengths | MCDformer/MAMC and feature-pyramid work motivate temporal modeling | Same architecture supports 128/256/512/1024; one fixed length per batch/run |
| QAM/PSK/FSK separation | Raw IQ temporal features plus verified higher-order expert statistics | Learned IQ branch + gated 11-feature expert branch |
| RF invariances | TorchSig and AMC augmentation work | Phase rotation, conjugation, mild CFO/IQ image/dropout on training only |
| Calibration | Guo et al., ICML 2017 | Temperature selected by calibration NLL, never validation/test |
| Unknown rejection | Energy OOD plus AMC open-set prototype literature | Energy threshold fitted for 95% known acceptance using calibration-only held-out families |
| Safe failure | Open-set standards require explicit rejection | Rejection automatically disabled if unknown count or calibration performance is inadequate |
| Domain shift | Cross-dataset AMC and SigDA | Required independent dataset/OTA acceptance test; no internal metric can waive it |
| Signal discovery | Cited spectrum-management patents | Detector/channelizer remains upstream; patents do not prove the classifier architecture |

## Architecture

`hybrid_xcit` uses:

1. Four strided 1-D convolutions for local RF structure and manageable token count.
2. Two residual convolution blocks for symbol transitions and pulse shapes.
3. Three XCiT-style cross-covariance-attention blocks. Attention is across
   feature channels and therefore scales linearly with sequence length.
4. Learned attention pooling rather than uniform global averaging.
5. A gated expert branch with a native 11-dimensional input. No permanent
   zero-padding or invented statistics are passed through LayerNorm.
6. A supervised softmax modulation head. ArcFace remains an explicit ablation,
   disabled by default.

The original residual model remains selectable. V5 does not declare the hybrid
better until both models complete an identical controlled comparison.

## Unknown-signal contract

- Unsupported, explicitly named Sig53 families are excluded from known-class
  CE training and used as open-set calibration/evaluation examples.
- Temperature and the energy threshold are fitted on the 10% calibration
  partition only.
- The threshold targets 95% acceptance of known calibration samples.
- Open-set deployment is enabled only with at least 32 calibration unknowns,
  calibration unknown rejection >= 50%, and calibration AUROC >= 0.65.
- AUROC, FPR95, known acceptance, counts and separate validation unknown
  rejection are stored in every checkpoint.
- If these conditions fail, the checkpoint reports the reason and neural
  unknown rejection stays disabled. No hard-coded universal threshold is used.

## Required acceptance gates

A checkpoint is not production-ready unless all of these are reported:

1. Internal known-class macro accuracy and per-class support/confusion.
2. Per-SNR results, especially low-SNR performance.
3. Calibration known acceptance and temperature.
4. Unknown AUROC and FPR95 on a disjoint unknown-family validation partition.
5. Accuracy and unknown metrics on an entirely independent RadioML/OTA corpus.
6. Receiver metadata: sample rate, bandwidth, center frequency, capture chain
   and preprocessing contract.
7. A same-data, same-split comparison of `baseline_resnet` and `hybrid_xcit`.

Training accuracy or paper accuracy alone cannot satisfy these gates.

## Leakage, conditioning and synthetic-data contract (V5.2)

- Adjacent HDF5 rows are kept together in deterministic 64-row groups before
  the 80/10/10 train/calibration/validation assignment. This is safer than the
  previous row-modulo split. Because the exports contain no capture/session
  identifier, only an external corpus can prove complete independence.
- ModRec files under one receiver/session directory are now inseparable during
  splitting. Other hardware files are kept whole because their exports provide
  no stronger session identity.
- Training, calibration, validation and deployment share non-finite repair,
  phase-preserving impulsive-noise winsorization and RMS normalization. CFO and
  timing correction stay disabled at both ends because blind correction can
  alter FSK/CPM structure and real carrier components.
- The synthetic anchor is 10% of each batch by default, drawn from 8,192
  disjointly partitioned base waveforms and independently RF-augmented during
  training. It cannot dominate the multi-million-sample measured corpora.
- Missing SNR metadata is represented by NaN and masked from regression. It is
  never replaced by an assumed 15 dB value.

## Primary sources and author code

- TorchSig/Sig53: https://arxiv.org/abs/2207.09918
- Official TorchSig: https://github.com/TorchDSP/torchsig
- Official TorchSig Models / XCiT: https://github.com/TorchDSP/torchsig-models
- XCiT: https://arxiv.org/abs/2106.09681
- Residual AMC author code: https://github.com/drexelwireless/RadioML
- Cross-dataset raw-IQ AMC: https://arxiv.org/abs/2307.02450
- MCDformer author code: https://github.com/JHL-HUST/MCDformer
- MAMC author code: https://github.com/ZhangYezhuo/MAMC
- SigDA author code: https://github.com/HantongXING/SigDA
- Contrastive AMC author code: https://github.com/dumingyang20/CSSL-AMC-Pytorch
- Open-ICL: https://doi.org/10.1109/TNNLS.2024.3414942
- Fine-grained open-set AMC: https://doi.org/10.1109/TCCN.2025.3549608
- Few-shot open-set AMC: https://arxiv.org/abs/2410.10265
- Energy OOD: https://arxiv.org/abs/2010.03759
- Official Energy OOD code: https://github.com/weitliu/energy_ood
- Temperature scaling: https://arxiv.org/abs/1706.04599
- OpenMax: https://arxiv.org/abs/1511.06233
- RF data augmentation: https://arxiv.org/abs/1912.03026
- Patent US10459020B2: https://patents.google.com/patent/US10459020B2/en
- Patent US11076308B2: https://patents.google.com/patent/US11076308B2/en

## Explicit limitations

- Structural HDF5 checks establish internal label alignment, not independent
  physical verification of every generated waveform.
- Synthetic-to-OTA domain shift remains possible even with augmentation.
- Unknown rejection detects novelty; it does not infer an unseen standard's name.
- XCiT superiority on Sig53 is evidence for a candidate, not proof on this
  merged 35-class corpus. The controlled benchmark decides deployment.
