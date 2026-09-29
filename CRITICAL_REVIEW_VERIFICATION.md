# Independent verification of `AMC_RESEARCH_V5_FULL_REVIEW.md`

Date: 2026-09-26

## Bottom line

The external review inspected real files, but it combined two different model
stacks. The Kaggle production path is
`notebooks/kaggle_grand_corpus_training.py` plus `signal9/train/inference.py`.
It does not use the IQ+spectrogram `signal9/models/Signal9Net`,
`signal9/models/Signal9Loss`, or `signal9/train/trainer.py` during the requested
grand-corpus run. Therefore several findings were true only in an unused or
secondary path, while several claims about the Kaggle path were false.

## Claim-by-claim result

| Review claim | Verified result | Kaggle impact | Repair |
|---|---|---:|---|
| Auxiliary SNR/baud/outlier losses absent | True in `signal9/models/Signal9Loss`; false for Kaggle, which already trains SNR, baud and presence | None for Kaggle | Added optional masked auxiliary losses to the secondary model loss |
| ModalityDropout can erase IQ and spectrogram together | True in the unused IQ+spectrogram model | None for Kaggle | Joint masks now guarantee at least one signal branch remains |
| AMP disabled | True in secondary `signal9/train/trainer.py`; false in Kaggle (`autocast` + `GradScaler` were already active) | None for Kaggle | Added AMP to secondary trainer |
| RF augmentations unused | True in secondary trainer; false in Kaggle (phase, conjugation, CFO, IQ image and sample dropout already active) | None for Kaggle | Wired augmentation into secondary trainer and recompute coupled expert features |
| Data leakage from random/row split | Partly true. The review cited the wrong trainer, but Kaggle's adjacent-row modulo split was leakage-prone | High | HDF and Panoradio rows now use deterministic, source-specific contiguous 64-row group splits |
| `max_pseudo_fraction` ignored | True | None for supervised Kaggle run | Enforced a confidence-sorted per-class budget |
| Hard-coded sampler seed | True | None for Kaggle | Added caller-controlled seed; `None` gives a fresh generator |
| Train/serve CFO mismatch | False for Kaggle. Checkpoints explicitly save `alignment_preprocess=False`; inference corrections remain off | None | No change |
| AX.25 unstuffing before flag search | True for ndarray bitstreams | Protocol stage only | Flags are found first; only between-flag payload is unstuffed |
| Protocols require byte alignment | True for ndarray bitstreams | Protocol stage only | AX.25, CCSDS and ISM now accept arbitrary 0–7 bit slip |
| ONNX sequence fixed at 4096 | True for the separate multimodal ONNX exporter | Export only | IQ sequence and spectrogram spatial axes are dynamic |

## Additional defects found independently

1. Missing SNR values in CSV manifests and Sig53-without-SNR were replaced by
   an invented 15 dB target. They are now NaN and masked from the SNR loss.
2. The Kaggle network received 32 expert inputs although only 11 were
   populated. New checkpoints use a native 11-dimensional contract.
3. A 2,048-waveform synthetic pool supplied 25% of roughly 500,000 examples
   per epoch, causing severe base-waveform reuse. The default is now an
   8,192-waveform pool at 10% per batch, with disjoint 80/10/10 pools and
   online RF augmentation.
4. Empty/invalid EVM and convolutional-code scoring paths emitted numerical
   warnings. They now return explicit NaN without invalid reductions.

## Claims that were not accepted as proven

- “Best/SOTA architecture in 2026” is not established by the review or by any
  single paper. V5 is an XCiT-inspired candidate and must beat the residual
  baseline on the identical split and an external corpus.
- Cross-modal attention, Mamba, focal loss, self-supervised pretraining,
  adversarial training, and a Costas loop are experiment proposals, not
  correctness fixes. They were not added without controlled evidence on this
  corpus.
- A 96–97% result on a different taxonomy/SNR slice cannot be used as an
  expected score for this 35-class mixed-source task.
- HDF exports do not expose capture/session IDs. The grouped split reduces the
  plausible leakage mechanism but cannot prove independence. An untouched
  independently generated or OTA test remains mandatory.

## Primary evidence used

- TorchSig/Sig53 reports that domain-specific augmentation helps and that its
  tested Transformers outperform its tested ConvNets:
  https://arxiv.org/abs/2207.09918
- Official TorchSig transforms implement RF/channel impairment augmentation:
  https://github.com/TorchDSP/torchsig
- XCiT defines cross-covariance attention and linear token-length scaling:
  https://arxiv.org/abs/2106.09681
- PyTorch AMP requires autocast together with gradient scaling:
  https://docs.pytorch.org/docs/stable/notes/amp_examples.html
- Independent-dataset AMC evaluation demonstrates why internal validation is
  insufficient:
  https://arxiv.org/abs/2307.02450
- AMC augmentation study supports rotation/flip/noise ablations:
  https://arxiv.org/abs/1912.03026
- Energy-based OOD detection supports the calibrated energy score, not a
  universal threshold:
  https://arxiv.org/abs/2010.03759

## Acceptance rule before long training

The code is ready for a controlled Kaggle run only after all local tests,
dataset verification, a real-data forward/backward/checkpoint smoke cycle, and
training/deployment state-dict parity pass. A resulting checkpoint is not
production-ready until it also passes an independent RadioML/OTA evaluation.

