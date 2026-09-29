# Independent verification of `AMC_RESEARCH_V5_1_FULL_REVIEW.md`

Date: 2026-09-26

## Bottom line

The external review found three useful defects, one useful maintenance warning,
and two weak tests. It also misidentified the Kaggle execution graph and made
several claims that are contradicted by the code. The supported repairs below
were applied without changing the production CNN/XCiT backbone, vocabulary,
input lengths or label translations.

This audit does **not** certify “SOTA”, “publication-ready”, “competition
winner”, or guaranteed unknown-signal accuracy. Those are experimental claims
that require an independent over-the-air test corpus, receiver-disjoint results,
confidence intervals and comparison against baselines.

## Claim-by-claim result

| Review claim | Verdict | Evidence and action |
|---|---|---|
| RTL-SDR augmentation clips de-scaled samples to `[-1,1]` | **True** in the secondary augmenter | Fixed by saturating signed 8-bit ADC codes and then undoing gain. This follows the official TorchSig quantizer's scale → saturate/quantize → unscale pattern. Added a regression test proving physical amplitude is not arbitrarily capped at one. |
| The code contains two model families | **True as a maintenance issue** | The Kaggle graph, library trainer and experimental multimodal graph are now explicitly mapped in `MODEL_ARCHITECTURE_MAP.md`. Deletion was rejected because `signal9.models` is imported and tested public code. |
| `Signal9Loss` ArcFace weights are disjoint from returned `CosineHead` logits | **True** for the experimental model | Fixed: returned L1/L2/L3 logits receive masked CE. ArcFace is now optional auxiliary supervision. Tests require non-zero gradients in all three returned classification logits. |
| Existing tests validate only the wrong model | **False as a blanket statement** | `tests/L18_signal_infinity` already tests `SignalInfinityLoss` and its training step. New tests now also exercise the actual notebook-local Kaggle heads, all production loss branches and checkpoint compatibility. |
| `np.median(complex)` crashes on NumPy ≥1.24 | **False** | Reproduced successfully on NumPy 2.5.3. NumPy defines complex sorting and current `median` accepts complex arrays. The implementation was still changed to separate I/Q medians because I and Q are independent ADC coordinates and component-wise robust replacement is semantically preferable. |
| Ring-buffer wait is vulnerable to a spurious wakeup | **True** | Both producer and consumer now use `Condition.wait_for(predicate, timeout)`, as required by Python's condition-variable guidance. A deliberate spurious-notify regression test confirms that a full producer stays blocked. |
| Inference always applies CFO correction and SPS resampling | **False for corrected checkpoints** | Both operations are gated by `alignment_preprocess`; the Kaggle checkpoint stores `False`. They are therefore disabled in serving. Blind alignment was not enabled because it may alter FSK/CPM tones or remove a real carrier. |
| Training and serving conditioning differ | **Partly true** | CFO/SPS did not differ, but serving repaired non-finite samples and mitigated strong impulsive contamination while training only normalized RMS. Both Kaggle and library training now use the exact deployment conditioner. A numerical parity test covers NaN and impulsive examples. |
| Receiver/session isolation is absent | **Partly true** | HDF exports do not contain trustworthy session IDs, so that information cannot be invented. Contiguous 64-row groups remain inseparable. For ModRec, the real directory-level receiver/session identity is now used, and all sibling modulations stay in one partition. Other hardware captures remain file-grouped. |
| Generate arbitrary FEC/scrambler labels for synthetic examples | **Unsafe recommendation** | Rejected. The current generators do not encode a verified FEC/scrambler chain, so `-1` correctly masks unavailable labels. Inventing labels would train false correlations. The Kaggle heads are frozen where no verified target exists. |
| Add PLL/Costas, AGC, adversarial training, codec directory | **Research proposals, not proven correctness fixes** | Not inserted before training. A blind PLL/AGC changes the signal distribution and must be trained and ablated end-to-end. Adversarial robustness also needs an RF-valid threat model. These additions could reduce ordinary and cross-domain accuracy. |
| L1 test catches its own assertion | **True** | Fixed so only decoder exceptions are accepted; a non-negative error count now fails the test. |
| L10 checks only a generic word | **True** | Strengthened to require the missing filename and absence of a traceback. |
| `strict=False` xfail and 0.9-radian metamorphic test are necessarily defective | **Not established** | The xfail call is explicit and the phase rotation is a valid modulation-invariance test. No behavior was changed without evidence of a defect. |
| The Kaggle model uses ten-task hierarchical CE + SupCon + SNR-gated fusion | **False** | The actual notebook loss is flat modulation CE plus masked SNR/symbol-rate/presence losses; ArcFace is optional and off by default. The hybrid XCiT notebook backbone uses learned expert gating, not the legacy SNR gate. |

## Repairs applied

1. Correct signed 8-bit quantization and a directly testable helper.
2. Component-wise I/Q median repair for invalid samples.
3. One shared conditioning contract for Kaggle training, library training,
   calibration, validation and inference.
4. Resume rejection for pre-contract checkpoints.
5. ModRec receiver/session-grouped 80/10/10 partitions.
6. Predicate-based ring-buffer waits.
7. Actual classification-head CE in the experimental `Signal9Loss`.
8. Actual Kaggle head gradient tests and training/deployment parity tests.
9. Stronger Reed–Solomon failure and CLI failure tests.
10. A canonical architecture map rather than destructive removal of public code.
11. Parent/child source-overlap protection: native VariableRF excludes direct
    Sig53 at 128/256/512; Sig53 remains active at 1024.

## Verification performed

- Python compilation: passed.
- Focused repaired-path suite: 50 tests passed before the final library parity
  addition.
- Complete suite after every repair: 465 passed, 4 explicitly skipped.
- Actual variable-length corpus: all 15,000,000 labels across the three groups
  physically present in the file (128/256/512) passed shape, label-range,
  balance and class-coverage checks; stratified IQ/SNR blocks passed finite-value
  checks. The file has no `len_1024` group, so 1024 training must come from the
  separately attached Sig53 and RadioML corpora.
- Actual Sig53 file: 5,000,000 rows passed embedded-class-table, label-range,
  mapping and sampled IQ checks.
- One-step CPU training on real mounted data completed, produced best/last/
  production checkpoints, and the production checkpoint loaded and predicted
  through `MultiTaskClassifier` with the new conditioning contract.
- Conditioner benchmark: approximately 5.7 ms for a 256×1024 CPU batch on this
  machine, far below the observed model/data training time.

## Primary references used for decisions

- TorchSig/Sig53 paper: https://arxiv.org/abs/2207.09918
- Official TorchSig quantization implementation:
  https://github.com/TorchDSP/torchsig/blob/main/torchsig/transforms/functional.py
- XCiT: https://arxiv.org/abs/2106.09681
- RF augmentation for modulation classification:
  https://arxiv.org/abs/1912.03026
- RF training-data/augmentation study: https://arxiv.org/abs/2010.00178
- Independent raw-IQ dataset evaluation: https://arxiv.org/abs/2307.02450
- Energy OOD detection: https://arxiv.org/abs/2010.03759
- Python condition-variable requirements:
  https://docs.python.org/3/library/threading.html#condition-objects
- NumPy complex ordering: https://numpy.org/doc/stable/reference/generated/numpy.sort_complex.html
- NumPy median: https://numpy.org/doc/stable/reference/generated/numpy.median.html

## Go/no-go decision

**Go for a fresh controlled Kaggle training run.** Do not resume a V5.1
checkpoint: it lacks the new conditioning and session-split contracts. This
means “the code and available dataset contracts passed the checks above,” not
“field accuracy is guaranteed.” Promote a trained checkpoint only after it
passes the fixed validation split and a separately captured receiver-disjoint
OTA test set.
