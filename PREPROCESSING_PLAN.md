# Classifier Input Conditioning Plan

The aim is stable, information-preserving I/Q at the modulation classifier,
not a claim that a universal "perfect" waveform exists.  Every correction is
gated and recorded so it can be measured against a no-correction baseline.

## Ordered runtime contract

1. **Ingest and preserve provenance.** Decode the sample format without
   clipping, retain sample rate / center frequency / zero-IF state, and repair
   only non-finite float samples.  A missing sample rate must remain labelled
   as assumed.
2. **Wideband health before detection.** Measure clipping, DC/LO evidence,
   IQ-image evidence and AGC ruptures on the capture.  Apply the existing
   zero-IF, wideband-only GSOP IQ correction only when mirror-image evidence
   supports it.  Do not infer IQ imbalance from a channelized BPSK signal.
3. **Adaptive spectral baseline and detection.** Build a robust time-frequency
   noise floor, whiten against it, then use CFAR/tracking to isolate each
   emitter.  This follows the supplied patent's useful principle: establish a
   baseline first, then retain target-band I/Q and its temporal parameters.
   The runtime also keeps a slow, per-frequency temporal baseline across file
   chunks; it adds novelty evidence for a newly elevated bin without replacing
   the local whitening decision or learning an active signal as noise.  A
   robust common-mode correction excludes receiver-wide AGC shifts from that
   novelty signal.
4. **Channelize with guard.** Frequency-shift the selected emission to zero,
   low-pass/resample with a polyphase anti-alias filter, preserve a guard band,
   and retain the estimated bandwidth, SNR, center frequency and duration.
5. **Conservative per-emitter conditioning.** The new
   `condition_for_classifier` repair removes NaN/Inf samples, detects clear
   impulsive contamination, uses phase-preserving robust winsorization only
   when that gate fires, then RMS-normalizes.  It intentionally preserves DC:
   after channelization a DC component can be a wanted AM/CW/OOK carrier.
6. **Parameter-aware corrections.** Estimate CFO, timing/symbol rate and
   samples-per-symbol.  Correct CFO only with a confidence gate and never by
   treating FSK tone separation as CFO.  Resample only when the symbol-rate
   estimate is credible; record the rational ratio.
7. **Hypothesis-specific equalization.** Do not run a CMA/MMA blindly before
   AMC.  Constant envelope is shared by PSK and FSK/CPM.  Use the classifier's
   candidate/hypothesis to decide whether an equalizer is applicable, accept
   it only when held-out EVM/classification metrics improve, and retain both
   the original and equalized branch during evaluation.
8. **Multi-window classifier input and quality gate.** Classify several
   separated windows, aggregate calibrated posteriors, and emit UNKNOWN when
   SNR, clipping, AGC instability or disagreement makes the result unreliable.

## Training-data acquisition and release gate

Use one frozen external hold-out corpus for every model promotion.  The
repository's `iq_test_dataset` is suitable as a smoke/contract set but is too
small to be the only test set.  Acquire the following into ignored `dataset/`
or `data/` directories, retaining each source's licence and a SHA-256 manifest:

1. DeepSig RadioML 2018.01A from the official DeepSig dataset portal for
   1024-sample, labelled I/Q across 24 classes and SNR buckets.
2. RadioML 2016.10a/10b for an independent, smaller validation domain.
3. Labeled real SDR/SigMF anchors that cover the intended receivers, formats,
   1-16 samples/symbol, CFO, pulse shaping and channel conditions.  Keep at
   least 10 percent anchors in training and reserve whole capture sessions for
   test to avoid leakage.
4. Synthetic waveforms only to fill documented class/impairment gaps.  Render
   the same preprocessing and short-window policy used by inference.

For every candidate checkpoint, save source manifests/hashes, class and SNR
counts, split seed and group key, train history, validation count, per-class
and per-SNR accuracy, ECE, confusion matrix, and the locked-corpus result.
Never promote from a single overall validation accuracy.

## Verification matrix

For every change, compare the full pipeline with the correction disabled on
clean data and on controlled impairments: DC/LO, IQ imbalance, CFO, timing
error, two-path multipath, impulsive noise, clipping, AGC steps, co-channel
interference and FSK.  Require no clean-set regression and report accuracy by
modulation, SNR and impairment bucket.  The preprocessing audit returned with
each trained classification makes this testable in production.

## Sources and scope

- The supplied US 11,076,308 document is a spectrum-management patent.  Its
  directly applicable content is wideband sweep/baseline construction,
  channelized I/Q retention, and symbol/protocol parameter handling; it does
  not prescribe a universal AMC equalizer.
- Weber, Peter and Felhauer (2015), *Automatic modulation classification
  technique for radio monitoring*, identifies CFO, phase, timing and power as
  the core AMC preprocessing estimates.
- Analog Devices AN-1039 documents that DC offsets and IQ gain/phase imbalance
  distort constellations, especially at higher QAM orders.
- The MITRE AMC patent application US 20240243952 is corroborating evidence
  for resampling plus constant-modulus blind equalization, but only for its
  stated constant-modulus setting.  That limitation is why this plan keeps
  equalization hypothesis-gated.
- US 10,459,020B2 independently reinforces the adaptive-detection portion:
  it uses per-bin power distributions learned over time, gradients to bound
  signals, and baseline subtraction.  Its temporal-baseline concept is used
  only to improve emitter isolation before AMC, not as evidence for a
  modulation label.
