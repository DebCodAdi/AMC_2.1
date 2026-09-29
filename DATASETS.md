# Datasets required to train Signal 9

This file lists every dataset the training code reads, where to get it, where to put it, and
what the loader does with it. The loaders live in `signal9/train/trainer.py`. Their labels are
mapped by **name** into the canonical vocabulary in `signal9/train/labels.py`, and that class
list is saved inside every checkpoint.

No dataset is committed to this repository (`.gitignore` excludes `dataset/`, `data/`, `*.raw`,
`*.iq`, `*.wav`, `*.sigmf-*`). Download them separately.

## Summary

| Dataset | Needed for | Size | Classes | Window | Loader |
|---|---|---|---|---|---|
| **DeepSig RadioML 2018.01A** (primary) | Neural modulation classifier | ~20 GB HDF5, 2,555,904 examples | 24 | 1024 × IQ | `RadioML2018LazyDataset` |
| DeepSig RadioML 2016.10a / 10b | Secondary / quick experiments | ~600 MB / ~1.2 GB pickle | 11 / 10 | 128 × IQ | `KaggleRadioMLDataset` |
| Synthetic (generated) | Always available; FSK coverage; tests | none (generated) | 10 | configurable (1024) | `SyntheticRFDataset` |
| Own real captures ("anchors", Rule R15) | Real-hardware anchor, ≥ 10 % of batches | any | per filename | 1024 crop | `SyntheticBatchDataset` |

The **exploitation chain** (demodulation, de-interleaving, FEC, frame/CRC analysis) needs **no
training data**. It is measurement- and algebra-based, and it works on any capture as soon as the
repository is installed. The datasets above train only the neural modulation classifier. Its label
is reported alongside the chain's measured modulation, and it can act as a tie-breaking hint.

---

## 1. DeepSig RadioML 2018.01A — primary training set

* **Get it:** <https://www.deepsig.ai/datasets> (free registration, CC BY-NC-SA 4.0). It is also
  mirrored on Kaggle, where the notebooks find it automatically under `/kaggle/input`.
* **File:** `GOLD_XYZ_OSC.0001_1024.hdf5` with datasets `X` (N × 1024 × 2 float32 I/Q), `Y`
  (N × 24 one-hot) and `Z` (N × 1 SNR in dB, −20 … +30 dB in 2 dB steps).
* **Put it at:** `dataset/GOLD_XYZ_OSC.0001_1024.hdf5` (or `data/…`, or anywhere under
  `/kaggle/input`). Alternatively pass the path explicitly:
  `run_training(dataset="path/to/GOLD_XYZ_OSC.0001_1024.hdf5")`.
* **Class order.** One-hot position *i* is `RML2018_CLASSES_FIXED[i]`:
  `OOK, 4ASK, 8ASK, BPSK, QPSK, 8PSK, 16PSK, 32PSK, 16APSK, 32APSK, 64APSK, 128APSK, 16QAM,
  32QAM, 64QAM, 128QAM, 256QAM, AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, AM-DSB-SC, FM, GMSK, OQPSK`.
  This is DeepSig's *classes-fixed* order. The `classes.txt` shipped with the original release
  lists a different order that does **not** match the labels, so the loader never reads it. If a
  `classes-fixed.json` / `classes-fixed.txt` is present next to the HDF5, it is used instead.
* **Loading:** lazy, one example at a time from the HDF5, so the file is never loaded into RAM.
  Windows are used at their native 1024 samples. `max_samples` subsamples with an even stride,
  which keeps the class/SNR balance because the file is stored in (class, SNR) blocks.

## 2. DeepSig RadioML 2016.10a / 2016.10b

* **Get it:** <https://www.deepsig.ai/datasets> (`RML2016.10a.tar.bz2` / `RML2016.10b.tar.bz2`).
* **File:** `RML2016.10a_dict.pkl` (or `RML2016.10b.dat`), a pickled dict
  `{(modulation, snr): ndarray(N, 2, 128)}`.
* **Put it at:** `dataset/RML2016.10a_dict.pkl` (the filename must contain `rml`, `radioml`,
  `deepsig` or `2016` for auto-detection), or pass the path to `run_training(dataset=...)`.
* **Classes:** `8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM`
  (10b has no AM-SSB). Every name maps explicitly; an unknown name raises instead of silently
  becoming class 0, which is how PAM4 used to be labelled BPSK.
* **Loading:** native 128-sample windows, with an equal share of every (modulation, SNR) cell
  under `max_samples`.

## 3. Synthetic data (no download)

`SyntheticRFDataset` (see `signal9/train/synth.py`) renders examples on the fly from a single
parameter draw, so every label describes the actual signal:

* **Modulations:** BPSK, QPSK, 8PSK, 16QAM, 64QAM (Gray-mapped, RRC pulse shaping with roll-off
  0.1–0.5 at 2–16 samples/symbol), 2FSK, 4FSK, MSK, GFSK (continuous phase), and NOISE.
* **Channel:** random phase, CFO up to ±0.02 cycles/sample, optional two-ray multipath, optional
  impulsive noise, and AWGN at the labelled SNR (−4 … +24 dB). The label matches the measured SNR
  to within 0.13 dB.
* **Labels:** modulation, symbol rate (symbols/sample), roll-off, CFO, SNR and presence. FEC and
  scrambler are left unlabelled (−1): the chain identifies them algebraically.

It is the fallback when no RadioML file is found. It is also the only source of FSK/MSK/GFSK
examples besides RML2016's CPFSK/GFSK, so mix it in if you need FSK coverage.

## 4. Real "anchor" captures (Rule R15)

To keep the model grounded in real hardware, put your own recordings in `dataset/` as complex64
`*.cf32` / `*.raw` files whose **filename names the modulation**, for example
`qpsk_vhf_2026-03-01.cf32` or `2fsk_aprs_144800k.cf32`. `SyntheticBatchDataset` then draws at least
`anchor_fraction` (default 15 %) of each epoch from them. Files that name no known modulation are
skipped rather than guessed. Good free sources include KiwiSDR recordings (HF), RTL-SDR captures
(VHF/UHF), the SigMF / IQEngine community recordings, and sigidwiki samples.

---

## Training

```bash
pip install -r requirements.txt

# RadioML 2018 (auto-detected in dataset/, data/ or /kaggle/input):
python -c "from signal9.train.trainer import run_training; \
           print(run_training(epochs=10, batch_size=256, max_samples=2555904, lr=1e-3, \
                              device='auto', num_workers=4, \
                              save_path='models/s9_radioml2018.ckpt')['val_acc'])"

# Or use the notebooks (Kaggle / Colab, GPU):
#   notebooks/kaggle_radioml2018_training.ipynb
#   notebooks/signal_infinity_cloud_training.ipynb
```

Each epoch evaluates the EMA weights on a stratified 10 % hold-out split and saves the best epoch.
The checkpoint stores its class vocabulary, input length, dimensions and validation accuracy per
2 dB SNR bucket. To use a trained model:

```bash
python main.py capture.wav --checkpoint models/s9_radioml2018.ckpt
python -m signal9.gui            # "Checkpoint" field in the top bar
```

**Smoke test (CPU, synthetic, 4 epochs, 3,600 examples):** validation accuracy rose from 10 %
(chance) to 57 %. This only shows that the loop learns. It is **not** a performance claim.
Accuracy figures must come from a real RadioML run and be quoted with their SNR range.
