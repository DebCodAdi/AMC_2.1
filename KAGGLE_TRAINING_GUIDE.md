# Correct Kaggle training procedure

This project trains one fixed-length classifier per IQ length. A batch never
mixes 128, 256, 512 and 1024-point records. The four resulting checkpoints can
be selected by capture length or combined later by a calibrated ensemble.

The mounted `variable_length_rf_corpus.h5` is Sig53-derived. For 128, 256 and
512, the trainer uses the native VariableRF group and automatically excludes
the direct Sig53 stream, because the exports do not contain waveform IDs needed
to prove parent/child split independence. For 1024, where no native VariableRF
group exists, the trainer uses Sig53 directly.

Research V5 provides `baseline_resnet` and `hybrid_xcit`. Train both on the
same length, seed, steps and data before choosing one. The default candidate is
`hybrid_xcit`; the residual model is the control.

## 1. Kaggle inputs

Attach these datasets to the notebook:

1. The new `AMC_RESEARCH_V5_4-kaggle.zip` (Kaggle exposes its extracted folder).
2. `variable_length_rf_corpus.h5` for lengths 128, 256 and 512.
3. `sig53_master_5million.h5`.
4. RadioML 2018.01A (`GOLD_XYZ_OSC.0001_1024.hdf5`) and its
   `classes-fixed.json` file.
5. The verified real-capture datasets already used by the project.

Do not upload old `.ckpt` files into a data directory for this run. V5.2's
conditioning and session-split contracts require a fresh training start.

## 2. Locate the code

```python
from pathlib import Path

root = Path("/kaggle/input/datasets/adityakrak")
scripts = list(root.rglob("kaggle_grand_corpus_training.py"))
requirements = list(root.rglob("requirements-kaggle.txt"))
print("Scripts:", scripts)
print("Requirements:", requirements)
assert scripts and requirements
SCRIPT = scripts[0]
REQ = requirements[0]
```

## 3. Install requirements and verify datasets

Kaggle normally preinstalls every training dependency. Verify imports first:

```python
import torch, numpy, scipy, h5py, yaml
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
```

Only if an import is missing, enable Internet and install the GUI-free list:

```python
KAGGLE_REQ = next(root.rglob("requirements-kaggle.txt"))
!python -m pip install -q -r "{KAGGLE_REQ}"
```

Do not install the desktop `requirements.txt` in Kaggle; it contains PySide6
and test-only packages that the trainer does not import.

```python
from pathlib import Path

input_root = Path("/kaggle/input")
required = {
    "Variable RF": list(input_root.rglob("variable_length_rf_corpus.h5")),
    "Sig53": list(input_root.rglob("sig53_master_5million.h5")),
    "RadioML 2018": list(input_root.rglob("GOLD_XYZ_OSC.0001_1024.hdf5")),
}
for name, paths in required.items():
    print(name, paths)
    assert paths, f"Missing {name}"
```

Run the fail-fast checker before spending GPU time:

```python
VERIFY = next(root.rglob("verify_kaggle_training.py"))
VARIABLE = required["Variable RF"][0]
SIG53 = required["Sig53"][0]
!python "{VERIFY}" --script "{SCRIPT}" --variable-rf "{VARIABLE}" --sig53 "{SIG53}"
```

## 4. Train one length

Use a separate output directory for each length. `1953` steps with batch size
`256` means approximately 500,000 examples per epoch; it is not a reduced
100-step performance test.

### 128 samples

```python
!python "{SCRIPT}" \
  --data-root /kaggle/input \
  --input-length 128 \
  --epochs 10 \
  --steps-per-epoch 1953 \
  --batch-size 256 \
  --lr 0.001 \
  --val-batches 64 \
  --arcface-weight 0 \
  --architecture hybrid_xcit \
  --synthetic-fraction 0.10 \
  --synthetic-pool-size 8192 \
  --output-dir /kaggle/working/satvik-L128
```

For the other models, repeat the command with:

- `--input-length 256 --output-dir /kaggle/working/satvik-L256`
- `--input-length 512 --output-dir /kaggle/working/satvik-L512`
- `--input-length 1024 --output-dir /kaggle/working/satvik-L1024`

Run each length in a separate Kaggle session if one session cannot finish all
four. The number of examples per epoch remains 1953 × 256.

The V5.4 input path is optimized for the supplied compressed HDF5 layout. It
reads randomized short runs from 16 independently selected, split-safe groups
instead of inflating one large gzip chunk for almost every individual row. It
also pins the background-prefetched CPU batch before the asynchronous transfer
to CUDA. This does **not** reduce steps, batch size, epochs, active datasets or
examples per epoch. At startup, confirm this line appears:

```text
>> [INPUT PIPELINE] gzip-aware 16-group HDF5 reads; background prefetch + pinned-memory transfer: ON
```

Kaggle's utilization card is an instantaneous sample and can still briefly
show 0% between kernels. Judge utilization with `nvidia-smi dmon` over time and
with the trainer's sustained `smp/s`, not one dashboard snapshot. The model is
small enough that `DataParallel` will not keep both T4s at 100%; the purpose of
the change is to remove the verified input stall without changing the training
experiment.

For the controlled baseline, repeat with `--architecture baseline_resnet` and
a different output directory. Do not compare runs with different examples,
seeds, epochs, losses or validation batches.

## 5. Resume after Kaggle stops

Before the session ends, save both `*_last.ckpt` and `*_best.ckpt` using Kaggle
notebook output versioning. Attach both files together to the next notebook,
then use the last checkpoint with the same total epoch target and input length:

```python
!python "{SCRIPT}" \
  --data-root /kaggle/input \
  --input-length 256 \
  --epochs 10 \
  --steps-per-epoch 1953 \
  --batch-size 256 \
  --lr 0.001 \
  --val-batches 64 \
  --arcface-weight 0 \
  --architecture hybrid_xcit \
  --synthetic-fraction 0.10 \
  --synthetic-pool-size 8192 \
  --resume /kaggle/input/YOUR-CHECKPOINT-DATASET/s9_hybrid_xcit_L256_last.ckpt \
  --output-dir /kaggle/working/satvik-L256
```

Do not change batch size, total epochs, steps per epoch, class order or input
length or ArcFace weight during an exact resume. Old checkpoints without the
corrected feature/split contracts must not be resumed. In particular, a
checkpoint reporting `expert11-padded32-paprdb-v2` or
`fixed-80-train-10-cal-10-val-v2` is intentionally rejected. V5.2 also rejects
checkpoints without the shared conditioning contract and session-grouped v4
validation split; start a fresh run with this archive.

## 6. Files to keep

For every length, keep:

- `s9_<architecture>_Lxxx_best.ckpt`: fixed-holdout macro-accuracy winner.
- `s9_<architecture>_Lxxx_last.ckpt`: exact training-resume state.
- `s9_<architecture>_Lxxx_production_bundle.ckpt`: copy of the best checkpoint.

The best checkpoint is selected using a fixed 80/10/10 train/calibration/
validation split. The default trustworthy baseline uses the supervised
cross-entropy head only (`--arcface-weight 0`). ArcFace/prototype blending is
experimental; if enabled explicitly, its blend is selected on calibration
data and scored on separate validation data.

## 7. Minimum acceptance checks

Do not call a run successful using training accuracy alone. Confirm that:

1. Every epoch prints Train, Validation and Macro accuracy.
2. A resumable checkpoint is saved after every epoch.
3. `train_class_counts` contains examples for every class expected at that
   length.
4. Validation support is non-zero for the important QAM, PSK and FSK classes.
5. Per-class accuracy and the confusion matrix improve, not only overall
   accuracy.
6. Final evaluation is performed on an untouched external RadioML/OTA test set.
7. Compare the CE baseline against any ArcFace, Transformer or Mamba variant
   on exactly the same saved split before adopting the more complex model.
