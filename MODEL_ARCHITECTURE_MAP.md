# Signal9 model-path map

There are two intentionally separate neural paths. They must not be mixed in a
checkpoint or in a test conclusion.

## Production Kaggle AMC path

- Training entry point: `notebooks/kaggle_grand_corpus_training.py`
- Default architecture: `hybrid_xcit`
- Training classes: notebook-local `ResearchHybridXCiTBackbone` and
  `SignalInfinityMultiTaskHeads`
- Deployment-compatible definitions: `signal9/train/backbone.py`,
  `signal9/train/heads.py`, and `signal9/train/inference.py`
- Checkpoint format: `signal9-multitask-v1`
- Trained outputs by default: modulation, symbol rate, SNR and signal presence
- Optional ArcFace: disabled by default (`--arcface-weight 0`)
- Unlabelled outputs (roll-off, CFO, FEC and scrambler) are frozen in the
  Kaggle trainer and are not advertised as trained.

The Kaggle trainer's loss is ordinary modulation cross-entropy plus masked
Smooth-L1 SNR/symbol-rate losses and presence BCE. It does not execute
`signal9/train/losses.py::SignalInfinityLoss`, supervised contrastive loss or
hierarchical family CE.

## Library research trainer

- Entry point: `signal9/train/trainer.py`
- Architecture: `signal9/train/backbone.py::SignalInfinityBackbone`
- Loss: `signal9/train/losses.py::SignalInfinityLoss`
- Purpose: smaller programmatic experiments and library tests

It writes the same checkpoint envelope, but it is not the script used by the
documented multi-dataset Kaggle workflow.

## Experimental multimodal model

- Package: `signal9/models/`
- Main class: `signal9.models.Signal9Net`
- Purpose: IQ/spectrogram/expert-feature architecture research

This model is not loaded by the Kaggle trainer. Its code remains because it has
independent tests and public imports; deleting it would be a breaking API
change, not a correctness repair. `Signal9Loss` now trains the logits returned
by `Signal9Net`; private ArcFace classifiers are optional auxiliary losses.

