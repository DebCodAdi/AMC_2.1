"""Train/serve feature-contract regressions for the Kaggle AMC model."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from signal9.train.features import expert_features


def test_kaggle_and_deployment_expert_features_match():
    script = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_grand_corpus_training.py"
    spec = importlib.util.spec_from_file_location("s9_kaggle_trainer_contract", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    rng = np.random.default_rng(20260924)
    iq = (rng.normal(size=1024) + 1j * rng.normal(size=1024)).astype(np.complex64)
    batch = torch.from_numpy(np.stack([iq.real, iq.imag])[None])
    training = module.extract_expert_32d_batch(batch)[0].numpy()
    deployment = expert_features(iq)

    np.testing.assert_allclose(training, deployment, rtol=2e-5, atol=3e-6)
