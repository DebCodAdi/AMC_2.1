"""Experimental multi-modal Signal9 model components.

The production Kaggle AMC path is deliberately separate: training is defined
in ``notebooks/kaggle_grand_corpus_training.py`` and the matching deployment
classes live in ``signal9.train.backbone`` and ``signal9.train.heads``.  The
``Signal9Net`` exported here is a research model and is not the network loaded
from ``signal9-multitask-v1`` Kaggle checkpoints.
"""
from .rope import RotaryTable, apply_rope, RoPEAttention, TransformerBlock
from .iq_branch import IQBranch
from .spec_branch import SpecBranch
from .fusion import ModalityFusion
from .heads import CosineHead, HierarchicalHeads
from .openset import energy_score, evaluate_decision_state
from .backbone import Signal9Net

__all__ = [
    "RotaryTable",
    "apply_rope",
    "RoPEAttention",
    "TransformerBlock",
    "IQBranch",
    "SpecBranch",
    "ModalityFusion",
    "CosineHead",
    "HierarchicalHeads",
    "energy_score",
    "evaluate_decision_state",
    "Signal9Net",
]
