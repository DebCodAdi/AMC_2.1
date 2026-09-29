"""RoPE attention relative-position property unit test.
Fixes B3: Verifies that Rotary Position Embedding preserves relative-distance invariance.
"""
import unittest
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal9.models.rope import RoPEAttention, RotaryTable, apply_rope

class TestRoPE(unittest.TestCase):
    def test_relative_position_property(self):
        d_head = 16
        seq_len = 32
        table = RotaryTable(head_dim=d_head)
        cos, sin = table(seq_len, device=torch.device("cpu"))

        # Create two identical query and key vectors
        torch.manual_seed(42)
        q_vec = torch.randn(1, 1, 1, d_head)
        k_vec = torch.randn(1, 1, 1, d_head)

        # Place them at positions (pos_q=2, pos_k=5) -> distance = 3
        q1 = apply_rope(q_vec, cos[2:3], sin[2:3])
        k1 = apply_rope(k_vec, cos[5:6], sin[5:6])
        score1 = (q1 * k1).sum()

        # Shift both positions by +10: (pos_q=12, pos_k=15) -> distance = 3
        q2 = apply_rope(q_vec, cos[12:13], sin[12:13])
        k2 = apply_rope(k_vec, cos[15:16], sin[15:16])
        score2 = (q2 * k2).sum()

        # Dot product under RoPE must be identical under translation
        self.assertAlmostEqual(score1.item(), score2.item(), places=5)

    def test_rope_attention_forward(self):
        att = RoPEAttention(d_model=128, n_heads=8)
        x = torch.randn(2, 256, 128)
        out = att(x)
        self.assertEqual(out.shape, (2, 256, 128))
        self.assertTrue(torch.isfinite(out).all())

if __name__ == "__main__":
    unittest.main()
