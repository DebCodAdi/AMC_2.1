"""Tests mixed-precision AMP forward and backward pass stability."""
import unittest
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal9.models.backbone import Signal9Net
from signal9.train.losses import hier_loss

class TestAMP(unittest.TestCase):
    def test_amp_numerical_stability(self):
        model = Signal9Net(n_l1=9, n_l2=24, n_l3=19)
        iq = torch.randn(2, 4, 4096)
        spec = torch.randn(2, 1, 224, 224)
        expert = None
        meta = torch.randn(2, 16)
        
        targets = {
            "l1": torch.tensor([0, 2], dtype=torch.long),
            "l2": torch.tensor([1, 5], dtype=torch.long),
            "l3": torch.tensor([0, -1], dtype=torch.long),
            "snr": torch.tensor([10.0, 15.0]),
            "log_baud": torch.tensor([2.0, 3.0]),
            "bw_frac": torch.tensor([0.2, 0.4]),
        }

        # Run forward under autocast
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, enabled=True):
            outputs = model(iq, spec, expert, meta)
            loss = hier_loss(outputs, targets)

        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        for name, p in model.named_parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all(), f"NaN/Inf gradient in {name}")

if __name__ == "__main__":
    unittest.main()
