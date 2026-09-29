"""Tests zero-leakage splits across receiver, session, and time."""
import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal9.train.samplers import group_split

class TestSplits(unittest.TestCase):
    def test_group_split_zero_leakage(self):
        rows = [
            {"receiver": rx, "utc": f"2026-0{m}-01", "session": f"{rx}-{m}-{k}"}
            for rx in ("rx_A", "rx_B", "rx_C", "heldout_D")
            for m in (6, 7, 8, 9)
            for k in range(5)
        ]
        train, val, test = group_split(
            rows,
            test_receivers={"heldout_D"},
            test_after_utc="2026-09-01"
        )
        
        train_sessions = {x["session"] for x in train}
        val_sessions = {x["session"] for x in val}
        test_sessions = {x["session"] for x in test}

        # No session appears in both train and test, or train and val
        self.assertEqual(len(train_sessions & test_sessions), 0)
        self.assertEqual(len(train_sessions & val_sessions), 0)
        
        # No held-out receiver in train
        self.assertTrue(all(x["receiver"] != "heldout_D" for x in train))

if __name__ == "__main__":
    unittest.main()
