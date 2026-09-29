"""GroupShuffleSplit dataset partitioner enforcing Rule R11 (No-Leakage).

Ensures zero session, device serial, or emitter identity overlap between
train, validation, and test splits.

Spec reference: Rule R11 (SIGNAL10 / PS26147).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def group_shuffle_split(items: list[dict],
                        group_keys: list[str] | None = None,
                        test_size: float = 0.2,
                        val_size: float = 0.1,
                        seed: int = 42) -> dict[str, list[dict]]:
    """Partition items strictly by groups with zero overlap across splits.

    Parameters
    ----------
    items : list of dicts representing corpus entries
    group_keys : list of str, default ['session_id', 'device_serial']
    test_size : float
    val_size : float
    seed : int

    Returns
    -------
    {'train': [...], 'val': [...], 'test': [...]}
    """
    if not items:
        return {"train": [], "val": [], "test": []}

    keys = group_keys or ["session_id", "device_serial"]
    rng = np.random.default_rng(seed)

    # Build connected components of groups:
    # If item A shares a session with B, and B shares a device with C,
    # A, B, and C must belong to the same partition!
    adj: dict[int, set[int]] = {i: set() for i in range(len(items))}
    key_to_indices: dict[tuple[str, str], list[int]] = {}

    for idx, item in enumerate(items):
        for k in keys:
            val = str(item.get(k, f"missing_{k}_{idx}"))
            pair = (k, val)
            if pair not in key_to_indices:
                key_to_indices[pair] = []
            key_to_indices[pair].append(idx)

    for idx_list in key_to_indices.values():
        for i in idx_list:
            adj[i].update(idx_list)

    # Find connected components (clusters of linked items)
    visited = set()
    clusters: list[list[int]] = []
    for i in range(len(items)):
        if i not in visited:
            cluster = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop()
                cluster.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(cluster)

    # Shuffle clusters randomly
    rng.shuffle(clusters)

    n_total = len(items)
    n_test_target = int(np.round(n_total * test_size))
    n_val_target = int(np.round(n_total * val_size))

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for cluster in clusters:
        if len(test_indices) < n_test_target:
            test_indices.extend(cluster)
        elif len(val_indices) < n_val_target:
            val_indices.extend(cluster)
        else:
            train_indices.extend(cluster)

    splits = {
        "train": [items[i] for i in train_indices],
        "val": [items[i] for i in val_indices],
        "test": [items[i] for i in test_indices],
    }

    # Verify no-leak invariant (Rule R11)
    for k in keys:
        for split_a, split_b in [("train", "val"), ("train", "test"), ("val", "test")]:
            vals_a = {str(item[k]) for item in splits[split_a] if k in item}
            vals_b = {str(item[k]) for item in splits[split_b] if k in item}
            overlap = vals_a.intersection(vals_b)
            if overlap:
                raise ValueError(
                    f"Rule R11 violation! Leakage detected on key '{k}' "
                    f"between {split_a} and {split_b}: {overlap}"
                )

    return splits


def main():
    parser = argparse.ArgumentParser(description="GroupShuffleSplit Dataset Partitioner (Rule R11)")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--out", type=str, default="reports/splits.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("entries", data) if isinstance(data, dict) else data
    splits = group_shuffle_split(items, seed=args.seed)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"Splits created: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    print(f"Written to {out_p}")


if __name__ == "__main__":
    main()
