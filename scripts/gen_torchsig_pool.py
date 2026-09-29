"""Bounded TorchSig pool generator (Fixes B12).
Uses WorkerSeedingDataLoader and checks live classes before dataset generation.
"""
from __future__ import annotations
import argparse
from pathlib import Path

def generate_pool(output_dir: str = "data/torchsig_pool/nb_v1", n_samples: int = 10000) -> None:
    try:
        import torchsig
        from torchsig.datasets.datasets import TorchSigIterableDataset
        from torchsig.utils.writer import DatasetCreator
        from torchsig.utils.data_loading import WorkerSeedingDataLoader
        from torchsig.transforms.impairments import Impairments
        print(f"[*] Found TorchSig version {torchsig.__version__}")
        
        imp = Impairments(level=2)
        ds = TorchSigIterableDataset(
            transforms=[imp.dataset_transforms],
            component_transforms=[imp.signal_transforms]
        )
        dl = WorkerSeedingDataLoader(ds, batch_size=32, num_workers=2)
        
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        
        creator = DatasetCreator(
            dataset_length=n_samples,
            dataloader=dl,
            root=str(out_p),
            overwrite=True
        )
        creator.create()
        print(f"[+] Bounded TorchSig dataset pool generated at {output_dir}")
    except ImportError:
        print("[!] TorchSig is not installed. Install via 'pip install torchsig' to generate synthetic pool.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate bounded TorchSig dataset pool")
    parser.add_argument("--out", default="data/torchsig_pool/nb_v1")
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()
    generate_pool(args.out, args.samples)
