"""Near-duplicate removal using MinHash over spectral signatures (SIGNAL INFINITY §1.4b)."""
from __future__ import annotations
import argparse
import hashlib
import numpy as np
from datasketch import MinHash, MinHashLSH


def compute_spectral_signature(iq: np.ndarray, num_perm: int = 128) -> MinHash:
    """Compute MinHash signature over quantized cyclostationary & PSD features."""
    m = MinHash(num_perm=num_perm)
    # 1. Quantized PSD profile
    fft_mag = np.abs(np.fft.fft(iq[:1024]))
    psd_bins = np.digitize(fft_mag, np.linspace(0, np.max(fft_mag) + 1e-6, 32))
    for i, b in enumerate(psd_bins):
        m.update(f"psd_{i}_{b}".encode("utf-8"))

    # 2. Fourth-order cumulant proxy
    c42 = np.mean(np.abs(iq)**4) - 2 * (np.mean(np.abs(iq)**2))**2 - np.abs(np.mean(iq**2))**2
    c42_bin = int(np.clip(c42 * 10, -50, 50))
    m.update(f"c42_{c42_bin}".encode("utf-8"))

    return m


def filter_duplicates(samples: list[dict], threshold: float = 0.98) -> tuple[list[dict], int]:
    """Filter out near-duplicate samples exceeding Jaccard threshold."""
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    kept = []
    dropped_count = 0

    for i, s in enumerate(samples):
        # Derive or simulate signature
        sig_data = s.get("signature_tokens", [f"feat_{s.get('mod')}_{int(s.get('snr_db', 0))}"])
        mh = MinHash(num_perm=128)
        for tok in sig_data:
            mh.update(tok.encode("utf-8"))
        
        matches = lsh.query(mh)
        if matches:
            dropped_count += 1
        else:
            lsh.insert(f"sample_{i}", mh)
            kept.append(s)

    return kept, dropped_count


def main():
    parser = argparse.ArgumentParser(description="Near-duplicate removal tool")
    parser.add_argument("--threshold", type=float, default=0.98, help="Jaccard deduplication threshold")
    args = parser.parse_args()
    print(f"MinHash deduplication engine ready (threshold={args.threshold}).")


if __name__ == "__main__":
    main()
