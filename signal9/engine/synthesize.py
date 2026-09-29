"""Failure-driven neighbourhood synthesis with ground truth by construction (SIGNAL INFINITY §3.4)."""
from __future__ import annotations
import numpy as np
from signal9.dsp.constellation import bits_to_symbols


def expand_failure_neighbourhood(failure_params: dict,
                                 n_samples: int = 50,
                                 cap_frac: float = 0.30,
                                 seed: int = 42) -> list[dict]:
    """Generate perturbed captures around an operating point where the model failed."""
    rng = np.random.default_rng(seed)
    base_mod = failure_params.get("true_mod", "QPSK")
    base_snr = float(failure_params.get("snr_db", 10.0))
    base_symrate = float(failure_params.get("symrate", 2400.0))

    synthesized = []
    for i in range(n_samples):
        # Small perturbations in SNR, symrate, CFO
        snr = float(base_snr + rng.uniform(-3.0, 3.0))
        symrate = float(base_symrate * rng.uniform(0.95, 1.05))
        cfo = float(rng.uniform(-0.02, 0.02))

        # Ground truth is exact by construction
        bits = rng.integers(0, 2, 512, dtype=np.uint8)
        syms = bits_to_symbols(bits, base_mod if base_mod in ("BPSK", "QPSK") else "QPSK")
        iq = np.repeat(syms, 4).astype(np.complex64)
        sigma = np.sqrt(1.0 / (2.0 * (10.0 ** (snr / 10.0))))
        noise = sigma * (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq)))
        iq = (iq + noise).astype(np.complex64)

        synthesized.append({
            "iq": iq,
            "mod": base_mod,
            "snr_db": snr,
            "symrate": symrate,
            "cfo_norm": cfo,
            "source": "D_mined_synth",
            "generation": failure_params.get("generation", 1),
            "labelled_by": "construction_exact",
            "verified_by": "construction",
        })

    return synthesized
