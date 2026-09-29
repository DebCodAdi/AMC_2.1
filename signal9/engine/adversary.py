"""Adversarial case synthesis hunting confident errors in physical parameter space (SIGNAL INFINITY §3.7)."""
from __future__ import annotations
import numpy as np
from signal9.dsp.constellation import bits_to_symbols


def hunt_adversarial_errors(model_fn,
                            budget: int = 100,
                            seed: int = 42) -> tuple[list[dict], float]:
    """Search for physically realisable signals where the model is confident but wrong."""
    rng = np.random.default_rng(seed)
    confident_errors = []

    # Parameter vector: [snr_db (0..25), cfo_norm (-0.2..0.2), phase (0..2pi), mod_idx (0..1)]
    # Target objective: maximize (confidence * (1 - is_correct))
    trials = 0
    for i in range(budget):
        snr_db = float(rng.uniform(2.0, 22.0))
        cfo_norm = float(rng.uniform(-0.15, 0.15))
        phase = float(rng.uniform(0, 2 * np.pi))
        mod_name = "BPSK" if (i % 2 == 0) else "QPSK"

        bits = rng.integers(0, 2, 512, dtype=np.uint8)
        syms = bits_to_symbols(bits, mod_name)
        iq = np.repeat(syms, 4).astype(np.complex64)
        t = np.arange(len(iq))
        iq = iq * np.exp(1j * (2 * np.pi * cfo_norm * t + phase))
        sigma = np.sqrt(1.0 / (2.0 * (10.0 ** (snr_db / 10.0))))
        noise = sigma * (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq)))
        iq = (iq + noise).astype(np.complex64)

        pred_label, conf = model_fn(iq)
        trials += 1

        # Confident error criterion
        if conf > 0.90 and pred_label != mod_name:
            confident_errors.append({
                "params": {"snr_db": snr_db, "cfo_norm": cfo_norm, "mod": mod_name},
                "pred_label": pred_label,
                "conf": conf,
            })

    hunter_yield = (len(confident_errors) / max(1, trials)) * 100000.0  # per 100k trials
    return confident_errors, hunter_yield
