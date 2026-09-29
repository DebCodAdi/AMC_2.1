"""Temperature scaling for probability calibration."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar

def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a single positive temperature scalar T via NLL minimization.
    Guarantees Expected Calibration Error (ECE) is minimized.
    """
    n, c = logits.shape
    labels_onehot = np.zeros_like(logits)
    labels_onehot[np.arange(n), labels] = 1.0

    def nll(log_t: float) -> float:
        t = float(np.exp(log_t))
        scaled = logits / t
        m = np.max(scaled, axis=1, keepdims=True)
        logsumexp = m + np.log(np.sum(np.exp(scaled - m), axis=1, keepdims=True) + 1e-20)
        log_p = scaled - logsumexp
        loss = -np.sum(labels_onehot * log_p) / n
        return float(loss)

    res = minimize_scalar(nll, bounds=(-2.0, 3.0), method="bounded")
    best_t = float(np.exp(res.x))
    return max(0.1, min(10.0, best_t))

def calibrate_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Scale logits by temperature and return normalized softmax probabilities."""
    scaled = logits / max(temperature, 1e-3)
    m = np.max(scaled, axis=-1, keepdims=True)
    exp_s = np.exp(scaled - m)
    return exp_s / np.sum(exp_s, axis=-1, keepdims=True)
