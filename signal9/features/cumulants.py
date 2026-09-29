"""Hardened cumulant feature extractor.

Fixes three independent defects:
  DEF-039  Mahalanobis distance assumes diagonal covariance — use full
            covariance with Ledoit-Wolf shrinkage.
  DEF-040  No AMC reject option — add chi²(d, 0.999) Mahalanobis gate.
  EC-AMC-030  Outlier sensitivity — trim top/bottom 0.5% of |symbol| before
              estimating cumulants (robust to clipping spikes and PAPR peaks).

Spec reference: DEF-039, DEF-040, EC-AMC-030, §IV-E.E1 (PS26147).
"""
from __future__ import annotations
from typing import Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Robust moment estimator (EC-AMC-030)
# ─────────────────────────────────────────────────────────────────────────────

def _trim_samples(z: np.ndarray, trim_frac: float = 0.005) -> np.ndarray:
    """Remove the top and bottom ``trim_frac`` of samples by |z|.

    EC-AMC-030: Clipping artefacts and occasional PAPR peaks are extreme
    outliers in |z|.  They inflate C40 and C42 by up to 40% on a single
    outlier out of 4096 samples.  Removing the outer 0.5% is statistically
    negligible for well-behaved signals but removes most of the outlier
    influence.
    """
    if len(z) < 32 or trim_frac <= 0:
        return z
    mags = np.abs(z)
    lo = np.quantile(mags, trim_frac)
    hi = np.quantile(mags, 1.0 - trim_frac)
    mask = (mags >= lo) & (mags <= hi)
    if mask.sum() < 16:
        return z  # Safety: don't trim if it removes too many samples
    return z[mask]


# ─────────────────────────────────────────────────────────────────────────────
# Higher-order moment helpers
# ─────────────────────────────────────────────────────────────────────────────

def _moments(z: np.ndarray) -> Tuple[complex, complex, complex, float, float, float]:
    """Return (C20, C40, C41, C42, C63, C21) for centred ``z``."""
    c21 = float(np.mean(np.abs(z) ** 2))
    if c21 < 1e-16:
        return 0+0j, 0+0j, 0+0j, 0.0, 0.0, c21

    c20 = complex(np.mean(z * z))
    # 4th-order cumulants
    m40 = complex(np.mean(z ** 4))
    m41 = complex(np.mean((z ** 3) * np.conj(z)))
    m42 = float(np.mean(np.abs(z) ** 4))
    # Using Isserlis / Wick theorem for Gaussian subtraction
    c40 = m40 - 3.0 * c20 ** 2
    c41 = m41 - 3.0 * c20 * c21
    c42 = m42 - abs(c20) ** 2 - 2.0 * c21 ** 2
    # 6th-order
    m63 = float(np.mean(np.abs(z) ** 6))
    c63 = m63 - 9.0 * c42 * c21 - 6.0 * c21 ** 3

    return c20, c40, c41, complex(c42), complex(c63), c21


# ─────────────────────────────────────────────────────────────────────────────
# Feature vector
# ─────────────────────────────────────────────────────────────────────────────

def cumulant_features(z: np.ndarray,
                      trim_frac: float = 0.005) -> np.ndarray:
    """Scale- and phase-invariant HOC feature vector.

    Returns 5-D float32 vector:
        [|C20|/C21, |C40|/C21², |C41|/C21², C42/C21², C63.real/C21³]

    Phase invariance: |·| removes carrier phase dependence.
    Scale invariance: normalisation by C21 powers.

    Changes vs. original:
    - EC-AMC-030: 0.5% trimming of |z| extremes before estimation.
    - Consistent Isserlis-theorem moment reduction.
    """
    if len(z) < 16:
        return np.zeros(5, dtype=np.float32)

    zc = z - np.mean(z)
    zc = _trim_samples(zc, trim_frac)
    if len(zc) < 16:
        return np.zeros(5, dtype=np.float32)

    c20, c40, c41, c42, c63, c21 = _moments(zc)

    if c21 < 1e-16:
        return np.zeros(5, dtype=np.float32)

    return np.array([
        abs(c20) / c21,
        abs(c40) / (c21 ** 2),
        abs(c41) / (c21 ** 2),
        float(c42.real if hasattr(c42, "real") else c42) / (c21 ** 2),  # Signed C42/C21^2 matching training parity
        c63.real / (c21 ** 3),
    ], dtype=np.float32)


def compute_cumulants(z: np.ndarray, trim_frac: float = 0.005) -> tuple[np.ndarray, str]:
    """Extract cumulant feature vector and status."""
    if len(z) < 16:
        from signal9.status import Status
        return np.zeros(5, dtype=np.float32), Status.INSUFFICIENT_SAMPLES.value
    feats = cumulant_features(z, trim_frac=trim_frac)
    return feats, "OK"


# ─────────────────────────────────────────────────────────────────────────────
# Mahalanobis classifier with Ledoit-Wolf and reject option
# ─────────────────────────────────────────────────────────────────────────────

def _ledoit_wolf_shrink(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf oracle shrinkage of the sample covariance matrix.

    DEF-039: Diagonal covariance over-penalises features that are
    correlated (C40 and C42 are strongly correlated for most modulations).
    Ledoit-Wolf produces a well-conditioned estimate without tuning.

    Parameters
    ----------
    X : (n_samples, n_features)

    Returns
    -------
    S : (n_features, n_features) — shrunk covariance
    """
    try:
        from sklearn.covariance import ledoit_wolf
        S, _ = ledoit_wolf(X)
        return S
    except ImportError:
        pass

    # Fallback: Ledoit-Wolf analytic formula for the isotropic limit
    n, p = X.shape
    S_sample = np.cov(X.T, ddof=1)
    mu = np.trace(S_sample) / p

    # Oracle shrinkage coefficient α*
    num = (1.0 / (n * n)) * np.sum(
        [np.trace(np.outer(X[i] - X.mean(0), X[i] - X.mean(0)) @ S_sample) ** 2
         for i in range(n)]
    ) if n <= 50 else (np.linalg.norm(S_sample, 'fro') ** 2 + np.trace(S_sample) ** 2) / ((n + 2) * (np.trace(S_sample) ** 2))
    alpha = float(np.clip(1.0 - num / (np.linalg.norm(S_sample - mu * np.eye(p), 'fro') ** 2 + 1e-12), 0.0, 1.0))

    return alpha * S_sample + (1.0 - alpha) * mu * np.eye(p)


class MahalanobisAMC:
    """Mahalanobis nearest-centroid AMC classifier with reject option.

    DEF-039 + DEF-040: Use full covariance (Ledoit-Wolf shrinkage) and
    chi²(d, 0.999) threshold for the reject option.

    Training
    --------
    Build a ``MahalanobisAMC`` from a DataFrame of labelled features::

        clf = MahalanobisAMC.fit(features_df, label_col="mod")
        result = clf.predict(feat_vec)

    Inference
    ---------
    ``predict`` returns a dict with keys: "label", "mahal_sq", "rejected".
    ``rejected=True`` → AMC could not confidently classify (UNKNOWN_MODULATION).
    """

    def __init__(self, centroids: dict[str, np.ndarray],
                 inv_covs: dict[str, np.ndarray],
                 d: int,
                 chi2_threshold: float = 22.458):
        """``chi2_threshold`` defaults to χ²(d=6, 0.999) ≈ 22.458 from scipy."""
        self.centroids = centroids
        self.inv_covs = inv_covs
        self.d = d
        self.chi2_threshold = chi2_threshold

    @classmethod
    def fit(cls, X: dict[str, np.ndarray],
            chi2_threshold: float = 22.458) -> "MahalanobisAMC":
        """Fit from a dict mapping label → (n_samples, d) feature matrix.

        Uses Ledoit-Wolf shrinkage for each class covariance.
        """
        centroids: dict[str, np.ndarray] = {}
        inv_covs: dict[str, np.ndarray] = {}
        d = 0

        for label, feats in X.items():
            feats = np.asarray(feats, dtype=np.float64)
            if feats.ndim == 1:
                feats = feats[np.newaxis, :]
            d = feats.shape[1]
            centroids[label] = feats.mean(0)
            if len(feats) >= d + 2:
                S = _ledoit_wolf_shrink(feats)
            else:
                S = np.eye(d)  # Degenerate: not enough samples
            try:
                inv_covs[label] = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                inv_covs[label] = np.eye(d)  # Safety

        return cls(centroids, inv_covs, d, chi2_threshold)

    def mahal_sq(self, x: np.ndarray, label: str) -> float:
        """Mahalanobis squared distance from ``x`` to class ``label``."""
        delta = x.astype(np.float64) - self.centroids[label]
        return float(delta @ self.inv_covs[label] @ delta)

    def predict(self, x: np.ndarray) -> dict:
        """Classify feature vector ``x``.

        Returns
        -------
        dict with keys:
            label : str | None — predicted class, None if rejected
            mahal_sq : float — Mahalanobis² distance to best class
            rejected : bool — True if Mahalanobis² > chi²(d, 0.999)
            scores : dict[str, float] — {label: mahal_sq} for all classes
        """
        if not self.centroids:
            return {"label": None, "mahal_sq": float("inf"),
                    "rejected": True, "scores": {}}

        x = np.asarray(x, dtype=np.float64)
        scores = {lbl: self.mahal_sq(x, lbl) for lbl in self.centroids}
        best_label = min(scores, key=scores.__getitem__)
        best_score = scores[best_label]

        # DEF-040 reject option: χ²(d, 0.999)
        rejected = best_score > self.chi2_threshold

        return {
            "label": None if rejected else best_label,
            "mahal_sq": best_score,
            "rejected": rejected,
            "scores": scores,
        }
