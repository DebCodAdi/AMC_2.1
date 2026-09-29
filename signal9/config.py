"""Pipeline configuration — all feature flags and named thresholds.

Rules enforced by this module:
  R2  Every new algorithm defaults to OFF until it wins an A/B gate.
  R5  Every numerical routine gets a seeded RNG via ``Config.rng()``.

No magic-number float literals (other than 0, 1, 2, 0.5, -1) may appear
anywhere in signal9/ outside this file.  Enforce with the grep-based lint
in L0_contracts/test_no_magic_numbers.py.

Spec reference: §9.2 (PS26147).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Config:
    """Immutable-ish configuration object.  Pass one instance through the
    full pipeline so every stage uses the same flags and thresholds.

    Mutating a ``Config`` after construction is allowed during testing but
    should be considered a code smell in production.
    """

    # ── Reproducibility ─────────────────────────────────────────────────────
    seed: int = 0
    """Master RNG seed.  Every stage must call ``self.cfg.rng()`` rather
    than touching ``np.random`` module-level functions (DEF-043)."""

    # ── Feature flags (R2: default OFF until proven) ─────────────────────
    enable_matched_filter: bool = False
    """Insert an RRC matched filter after resampling to 2 SPS (IV-B)."""

    enable_rolloff_estimation: bool = False
    """Estimate roll-off α blind before building the matched filter (IV-B,
    DEF-005)."""

    enable_mma_equalizer: bool = False
    """Use MMA / dual-mode CMA→DD equaliser instead of plain CMA (IV-D,
    DEF-037)."""

    enable_differential_track: bool = False
    """Add Track H (differential demapping) to the blackboard (IV-L,
    DEF-024)."""

    enable_segmented_rotation: bool = False
    """Segmented rotation search at detected slip indices (IV-L,
    DEF-003)."""

    enable_cfo_hypotheses: bool = False
    """Carry top-2 (M, f_cfo) hypotheses forward instead of committing to
    one (IV-C, DEF-001)."""

    enable_scf_baud: bool = False
    """Add SCF and zero-crossing Rs estimators, fused with the envelope
    estimator (IV-A, DEF-011)."""

    enable_rank_fec_id: bool = False
    """Use the rank-criterion / Soft-GJETP Stage 6.5 FEC identifier
    (IV-G)."""

    enable_soft_syndrome_sync: bool = False
    """Replace the bit-slip sweep with soft-syndrome frame
    synchronisation (IV-I, DEF-007)."""

    enable_interleaver_detect: bool = False
    """Use the KS-test algebraic interleaver detector instead of a guess
    list (IV-H, DEF-019)."""

    enable_nms_ldpc: bool = False
    """Normalised min-sum LDPC decoding; candidate-set identification
    (IV-J, DEF-016)."""

    enable_learned_amc: bool = False
    """Use the learned AMC (LightGBM / CNN) as a second opinion (IV-E)."""

    enable_nda_snr: bool = False
    """Three-way fused NDA SNR estimation (IV-F)."""

    enable_timing_loop: bool = False
    """Closed-loop Gardner + Farrow interpolator for SCO tracking (DEF-012)."""

    enable_bit_order_sweep: bool = False
    """Sweep MSB-first vs LSB-first and IQ bit order in demapper (DEF-023)."""

    enable_windowed_crc: bool = False
    """Sliding-window search for CRC-32 across candidate frame boundaries (DEF-022)."""

    # ── Named thresholds (no magic numbers in src/) ──────────────────────
    mme_pfa: float = 1e-3
    """MME false-alarm probability used for Tracy-Widom threshold (DEF-009)."""

    mme_min_p_over_l: int = 20
    """Minimum P/L ratio before trusting MME; below this return
    INSUFFICIENT_SAMPLES_FOR_DETECTION (DEF-009)."""

    amc_reject_chi2: float = 22.458
    """χ²₆(0.999) threshold for the AMC reject option (DEF-040).
    Computed as scipy.stats.chi2.ppf(0.999, df=6)."""

    llr_clip: float = 8.0
    """LLR saturation limit ±llr_clip for Viterbi / LDPC inputs (DEF-017).
    Prevents float32 metric overflow."""

    sigma2_floor: float = 1e-10
    """Minimum noise-variance estimate to prevent ±Inf LLRs (DEF-017,
    EC-FEC-038)."""

    viterbi_traceback: int = 96
    """Minimum Viterbi traceback depth.  Conventionally 96–128 for K=7
    (DEF-018)."""

    nms_alpha: float = 0.75
    """Normalisation factor for normalised min-sum LDPC (DEF-016)."""

    cma_taps: int = 21
    """CMA / MMA equaliser tap count for ≤8PSK.  Use 31 for 16/64-QAM."""

    cma_divergence_factor: float = 10.0
    """If ‖w‖ exceeds this multiple of ‖w₀‖, halve μ and restart
    (DEF-036)."""

    gsop_rho_clip: float = 0.999
    """Maximum |ρ| in the GSOP corrector to avoid division by ~0
    (EC-SIG-030/031, DEF-010)."""

    branch_timeout_s: float = 5.0
    """Per-branch wall-clock timeout in the blackboard (EC-ARB-004)."""

    max_branches: int = 256
    """Hard cap on the total number of concurrent blackboard branches."""

    rolloff_grid: tuple[float, ...] = (
        0.1, 0.15, 0.2, 0.22, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0
    )
    """Candidate roll-off values for PSD-shape matching (IV-B, DEF-005)."""

    rrc_span_symbols: int = 10
    """RRC filter span in symbols; at 2 SPS this is ``2 * span + 1`` taps
    (IV-B, DEF-004)."""

    fs_candidates_k: int = 3
    """Number of Fs/Rs hypotheses to carry from Stage 2 into the blackboard
    (DEF-011)."""

    scf_baud_fusion_tol: float = 0.001
    """Relative tolerance for the three-estimator Rs fusion gate (IV-A).
    Two estimators agreeing within this fraction → high confidence."""

    entropy_plausibility_weights: tuple[float, float, float, float, float] = (
        0.25, 0.35, 0.15, 0.20, 0.05
    )
    """Logistic-regression weights (z_entropy, printable_ratio,
    bigram_loglik, crc_valid, log_n_branches) for the payload plausibility
    score (DEF-006, IV-M).  Recalibrate after Phase 1."""

    nan_repair_max_run: int = 10_000
    """NaN runs longer than this are treated as gaps (nearest-fill), not
    interpolated (DEF-033, EC-SIG-034)."""

    nan_density_limit: float = 0.50
    """Fraction of NaN samples above which ingestion returns TOO_CORRUPT
    (DEF-034, EC-SIG-035)."""

    min_samples: int = 256
    """Absolute minimum sample count below which INSUFFICIENT_SAMPLES is
    returned (EC-FILE-005/006)."""

    # ── Reporting / diagnostic flags ─────────────────────────────────────
    early_exit_on_crc: bool = True
    """Abort remaining branches once any branch locks a CRC (DEF-021).
    Set False in diagnostic / test runs to count all agreeing branches."""

    emit_all_branch_results: bool = False
    """Attach results from every branch to the output dict.  Expensive."""

    # ──────────────────────────────────────────────────────────────────────
    def rng(self) -> np.random.Generator:
        """Return a fresh seeded Generator.  Call once per top-level
        pipeline invocation and thread the result through all stages."""
        return np.random.default_rng(np.random.SeedSequence(self.seed))

    def rng_for(self, subsystem: str) -> np.random.Generator:
        """Return a deterministically derived child Generator for a named
        subsystem (e.g. 'augment', 'blackboard').  Isolates subsystems so
        changing one does not perturb another."""
        child_seed = abs(hash(subsystem)) ^ self.seed
        return np.random.default_rng(np.random.SeedSequence(child_seed))


# Singleton default used by callers that do not need customisation.
DEFAULT_CONFIG: Config = Config()
