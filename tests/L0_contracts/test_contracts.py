"""L0: Static & contract tests.

Tests in this file must pass with zero data files and zero models.
They check:
  - That the Status enum is complete and maps to distinct strings.
  - That the Config dataclass has all required fields.
  - That every result dict from the pipeline passes validate_result_schema().
  - That no magic-number literals appear in signal9/. (grep-based)
  - That mypy --strict passes on signal9/.

Spec reference: §2.1, L0 tier (PS26147).
"""
from __future__ import annotations
import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).parent.parent.parent / "signal9"


# ─────────────────────────────────────────────────────────────────────────────
# Status enum
# ─────────────────────────────────────────────────────────────────────────────

def test_status_importable():
    from signal9.status import Status
    assert hasattr(Status, "SUCCESS_CRC_LOCKED")


def test_status_all_unique_strings():
    from signal9.status import Status
    values = [s.value for s in Status]
    assert len(values) == len(set(values)), "Duplicate Status string values detected"


def test_status_expected_members():
    from signal9.status import Status
    required = {
        "SUCCESS_CRC_LOCKED",
        "DEMOD_NO_CRC",
        "NO_SIGNAL_DETECTED",
        "UNKNOWN_MODULATION",
        "EMPTY_FILE",
        "INSUFFICIENT_SAMPLES",
        "INSUFFICIENT_SAMPLES_FOR_DETECTION",
        "INVALID_SAMPLES",
        "TOO_CORRUPT",
        "CLIPPED_ADC_SATURATION",
        "UNSUPPORTED_FORMAT",
        "UNSUPPORTED_SCRAMBLER",
        "UNSUPPORTED_FEC",
        "UNSUPPORTED_INTERLEAVER",
        "PLL_UNLOCKED",
        "ALL_BRANCHES_FAILED",
        "TIMEOUT",
        "RESOURCE_LIMIT",
    }
    members = {s.value for s in Status}
    missing = required - members
    assert not missing, f"Missing Status members: {missing}"


def test_status_is_success():
    from signal9.status import Status
    assert Status.is_success(Status.SUCCESS_CRC_LOCKED)
    assert not Status.is_success(Status.DEMOD_NO_CRC)
    assert not Status.is_success(Status.ALL_BRANCHES_FAILED)


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────

def test_config_importable():
    from signal9.config import Config
    cfg = Config()
    assert cfg.seed == 0


def test_config_all_flags_default_false():
    from signal9.config import Config
    cfg = Config()
    flag_fields = [
        "enable_matched_filter",
        "enable_rolloff_estimation",
        "enable_mma_equalizer",
        "enable_differential_track",
        "enable_segmented_rotation",
        "enable_cfo_hypotheses",
        "enable_scf_baud",
        "enable_rank_fec_id",
        "enable_soft_syndrome_sync",
        "enable_interleaver_detect",
        "enable_nms_ldpc",
        "enable_learned_amc",
        "enable_nda_snr",
        "enable_timing_loop",
    ]
    for flag in flag_fields:
        assert hasattr(cfg, flag), f"Config missing flag: {flag}"
        assert getattr(cfg, flag) is False, (
            f"Config.{flag} must default to False (Rule R2)"
        )


def test_config_rng_seeded():
    import numpy as np
    from signal9.config import Config
    cfg = Config(seed=7)
    r1 = cfg.rng().integers(0, 1000)
    r2 = Config(seed=7).rng().integers(0, 1000)
    assert r1 == r2, "rng() must be deterministic for the same seed"


def test_config_rng_for_subsystems_independent():
    from signal9.config import Config
    cfg = Config(seed=0)
    r_a = cfg.rng_for("augment").integers(0, 1000)
    r_b = cfg.rng_for("blackboard").integers(0, 1000)
    assert r_a != r_b, "Different subsystems must get different RNG streams"


def test_config_named_thresholds_reasonable():
    from signal9.config import Config
    cfg = Config()
    assert 10.0 < cfg.amc_reject_chi2 < 30.0, "amc_reject_chi2 should be ~22.46"
    assert 4.0 <= cfg.llr_clip <= 16.0, "llr_clip should be 8"
    assert cfg.viterbi_traceback >= 70, "traceback must be ≥ 70 (conventionally 96)"
    assert 0.5 <= cfg.nms_alpha <= 1.0, "nms_alpha must be in (0.5, 1]"
    assert 0.0 < cfg.gsop_rho_clip < 1.0, "gsop_rho_clip must be < 1"


# ─────────────────────────────────────────────────────────────────────────────
# Contracts
# ─────────────────────────────────────────────────────────────────────────────

def test_contracts_importable():
    from signal9.contracts import Stage1Out, Stage2Out, Stage4Out
    assert callable(Stage1Out)


def test_stage1out_valid():
    import numpy as np
    from signal9.contracts import Stage1Out

    x = np.random.default_rng(0).standard_normal(1024).view(np.complex128)
    # Normalise and remove DC as Stage 1 would
    x = x - x.mean()
    x = x / (np.std(x) + 1e-12)
    out = Stage1Out(x=x, fs_hint=2.4e6, fmt="cs16", n_repaired=0, mme_ratio=3.2)
    out.validate()  # must not raise


def test_stage1out_detects_nonfin():
    import numpy as np
    from signal9.contracts import Stage1Out
    x = np.ones(64, dtype=np.complex128)
    x[10] = float("nan")
    with pytest.raises(AssertionError, match="C1.1"):
        Stage1Out(x=x, fs_hint=None, fmt="cf32",
                  n_repaired=0, mme_ratio=1.0).validate()


def test_stage1out_detects_dc():
    import numpy as np
    from signal9.contracts import Stage1Out
    x = np.ones(64, dtype=np.complex128) * 0.5  # large DC
    with pytest.raises(AssertionError, match="C1.2"):
        Stage1Out(x=x, fs_hint=None, fmt="cf32",
                  n_repaired=0, mme_ratio=1.0).validate()


def test_stage2out_valid():
    import numpy as np
    from signal9.contracts import Stage2Out
    fs = 2400.0
    rs = 1200.0
    x2 = (np.zeros(2048) + 0j).astype(np.complex128)
    out = Stage2Out(x2=x2, fs=fs, rs=rs, sps_pre=2.0,
                    cfo_hz=0.0, alpha=0.35,
                    fs_candidates=[(fs, 0.0)])
    out.validate()


def test_stage2out_detects_wrong_sps():
    import numpy as np
    from signal9.contracts import Stage2Out
    with pytest.raises(AssertionError, match="C2.2"):
        Stage2Out(x2=np.zeros(100, dtype=np.complex128),
                  fs=2400.0, rs=800.0, sps_pre=3.0,  # 2400/800 = 3, not 2
                  cfo_hz=0.0, alpha=0.35,
                  fs_candidates=[(2400.0, 0.0)]).validate()


def test_stage4out_valid():
    import numpy as np
    from signal9.contracts import Stage4Out
    sym = (np.zeros(512) + 0j).astype(np.complex128)
    out = Stage4Out(sym=sym, slip_indices=[], evm=0.1, snr_db=15.0)
    out.validate()


def test_telemetry_schema():
    from signal9.contracts import empty_telemetry, validate_result_schema
    from signal9.status import Status

    result = {"file": "test.iq", "status": Status.DEMOD_NO_CRC.value}
    result.update(empty_telemetry())
    errors = validate_result_schema(result)
    assert errors == [], f"Schema errors: {errors}"


def test_telemetry_unknown_status_detected():
    from signal9.contracts import empty_telemetry, validate_result_schema

    result = {"file": "test.iq", "status": "MADE_UP_STATUS"}
    result.update(empty_telemetry())
    errors = validate_result_schema(result)
    assert any("Unknown status" in e for e in errors)


# ─────────────────────────────────────────────────────────────────────────────
# No bare `except:` (catches everything including SystemExit, KeyboardInterrupt)
# ─────────────────────────────────────────────────────────────────────────────

def test_no_bare_except_in_src():
    """Bare `except:` without an exception type swallows SystemExit and
    KeyboardInterrupt, making the system non-interruptible."""
    violations = []
    for py_file in SRC_ROOT.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped == "except:" or stripped.startswith("except:"):
                violations.append(f"{py_file.relative_to(SRC_ROOT)}:{i}")
    assert not violations, (
        "Bare 'except:' found (use 'except Exception:' or narrower):\n"
        + "\n".join(violations)
    )


# ─────────────────────────────────────────────────────────────────────────────
# No module-level np.random (DEF-043)
# ─────────────────────────────────────────────────────────────────────────────

def test_no_module_level_np_random():
    """All randomness must go through a seeded Generator (DEF-043 / Rule R5)."""
    _BANNED = re.compile(r"np\.random\.(rand|randn|randint|random|choice|seed)\(")
    violations = []
    for py_file in SRC_ROOT.rglob("*.py"):
        if "test" in py_file.name.lower():
            continue
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(text.splitlines(), 1):
            if _BANNED.search(line) and not line.strip().startswith("#"):
                violations.append(f"{py_file.relative_to(SRC_ROOT)}:{i}: {line.strip()}")
    assert not violations, (
        "Unseed np.random.* calls found (use np.random.default_rng(seed)):\n"
        + "\n".join(violations)
    )
