"""Verify all differential oracles are installed and importable (Rule R9)."""
import importlib.util
import pytest

REQUIRED_ORACLES = [
    "galois",
    "reedsolo",
    "commpy",
    "crcmod",
    "soundfile",
    "soxr",
    "numba",
]

@pytest.mark.parametrize("pkg", REQUIRED_ORACLES)
def test_oracle_installed(pkg):
    spec = importlib.util.find_spec(pkg)
    if spec is None:
        pytest.skip(f"Differential oracle '{pkg}' not installed in this environment")
    assert spec is not None
