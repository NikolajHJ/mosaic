"""Home artifact gating must reject unknown or exhausted quota formats."""

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/hpc/zhome_quota_preflight.py"
    spec = importlib.util.spec_from_file_location("home_quota", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_home_quota_rounds_away_from_optimistic_headroom():
    quota = _module().parse_home("You are using 4.97G of 30.00G quota")
    assert quota == {"used_bytes_upper": 5 * 1024**3, "hard_bytes_lower": 30 * 1024**3}


@pytest.mark.parametrize(
    "text", ["unlimited", "using 31G of 30G quota", "using 0G of 0G quota"]
)
def test_home_quota_rejects_unknown_or_over_capacity(text):
    with pytest.raises(ValueError):
        _module().parse_home(text)
