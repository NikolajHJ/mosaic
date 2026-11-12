"""Explicit hardware placement preserves matched seed/start blocks."""

from collections import Counter
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "stage3_dispatch", Path(__file__).parents[1] / "scripts/stage3_dispatch.py")
dispatch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dispatch)


def fixture(tmp_path):
    plan = {"mock": False, "bundles": [
        {"bundle_id": f"{seed}-{initialization}-{arm}", "seed": seed,
         "initialization": initialization, "run_ids": [f"r-{seed}-{initialization}-{arm}"]}
        for seed in (17, 23, 42) for initialization in ("sparse", "dense", "fragmented")
        for arm in (0, 1)
    ]}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    (tmp_path / "bundles").mkdir()
    placement = {
        "17:sparse": "v100", "23:sparse": "a100", "42:sparse": "l40s",
        "17:dense": "a100", "23:dense": "v100", "42:dense": "a10",
        "17:fragmented": "v100", "23:fragmented": "a100", "42:fragmented": "a40",
    }
    return path, placement


def prepare(tmp_path, placement, profiles=None):
    path, _ = fixture(tmp_path)
    return dispatch.prepare(path, tmp_path / "dispatch", "/remote/source", "/env/bin/python",
                            "/remote/assets.json", profiles or list(dispatch.PROFILES), placement=placement)


def test_explicit_placement_uses_every_gpu_family_with_two_thirds_on_v100_a100(tmp_path):
    path, placement = fixture(tmp_path)
    original = dict(placement)
    result = dispatch.prepare(path, tmp_path / "dispatch", "/remote/source", "/env/bin/python",
                              "/remote/assets.json", list(dispatch.PROFILES), placement=placement)
    assert result["placement"] == original == placement
    assert result["placement_policy"] == "explicit"
    counts = Counter(job["gpu_profile"] for job in result["jobs"])
    assert counts == {"v100": 6, "a100": 6, "l40s": 2, "a10": 2, "a40": 2}
    assert (counts["v100"] + counts["a100"]) / len(result["jobs"]) == pytest.approx(2 / 3)
    for job in result["jobs"]:
        expected = original[f"{job['seed']}:{job['initialization']}"]
        assert job["gpu_profile"] == expected
        assert job["queue"] == dispatch.PROFILES[expected][0]
        assert f"export MOSAIC_GPU_PROFILE={expected}" in Path(job["script"]).read_text()
        assert "mode=exclusive_process:j_exclusive=yes:mps=no" in " ".join(job["bsub"])


@pytest.mark.parametrize("mutation", ["missing", "extra", "unknown_profile", "unselected_profile", "not_object"])
def test_invalid_placement_fails_before_creating_output(tmp_path, mutation):
    path, placement = fixture(tmp_path)
    profiles = list(dispatch.PROFILES)
    if mutation == "missing":
        placement.pop("17:sparse")
    elif mutation == "extra":
        placement["101:sparse"] = "v100"
    elif mutation == "unknown_profile":
        placement["17:sparse"] = "h100"
    elif mutation == "unselected_profile":
        profiles.remove("v100")
    else:
        placement = list(placement)
    with pytest.raises(ValueError, match="Placement"):
        dispatch.prepare(path, tmp_path / "dispatch", "/remote/source", "/env/bin/python",
                         "/remote/assets.json", profiles, placement=placement)
    assert not (tmp_path / "dispatch").exists()


def test_default_assignment_is_unchanged_and_recorded(tmp_path):
    path, _ = fixture(tmp_path)
    profiles = ["a10", "a100", "v100"]
    result = dispatch.prepare(path, tmp_path / "dispatch", "/remote/source", "/env/bin/python",
                              "/remote/assets.json", profiles)
    assert result["placement_policy"] == "deterministic_block_hash"
    assert all(job["gpu_profile"] == dispatch.assignment(job["seed"], job["initialization"], profiles)
               for job in result["jobs"])
