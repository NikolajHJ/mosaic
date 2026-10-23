"""Metadata-only promotion fixtures; executed in allocated CPU readiness."""
from copy import deepcopy
import hashlib
import json

import pytest

from latentmol.failure_guidance_selection import authenticate_readiness, select_candidate


def observations():
    values = []
    for name, extra, seconds in (("legacy3", 0, 20000.), ("fallback", 0, 21000.), ("bond6", 3, 25000.)):
        counts = dict(attempts=1024, original_accepted=500, already_valid=400, original_censored=124,
                      unsupported=0, extra_valid=extra, failed_with_nonzero_gradient=120)
        values.append(dict(id=name, eligible=True, selection_epoch_seconds=seconds,
            readiness=dict(native_panel=dict(source_snapshot_chain_sha256="a" * 64, counts=counts))))
    return values


def test_prefer_complete_repairs_then_useful_signal_then_runtime():
    rows = observations()
    assert select_candidate(rows)["winner"] == "bond6"
    rows[2]["readiness"]["native_panel"]["counts"]["extra_valid"] = 0
    assert select_candidate(rows)["winner"] == "fallback"


def test_no_automatic_promotion_when_control_or_all_new_candidates_fail():
    rows = observations()
    rows[0]["eligible"] = False
    assert select_candidate(rows)["winner"] is None
    rows = observations()
    rows[1]["eligible"] = rows[2]["eligible"] = False
    assert select_candidate(rows)["winner"] is None


def test_incomplete_grid_or_different_native_snapshots_cannot_be_compared():
    with pytest.raises(ValueError, match="Incomplete"):
        select_candidate(observations()[:2])
    rows = observations()
    rows[2]["readiness"]["native_panel"]["source_snapshot_chain_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="identical"):
        select_candidate(rows)
    rows = observations()
    rows[2]["readiness"]["native_panel"]["counts"]["original_censored"] -= 1
    with pytest.raises(ValueError, match="coverage"):
        select_candidate(rows)


def test_readiness_hash_required_and_atom_edits_cannot_be_promoted(tmp_path):
    path = tmp_path / "failure_readiness.json"
    path.write_text("{}")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    plan = dict(campaign=str(tmp_path), protocol=dict(failure_guidance=dict(progress_search=dict(allow_atom_edits=True))))
    with pytest.raises(ValueError, match="diagnostic-only"):
        authenticate_readiness(plan, sha)
    with pytest.raises(ValueError, match="identity"):
        authenticate_readiness(plan, "0" * 64)


def test_missing_scientific_gates_rejected_even_with_a_correct_file_hash(tmp_path):
    from latentmol.failure_guidance_selection import VERSION
    path = tmp_path / "failure_readiness.json"
    path.write_text(json.dumps(dict(version=VERSION, status="passed", gates={})))
    plan = dict(campaign=str(tmp_path), protocol=dict(failure_guidance=dict(progress_search=None)),
                _plan_sha256="a" * 64, archive_sha256="b" * 64, manifest_sha256="c" * 64)
    with pytest.raises(ValueError, match="every declared"):
        authenticate_readiness(plan, hashlib.sha256(path.read_bytes()).hexdigest())
