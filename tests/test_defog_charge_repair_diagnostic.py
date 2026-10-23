"""Allocated-HPC fixtures for replay integrity; no claim of native efficacy."""
from copy import deepcopy

import pytest

from latentmol.defog.charge_repair_diagnostic import replay_request, execution_completeness
from latentmol.defog.codec import PROJECT_CHARGE_POLICY
from latentmol.repair_guidance.provider import request_digest
from test_defog_candidate_repair import _candidate, _request


@pytest.mark.parametrize("projected", [False, True])
def test_replay_changes_only_explicit_transport_and_policy(projected):
    spec, graph = _candidate(charge=1)
    original = _request(spec, graph, sampling={}, initialization="direct_candidate_no_corruption")
    saved = deepcopy(original)
    replay = replay_request(original, "b" * 64, projected=projected)
    assert original == saved
    assert replay["request_sha256"] == request_digest(replay)
    changed = {key for key in set(original) | set(replay) if original.get(key) != replay.get(key)}
    assert changed == ({"adapter_source_sha256", "request_sha256", "version", "charge_policy"}
                       if projected else {"adapter_source_sha256", "request_sha256"})
    assert replay.get("charge_policy") == (PROJECT_CHARGE_POLICY if projected else None)
    for key in ("candidate", "spec", "seed", "occupied_slots", "t0", "steps", "sampling", "initialization"):
        assert replay[key] == original[key]


def test_changed_original_request_never_replayed():
    spec, graph = _candidate()
    original = _request(spec, graph)
    original["seed"] += 1
    with pytest.raises(ValueError, match="digest"):
        replay_request(original, "b" * 64, projected=True)


@pytest.mark.parametrize("failed", [None, "strict_timeout", "native_timeout", "incomplete_counter"])
def test_one_accepted_target_cannot_hide_incomplete_prescribed_workers(failed):
    counters = ("native_forward_calls", "native_transitions", "attempted_native_transitions")
    cases = []
    for charged in (False, True, True, True, True):
        attempts = []
        for mode in ("strict_v1", "projected_v2"):
            reject = charged and mode == "strict_v1"
            attempts.append({"mode": mode, "status": "rejected" if reject else "accepted",
                "reason": "input_representation_unsupported" if reject else "chemical_and_representation_gates_passed",
                "process_returncode": 0, "cost": {key: 0 if reject else 447 for key in counters}})
        cases.append({"charged_input": charged, "attempts": attempts})
    if failed == "strict_timeout":
        cases[1]["attempts"][0].update(status="timeout", process_returncode=None)
    elif failed == "native_timeout":
        cases[1]["attempts"][1].update(status="timeout", process_returncode=None)
    elif failed == "incomplete_counter":
        cases[1]["attempts"][1]["cost"]["native_forward_calls"] = 446
    report = execution_completeness(cases)
    assert (report["strict_charged_controls_passed"] and report["supported_continuations_completed"]) == (failed is None)
