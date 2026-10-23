"""HPC-only metadata and native-output boundary fixtures; no pretrained load."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from latentmol.boltz.active_set import ContextBank, make_branch, select_slots
from latentmol.boltz.active_set_diagnostic import _probe
from latentmol.boltz.step_control_comparison import locate_validation, validation_evidence
from latentmol.config_identity import CONFIG_HASH_VERSION, config_sha256
from latentmol.multimodel.engine import Counters
from latentmol.optimization import NonfiniteObjective
from test_boltz_active_set_runtime import SPEC, _Decoder, _Scorer, _config


ARCHIVE = "a" * 64


def audit_fixture(tmp_path):
    parent = tmp_path / "parent"
    production = tmp_path / "original-source"
    case = {"id": "random-0", "parent_run": str(parent.resolve()),
            "production_release": str(production.resolve()), "arm": "active_set", "step": 0}
    root = tmp_path / "validation"
    folder = root / "controller" / case["id"] / "audit"
    folder.mkdir(parents=True)
    cfg = {"parent_run": case["parent_run"], "production_release": case["production_release"],
           "parent_arm": case["arm"], "parent_step": case["step"],
           "parent_artifact_sha256": {"saved-state.pt": "b" * 64}}
    record = {"status": "completed", "source_archive_sha256": ARCHIVE, "config": cfg,
              "config_hash_version": CONFIG_HASH_VERSION, "config_sha256": config_sha256(cfg)}
    identities = {"point_sha256": "c" * 64, "context_sha256": "d" * 64,
                  "decoder_state_sha256": "e" * 64, "scorer_state_sha256": "f" * 64}
    report = {"status": "completed", "validation_passed": True, "parent_unchanged": True,
              "frozen_state_unchanged": True, "repeatability_passed": True,
              "native_seed_parity": {"affinity_probability_binary": True},
              "identities_before": identities, "identities_after": deepcopy(identities),
              "runtime": {"cuda_device_name": "fixture-device"},
              "repeats": [{}, {}, {"gradient": {"sha256": "1" * 64, "norm_float64": 1.0}}],
              "local_fd": {"policy": {"version": "local_v1"}, "passed": True, "status": "passed",
                  "direction_results": {name: {"passed": True} for name in
                      ("gradient", "random_1", "random_2", "random_3")}}}
    path = folder / "validation.json"
    (folder / "run.json").write_text(json.dumps(record))
    path.write_text(json.dumps(report))
    return root, path, case, record, report


def test_valid_audit_is_bound_to_exact_state_archive_and_immutable_receipts(tmp_path):
    root, path, case, record, report = audit_fixture(tmp_path)
    evidence = locate_validation(root, case, ARCHIVE)
    assert evidence["path"] == str(path.resolve())
    assert evidence["parent_artifact_sha256"] == record["config"]["parent_artifact_sha256"]
    assert evidence["scientific_identities"] == report["identities_before"]
    assert len(evidence["sha256"]) == 64 and len(evidence["run_sha256"]) == 64
    assert evidence["validated_gradient"]["norm_float64"] == 1.0


@pytest.mark.parametrize("field", ["validation_passed", "frozen_state_unchanged", "parent_unchanged",
                                   "repeatability_passed"])
def test_failed_or_unfinished_derivative_checks_are_withheld(tmp_path, field):
    _, path, case, _, report = audit_fixture(tmp_path)
    report[field] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="did not pass"):
        validation_evidence(path, case, ARCHIVE)


def test_one_failed_direction_cannot_be_hidden_by_overall_pass_flag(tmp_path):
    _, path, case, _, report = audit_fixture(tmp_path)
    report["local_fd"]["direction_results"]["random_2"]["passed"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="did not pass"):
        validation_evidence(path, case, ARCHIVE)


def test_different_archive_state_and_tampered_metadata_are_rejected(tmp_path):
    _, path, case, record, _ = audit_fixture(tmp_path)
    with pytest.raises(ValueError, match="immutable source"):
        validation_evidence(path, case, "2" * 64)
    with pytest.raises(ValueError, match="different saved state"):
        validation_evidence(path, {**case, "step": 4}, ARCHIVE)
    record["config"]["parent_step"] = 4
    (path.parent / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hash changed"):
        validation_evidence(path, case, ARCHIVE)


def test_multiple_audits_are_not_selected_after_seeing_their_results(tmp_path):
    root, path, case, _, _ = audit_fixture(tmp_path)
    duplicate = path.parent.parent / "another-audit" / "validation.json"
    duplicate.parent.mkdir()
    duplicate.write_text(path.read_text())
    with pytest.raises(ValueError, match="exactly one"):
        locate_validation(root, case, ARCHIVE)


@pytest.mark.parametrize("malformed", [False, True])
def test_probe_classifies_observed_nan_scalar_without_swallowing_wrong_shape(tmp_path, malformed):
    class BadOutput(_Scorer):
        def forward(self, features, types, coordinates):
            result = super().forward(features, types, coordinates)
            result["affinity_pred_value"] = torch.tensor([1., 2.]) if malformed else torch.tensor(float("nan"))
            return result
    config = _config(tmp_path)
    decoder = _Decoder()
    point = torch.zeros((1, 1), dtype=torch.float64)
    selection = select_slots(decoder.decode(point), temperature=1.0, threshold=0.5)
    branch = make_branch(BadOutput(), ContextBank(config, SPEC), selection, config, "3" * 64)
    if malformed:
        with pytest.raises(ValueError) as error:
            _probe(decoder, point, branch, config, Counters(), grad_enabled=False,
                   reject_nonfinite_objective=True)
        assert not isinstance(error.value, NonfiniteObjective)
    else:
        with pytest.raises(NonfiniteObjective, match="Observed nonfinite scalar"):
            _probe(decoder, point, branch, config, Counters(), grad_enabled=False,
                   reject_nonfinite_objective=True)
