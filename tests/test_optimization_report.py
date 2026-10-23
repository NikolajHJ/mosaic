"""Pure saved-artifact presentation checks; synthetic values are not model results.

These fixtures exercise scientific reporting boundaries without importing a
decoder, scorer, chemistry toolkit or tensor library. They do not validate any
native molecular experiment or chemical validity assertion.
"""

from copy import deepcopy
import builtins
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path

import pytest

from latentmol.reporting import optimization as report


SPEC = {"version": 1, "max_nodes": 3, "atomic_numbers": [0, 6, 8],
        "formal_charges": [-1, 0, 1],
        "bond_names": ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]}


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _metadata(name="synthetic", status="completed"):
    return {"run_id": name, "name": "SYNTHETIC ARTIFACT TEST", "status": status,
            "source": {"sha256": _sha("source-" + name)},
            "config_sha256": _sha("config-" + name),
            "decoder_state_sha256": _sha("same-decoder"),
            "scorer_state_sha256": _sha("same-scorer"),
            "target_sequence_sha256": _sha("same-target"),
            "msa_sha256": _sha("same-msa"), "graph_spec_sha256": _sha("same-schema"),
            "config": {"objective": {"output": "dbmol_total_loss", "direction": "minimize"},
                       "optimizer": {"steps": 5, "learning_rate": .1},
                       "temperature": 1., "threshold": .5, "anchor_count": 2,
                       "recycling_steps": 0, "start": {"kind": "standard_normal", "seed": 17},
                       "context_manifest": {"path": "/saved/contexts.json", "sha256": _sha("context")},
                       "diagnostics": {"hard_samples": 1, "seed": 23},
                       "fd_policy": "pointwise_native_ad_v4"}}


def _candidate(step, *, draw=0, valid=False):
    # Raw O#C is intentionally retained even though its saved check failed.
    labels = {"atoms": [2, 1, 0], "charges": [1, 1, 1],
              "bonds": [[0, 3, 0], [3, 0, 0], [0, 0, 0]]}
    if valid:
        labels["bonds"] = [[0, 2, 0], [2, 0, 0], [0, 0, 0]]
    return {"candidate_id": f"fixture-s{step:04d}-d{draw:03d}", "arm": "fixed_initial_mask",
            "step": step, "draw": draw, "kind": "argmax" if draw == 0 else "categorical",
            "graph_sha256": _sha(f"graph-{step}-{draw}"), "graph_spec": deepcopy(SPEC),
            "labels": labels, "smiles": "C=O" if valid else None,
            "failure": None if valid else "saved synthetic valence failure", "heavy_atoms": 2}


def _metrics(step):
    return {"step": step, "arm": "fixed_initial_mask", "context_sha256": _sha("pose-0"),
            "selection": {"active_slots": [0, 1], "count": 2},
            "natural_selection": {"active_slots": [0, 1], "count": 2},
            "latent_displacement": .1 * step,
            "score": {"loss": 10. - step, "binding_loss": 8. - .1 * step,
                      "gradient_norm": 0. if step == 3 else .2,
                      "components": {"dbmol_total_loss": 8. - .1 * step,
                                     "dbmol_affinity_loss": 1., "dbmol_contact_loss": 2.,
                                     "dbmol_confidence_loss": .8},
                      "repair_guidance": {"normalized_nll": 20., "applied_weight": .1,
                                          "target_sha256": _sha("target-0")}}}


def _probabilities(step):
    none = [1., 0., 0., 0., 0.]
    pair = [.2, .3, .5, 0., 0.]
    return {"snapshot_version": "shared-relaxed-graph-v1", "step": step,
            "scope": "SYNTHETIC saved conditioned shared graph; no new model calculation",
            "temperature": 1., "graph_spec": deepcopy(SPEC), "row_slots": [0, 1, 2],
            "active_slots": [0, 1],
            "atoms": [[0., .2, .8], [0., .9, .1], [1., 0., 0.]],
            "charges": [[0., 1., 0.], [0., 1., 0.], [0., 1., 0.]],
            "bonds": [[deepcopy(none), deepcopy(pair), deepcopy(none)],
                      [deepcopy(pair), deepcopy(none), deepcopy(none)],
                      [deepcopy(none), deepcopy(none), deepcopy(none)]]}


def _run(tmp_path, name="synthetic", *, status="completed", steps=(0, 3), full=False):
    root = tmp_path / name
    metadata = _metadata(name, status)
    if status != "completed":
        metadata["error"] = "Synthetic stop before requested update budget"
    _write(root / "run.json", metadata)
    arm = root / "fixed_initial_mask"
    _write(root / "summary.json", {"claim_limit": "SYNTHETIC: not a chemical/model result",
           "arms": {arm.name: {"optimization_status": "failed" if status == "failed" else "completed_budget",
                               "requested_steps": 5, "accepted_updates": 2}}})
    candidates = []
    for step in steps:
        _write(arm / f"metrics-{step:04d}.json", _metrics(step))
        candidates.extend([_candidate(step), _candidate(step, draw=1, valid=True)])
        if full:
            _write(arm / f"probabilities-{step:04d}.json", _probabilities(step))
    (arm / "candidates.jsonl").write_text("".join(json.dumps(c) + "\n" for c in candidates), encoding="utf-8")
    return root


def _one(data):
    return data["runs"][0]["trajectories"][0]


def test_failed_raw_graphs_are_retained_without_chemical_repair(tmp_path):
    root = _run(tmp_path)
    original = (root / "fixed_initial_mask/candidates.jsonl").read_bytes()
    trajectory = _one(report.build_report([root]))
    first = trajectory["frames"][0]["candidates"][0]
    assert first["validity"] == "invalid"
    assert first["smiles"] is None and "valence failure" in first["failure"]
    assert first["graph"]["raw_labels"] == _candidate(0)["labels"]
    assert first["graph"]["bonds"] == [{"i": 0, "j": 1, "type": "TRIPLE"}]
    assert [atom["slot"] for atom in first["graph"]["atoms"]] == [0, 1]
    assert trajectory["summary"]["argmax_valid"] == 0
    assert trajectory["summary"]["sampled_valid"] == 2
    assert trajectory["summary"]["sampled_total"] == 2
    assert (root / "fixed_initial_mask/candidates.jsonl").read_bytes() == original


def test_partial_failed_run_preserves_step_gaps_and_budget(tmp_path):
    root = _run(tmp_path, status="failed")
    data = report.build_report([root])
    trajectory = _one(data)
    assert data["runs"][0]["status"] == "failed"
    assert "Synthetic stop" in data["runs"][0]["error"]
    assert trajectory["status"] == "failed"
    assert [frame["step"] for frame in trajectory["frames"]] == [0, 3]
    assert trajectory["summary"]["gaps"] == [1, 2]
    assert trajectory["summary"]["requested_steps"] == 5
    assert trajectory["summary"]["accepted_updates"] == 2
    assert any("missing step" in warning for warning in data["runs"][0]["warnings"])
    assert any("not a scientific pass" in warning for warning in data["runs"][0]["warnings"])


def test_missing_probabilities_are_not_reconstructed_from_hard_labels(tmp_path):
    trajectory = _one(report.build_report([_run(tmp_path)]))
    assert all(frame["probabilities"] is None for frame in trajectory["frames"])
    assert trajectory["frames"][-1]["metrics"]["gradient_norm"] == 0.
    assert trajectory["frames"][0]["metrics"]["natural_count"] == 2


def test_historical_conditioned_atom_probabilities_are_recovered_only(tmp_path):
    root = _run(tmp_path, steps=(0,))
    path = root / "fixed_initial_mask/metrics-0000.json"
    row = json.loads(path.read_text())
    saved = [[0., .25, .75], [0., 1., 0.]]
    row["score"]["elements"] = {"atomic_numbers": [0, 6, 8], "active_slots": [0, 1],
                                  "conditional_probabilities": saved}
    _write(path, row)
    probabilities = _one(report.build_report([root]))["frames"][0]["probabilities"]
    assert probabilities["atoms"] == saved and probabilities["row_slots"] == [0, 1]
    assert probabilities["charges"] is None and probabilities["bonds"] is None
    assert probabilities["atoms"][1][2] == 0.
    assert "conditioned" in probabilities["scope"]


def test_full_snapshots_preserve_charge_atom_and_typed_bond_vocabulary(tmp_path):
    data = report.build_report([_run(tmp_path, full=True, steps=(0,))])
    probability = _one(data)["frames"][0]["probabilities"]
    assert probability["graph_spec"] == SPEC
    assert probability["row_slots"] == [0, 1, 2]
    assert probability["active_slots"] == [0, 1]
    assert probability["bonds"][0][1] == [.2, .3, .5, 0., 0.]
    assert probability["bonds"][0][1] == probability["bonds"][1][0]
    assert probability["charges"][0] == [0., 1., 0.]
    assert not data["runs"][0]["warnings"]


@pytest.mark.parametrize("change", ["atom_rows", "atom_sum", "charge_vocab", "bond_shape", "bond_asymmetry", "diagonal", "row_slots"])
def test_malformed_full_distributions_are_rejected(tmp_path, change):
    root = _run(tmp_path, steps=(0,), full=True)
    value = _probabilities(0)
    if change == "atom_rows": value["atoms"].pop()
    elif change == "atom_sum": value["atoms"][0] = [0., .1, .1]
    elif change == "charge_vocab": value["charges"][0] = [0., 1.]
    elif change == "bond_shape": value["bonds"][2].pop()
    elif change == "bond_asymmetry": value["bonds"][1][0] = [1., 0., 0., 0., 0.]
    elif change == "diagonal": value["bonds"][0][0] = [0., 1., 0., 0., 0.]
    else: value["row_slots"] = [0, 0, 2]
    _write(root / "fixed_initial_mask/probabilities-0000.json", value)
    with pytest.raises(ValueError):
        report.build_report([root])


@pytest.mark.parametrize("change", ["snapshot_version", "step", "schema", "mask"])
def test_full_snapshot_is_bound_to_its_actual_saved_state(tmp_path, change):
    root = _run(tmp_path, steps=(0,), full=True)
    value = _probabilities(0)
    if change == "snapshot_version": value["snapshot_version"] = "unrecognized"
    elif change == "step": value["step"] = 3
    elif change == "schema": value["graph_spec"]["atomic_numbers"] = [0, 6, 7]
    else: value["active_slots"] = [0, 2]
    _write(root / "fixed_initial_mask/probabilities-0000.json", value)
    with pytest.raises(ValueError):
        report.build_report([root])


def test_changed_context_target_and_weight_are_explicit_events(tmp_path):
    root = _run(tmp_path)
    path = root / "fixed_initial_mask/metrics-0003.json"
    row = json.loads(path.read_text())
    row["context_sha256"] = _sha("pose-1")
    row["score"]["repair_guidance"].update(target_sha256=_sha("target-1"), applied_weight=.5)
    _write(path, row)
    frame = _one(report.build_report([root]))["frames"][1]
    events = " ".join(frame["events"]).lower()
    assert frame["context_changed"] and "context" in events
    assert "target changed" in events and "coefficient changed" in events
    assert frame["metrics"]["binding_loss"] == 7.7
    assert frame["metrics"]["repair_weight"] == .5


def test_saved_v2_outcomes_and_best_binding_state_are_explicit_without_recalculation(tmp_path):
    root = _run(tmp_path, steps=(0, 1, 2, 3))
    metadata = json.loads((root / "run.json").read_text())
    metadata["config"]["optimizer"]["steps"] = 3
    metadata["config"]["step_control"] = {"version": "best_trial_v2", "max_trials": 16}
    _write(root / "run.json", metadata)
    context = _sha("pose-0")
    context_mask = _sha("pose-0-and-slots-0-1")
    best = {context_mask: {"step": 1, "loss": 7.5, "state_path": "state-0001.pt",
                           "context_sha256": context, "active_slots": [0, 1], "context_mask_sha256": context_mask}}
    summary = json.loads((root / "summary.json").read_text())
    summary["arms"]["fixed_initial_mask"].update(requested_steps=3, accepted_updates=3,
        applied_updates=3, improving_updates=1, forced_fallback_updates=2, no_op_updates=1,
        best_binding_states=best)
    _write(root / "summary.json", summary)
    outcomes = [("first_improvement", True, .5, False),
                ("best_finite_fallback", False, -.2, False),
                ("best_finite_fallback", False, 0., True)]
    binding_losses = [8., 7.5, 7.7, 7.7]
    for step in range(4):
        path = root / f"fixed_initial_mask/metrics-{step:04d}.json"
        row = json.loads(path.read_text())
        row["score"].update(loss=binding_losses[step] + 2., binding_loss=binding_losses[step])
        if step < 3:
            reason, improved, gain, no_op = outcomes[step]
            row.update(update_status="accepted", selection_reason=reason,
                       strictly_improved=improved, applied_loss_improvement=gain, no_op=no_op)
        _write(path, row)
    before = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    data = report.build_report([root])
    trajectory = _one(data)
    assert trajectory["summary"]["step_control_version"] == "best_trial_v2"
    assert [trajectory["summary"][key] for key in
            ("applied_updates", "improving_updates", "forced_fallback_updates", "no_op_updates")] == [3, 1, 2, 1]
    assert trajectory["summary"]["best_binding_states"] == best
    frames = trajectory["frames"]
    assert frames[1]["step_selection"] == dict(selection_reason="best_finite_fallback",
        strictly_improved=False, applied_loss_improvement=-.2, no_op=False)
    assert any("higher objective" in event for event in frames[1]["events"])
    assert any("equal objective" in event for event in frames[2]["events"])
    assert any("no-op" in event for event in frames[2]["events"])
    assert any("Saved best binding state" in event for event in frames[1]["events"])
    assert "step_selection" not in frames[3]  # Terminal state has no outgoing update.
    assert any("not a count of strict improvements" in warning for warning in data["warnings"])
    assert not any(source["path"].endswith(".pt") for source in data["sources"])
    assert before == {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("no_op", [False, True])
def test_saved_equal_full_step_is_neither_strict_improvement_nor_fallback(tmp_path, no_op):
    root = _run(tmp_path, steps=(0, 1))
    metadata = json.loads((root / "run.json").read_text())
    metadata["config"]["optimizer"]["steps"] = 1
    metadata["config"]["step_control"] = {"version": "best_trial_v2", "max_trials": 16}
    _write(root / "run.json", metadata)
    summary = json.loads((root / "summary.json").read_text())
    summary["arms"]["fixed_initial_mask"].update(requested_steps=1, accepted_updates=1,
        applied_updates=1, improving_updates=0, forced_fallback_updates=0, no_op_updates=int(no_op))
    _write(root / "summary.json", summary)
    selection = dict(selection_reason="full_step_nonincrease", strictly_improved=False,
                     applied_loss_improvement=0., no_op=no_op)
    path = root / "fixed_initial_mask/metrics-0000.json"
    row = json.loads(path.read_text())
    row.update(update_status="accepted", **selection)
    _write(path, row)
    trajectory = _one(report.build_report([root]))
    frame = trajectory["frames"][0]
    assert frame["step_selection"] == selection
    assert all(frame["raw_metrics"][key] == value for key, value in selection.items())
    assert any("full step applied without objective increase" in event for event in frame["events"])
    assert not any("strictly lower" in event or "fallback" in event for event in frame["events"])
    assert any("no-op" in event for event in frame["events"]) is no_op
    assert [trajectory["summary"][key] for key in
            ("applied_updates", "improving_updates", "forced_fallback_updates", "no_op_updates")] == [1, 0, 0, int(no_op)]


def test_older_missing_update_fields_and_best_states_remain_missing(tmp_path):
    trajectory = _one(report.build_report([_run(tmp_path)]))
    for key in ("applied_updates", "improving_updates", "forced_fallback_updates", "no_op_updates", "best_binding_states"):
        assert key not in trajectory["summary"]
    assert all("step_selection" not in frame for frame in trajectory["frames"])
    assert trajectory["summary"]["accepted_updates"] == 2


def test_best_binding_receipt_does_not_invent_a_missing_frame_or_cross_context(tmp_path):
    root = _run(tmp_path, steps=(0,))
    summary = json.loads((root / "summary.json").read_text())
    saved_best = {_sha("other-pose"): {"step": 0, "loss": 1., "state_path": "unread-state.pt"},
                  _sha("pose-0"): {"step": 9, "loss": 2., "state_path": "missing-state.pt"},
                  _sha("other-mask"): {"step": 0, "loss": 1.5, "state_path": "other-mask.pt",
                                        "context_sha256": _sha("pose-0"), "active_slots": [1, 2]}}
    summary["arms"]["fixed_initial_mask"]["best_binding_states"] = saved_best
    _write(root / "summary.json", summary)
    trajectory = _one(report.build_report([root]))
    assert trajectory["summary"]["best_binding_states"] == saved_best
    assert [frame["step"] for frame in trajectory["frames"]] == [0]
    assert not any("Saved best binding state" in event for event in trajectory["frames"][0]["events"])


def test_distinct_source_and_config_receipts_are_not_merged(tmp_path):
    a, b = _run(tmp_path, "left"), _run(tmp_path, "right")
    data = report.build_report([a, b], labels=["Control", "Guided"], require_matched=True)
    assert [run["label"] for run in data["runs"]] == ["Control", "Guided"]
    assert len({run["identity"]["source_sha256"] for run in data["runs"]}) == 2
    assert len({run["identity"]["config_sha256"] for run in data["runs"]}) == 2
    assert len([source for source in data["sources"] if source["path"].endswith("run.json")]) == 2
    assert data["comparison"]["policy"] == "matched_identity_checks"


@pytest.mark.parametrize("change", ["decoder", "msa", "context", "anchor", "budget", "samples", "failed", "initialization"])
def test_matched_label_rejects_scientifically_different_inputs(tmp_path, change):
    a, b = _run(tmp_path, "left"), _run(tmp_path, "right")
    path = b / "run.json"; value = json.loads(path.read_text())
    if change == "decoder": value["decoder_state_sha256"] = _sha("other-decoder")
    elif change == "msa": value["msa_sha256"] = _sha("other-msa")
    elif change == "context": value["config"]["context_manifest"]["sha256"] = _sha("other-pose")
    elif change == "anchor": value["config"]["anchor_count"] = 3
    elif change == "budget": value["config"]["optimizer"]["steps"] = 100
    elif change == "samples": value["config"]["diagnostics"]["hard_samples"] = 8
    elif change == "failed": value["status"] = "failed"
    else: value["config"]["start"]["seed"] = 99
    _write(path, value)
    with pytest.raises(ValueError):
        report.build_report([a, b], require_matched=True)
    descriptive = report.build_report([a, b])
    assert len(descriptive["runs"]) == 2
    assert descriptive["comparison"]["policy"] == "descriptive_overlay"


def test_early_native_gate_failure_can_be_presented_without_fabricated_states(tmp_path):
    root = tmp_path / "early-failure"
    value = _metadata("early-failure", "failed")
    value["config"].update(decoder={"path": "/saved/decoder.pt", "sha256": _sha("decoder")},
                           contexts={"2": {"path": "/saved/prepared.pt", "sha256": _sha("context")}})
    value["error"] = "Synthetic native gate stopped before first metric"
    _write(root / "run.json", value)
    _write(root / "fixed_initial_mask/gate-0000.json", {"passed": False, "status": "failed"})
    data = report.build_report([root])
    assert data["runs"][0]["status"] == "failed"
    assert "before first metric" in data["runs"][0]["error"]
    frames = _one(data)["frames"]
    assert len(frames) == 1 and frames[0]["step"] == 0
    assert frames[0]["metrics"]["loss"] is None
    assert frames[0]["candidates"] == [] and frames[0]["probabilities"] is None
    assert frames[0]["audit"]["passed"] is False
    assert frames[0]["raw_metrics"]["missing_scored_metrics"] is True


def test_saved_candidate_before_scoring_failure_remains_visible(tmp_path):
    root = _run(tmp_path, status="failed", steps=(0,))
    path = root / "fixed_initial_mask/candidates.jsonl"
    path.write_text(path.read_text() + json.dumps(_candidate(4)) + "\n")
    frame = _one(report.build_report([root]))["frames"][-1]
    assert frame["step"] == 4 and frame["metrics"]["loss"] is None
    assert frame["candidates"][0]["graph"]["raw_labels"] == _candidate(4)["labels"]
    assert frame["candidates"][0]["validity"] == "invalid"
    assert frame["raw_metrics"]["missing_scored_metrics"] is True


def test_comparison_exceptions_are_explicit_and_do_not_relax_other_inputs(tmp_path):
    a, b = _run(tmp_path, "left"), _run(tmp_path, "right")
    path = b / "run.json"; value = json.loads(path.read_text())
    value["config"]["start"]["seed"] = 99
    _write(path, value)
    assert len(report.build_report([a, b], require_matched=True,
               allow_initialization_comparison=True)["runs"]) == 2
    value["config"]["optimizer"]["learning_rate"] = .3
    _write(path, value)
    with pytest.raises(ValueError):
        report.build_report([a, b], require_matched=True, allow_initialization_comparison=True)


def test_guidance_comparison_keeps_provider_identity_and_native_receipts(tmp_path):
    a, b = _run(tmp_path, "control"), _run(tmp_path, "guided")
    for root, weight in ((a, 0.), (b, .1)):
        path = root / "run.json"; value = json.loads(path.read_text())
        value["config"].update(config_path=str(root / "native.yaml"), config_file_sha256=_sha(root.name),
            repair_guidance={"weight": weight, "weight_schedule": {"end_weight": weight},
                             "provider": {"checkpoint_sha256": _sha("same-repair-model")},
                             "refresh_interval": 5, "attempts_per_refresh": 4})
        _write(path, value)
    with pytest.raises(ValueError): report.build_report([a, b], require_matched=True)
    data = report.build_report([a, b], require_matched=True, allow_guidance_comparison=True)
    assert [run["config"]["repair_guidance"]["weight"] for run in data["runs"]] == [0., .1]
    assert len({run["config"]["config_file_sha256"] for run in data["runs"]}) == 2
    path = b / "run.json"; value = json.loads(path.read_text())
    value["config"]["repair_guidance"]["provider"]["checkpoint_sha256"] = _sha("different-model")
    _write(path, value)
    with pytest.raises(ValueError):
        report.build_report([a, b], require_matched=True, allow_guidance_comparison=True)


def test_duplicate_memberships_and_nonfinite_saved_json_are_rejected(tmp_path):
    root = _run(tmp_path, steps=(0,))
    path = root / "fixed_initial_mask/candidates.jsonl"
    path.write_text(path.read_text() + json.dumps(_candidate(0)) + "\n")
    with pytest.raises(ValueError): report.build_report([root])
    other = _run(tmp_path, "nonfinite", steps=(0,))
    path = other / "fixed_initial_mask/metrics-0000.json"
    path.write_text('{"step":0,"score":{"loss":NaN}}')
    with pytest.raises(ValueError): report.build_report([other])


def test_referenced_artifact_symlink_cannot_escape_declared_root(tmp_path):
    root = _run(tmp_path)
    outside = tmp_path / "outside.json"
    _write(outside, _metrics(0))
    target = root / "fixed_initial_mask/metrics-0000.json"
    target.unlink()
    try:
        target.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("This filesystem/user cannot create symlinks")
    with pytest.raises(ValueError): report.build_report([root])


def test_path_escape_is_rejected_before_reading_any_outside_bytes(tmp_path):
    root = tmp_path / "inside"; root.mkdir()
    outside = tmp_path / "outside.json"; _write(outside, {"unexpected": "outside"})
    reader = report.SavedArtifacts()
    with pytest.raises(ValueError): reader.read(root / "../outside.json", root)
    assert reader.sources == {}


class _HTMLScripts(HTMLParser):
    def __init__(self):
        super().__init__(); self.payloads = []; self.external = []; self._json = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script":
            self._json = attrs.get("type") == "application/json"
            if attrs.get("src"): self.external.append(attrs["src"])
        if tag == "link" and attrs.get("href", "").startswith(("http:", "https:", "//")):
            self.external.append(attrs["href"])

    def handle_endtag(self, tag):
        if tag == "script": self._json = False

    def handle_data(self, data):
        if self._json: self.payloads.append(data)


def test_offline_html_hash_manifest_and_script_escaping(tmp_path):
    root = _run(tmp_path, full=True)
    hostile = '</script><img src=x onerror="alert(1)">\u2028\u2029&'
    output = tmp_path / "report"
    html_path = report.render_report([root], output, title=hostile, labels=[hostile])
    html = html_path.read_text(encoding="utf-8")
    data = json.loads((output / "report.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert data["title"] == hostile and data["runs"][0]["label"] == hostile
    assert hostile not in html and '@@REPORT_' not in html
    parsed = _HTMLScripts(); parsed.feed(html)
    assert not parsed.external
    embedded = [json.loads(text) for text in parsed.payloads]
    assert any(value.get("report_version") == data["report_version"] and value["title"] == hostile for value in embedded)
    for name, expected in manifest["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    for source in manifest["inputs"]:
        raw = Path(source["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source["sha256"] and len(raw) == source["bytes"]
    assert "optimization.py" in manifest["renderer"]
    with pytest.raises(FileExistsError): report.render_report([root], output)


def test_presentation_needs_no_hpc_or_scientific_library_imports(tmp_path, monkeypatch):
    root = _run(tmp_path, full=True)
    monkeypatch.delenv("LSB_JOBID", raising=False)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "numpy", "rdkit"} or name.startswith("latentmol.models"):
            raise AssertionError("Saved-artifact presentation imported a scientific model dependency: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    data = report.build_report([root])
    assert data["runs"][0]["trajectories"]
    assert report.render_report([root], tmp_path / "offline").is_file()


def test_default_presentation_failure_retains_original_failed_scientific_record(tmp_path, monkeypatch):
    root = _run(tmp_path, status="failed")
    before = (root / "run.json").read_bytes()

    def fail(*args, **kwargs):
        raise ValueError("Synthetic presentation-only error")

    monkeypatch.setattr(report, "render_report", fail)
    receipt = report.default_report(root)
    assert receipt["status"] == "failed" and receipt["error_type"] == "ValueError"
    assert (root / "run.json").read_bytes() == before
    assert json.loads((root / "interactive-report-status.json").read_text())["status"] == "failed"


def test_experiment_hook_preserves_scientific_exception_when_renderer_fails(tmp_path, monkeypatch):
    from latentmol.experiments import ExperimentRun

    source = tmp_path / "source"; source.mkdir()
    scientific_error = RuntimeError("Original synthetic scientific failure")

    def fail(*args, **kwargs):
        raise ValueError("Separate synthetic HTML assembly failure")

    monkeypatch.setattr(report, "render_report", fail)
    with pytest.raises(RuntimeError) as caught:
        with ExperimentRun(tmp_path / "runs", "report-hook-failure", {}, source,
                           report_kind="molecule_optimization") as run:
            raise scientific_error
    assert caught.value is scientific_error
    metadata = json.loads((run.path / "run.json").read_text())
    assert metadata["status"] == "failed" and metadata["error"] == str(scientific_error)
    trace = (run.path / "traceback.txt").read_text()
    assert str(scientific_error) in trace and "HTML assembly failure" not in trace
    assert json.loads((run.path / "interactive-report-status.json").read_text())["status"] == "failed"


def test_terminal_experiment_record_hash_stays_bound_after_successful_report_hook(tmp_path):
    from latentmol.experiments import ExperimentRun

    source = tmp_path / "source"; source.mkdir()
    scientific_error = RuntimeError("Synthetic failure with a renderable saved state")
    with pytest.raises(RuntimeError) as caught:
        with ExperimentRun(tmp_path / "runs", "terminal-record-hash", _metadata()["config"], source,
                           report_kind="molecule_optimization") as run:
            arm = run.path / "fixed_initial_mask"
            _write(arm / "metrics-0000.json", _metrics(0))
            (arm / "candidates.jsonl").write_text(json.dumps(_candidate(0)) + "\n")
            raise scientific_error
    assert caught.value is scientific_error
    receipt = json.loads((run.path / "interactive-report-status.json").read_text())
    assert receipt["status"] == "generated"
    directory = run.path / "interactive-report"
    manifest = json.loads((directory / "manifest.json").read_text())
    bound = next(item for item in manifest["inputs"] if Path(item["path"]) == run.path / "run.json")
    assert bound["sha256"] == hashlib.sha256((run.path / "run.json").read_bytes()).hexdigest()
    saved = json.loads((directory / "report.json").read_text())
    assert saved["runs"][0]["status"] == "failed"
    assert saved["runs"][0]["error"] == str(scientific_error)
    assert "renderable saved state" in (run.path / "traceback.txt").read_text()
