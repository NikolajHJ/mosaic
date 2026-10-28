"""Saved-only data/FP32 fixtures; execute on allocated DTU HPC CPU only."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "report_data.py"
SPEC = importlib.util.spec_from_file_location("jq1_live_report_data_fixture", SOURCE)
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)


def bind(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": data.sha(path)}


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")
    return bind(path)


def save_rows(path, rows, suffix=b""):
    path.write_bytes(b"".join((json.dumps(row, allow_nan=False) + "\n").encode()
                             for row in rows) + suffix)
    return bind(path)


def make_segment(root, name="segment-0", kind="fixed", offset=0, n=20,
                 requested=None, status="failed"):
    requested = n if requested is None else requested
    directory = root / name
    directory.mkdir()
    snapshots = directory / "snapshots"
    snapshots.mkdir()
    rate = 0.3
    states, updates, retained = [], [], []
    pairs = [[i, j] for i in range(25) for j in range(i + 1, 25)]
    for index in range(n + 1):
        value = float(np.float32(0.5 + (offset + index) / 128))
        keep = index == 0 or index == requested or index % 20 == 0
        state = {"step": index, "snapshot": None, "snapshot_sha256": None,
                 "snapshot_retained": keep, "contact_loss": value,
                 "readout": {"sanitized": False, "canonical_smiles": None,
                             "conversion_error": "Original fixture raw graph is disconnected",
                             "disconnected": True}}
        if kind == "fixed":
            state.update(global_step=offset + index, refreshed_surrogate=value,
                         autodiff_surrogate=value, repair_weight=0.0,
                         components={"contact_loss": value, "binder_probability": 0.25},
                         preparation_identity="a" * 64,
                         previous_fixed_surrogate_after=value if index else None)
        if keep:
            x = np.zeros((25, 6), np.float32)
            x[:, 0] = 1
            e = np.zeros((300, 5), np.float32)
            e[:, 0] = 1
            path = snapshots / ("step-" + str(index).zfill(4) + ".npz")
            fields = dict(X=x, E_upper=e, atom_logits=np.zeros_like(x),
                          bond_logits=np.zeros_like(e), key_data=np.asarray([0, 17], np.uint32),
                          gradient_atom_logits=np.zeros_like(x),
                          gradient_bond_logits=np.zeros_like(e),
                          contact_primal=np.asarray(value, np.float32))
            if kind == "fixed":
                fields.update({key: np.asarray(state[key], np.float64)
                               for key in ("refreshed_surrogate", "autodiff_surrogate", "repair_weight")})
            np.savez(path, **fields)
            state.update(snapshot="snapshots/" + path.name, snapshot_sha256=data.sha(path))
            state["readout"].update(element_indices=[0] * 25, bond_upper_indices=[0] * 300,
                                    elements=["C"] * 25, pairs=pairs)
            retained.append({"step": index, "global_step": offset + index,
                             "snapshot": state["snapshot"], "snapshot_sha256": state["snapshot_sha256"]})
        states.append(state)
    for index in range(n):
        before, after = states[index]["contact_loss"], states[index + 1]["contact_loss"]
        if kind == "noisy":
            update = {"update": index + 1, "before": before, "after": after,
                      "uphill": after > before, "equal": after == before, "no_op": False,
                      "sigma": 0.01, "learning_rate": rate, "independent_primal_parity": True,
                      "noise_sha256": "c" * 64, "noise_snapshot_retained": False,
                      "noise_snapshot": None, "elapsed_seconds": 0.5}
        else:
            update = {"step": index, "global_step": offset + index,
                      "preparation_identity": "a" * 64, "repair_weight": 0.0,
                      "fixed_before": before, "fixed_after": after,
                      "contact_before": before, "contact_after": after,
                      "attempts": [{"trial": 0, "rate": rate, "autodiff_surrogate": after,
                                    "independent_primal_surrogate": after,
                                    "components": {"contact_loss": after}, "repair_weight": 0.0,
                                    "strict": after < before}],
                      "selection_policy": "fixed_step", "selected_trial": 0,
                      "accepted_step_size": rate, "fallback_used": False,
                      "fixed_increased": after > before, "no_op": False,
                      "best_observed_surrogate_for_this_preparation": min(before, after),
                      "elapsed_seconds": 0.5}
        updates.append(update)
    config = save(directory / "config.json", {"learning_rate" if kind == "noisy" else "initial_step_size": rate})
    summary = {"status": status, "source_manifest_sha256": "b" * 64,
               "config_sha256": config["sha256"], "job_id": "123",
               "error": "Original mandatory terminal gradient check failed" if status == "failed" else None}
    if kind == "fixed":
        summary["cases"] = [{"label": "fixture", "states": states, "updates": updates,
                             "status": "passed"}]
    segment = {"id": name, "global_offset": offset, "requested_updates": requested,
               "case_directory": str(directory), "anchor_directory": str(directory),
               "states": save_rows(directory / "states.jsonl", states),
               "updates": save_rows(directory / "updates.jsonl", updates),
               "summary": save(directory / "summary.json", summary), "config": config,
               "retained_index": save(directory / "retained-frames.json",
                                      {"retained_frames": retained, "scalar_states": n + 1,
                                       "scalar_updates": n}),
               "source_manifest_sha256": "b" * 64, "job_id": "123"}
    arm = {"id": "protenix", "label": "Fixture Protenix", "kind": kind, "rate": rate,
           "target_updates": offset + requested, "global_start": offset,
           "global_target": offset + requested, "scientific_status": status,
           "status_detail": summary["error"], "metric_labels": {"contact_loss": "Contact loss"},
           "segments": [segment]}
    return arm, segment, states, updates


def rewrite_record(segment, field, rows):
    segment[field] = save_rows(Path(segment[field]["path"]), rows)


def rewrite_summary(segment, mutate):
    value = data.load(segment["summary"]["path"])
    mutate(value)
    segment["summary"] = save(Path(segment["summary"]["path"]), value)


def rewrite_npz(path, mutate):
    with np.load(path, allow_pickle=False) as original:
        arrays = {key: original[key].copy() for key in original.files}
    mutate(arrays)
    np.savez(path, **arrays)


def test_newline_commit_excludes_unfinished_record(tmp_path):
    binding = save_rows(tmp_path / "rows.jsonl", [{"step": 0}], b'{"step":1')
    rows, proof = data.committed_jsonl(binding)
    assert rows == [{"step": 0}]
    assert proof["committed_rows"] == 1 and proof["uncommitted_suffix_bytes"] == 9


@pytest.mark.parametrize("raw", [b"\n", b"[]\n", b'{"x":NaN}\n', b'{"x":Infinity}\n'])
def test_reject_malformed_committed_record(tmp_path, raw):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        data.committed_jsonl(bind(path))


@pytest.mark.parametrize("field,value", [("bytes", 1), ("sha256", "0" * 64),
                                         ("path", "relative.json")])
def test_original_binding_rejects_wrong_identity(tmp_path, field, value):
    path = tmp_path / "original.json"
    binding = save(path, {"x": 1})
    binding[field] = value
    with pytest.raises(ValueError):
        data.bound_file(binding)


def test_empty_committed_stream_disclosed(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")
    assert data.committed_jsonl(bind(path))[0] == []


def test_exact_frame_preserves_fp32_raw_graph_and_original_readout(tmp_path):
    _, segment, states, _ = make_segment(tmp_path)
    path = data._saved_path(segment["anchor_directory"], states[0]["snapshot"])
    frame, arrays, proof = data.validate_frame(path, states[0], 0)
    assert np.array_equal(np.asarray(frame["atoms"], np.float32), arrays["X"])
    assert np.array_equal(np.asarray(frame["bonds"], np.float32), arrays["E_upper"])
    assert frame["graph"]["edges"] == [] and len(frame["graph"]["nodes"]) == 25
    assert frame["readout"] == states[0]["readout"]
    assert frame["readout"]["disconnected"] is True and proof["argmax_correspondence"]


FRAME_MUTATIONS = [
    ("negative", lambda a: a["X"].__setitem__((0, 0), -0.1)),
    ("above_one", lambda a: a["X"].__setitem__((0, 0), 1.1)),
    ("simplex", lambda a: a["X"].__setitem__((0, 0), 0.9)),
    ("dtype", lambda a: a.__setitem__("X", a["X"].astype(np.float64))),
    ("shape", lambda a: a.__setitem__("E_upper", a["E_upper"][:-1])),
    ("key_dtype", lambda a: a.__setitem__("key_data", a["key_data"].astype(np.int64))),
    ("missing_logits", lambda a: a.pop("atom_logits")),
    ("nonfinite_gradient", lambda a: a["gradient_atom_logits"].__setitem__((0, 0), np.nan)),
    ("unsafe_object", lambda a: a.__setitem__("unexpected", np.asarray([object()], dtype=object))),
    ("primal_scalar", lambda a: a.__setitem__("contact_primal", np.asarray(2.0))),
    ("missing_primal", lambda a: a.pop("contact_primal")),
    ("missing_surrogate", lambda a: a.pop("refreshed_surrogate")),
]


@pytest.mark.parametrize("name,mutate", FRAME_MUTATIONS, ids=[row[0] for row in FRAME_MUTATIONS])
def test_frame_rejects_corrupt_probability_or_correspondence(tmp_path, name, mutate):
    _, segment, states, _ = make_segment(tmp_path)
    state = states[0]
    path = data._saved_path(segment["anchor_directory"], state["snapshot"])
    rewrite_npz(path, mutate)
    state["snapshot_sha256"] = data.sha(path)
    with pytest.raises(ValueError):
        data.validate_frame(path, state, 0)


@pytest.mark.parametrize("key,value", [("element_indices", [1] * 25),
                                       ("bond_upper_indices", [1] * 300),
                                       ("pairs", [[0, 2]]), ("elements", ["N"] * 25)])
def test_frame_rejects_saved_argmax_disagreement(tmp_path, key, value):
    _, segment, states, _ = make_segment(tmp_path)
    states[0]["readout"][key] = value
    path = data._saved_path(segment["anchor_directory"], states[0]["snapshot"])
    with pytest.raises(ValueError, match="argmax"):
        data.validate_frame(path, states[0], 0)


def test_frame_sha_is_authenticated_before_array_loading(tmp_path):
    _, segment, states, _ = make_segment(tmp_path)
    path = data._saved_path(segment["anchor_directory"], states[0]["snapshot"])
    states[0]["snapshot_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA"):
        data.validate_frame(path, states[0], 0)


def test_sparse_rows_retained_only_at_actual_stride_and_terminal(tmp_path):
    arm, segment, _, _ = make_segment(tmp_path, n=21)
    result = data.collect_segment(segment, arm)
    assert [frame["global_step"] for frame, _ in result["retained"]] == [0, 20, 21]
    assert len(result["states"]) == 22 and len(result["updates"]) == 21


@pytest.mark.parametrize("mutation", ["index", "absolute", "retention", "invented_snapshot"])
def test_state_prefix_rejects_gap_or_invented_frame(tmp_path, mutation):
    arm, segment, states, _ = make_segment(tmp_path)
    if mutation == "index":
        states[1]["step"] = 2
    elif mutation == "absolute":
        states[1]["global_step"] = 99
    elif mutation == "retention":
        states[1]["snapshot_retained"] = True
    else:
        states[1]["snapshot"] = "invented.npz"
    rewrite_record(segment, "states", states)
    with pytest.raises(ValueError):
        data.collect_segment(segment, arm)


@pytest.mark.parametrize("mutation", ["two_attempts", "fallback", "rate", "native_scalar", "uphill", "policy"])
def test_fixed_policy_and_scalar_transition_are_authenticated(tmp_path, mutation):
    arm, segment, _, updates = make_segment(tmp_path)
    if mutation == "two_attempts":
        updates[0]["attempts"].append(deepcopy(updates[0]["attempts"][0]))
    elif mutation == "fallback":
        updates[0]["fallback_used"] = True
    elif mutation == "rate":
        updates[0]["attempts"][0]["rate"] = 0.03
    elif mutation == "native_scalar":
        updates[0]["contact_after"] = -99
    elif mutation == "uphill":
        updates[0]["fixed_increased"] = False
    else:
        updates[0]["selection_policy"] = "best_primal"
    rewrite_record(segment, "updates", updates)
    with pytest.raises(ValueError):
        data.collect_segment(segment, arm)


def test_failed_terminal_preserves_complete_optimizer_prefix_and_all_scalars(tmp_path):
    arm, segment, states, updates = make_segment(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    result = data.collect_arm(arm, output)
    rows = data.load(output / result["scalars"]["path"])["rows"]
    assert result["scientific_status"] == "failed" and result["committed_updates"] == 20
    assert result["update_counts"]["uphill"] == 20 and result["update_counts"]["forced_fallback"] == 0
    assert [row["state"] for row in rows] == states
    assert [row["update"] for row in rows[1:]] == updates
    assert rows[1]["losses"]["components.binder_probability"] == 0.25
    assert rows[1]["losses"]["update.elapsed_seconds"] == 0.5


@pytest.mark.parametrize("field,value", [("source_manifest_sha256", "0" * 64),
                                         ("config_sha256", "0" * 64), ("job_id", "999")])
def test_summary_cannot_change_source_config_or_original_job(tmp_path, field, value):
    arm, segment, _, _ = make_segment(tmp_path)
    rewrite_summary(segment, lambda row: row.__setitem__(field, value))
    with pytest.raises(ValueError):
        data.collect_segment(segment, arm)


def test_case_and_committed_jsonl_require_exact_saved_equality(tmp_path):
    arm, segment, _, _ = make_segment(tmp_path)
    rewrite_summary(segment, lambda row: row["cases"][0]["states"][1].__setitem__("contact_loss", -99))
    with pytest.raises(ValueError, match="case"):
        data.collect_segment(segment, arm)


def test_noisy_schema_retains_stochastic_fields_without_fixed_policy_relabel(tmp_path):
    arm, segment, _, _ = make_segment(tmp_path, kind="noisy", status="passed")
    output = tmp_path / "out"
    output.mkdir()
    result = data.collect_arm(arm, output)
    rows = data.load(output / result["scalars"]["path"])["rows"]
    assert result["kind"] == "noisy" and result["scientific_status"] == "passed"
    assert rows[1]["losses"]["update.sigma"] == 0.01
    assert rows[1]["update"]["noise_sha256"] == "c" * 64
    assert "fixed_after" not in rows[1]["update"]


def test_continuation_uses_absolute_steps_and_requested_global_target(tmp_path):
    arm, _, _, _ = make_segment(tmp_path, offset=10000)
    arm["target_updates"] = arm["global_target"] = 50000
    output = tmp_path / "out"
    output.mkdir()
    result = data.collect_arm(arm, output)
    assert result["global_start"] == 10000 and result["latest_global_step"] == 10020
    assert result["committed_updates"] == 20 and result["target_updates"] == 50000
    assert result["frames"]["steps"] == [10000, 10020]


def test_exact_segment_boundary_dedupes_only_frame_and_preserves_both_scalar_observations(tmp_path):
    arm, _, _, _ = make_segment(tmp_path, status="passed")
    _, second, _, _ = make_segment(tmp_path, name="segment-1", offset=20, status="failed")
    arm.update(target_updates=40, global_target=40, scientific_status="failed")
    arm["segments"].append(second)
    output = tmp_path / "out"
    output.mkdir()
    result = data.collect_arm(arm, output)
    scalar = data.load(output / result["scalars"]["path"])
    assert result["committed_updates"] == 40 and result["frames"]["steps"] == [0, 20, 40]
    assert len(scalar["rows"]) == 41 and scalar["boundary_checks"][0]["exact_arrays_and_key"]
    assert scalar["rows"][20]["boundary_observations"][0]["step"] == 0


@pytest.mark.parametrize("field", ["X", "E_upper", "atom_logits", "bond_logits", "key_data"])
def test_boundary_rejects_any_changed_original_array_or_rng(field):
    shapes = {"X": (25, 6), "E_upper": (300, 5), "atom_logits": (25, 6),
              "bond_logits": (300, 5), "key_data": (2,)}
    previous = {key: np.zeros(shape, np.uint32 if key == "key_data" else np.float32)
                for key, shape in shapes.items()}
    current = {key: value.copy() for key, value in previous.items()}
    current[field].flat[0] = 1
    with pytest.raises(ValueError, match="boundary"):
        data._boundary_equal(previous, current)


def test_empty_arm_has_no_molecule_or_fabricated_progress(tmp_path):
    arm = {"id": "empty", "label": "Queued", "kind": "fixed", "rate": 0.3,
           "target_updates": 20, "global_start": 0, "global_target": 20,
           "scientific_status": "not_started", "segments": []}
    result = data.collect_arm(arm, tmp_path)
    assert result["frames"] == {"count": 0, "steps": [], "chunks": []}
    assert result["latest"] is None and result["scalars"]["count"] == 0


def test_chunks_bounded_to_one_hundred_actual_retained_frames(tmp_path):
    arm, _, _, _ = make_segment(tmp_path, n=2020)
    output = tmp_path / "out"
    output.mkdir()
    result = data.collect_arm(arm, output)
    assert [row["count"] for row in result["frames"]["chunks"]] == [100, 2]
    assert result["frames"]["count"] == 102 and result["scalars"]["count"] == 2021
    assert result["latest"]["global_step"] == 2020


def test_original_failure_cannot_be_called_scientific_pass(tmp_path):
    arm, _, _, _ = make_segment(tmp_path)
    arm["scientific_status"] = "passed"
    with pytest.raises(ValueError, match="outcomes"):
        data.collect_arm(arm, tmp_path)


def test_snapshot_path_cannot_escape_original_study(tmp_path):
    path = tmp_path / "outside.npz"
    path.write_bytes(b"not an array")
    inside = tmp_path / "inside"
    inside.mkdir()
    with pytest.raises(ValueError, match="escaped"):
        data._saved_path(inside, "../outside.npz")


def test_main_requires_allocated_hpc_cpu_before_reading_plan(tmp_path):
    with patch.dict(data.os.environ, {}, clear=True), pytest.raises(ValueError, match="allocated"):
        data.main(["--plan", str(tmp_path / "absent.json"), "--output", str(tmp_path / "out")])


def test_render_refuses_existing_immutable_edition(tmp_path):
    arms = [{"id": name, "label": name, "kind": "fixed", "rate": 0.3,
             "target_updates": 20, "global_start": 0, "global_target": 20,
             "scientific_status": "not_started", "segments": []}
            for name in ("protenix", "boltz", "dbmol", "noisy")]
    plan = {"version": data.VERSION, "edition": "edition-0001", "captured_at": "2026-10-25T00:00:00Z", "arms": arms}
    with pytest.raises(FileExistsError):
        data.render(plan, tmp_path)


def test_render_produces_four_saved_only_arms_and_failed_verdict_separately(tmp_path):
    arm, _, _, _ = make_segment(tmp_path)
    arms = []
    for name in ("protenix", "boltz", "dbmol", "noisy"):
        clone = deepcopy(arm)
        clone["id"] = name
        arms.append(clone)
    plan = {"version": data.VERSION, "edition": "edition-0001", "captured_at": "2026-10-25T00:00:00Z", "arms": arms}
    output = tmp_path / "out"
    manifest = data.render(plan, output)
    assessment = data.load(output / "assessment.json")
    assert len(manifest["arms"]) == 4 and all(row["scientific_status"] == "failed" for row in manifest["arms"])
    assert assessment["status"] == "passed_saved_only_render" and not assessment["scientific_success_claim"]
    assert assessment["all_scalar_records_retained"] and not assessment["native_inference_executed"]


def test_six_saved_arms_preserve_each_scientific_verdict_and_continuation_state(tmp_path):
    original, _, _, _ = make_segment(tmp_path)
    arms = []
    for name in ("protenix", "boltz", "dbmol", "protenix_noisy", "protenix_chemistry", "boltz_chemistry"):
        arm = deepcopy(original)
        arm.update(id=name, continuation_status="blocked")
        arms.append(arm)
    result = data.render({"version": data.VERSION, "edition": "edition-0004", "captured_at": "2026-10-28T00:00:00Z", "arms": arms}, tmp_path / "six")
    assert len(result["arms"]) == 6
    assert all(arm["scientific_status"] == "failed" and arm["continuation_status"] == "blocked" for arm in result["arms"])
    assert all(arm["scalars"]["count"] == 21 and arm["frames"]["count"] == 2 for arm in result["arms"])


def test_six_arm_refresh_rejects_duplicate_identity_before_creating_output(tmp_path):
    arms = [{"id": "duplicate"} for _ in range(6)]
    with pytest.raises(ValueError, match="Duplicate"):
        data.render({"version": data.VERSION, "edition": "edition-0004", "captured_at": "2026-10-28T00:00:00Z", "arms": arms}, tmp_path / "duplicate")
    assert not (tmp_path / "duplicate").exists()


def layout_nodes():
    return [{"id": index, "element": "C" if index % 3 else "Cl", "original_marker": index}
            for index in range(25)]


def layout_edges(kind):
    if kind == "path":
        pairs = [(index, index + 1) for index in range(24)]
    elif kind == "invalid_star":
        pairs = [(0, index) for index in range(1, 25)]
    elif kind == "disconnected":
        pairs = [(index, index + 1) for index in range(5)] + [(5, 0)]
        pairs += [(index, index + 1) for index in range(6, 11)]
    elif kind == "dense_invalid":
        pairs = [(first, second) for first in range(25) for second in range(first + 1, 25)]
    else:
        pairs = []
    return [{"source": first, "target": second,
             "label": data.BONDS[1 + number % 4]} for number, (first, second) in enumerate(pairs)]


@pytest.mark.parametrize("kind", ["path", "invalid_star", "disconnected", "dense_invalid", "isolated"])
def test_bond_layout_preserves_all_25_raw_atoms_and_every_original_edge(kind):
    nodes, edges = layout_nodes(), layout_edges(kind)
    original_nodes, original_edges = deepcopy(nodes), deepcopy(edges)
    graph = data.layout_raw_graph(nodes, edges)
    assert [{key: value for key, value in node.items() if key not in ("x", "y")}
            for node in graph["nodes"]] == original_nodes
    assert graph["edges"] == original_edges and nodes == original_nodes and edges == original_edges
    assert len(graph["nodes"]) == 25 and graph["layout"]["identity_preserved"]
    assert graph["layout"]["coordinate_system"] == "normalized_0_1"
    assert all(np.isfinite(node[axis]) and 0 <= node[axis] <= 1
               for node in graph["nodes"] for axis in ("x", "y"))
    assert graph["layout"]["width"] > 0 and graph["layout"]["height"] > 0


def test_raw_bond_layout_is_deterministic_and_independent_of_input_order():
    nodes, edges = layout_nodes(), layout_edges("invalid_star")
    first = data.layout_raw_graph(nodes, edges)
    again = data.layout_raw_graph(nodes, edges)
    reversed_graph = data.layout_raw_graph(list(reversed(nodes)), list(reversed(edges)))
    assert first == again
    assert {node["id"]: (node["x"], node["y"]) for node in first["nodes"]} == {
        node["id"]: (node["x"], node["y"]) for node in reversed_graph["nodes"]}
    assert reversed_graph["edges"] == list(reversed(edges))


def test_raw_chain_layout_is_a_bond_zigzag_instead_of_a_circle():
    graph = data.layout_raw_graph(layout_nodes(), layout_edges("path"))
    assert graph["layout"]["components"] == [{"nodes": list(range(25)), "method": "bond-path-zigzag"}]
    assert len({node["y"] for node in graph["nodes"]}) == 2
    assert all(first["x"] < second["x"] for first, second in zip(graph["nodes"], graph["nodes"][1:]))


def test_disconnected_layout_separates_components_without_adding_connecting_bonds():
    nodes = layout_nodes()
    edges = [{"source": index, "target": index + 1, "label": "SINGLE"}
             for index in list(range(11)) + list(range(12, 24))]
    graph = data.layout_raw_graph(nodes, edges)
    assert graph["layout"]["component_count"] == 2 and graph["edges"] == edges
    components = graph["layout"]["components"]
    boxes = []
    for component in components:
        selected = [node for node in graph["nodes"] if node["id"] in component["nodes"]]
        boxes.append((min(node["x"] for node in selected), max(node["x"] for node in selected),
                      min(node["y"] for node in selected), max(node["y"] for node in selected)))
    first, second = boxes
    assert first[1] < second[0] or second[1] < first[0] or first[3] < second[2] or second[3] < first[2]


def test_simple_ring_is_laid_out_by_its_actual_cycle_then_isolated_nodes_are_preserved():
    edges = [{"source": index, "target": (index + 1) % 6, "label": "AROMATIC"} for index in range(6)]
    graph = data.layout_raw_graph(layout_nodes(), edges)
    assert graph["layout"]["component_count"] == 20
    assert graph["layout"]["components"][0] == {"nodes": list(range(6)), "method": "bond-ring"}
    assert graph["edges"] == edges and len(graph["nodes"]) == 25


@pytest.mark.parametrize("edges", [[{"source": 0, "target": 0, "label": "SINGLE"}],
                                   [{"source": 0, "target": 25, "label": "SINGLE"}],
                                   [{"source": 0, "target": 1, "label": "SINGLE"},
                                    {"source": 1, "target": 0, "label": "DOUBLE"}]])
def test_display_layout_rejects_malformed_edges_without_silently_repairing_them(edges):
    with pytest.raises(ValueError):
        data.layout_raw_graph(layout_nodes(), edges)


def test_display_geometry_cache_is_bounded_and_never_returns_mutable_aliases(monkeypatch):
    monkeypatch.setattr(data, "_LAYOUT_CACHE", data.OrderedDict())
    original = data.layout_raw_graph(layout_nodes(), layout_edges("path"))
    changed = data.layout_raw_graph(layout_nodes(), layout_edges("path"))
    changed["nodes"][0]["x"] = -99
    changed["layout"]["components"][0]["nodes"].clear()
    assert data.layout_raw_graph(layout_nodes(), layout_edges("path")) == original
    for index in range(data._LAYOUT_CACHE_LIMIT + 1):
        nodes = layout_nodes()
        nodes[0]["original_marker"] = index + 10000
        data.layout_raw_graph(nodes, [])
    assert len(data._LAYOUT_CACHE) == data._LAYOUT_CACHE_LIMIT
