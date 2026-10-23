"""Allocated pilot guards and exact FP32 split/replay with FP64 loss scalars."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.chemistry import GraphLogits, mosaic_spec
from mosaic.small_molecule.chemistry import checks as chemistry_checks
from mosaic.small_molecule.chemistry import trajectory as chemistry_trajectory
from mosaic.small_molecule.chemistry.native_only import NativeOnlyPreparation
from mosaic.small_molecule.chemistry.trajectory import run_chemical_updates


def _protocol():
    try:
        import dbmol_protocol
        return dbmol_protocol
    except ModuleNotFoundError:
        path = Path(__file__).resolve().parents[1] / "runs" / "20261023-boltz-dbmol-pilot-01" / "dbmol_protocol.py"
        spec = importlib.util.spec_from_file_location("dbmol_protocol", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


protocol = _protocol()


@pytest.fixture
def config(monkeypatch):
    inherited = {"version": "original-fixture", "arm": "boltz2", "n_atoms": 25,
                 "initial_step_size": .005, "steps": 10000,
                 "acceptance_policy": "fixed_step", "max_backtracks": 1,
                 "temperature": 1., "seed": 17, "chemistry_enabled": False}
    original_read = protocol.read
    def read(path):
        if Path(path).name == "brd4_jq1_25_fixed_lrreset_boltz2_10000.json":
            return copy.deepcopy(inherited)
        return original_read(path)
    monkeypatch.setattr(protocol, "read", read)
    inherited.update(
        version=protocol.VERSION, initial_step_size=.0025,
        label="JQ125 / DBMol binding-only / fixedLR0.0025 / fresh10000",
        dbmol={"version": protocol.SUITE, "arithmetic": protocol.ARITHMETIC,
               "contact_weight": 2.5, "confidence_weight": 2.5,
               "affinity": "mean(value)-mean(raw_binary_logit)",
               "segment_updates": 2500, "chemistry_enabled": False})
    return inherited


def case_fixture():
    return {"acceptance_policy": "fixed_step", "updates": [{
        "fixed_before": 1., "fixed_after": 2., "repair_weight": 0.,
        "selected_trial": 0, "accepted_step_size": .0025,
        "fallback_used": False, "selection_policy": "fixed_step",
        "fixed_increased": True, "no_op": False,
        "attempts": [{"trial": 0, "rate": .0025, "autodiff_surrogate": 2.,
                      "independent_primal_surrogate": 2., "repair_weight": 0.,
                      "components": {name: 1. for name in protocol.COMPONENTS}}],
    }]}


def test_declared_fixed_pilot_parameters_and_finite_uphill_candidate(config):
    assert protocol.config25(config) == config
    assert protocol.update_parameters(config) == {"step_size": .0025,
        "max_backtracks": 1, "acceptance_policy": "fixed_step"}
    assert protocol.verify_attempt_rates(case_fixture(), config)["step_size"] == .0025


@pytest.mark.parametrize("fault", ("rate", "steps", "temperature", "seed", "chemistry", "contact_weight", "confidence_weight", "arithmetic", "extra"))
def test_config_rejects_undeclared_policy_changes(config, fault):
    if fault == "rate": config["initial_step_size"] = .001
    elif fault == "steps": config["steps"] = 30000
    elif fault == "temperature": config["temperature"] = .5
    elif fault == "seed": config["seed"] = 18
    elif fault == "chemistry": config["dbmol"]["chemistry_enabled"] = True
    elif fault in ("contact_weight", "confidence_weight"): config["dbmol"][fault] = 1.
    elif fault == "arithmetic": config["dbmol"][fault] = "native_v1"
    else: config["new_unreviewed_field"] = True
    with pytest.raises(ValueError): protocol.config25(config)


@pytest.mark.parametrize("fault", ("halving", "second_trial", "fallback", "wrong_selected", "false_increase", "nonfinite", "missing_component", "chemistry", "not_bool"))
def test_execution_cannot_mislabel_search_or_nonfinite_or_changing_objectives(config, fault):
    case = case_fixture()
    u, a = case["updates"][0], case["updates"][0]["attempts"][0]
    if fault == "halving": a["rate"] = .00125
    elif fault == "second_trial": u["attempts"].append(copy.deepcopy(a))
    elif fault == "fallback": u["fallback_used"] = True
    elif fault == "wrong_selected": u["selected_trial"] = 1
    elif fault == "false_increase": u["fixed_increased"] = False
    elif fault == "nonfinite": a["components"]["dbmol_total_loss"] = float("nan")
    elif fault == "missing_component": del a["components"]["dbmol_contact_loss"]
    elif fault == "chemistry": u["repair_weight"] = a["repair_weight"] = .001
    else: u["no_op"] = 0
    with pytest.raises(ValueError): protocol.verify_attempt_rates(case, config)


@pytest.mark.parametrize("offset", (0, 2500, 5000, 7500))
def test_exact_four_segment_ranges_and_predecessor_requirement(offset):
    snapshot, digest = (None, None) if offset == 0 else ("previous.npz", "a" * 64)
    record = protocol.scope(offset, 2500, snapshot, digest)
    assert (record["start"], record["end"], record["target"]) == (offset, offset + 2500, 10000)
    assert record["input_snapshot_sha256"] == digest


@pytest.mark.parametrize("args", ((0, 10000, None, None), (10000, 2500, "x", "a" * 64),
                                  (-2500, 2500, "x", "a" * 64), (2500, 2500, None, None),
                                  (0, 2500, "x", "a" * 64), (2500, 2500, "x", None)))
def test_segment_scope_never_implies_a_fresh_resume_or_extra_budget(args):
    with pytest.raises(ValueError): protocol.scope(*args)


def identity():
    return {
        "protocol": "mosaic-boltz-dbmol-v1", "objective_arithmetic": protocol.ARITHMETIC,
        "checkpoint_sha256": protocol.AFFINITY_CHECKPOINT,
        "structural_checkpoint_sha256": protocol.STRUCTURAL_CHECKPOINT,
        "affinity_model_digest": protocol.AFFINITY_MODEL,
        "context_digest": protocol.CONTEXT, "context_file_sha256": protocol.CONTEXT_FILE,
        "pocket_definition_sha256": protocol.POCKET, "diffusion_gradient": False,
        "recycling_steps": 0, "affinity_mw_correction": False, "n_atoms": 25,
        "optimized_output": "dbmol_total_loss", "scalar_dtype": "float64",
        "graph_dtype": "float32", "gradient_dtype": "float32",
        "model_digest": "a" * 64, "structural_model_digest": "b" * 64,
        "full_context_digest": "c" * 64,
    }


@pytest.mark.parametrize("name", ("checkpoint_sha256", "structural_checkpoint_sha256", "affinity_model_digest", "context_digest", "context_file_sha256", "pocket_definition_sha256", "scalar_dtype", "graph_dtype", "gradient_dtype"))
def test_native_identity_rejects_asset_context_and_precision_substitution(name):
    value = identity()
    protocol.verify_identity(value)
    value[name] = "changed"
    with pytest.raises(ValueError): protocol.verify_identity(value)


@pytest.mark.parametrize("name", ("model_digest", "affinity_model_digest", "structural_model_digest", "context_digest", "full_context_digest", "context_file_sha256", "checkpoint_sha256", "structural_checkpoint_sha256", "frozen"))
def test_frozen_identity_guard_rejects_any_mutation(name):
    before, after = identity(), {**identity(), "frozen": True}
    protocol.frozen(after, before)
    after[name] = False if name == "frozen" else "changed"
    with pytest.raises(ValueError): protocol.frozen(after, before)


def initial():
    assert jax.config.jax_enable_x64
    return GraphLogits(jnp.linspace(-.7, .9, 150, dtype=jnp.float32).reshape(25, 6),
                       jnp.linspace(-.8, 1.1, 1500, dtype=jnp.float32).reshape(300, 5))


def checkpoint(path, state, key):
    graph = state.graph()
    np.savez_compressed(path, X=np.asarray(graph.X), E_upper=np.asarray(graph.E_upper),
                        atom_logits=np.asarray(state.atoms), bond_logits=np.asarray(state.bonds),
                        key_data=np.asarray(jax.random.key_data(key)))


def test_restore_is_exact_original_fp32_logits_graph_and_model_key(tmp_path):
    state, key = initial(), jax.random.key(17)
    path = tmp_path / "checkpoint.npz"
    checkpoint(path, state, key)
    recovered = protocol.restore(path, protocol.sha(path), key)
    for left, right in ((state.atoms, recovered.atoms), (state.bonds, recovered.bonds),
                        (state.graph().X, recovered.graph().X), (state.graph().E_upper, recovered.graph().E_upper)):
        assert np.asarray(right).dtype == np.float32
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("fault", ("hash", "X", "E_upper", "atom_logits", "bond_logits", "key_data", "dtype", "nonfinite"))
def test_restore_rejects_tampered_snapshot_even_when_file_hash_is_recomputed(tmp_path, fault):
    state, key = initial(), jax.random.key(17)
    path = tmp_path / "checkpoint.npz"
    checkpoint(path, state, key)
    original_digest = protocol.sha(path)
    if fault == "hash":
        path.write_bytes(path.read_bytes() + b"changed")
        digest = original_digest
    else:
        with np.load(path, allow_pickle=False) as z: arrays = {k: z[k].copy() for k in z.files}
        if fault == "dtype": arrays["atom_logits"] = arrays["atom_logits"].astype(np.float64)
        elif fault == "nonfinite": arrays["bond_logits"][0, 0] = np.nan
        elif fault == "key_data": arrays[fault][0] ^= np.uint32(1)
        else: arrays[fault][0, 0] += np.float32(.01)
        np.savez_compressed(path, **arrays)
        digest = protocol.sha(path)
    with pytest.raises((ValueError, AssertionError)):
        protocol.restore(path, digest, key)


def test_real_native_only_trajectory_split_equals_uninterrupted_fp32_state_and_raw_gradients(tmp_path, config):
    state, key, spec = initial(), jax.random.key(17), mosaic_spec(25)
    def loss(graph, *, key):
        del key
        atom = (graph.X * jnp.arange(6, dtype=jnp.float32)).sum().astype(jnp.float64)
        bond = (graph.E_upper * jnp.arange(5, dtype=jnp.float32)).sum().astype(jnp.float64)
        components = {"dbmol_affinity_loss": atom - bond,
                      "dbmol_contact_loss": .3 * atom + .2 * bond,
                      "dbmol_confidence_loss": -.1 * atom + .4 * bond}
        total = components["dbmol_affinity_loss"] + 2.5 * components["dbmol_contact_loss"] + 2.5 * components["dbmol_confidence_loss"]
        return total, {**components, "dbmol_total_loss": total}
    options = {"objective_mode": "native_only", "repair_weight": 0., "element_js_weight": 0.,
               "parity_atol": 2e-5, "native_name": "dbmol_total"}
    common = dict(prepare=lambda value, update: NativeOnlyPreparation(spec.fingerprint(), update),
                  spec=spec, reference=jnp.full(6, 1 / 6, jnp.float32), key=key,
                  config=options, **protocol.update_parameters(config))
    full = run_chemical_updates(loss, state, directory=tmp_path / "full", steps=4, **common)
    first = run_chemical_updates(loss, state, directory=tmp_path / "first", steps=2, **common)
    saved = tmp_path / "first" / first["states"][-1]["snapshot"]
    replay = protocol.restore(saved, protocol.sha(saved), key)
    last = run_chemical_updates(loss, replay, directory=tmp_path / "last", steps=2, update_offset=2, **common)
    for report in (full, first, last):
        assert report["status"] == "passed"
        protocol.verify_attempt_rates(report, config)
        assert all(len(update["attempts"]) == 1 for update in report["updates"])
    assert [row["global_step"] for row in last["states"]] == [2, 3, 4]
    fpath = tmp_path / "full" / full["states"][-1]["snapshot"]
    lpath = tmp_path / "last" / last["states"][-1]["snapshot"]
    fields = ("X", "E_upper", "atom_logits", "bond_logits", "key_data",
              "dbmol_total_gradient_X", "dbmol_total_gradient_E_upper",
              "gradient_atom_logits", "gradient_bond_logits", "dbmol_total_primal")
    with np.load(fpath) as f, np.load(lpath) as l:
        for field in fields: np.testing.assert_array_equal(f[field], l[field])
        for field in fields[:-1]:
            assert f[field].dtype == (np.uint32 if field == "key_data" else np.float32)
        assert f["dbmol_total_primal"].dtype == np.float64


def _ops():
    try:
        import dbmol_ops
        return dbmol_ops
    except ModuleNotFoundError:
        path = Path(__file__).resolve().parents[1] / "runs" / "20261023-boltz-dbmol-pilot-01" / "dbmol_ops.py"
        spec = importlib.util.spec_from_file_location("dbmol_ops", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


@pytest.fixture
def ops(monkeypatch, tmp_path):
    module = _ops()
    base = tmp_path / "base"
    base.mkdir()
    for name, value in (("BASE", base), ("ROOT", base / "pilot"),
                        ("SOURCE", base / "release" / "source"),
                        ("EXTENDED", base / "extended"), ("BASELINE", base / "baseline"),
                        ("OLD", base / "old")):
        monkeypatch.setattr(module, name, value)
    module.ROOT.mkdir()
    return module


@pytest.mark.parametrize("phase,queue,slots,memory,wall", (
    ("cpu", "hpc", "4", "rusage[mem=4GB]", "01:00"),
    ("native", "gpul40s", "4", "rusage[mem=8GB]", "02:00"),
    ("segment", "gpul40s", "4", "rusage[mem=8GB]", "23:00"),
    ("after-cpu", "hpc", "2", "rusage[mem=4GB]", "00:30"),
    ("after-native", "hpc", "2", "rusage[mem=4GB]", "00:30"),
    ("after-segment", "hpc", "2", "rusage[mem=4GB]", "00:30"),
    ("dispatch", "hpc", "2", "rusage[mem=4GB]", "00:30")))
def test_controller_exact_declared_resource_vectors(ops, phase, queue, slots, memory, wall):
    vector = ops.resources(phase)
    assert vector[:4] == ["-q", queue, "-n", slots]
    assert memory in vector and vector[-2:] == ["-W", wall]
    if queue == "gpul40s":
        assert vector[vector.index("-gpu") + 1] == "num=1:mode=exclusive_process"
    else:
        assert "-gpu" not in vector


@pytest.mark.parametrize("phase,index", (("native", 0), ("after-native", 1), ("segment", 2), ("after-segment", 3)))
def test_declared_phase_labels_have_exact_command_source_manifest_and_dependency(ops, phase, index):
    name = ops.label(phase, index)
    assert ops.decode(name) == (phase, index)
    args = ops.command(name, phase, "m" * 64, "ended(123)")
    assert args[0] == "bsub"
    assert args[args.index("-w") + 1] == "ended(123)"
    assert args[-5:] == [str(ops.SOURCE), str(ops.ROOT), "m" * 64, phase, str(ops.ROOT / name)]
    assert args[args.index("-J") + 1] == "mosaic-dbmol-" + name


@pytest.mark.parametrize("phase,index", (("protenix", 0), ("run", 0), ("native", -1), ("native", 4), ("native", True)))
def test_controller_cannot_dispatch_undeclared_arm_or_index(ops, phase, index):
    with pytest.raises(ValueError): ops.label(phase, index)


@pytest.mark.parametrize("name,phase", (("wrong", "cpu"), ("cpu", "after-cpu"),
                                       ("dispatch", "dispatch"), ("pilot-native-0004", "native"),
                                       ("pilot-segment-0000", "native")))
def test_command_refuses_unknown_parent_or_phase_mismatch(ops, name, phase):
    with pytest.raises(ValueError): ops.command(name, phase, "a" * 64)


def test_unknown_submission_persists_exclusive_marker_and_never_blindly_duplicates(ops, monkeypatch):
    calls = []
    def failed(args, **kwargs):
        calls.append(args)
        raise ops.subprocess.TimeoutExpired(args, 60)
    monkeypatch.setattr(ops.subprocess, "run", failed)
    with pytest.raises(ValueError, match="Unknown submission"):
        ops.submit("pilot-native-0000", "native", "a" * 64)
    parent = ops.ROOT / "pilot-native-0000"
    assert (parent / "submission-started.json").is_file()
    assert ops.read(parent / "submission.json")["outcome"] == "unknown"
    with pytest.raises(FileExistsError):
        ops.submit("pilot-native-0000", "native", "a" * 64)
    assert len(calls) == 1


def test_controller_own_execution_authenticates_actual_submission_and_resources(ops, monkeypatch):
    parent, manifest = ops.ROOT / "cpu", "a" * 64
    parent.mkdir()
    receipt = {"outcome": "submitted", "job_id": "123", "phase": "cpu",
               "args": ops.command("cpu", "cpu", manifest), "dependency": None}
    (parent / "submission.json").write_text(json.dumps(receipt))
    monkeypatch.setenv("LSB_JOBID", "123")
    ops.executing(parent, "cpu", manifest)
    receipt["args"][receipt["args"].index("-n") + 1] = "16"
    (parent / "submission.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError): ops.executing(parent, "cpu", manifest)


def test_pending_peer_validation_allowance_is_explicitly_not_full_long_capacity(ops):
    peers, missing = ops.external_budgets(False)
    assert set(missing) == {"extended", "baseline"}
    assert sum(peer["remaining"] for peer in peers.values()) == 4 * 1024**3
    assert all("allowance only" in peer["scope"] for peer in peers.values())
    with pytest.raises(ValueError): ops.external_budgets(True)


@pytest.mark.parametrize("peer,name", (("EXTENDED", "random-segment-0000"),
                                      ("BASELINE", "boltz2-segment-0000")))
def test_existing_or_unknown_peer_optimization_cannot_use_native_allowance(ops, peer, name):
    root = getattr(ops, peer)
    (root / name).mkdir(parents=True)
    with pytest.raises(ValueError, match="lacks full projection"):
        ops.external_budgets(False)


def peer_projections(ops, monkeypatch):
    original_sha = ops.sha
    monkeypatch.setattr(ops, "sha", lambda p: "m" * 64 if Path(p).name == "source_manifest.json" else original_sha(p))
    monkeypatch.setattr(ops, "tree", lambda p: 0)
    for root, arms in ((ops.EXTENDED, ("random", "continuation")),
                       (ops.BASELINE, ("boltz2", "boltz2_chemistry"))):
        root.mkdir()
        (root / "cpu-certificate.json").write_text(json.dumps({"status": "passed", "source_manifest_sha256": "m" * 64, "proofs": []}))
        (root / "projection.json").write_text(json.dumps({arm: {"unit_bytes": 100} for arm in arms}))
        for arm in arms:
            label = arm + "-native-0000"
            pro = root == ops.EXTENDED
            source = ops.BASE / ("protenix-extended-release-01" if pro else "boltz-lr0025-release-02") / "source"
            config = source / "examples" / "configs" / (("extended_" if pro else "boltz_serial_") + arm + ".json")
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text("{}\n")
            ready = root / label / "payload" / "study" / "assessment.json"
            ready.parent.mkdir(parents=True)
            ready.write_text(json.dumps({"status": "passed", "source_manifest_sha256": "m" * 64,
                                         "config_sha256": ops.sha(config),
                                         "arm": "protenix" if pro else arm,
                                         "storage_projection": {"maximum_pilot_update_bytes": 100}}))
            (root / (label + "-certificate.json")).write_text(json.dumps({
                "status": "passed", "source_manifest_sha256": "m" * 64,
                "proofs": [{"path": str(ready), "sha256": ops.sha(ready)}]}))


def test_peer_full_projection_is_reconstructed_from_every_authenticated_native_maximum(ops, monkeypatch):
    peer_projections(ops, monkeypatch)
    peers, missing = ops.external_budgets(True)
    assert missing == []
    # Two Pro chains have five/four 10k outputs and two Bol chains three/three.
    assert peers["extended"]["remaining"] == 2 * 100 * 9 * 10000 + 9 * 1024**3
    assert peers["baseline"]["remaining"] == 2 * 100 * 6 * 10000 + 6 * 1024**3
    assert all(peer["scope"] == "authenticated complete peer projection" for peer in peers.values())


@pytest.mark.parametrize("fault", ("unit", "proof", "source", "status", "missing_native", "config", "arm"))
def test_peer_full_projection_tamper_or_missing_actual_native_evidence_rejects(ops, monkeypatch, fault):
    peer_projections(ops, monkeypatch)
    root = ops.EXTENDED
    ready = root / "random-native-0000" / "payload" / "study" / "assessment.json"
    cert = root / "random-native-0000-certificate.json"
    if fault == "unit":
        value = ops.read(root / "projection.json"); value["random"]["unit_bytes"] = 101
        (root / "projection.json").write_text(json.dumps(value))
    elif fault == "proof": ready.write_text(ready.read_text() + " ")
    elif fault == "source":
        value = ops.read(cert); value["source_manifest_sha256"] = "x" * 64
        cert.write_text(json.dumps(value))
    elif fault == "status":
        value = ops.read(cert); value["status"] = "failed"
        cert.write_text(json.dumps(value))
    elif fault == "missing_native": cert.unlink()
    else:
        value = ops.read(ready)
        value["config_sha256" if fault == "config" else "arm"] = "changed"
        ready.write_text(json.dumps(value))
        # Keep the byte proof genuine so the dedicated identity check, rather
        # than only an artifact-hash mismatch, rejects this malformed producer.
        certificate = ops.read(cert)
        certificate["proofs"][0]["sha256"] = ops.sha(ready)
        cert.write_text(json.dumps(certificate))
    with pytest.raises(ValueError): ops.external_budgets(True)


@pytest.mark.parametrize("fault", ("unit", "full", "source", "proof"))
def test_own_full_projection_cannot_be_changed_independently_of_authenticated_native_pilot(ops, monkeypatch, fault):
    native = ops.ROOT / "pilot-native-0000-certificate.json"
    native.write_text(json.dumps({"status": "passed", "proofs": []}))
    raw = {"status": "passed", "source_manifest_sha256": "m" * 64, "proofs": [],
           "unit_bytes": 100, "full_projected_bytes": 2 * 100 * 10000 + 4 * 1024**3}
    if fault == "unit": raw["unit_bytes"] = 101
    elif fault == "full": raw["full_projected_bytes"] += 1
    elif fault == "source": raw["source_manifest_sha256"] = "x" * 64
    else: raw["proofs"] = [{"path": str(native), "sha256": "x" * 64}]
    (ops.ROOT / "projection.json").write_text(json.dumps(raw))
    monkeypatch.setattr(ops, "sha", lambda p: "m" * 64)
    monkeypatch.setattr(ops, "readiness", lambda i, m: (None, {"storage_projection": {"maximum_pilot_update_bytes": 100}}))
    with pytest.raises(ValueError): ops.own_budget()


def peer_callbacks(ops, monkeypatch):
    monkeypatch.setattr(ops, "sha", lambda p: "m" * 64)
    ids = []
    for root, arms in ((ops.EXTENDED, ("random", "continuation")),
                       (ops.BASELINE, ("boltz2", "boltz2_chemistry"))):
        root.mkdir()
        jobs = {}
        for index, arm in enumerate(arms):
            native_id, callback_id = str(100 + len(ids)), str(200 + len(ids))
            jobs[arm + "-native-0000"], jobs[arm + "-after-native-0000"] = native_id, callback_id
            parent = root / (arm + "-after-native-0000")
            parent.mkdir()
            name = parent.name
            pro = root == ops.EXTENDED
            source = ops.BASE / ("protenix-extended-release-01" if pro else "boltz-lr0025-release-02") / "source"
            dependency = "ended(" + native_id + ")"
            args = ["bsub", "-q", "hpc", "-n", "2", "-R", "span[hosts=1]", "-R", "rusage[mem=4GB]", "-W", "00:30",
                    "-J", ("mosaic-pro50k-" if pro else "mosaic-boltz30k-") + name,
                    "-oo", str(parent / "stdout.%J.log"), "-eo", str(parent / "stderr.%J.log"),
                    "-w", dependency, "bash", "-l", str(source / "scripts" / "hpc" / ("extended_stage.lsf" if pro else "boltz_serial_stage.lsf")),
                    str(source), str(root), "m" * 64, "after-native", str(parent)]
            (parent / "submission.json").write_text(json.dumps({"outcome": "submitted", "job_id": callback_id,
                "phase": "after-native", "dependency": dependency, "args": args}))
            ids.append(callback_id)
        (root / "jobs.json").write_text(json.dumps(jobs))
    return ids


def test_peer_dispatch_waits_only_actual_authenticated_own_ended_callback_ids(ops, monkeypatch):
    ids = peer_callbacks(ops, monkeypatch)
    calls = []
    monkeypatch.setattr(ops, "submit", lambda *args: calls.append(args))
    ops.queue_peer_dispatch("m" * 64)
    assert calls == [("dispatch-0001", "dispatch", "m" * 64,
                      " && ".join("done(" + job + ")" for job in ids))]
    parent = ops.ROOT / "dispatch-0001"
    parent.mkdir()
    dependency = " && ".join("done(" + job + ")" for job in ids)
    (parent / "submission.json").write_text(json.dumps({"outcome": "submitted", "job_id": "555",
        "args": ops.command("dispatch-0001", "dispatch", "m" * 64, dependency)}))
    ops.queue_peer_dispatch("m" * 64)
    assert len(calls) == 1


@pytest.mark.parametrize("fault", ("missing_submission", "unknown_outcome", "changed_args"))
def test_existing_peer_watch_marker_with_unknown_or_mismatched_outcome_blocks_duplicates(ops, monkeypatch, fault):
    peer_callbacks(ops, monkeypatch)
    parent = ops.ROOT / "dispatch-0001"
    parent.mkdir()
    if fault != "missing_submission":
        (parent / "submission.json").write_text(json.dumps({
            "outcome": "unknown" if fault == "unknown_outcome" else "submitted", "args": []}))
    monkeypatch.setattr(ops, "submit", lambda *args: pytest.fail("Cannot duplicate a watcher with unknown outcome"))
    with pytest.raises(ValueError): ops.queue_peer_dispatch("m" * 64)


@pytest.mark.parametrize("fault", ("resources", "source", "wrapper", "manifest"))
def test_peer_watch_authenticates_callback_full_source_and_resource_vector(ops, monkeypatch, fault):
    peer_callbacks(ops, monkeypatch)
    path = ops.EXTENDED / "random-after-native-0000" / "submission.json"
    receipt = ops.read(path)
    if fault == "resources": receipt["args"][receipt["args"].index("-n") + 1] = "16"
    elif fault == "source": receipt["args"][-5] = "/another/source"
    elif fault == "wrapper": receipt["args"][receipt["args"].index("-l") + 1] = "/another/wrapper.lsf"
    else: receipt["args"][-3] = "x" * 64
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(ops, "submit", lambda *args: pytest.fail("No mismatched peer watcher submission"))
    with pytest.raises(ValueError): ops.queue_peer_dispatch("m" * 64)


def test_peer_dispatch_refuses_invented_or_mismatched_callback_dependency(ops, monkeypatch):
    root = ops.EXTENDED
    parent = root / "random-after-native-0000"
    parent.mkdir(parents=True)
    (root / "jobs.json").write_text(json.dumps({"random-native-0000": "123", "random-after-native-0000": "124"}))
    (parent / "submission.json").write_text(json.dumps({"outcome": "submitted", "job_id": "124",
                                                        "phase": "after-native", "dependency": "ended(999)"}))
    monkeypatch.setattr(ops, "submit", lambda *args: pytest.fail("No submission is allowed"))
    with pytest.raises(ValueError): ops.queue_peer_dispatch("m" * 64)


def native_report_fixture():
    component = {
        "status": "passed", "epsilons": [.01, .003, .001],
        "relative_tolerance": .15, "required_adjacent_passes": 2,
        "independent_primal": True, "raw_unclipped_gradients": True,
        "typed_bond_contrast_nonzero": True, "production_jax_transport": "passed",
        "rows": [{"field": field, "probes": [{"epsilon": epsilon,
                   "analytic": 1., "finite_difference": 1., "agrees": True}
                  for epsilon in (.01, .003, .001)]} for field in ("X", "E_upper")],
    }
    return {"status": "passed", "policy": "dbmol-native-raw-fd-v1",
            "components": {name: copy.deepcopy(component) for name in protocol.COMPONENTS},
            "all_components_and_total_raw_fd_passed": True,
            "hard_native_parity": {"status": "passed"}, "frozen_unchanged": True,
            "key_and_logit_row_shift_nulls": "passed",
            "live_primitive_recomposition": "passed_exact_scalar_and_raw_gradients"}


def test_independent_reader_checks_each_original_raw_gradient_agreement():
    report = native_report_fixture()
    assert protocol.verify_native_report(report) == report


@pytest.mark.parametrize("fault", ("policy", "component", "epsilon", "tolerance", "adjacency", "fabricated_agreement", "nonfinite", "typed_bond", "transport", "frozen"))
def test_reader_never_promotes_ad_only_or_relaxed_failed_fd_to_a_native_pass(fault):
    report = native_report_fixture()
    component = report["components"]["dbmol_total_loss"]
    if fault == "policy": report["policy"] = "pointwise_native_ad_v4"
    elif fault == "component": del report["components"]["dbmol_contact_loss"]
    elif fault == "epsilon": component["rows"][0]["probes"][0]["epsilon"] = .0001
    elif fault == "tolerance": component["relative_tolerance"] = .5
    elif fault == "adjacency":
        component["rows"][0]["probes"][1].update(finite_difference=-1., agrees=False)
    elif fault == "fabricated_agreement": component["rows"][0]["probes"][1]["finite_difference"] = -1.
    elif fault == "nonfinite": component["rows"][0]["probes"][0]["analytic"] = float("nan")
    elif fault == "typed_bond": component["typed_bond_contrast_nonzero"] = False
    elif fault == "transport": component["production_jax_transport"] = "not_run"
    else: report["frozen_unchanged"] = False
    with pytest.raises(ValueError): protocol.verify_native_report(report)


def test_checks_import_retains_real_combined_evaluator_across_temporary_trajectory_fixture_patch(monkeypatch):
    # Import checks at module collection, before any temporary trajectory stub.
    # A delayed first import from inside that stub can permanently capture its
    # Protenix-only closure in checks.evaluate_combined even after teardown.
    original = chemistry_trajectory.evaluate_combined
    assert chemistry_checks.evaluate_combined is original
    def temporary_protenix_only_fixture(*args, **kwargs):
        raise AssertionError("Temporary fixture must never become a production evaluator")
    with monkeypatch.context() as patch:
        patch.setattr(chemistry_trajectory, "evaluate_combined", temporary_protenix_only_fixture)
        assert chemistry_trajectory.evaluate_combined is temporary_protenix_only_fixture
        assert chemistry_checks.evaluate_combined is original
    assert chemistry_trajectory.evaluate_combined is original
    assert chemistry_checks.evaluate_combined is chemistry_trajectory.evaluate_combined
