"""Paired fresh refinement from saved logits and their verified valid repair.

The original screen run_loop, chemistry preparation, acceleration adapters and
optimizer arithmetic are imported unchanged from a sealed parent source.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import score_transfer_audit as audit


VERSION = "protenix-projection-paired-pilot-v1"
FORWARD_MODES = ("relaxed", "hard_forward_soft_backward")


def arm_protocol(plan):
    comparison = plan.get("comparison", "projection_start")
    if comparison == "projection_start":
        return (("original_logits", "original", "relaxed"),
                ("saved_repair", "repair", "relaxed"))
    if comparison == "binding_forward":
        return (("relaxed_forward", "repair", "relaxed"),
                ("hard_forward", "repair", "hard_forward_soft_backward"))
    if comparison == "projection_feedback":
        return (("archive_only", "repair", "relaxed"),
                ("project_feedback", "repair", "relaxed"))
    raise ValueError("Unknown causal comparison")


def binding_graph(graph, mode):
    """Hard primal with a deliberately biased identity probability pullback.

    This is not the derivative of argmax. The inherited softmax/logit pullback
    follows the identity estimator. Chemistry still uses relaxed probabilities.
    """
    audit.require(mode in FORWARD_MODES, "Unknown binding forward mode")
    if mode == "relaxed":
        return graph
    import jax
    import jax.numpy as jnp
    from mosaic.small_molecule.state import RelaxedGraph
    def leaf(p):
        hard = jax.nn.one_hot(jnp.argmax(p, axis=-1), p.shape[-1], dtype=p.dtype)
        return hard + (p - jax.lax.stop_gradient(p))
    return RelaxedGraph(leaf(graph.X), leaf(graph.E_upper))


def forward_controls():
    """Allocated-only analytic controls; no model evaluations or updates."""
    audit.require(os.environ.get("LSB_JOBID", "").isdigit(), "Forward controls require allocation")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.state import RelaxedGraph
    x = jnp.asarray([[.2, .8], [.7, .3]], dtype=jnp.float32)
    edge = jnp.asarray([[.1, .9]], dtype=jnp.float32)
    graph = RelaxedGraph(x, edge)
    hard = binding_graph(graph, "hard_forward_soft_backward")
    audit.require(np.array_equal(np.asarray(hard.X), [[0., 1.], [1., 0.]]) and
                  np.array_equal(np.asarray(hard.E_upper), [[0., 1.]]), "Hard forward differs")
    value, gradient = jax.value_and_grad(lambda g: jnp.sum(binding_graph(g, "hard_forward_soft_backward").X**2)
                                       + jnp.sum(binding_graph(g, "hard_forward_soft_backward").E_upper**2))(graph)
    audit.require(float(value) == 3. and np.array_equal(np.asarray(gradient.X), 2*np.asarray(hard.X))
                  and np.array_equal(np.asarray(gradient.E_upper), 2*np.asarray(hard.E_upper)),
                  "Declared identity surrogate is detached or scaled")
    _, grad = jax.value_and_grad(lambda g: jnp.sum(binding_graph(g, "relaxed").X**2))(graph)
    audit.require(np.allclose(np.asarray(grad.X), 2*np.asarray(x)) and
                  binding_graph(graph, "relaxed") is graph, "Relaxed control changed")
    return {"status": "passed_declared_forward_and_surrogate_controls", "checks": 3,
            "discrete_argmax_derivative_certified": False, "optimizer_updates": 0}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def validate_plan(plan):
    audit.require(plan["version"] == VERSION, "Unknown paired pilot plan")
    audit.require(plan["parent_step"] == 30000, "This contrast uses the declared saved30k parent")
    audit.require(type(plan["updates"]) is int and 1 <= plan["updates"] <= 20000,
                  "Require a bounded informative paired budget")
    for field, expected in (("temperature", .4), ("chemistry_coefficient", 30.),
                            ("learning_rate", .3), ("smoothing", .02)):
        audit.require(type(plan[field]) in (int, float) and plan[field] == expected,
                      "Changed declared paired protocol: " + field)
    internal_ceiling = 6600
    if "confirmation_pilot" in plan:
        audit.require(plan.get("comparison") == "projection_feedback" and plan["updates"] == 20000 and
                      plan.get("wall_minutes") == 480, "Invalid matched confirmation allocation")
        internal_ceiling = 28200
    audit.require(type(plan.get("internal_seconds", 3300)) is int and
                  1 <= plan.get("internal_seconds", 3300) <= internal_ceiling, "Invalid complete run budget")
    audit.require(type(plan.get("output_cap_bytes", 536870912)) is int and
                  0 < plan.get("output_cap_bytes", 536870912) <= 2147483648, "Invalid own output cap")
    audit.require(type(plan.get("file_cap", 10000)) is int and
                  0 < plan.get("file_cap", 10000) <= 30000, "Invalid own file cap")
    arm_protocol(plan)
    if plan.get("comparison") == "projection_feedback":
        audit.require(plan.get("block_updates") == 500 and plan["updates"] % 500 == 0,
                      "Require matched500-update blocks")
    return plan


def fixed_configuration(parent_config, plan):
    """The two arms share this entire policy; only their initial logits differ."""
    validate_plan(plan)
    config = deepcopy(parent_config)
    config.update(updates=plan["updates"], temperature=plan["temperature"],
                  chemistry={"mode": "constant", "coefficient": plan["chemistry_coefficient"]},
                  binding={"mode": "always"})
    config.pop("temperature_schedule", None)
    config.pop("logit_noise", None)
    audit.require(config.get("neighbor_environment") is None and config.get("severity_progress") is None,
                  "This comparison inherits structural repair without other guidance changes")
    audit.require(config.get("structural_repair") is not None, "Original structural builder required")
    return config


def smoothed_row(label, categories, *, smoothing=.02):
    audit.require(type(label) is int and type(categories) is int and categories >= 2 and
                  0 <= label < categories, "Invalid categorical target")
    audit.require(type(smoothing) in (int, float) and math.isfinite(smoothing) and
                  0 < smoothing < 1, "Finite positive support mixture required")
    return [(1. - smoothing if i == label else 0.) + smoothing / categories
            for i in range(categories)]


def inputs(plan, row):
    records = [plan["parent_plan"], plan["parent_snapshot"], plan["parent_reference"], plan["storage_module"],
               row["config"], row["context_ready"], row["pocket"], *plan.get("source_files", [])]
    if "confirmation_pilot" in plan:
        records.append(plan["confirmation_pilot"])
    return records


def confirmation_budget(plan, config):
    """Authenticate the matched pilot and budget the full paired allocation."""
    reference = plan["confirmation_pilot"]
    pilot = audit.read(audit.bound(reference))
    audit.require(pilot["job_id"] == reference["job_id"] and
                  pilot["status"] == "completed_paired_exploratory_refinement" and
                  pilot["comparison"] == "projection_feedback" and pilot["frozen_unchanged"] and
                  pilot["inputs_unchanged"] and pilot["frozen_before"] == plan["frozen_identity"] and
                  pilot["configuration"] == dict(config, updates=2000) and
                  plan["parent_snapshot"] in pilot["inputs_before"], "Confirmation pilot differs")
    audit.require([arm["arm"] for arm in pilot["arms"]] == ["archive_only", "project_feedback"] and
                  all(arm["applied_updates"] == 2000 and arm["hard_diagnostic_calls"] == 517
                      for arm in pilot["arms"]), "Incomplete matched pilot budget")
    elapsed = pilot["elapsed_seconds"]
    audit.require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0,
                  "Invalid measured pilot runtime")
    required = math.ceil(elapsed * (plan["updates"] / 2000) * 1.25 + 600)
    audit.require(required <= plan["internal_seconds"], "Measured confirmation does not fit allocation")
    return {"pilot_job_id": reference["job_id"], "pilot_pair_seconds": elapsed,
            "minimum_internal_seconds": required, "internal_seconds": plan["internal_seconds"]}


def parent_metadata(plan):
    validate_plan(plan)
    parent = audit.read(audit.bound(plan["parent_plan"]))
    audit.require(plan["source_root"] == parent["code_source_root"], "Use the exact sealed parent code source")
    row = parent["arms"][plan["parent_arm"]]
    audit.require(row["rate"] == .3 and row["model_key"] == 17 and row.get("composition_weight", 0) == 0
                  and not row.get("margin", False), "Parent scientific settings differ")
    audit.require(row["screen"]["init_seed"] == 17, "This comparison uses saved seed17")
    config = fixed_configuration(row["screen"], plan)
    for record in inputs(plan, row):
        audit.bound(record)
    if "confirmation_pilot" in plan:
        confirmation_budget(plan, config)
    return parent, row, config


def should_project(current_valid, current_score, current_identity, best_score, best_identity):
    """Feedback is a declared candidate-selection operation, not a descent gate."""
    audit.require(type(current_valid) is bool and math.isfinite(current_score) and math.isfinite(best_score),
                  "Invalid hard-candidate selection inputs")
    return best_identity != current_identity and (not current_valid or best_score > current_score)


def feedback_blocks(screen, loss, initial, prepare, spec, loss_config, key, config,
                    output, plan, tick, score_hard, *, feedback):
    """Equal update/proposal budgets; only feedback into the next block differs."""
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.state import graph_from_hard
    started = time.monotonic()
    logits = initial
    first = score_hard(graph_from_hard(np.asarray(initial.graph().X).argmax(-1),
                                       np.asarray(initial.graph().E_upper).argmax(-1)))
    audit.require(first["strict_valid_connected"], "Both feedback arms must start valid")
    best, valid_sources, resets, diagnostic_calls = first, 0, [], 1
    unique, blocks = {first["readout"]["canonical_smiles"]}, []
    signature = lambda p: (tuple(p["readout"]["element_indices"]), tuple(p["readout"]["bond_upper_indices"]))
    for offset in range(0, plan["updates"], plan["block_updates"]):
        tick()
        block = output / ("block-%04d" % (offset // plan["block_updates"]))
        block.mkdir(exist_ok=False)
        summary = screen.run_loop(loss, logits, prepare, spec, loss_config, key, config,
                                  block / "study", steps=plan["block_updates"], tick=tick)
        audit.require(summary["status"] == "completed_exploratory_updates" and
                      summary["applied_updates"] == plan["block_updates"], "Incomplete feedback block")
        # Source-state denominator excludes each block's extra evaluated endpoint.
        valid_sources += summary["strict_valid_states"] - int(summary["last_state"]["strict_valid_connected"])
        unique.update(summary["valid_unique_smiles"])
        with np.load(block / "study/latest-state.npz", allow_pickle=False) as point:
            logits = type(initial)(jnp.asarray(point["atom_logits"]), jnp.asarray(point["bond_logits"]), plan["temperature"])
        current = score_hard(graph_from_hard(np.asarray(logits.graph().X).argmax(-1),
                                             np.asarray(logits.graph().E_upper).argmax(-1)))
        diagnostic_calls += 1
        if current["strict_valid_connected"] and current["hard_contact_score"] > best["hard_contact_score"]:
            best = current
        sample_dir = block / "candidate-samples"
        sample_dir.mkdir()
        global_step = offset + plan["block_updates"]
        sampling = screen.sample_validity(logits, config["init_seed"], global_step, sample_dir)
        with np.load(sampling["raw_samples"]["path"], allow_pickle=False) as draws:
            atoms, bonds = draws["atom_indices"], draws["bond_indices"]
            audit.require(atoms.shape == (128, 25) and bonds.shape == (128, 300), "Candidate batch differs")
            atoms, bonds = atoms.copy(), bonds.copy()
        scores, valid_draws = [], 0
        # Score all128 draws, including invalid/duplicate ones, in BOTH arms.
        # This fixes the native-call budget independently of validity outcomes.
        for index, (a, b) in enumerate(zip(atoms, bonds)):
            tick()
            candidate = score_hard(graph_from_hard(a, b))
            diagnostic_calls += 1
            if candidate["strict_valid_connected"]:
                valid_draws += 1
                unique.add(candidate["readout"]["canonical_smiles"])
                if candidate["hard_contact_score"] > best["hard_contact_score"]:
                    best = candidate
            scores.append({"draw": index, "hard_contact_score": candidate["hard_contact_score"],
                           "valid": candidate["strict_valid_connected"],
                           "smiles": candidate["readout"].get("canonical_smiles")})
        audit.require(valid_draws == sampling["strict_valid_connected"], "Proposal validity count differs")
        audit.write(block / "candidate-scores.json", {"sampling": sampling, "scores": scores,
                    "scope": "All128 raw draws scored; invalid graphs excluded from candidate archive"})
        selected = feedback and global_step < plan["updates"] and should_project(
            current["strict_valid_connected"], current["hard_contact_score"], signature(current),
            best["hard_contact_score"], signature(best))
        if selected:
            labels = best["readout"]
            arrays = [np.asarray([smoothed_row(int(v), k, smoothing=plan["smoothing"]) for v in values], dtype=np.float32)
                      for values, k in ((labels["element_indices"], 6), (labels["bond_upper_indices"], 5))]
            logits = type(initial)(*(plan["temperature"] * jnp.log(jnp.asarray(a)) for a in arrays), plan["temperature"])
            from mosaic.small_molecule.io import hard_readout
            projected = hard_readout(logits.graph())
            audit.require(screen.valid_connected(projected) and
                          tuple(projected["element_indices"]) == signature(best)[0] and
                          tuple(projected["bond_upper_indices"]) == signature(best)[1],
                          "Projection changed the selected valid hard graph")
            resets.append({"after_update": global_step, "before": current, "selected": best,
                           "scope": "Explicit best-valid candidate projection plus2percent uniform; outside optimizer trial"})
        row = {"global_start": offset, "global_end": global_step,
               "report": audit.binding(block / "study/report.json"), "sampling": sampling,
               "candidate_scores": audit.binding(block / "candidate-scores.json"),
               "current_hard": current, "best_valid_archive": best, "feedback_applied": selected}
        blocks.append(row)
        audit.write(output / "progress.json", {"completed_updates": global_step,
                    "valid_update_sources": valid_sources, "best_valid_archive": best,
                    "projection_resets": resets, "blocks": blocks})
    audit.require(diagnostic_calls == 1 + (plan["updates"] // 500) * 129, "Hard-forward budgets differ")
    return {"status": "completed_block_refinement", "applied_updates": plan["updates"],
            "valid_update_sources": valid_sources, "update_source_denominator": plan["updates"],
            "valid_unique_smiles": sorted(unique), "hard_diagnostic_calls": diagnostic_calls,
            "best_valid_hard": best, "final_raw_hard": current, "blocks": blocks,
            "projection_resets": resets, "elapsed_seconds": time.monotonic() - started,
            "selection_scope": "Both arms deliver best valid archive; feedback only changes next-block logits",
            "scope": "Fixed-policy500-step blocks; local block steps plus explicit global offsets; all scalar records retained"}


def run(plan, plan_path):
    parent, row, config = parent_metadata(plan)
    audit.require(os.environ.get("LSB_JOBID", "").isdigit(), "Paired native refinement requires allocated GPU")
    if plan.get("queue") is not None:
        audit.require(os.environ.get("LSB_QUEUE") == plan["queue"], "Declared queue differs")
    audit.require(os.environ.get("XLA_FLAGS") == audit.PROFILE, "Original execution profile required")
    routing_root = plan.get("routing_source", parent["routing_source"])
    routing = module(Path(routing_root) / "scripts/cuda_ownership.py", "_paired_pilot_cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = routing.resolve_visibility(os.environ)["selected_cuda_visible_devices"]
    source = Path(plan["source_root"])
    original = Path(row["science_source"]["path"])
    adapter = Path(row.get("gpu_adapter_source", row["adapter_source"]["path"]))
    sys.path[:0] = [str(source), str(source / "src"), str(original / "scripts"), str(adapter / "scripts")]
    os.environ["PYTHONPATH"] = os.pathsep.join([str(source), str(source / "src"), str(original / "scripts"),
                                              str(adapter / "scripts"), os.environ.get("PYTHONPATH", "")])
    screen = module(source / "protenix_chemistry_screen.py", "_paired_pilot_original_screen")
    screen.validate_screen(config)
    storage = module(audit.bound(plan["storage_module"]), "_paired_pilot_storage")
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.protenix import FrozenProtenixTiny
    from mosaic.small_molecule.losses import NativeContactLoss
    from mosaic.small_molecule.io import load_context, hard_readout
    from mosaic.small_molecule.pocket import map_protenix_pocket
    from mosaic.small_molecule.gradient_checks import array_digest
    from mosaic.small_molecule.state import graph_from_hard, validate_graph
    from mosaic.small_molecule.chemistry.state import mosaic_spec
    from mosaic.small_molecule.chemistry.loss import FailureGuidanceConfig
    from mosaic.small_molecule.chemistry.preparation import prepare_failure_guidance
    from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig
    from mosaic.small_molecule.chemistry._reference.repair_progress import ProgressSearchConfig
    from mosaic.small_molecule.chemistry._reference.quality import graph_identity

    audit.require(not jax.config.x64_enabled and jax.default_backend() == "gpu" and len(jax.devices("gpu")) == 1,
                  "Require original FP32 and one allocated GPU")
    jax.config.update("jax_default_matmul_precision", "highest")
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    result = {"version": VERSION, "status": "running", "job_id": os.environ["LSB_JOBID"],
              "plan": audit.binding(plan_path), "parent_step": plan["parent_step"],
              "source_parent_preserved": True, "scope": "Fresh paired fixed-policy refinement; local steps0..budget, not an overwritten continuation",
              "configuration": config, "learning_rate": .3, "model_key": 17, "smoothing": plan["smoothing"],
              "projected_initialization": "T * log((1-smoothing)*one_hot(saved_valid_repair)+smoothing/K)",
              "runtime_profile": audit.PROFILE, "arms": [], "derivative_qualified": False,
              "warnings": ["Native geometry-off contact is a surrogate, not affinity or binding certification",
                           "Projection plus finite smoothing is the declared initial-state intervention",
                           "Second arm may reuse warmed native compilation; runtime comparison includes this ordering limitation"]}
    result["comparison"] = plan.get("comparison", "projection_start")
    if result["comparison"] == "binding_forward":
        result["warnings"].append("Hard forward uses a biased identity probability gradient, not the derivative of argmax")
    before_inputs = [audit.binding(record["path"]) for record in inputs(plan, row)]
    result["inputs_before"] = before_inputs
    begun, next_capacity = time.monotonic(), 0.
    frozen = None
    model = context = None

    def persist():
        audit.write(output / "result.json", result)

    def tick(state=None):
        nonlocal next_capacity
        audit.require(time.monotonic() - begun < plan.get("internal_seconds", 3300), "Complete paired pilot time budget exceeded")
        if time.monotonic() >= next_capacity:
            assessed = storage.assess_output(output, plan.get("output_cap_bytes", 536870912),
                                             plan.get("file_cap", 10000))
            audit.write(output / "capacity-latest.json", assessed)
            audit.require(assessed["status"] == "passed", "Own remaining output capacity unavailable")
            audit.require(assessed["own_output"]["actual_bytes"] <= plan.get("output_cap_bytes", 536870912) and
                          assessed["own_output"]["actual_files"] <= plan.get("file_cap", 10000), "Own output cap exceeded")
            next_capacity = time.monotonic() + 300

    try:
        tick()
        if result["comparison"] == "binding_forward":
            result["allocated_forward_controls"] = forward_controls()
        ready = audit.read(row["context_ready"]["path"])
        context = load_context(Path(row["context_ready"]["path"]).parent / "context", ready["context_files"])
        model = FrozenProtenixTiny.load(cache_dir=Path(plan["model_cache"]))
        frozen = {"model_digest": array_digest(model), "context_digest": array_digest(context)}
        audit.require(frozen == row["frozen_identity"], "Original frozen model/context differs")
        result["frozen_before"] = frozen
        loss = model.build_loss(context=context, loss=NativeContactLoss(8., map_protenix_pocket(context, audit.read(row["pocket"]["path"]))))
        key = jax.random.key(17)
        spec = mosaic_spec(25)
        with np.load(plan["parent_snapshot"]["path"], allow_pickle=False) as saved:
            audit.require(int(saved["step"]) == plan["parent_step"] and
                          np.array_equal(saved["key_data"], np.asarray(jax.random.key_data(key))), "Saved parent step/key differs")
            atoms, bonds = (np.asarray(saved[k]).copy() for k in ("atom_logits", "bond_logits"))
            audit.require(atoms.shape == (25, 6) and bonds.shape == (300, 5) and
                          atoms.dtype == bonds.dtype == np.float32 and np.isfinite(atoms).all() and np.isfinite(bonds).all(),
                          "Saved parent shapes/dtypes/finite logits differ")
            raw_atoms = np.asarray(saved["X"]).argmax(-1)
            raw_bonds = np.asarray(saved["E_upper"]).argmax(-1)
        with gzip.open(plan["parent_reference"]["path"], "rt", encoding="utf-8") as stream:
            teacher = audit.teacher_labels(json.load(stream), step=plan["parent_step"])
        audit.require(teacher["candidate"] is not None and teacher["route"] in (1, 2), "A saved complete repair is required")
        audit.require(teacher["receipt"]["schema_sha256"] == spec.fingerprint(), "Saved teacher schema differs")
        matrix = np.zeros((25, 25), dtype=np.int64)
        i, j = np.triu_indices(25, 1)
        matrix[i, j] = raw_bonds
        matrix[j, i] = raw_bonds
        audit.require(graph_identity(raw_atoms + 1, np.zeros(25, dtype=np.int64), matrix, spec) ==
                      teacher["receipt"]["rows"][0]["raw_sha256"], "Teacher belongs to another parent graph")
        kind = screen.logits_class(scheduled=False)
        original_initial = kind(jnp.asarray(atoms), jnp.asarray(bonds), plan["temperature"])
        target = teacher["candidate"]
        projected_probabilities = [np.asarray([smoothed_row(int(label), categories, smoothing=plan["smoothing"])
                                              for label in labels], dtype=np.float32)
                                   for labels, categories in ((target["element_indices"], 6), (target["bond_upper_indices"], 5))]
        projected_initial = kind(*(plan["temperature"] * jnp.log(jnp.asarray(p)) for p in projected_probabilities),
                                 plan["temperature"])
        audit.require(screen.valid_connected(hard_readout(projected_initial.graph())), "Projected finite-support initialization lost validity")
        source_config = audit.read(row["config"]["path"])
        search = ProbabilityRepairConfig(**source_config["search"])
        progress = ProgressSearchConfig(**source_config["progress_search"])

        def prepare(logits, state):
            return prepare_failure_guidance(logits.decoded(), spec, update=state, search=search,
                                            progress_search=progress, structural_repair=config["structural_repair"])

        @eqx.filter_jit
        def hard_forward(graph):
            scalar, aux = loss(graph, key=key)
            finite = jnp.isfinite(scalar)
            for leaf in jax.tree_util.tree_leaves(aux):
                if eqx.is_array(leaf):
                    finite &= jnp.all(jnp.isfinite(leaf))
            return scalar, finite

        def score_hard(graph):
            tick()
            validate_graph(graph)
            scalar, finite = hard_forward(graph)
            value = float(jax.block_until_ready(scalar))
            audit.require(bool(finite) and math.isfinite(value), "Nonfinite native hard-graph evaluation")
            readout = hard_readout(graph)
            readout.pop("pairs", None)
            return {"hard_contact_score": -value, "strict_valid_connected": screen.valid_connected(readout),
                    "readout": readout}

        def hard_archive(study, forward_mode):
            selected = [(step, study / "snapshots" / f"step-{step:05d}.npz")
                        for step in range(0, plan["updates"] + 1, 100)]
            if plan["updates"] % 100:
                selected.append((plan["updates"], study / "snapshots" / f"step-{plan['updates']:05d}.npz"))
            if plan.get("comparison", "projection_start") != "binding_forward" and (study / "best-valid-state.npz").is_file():
                with np.load(study / "best-valid-state.npz", allow_pickle=False) as point:
                    selected.append((int(point["step"]), study / "best-valid-state.npz"))
            records, seen = [], set()
            for local_step, path in selected:
                tick()
                with np.load(path, allow_pickle=False) as point:
                    hard = graph_from_hard(np.asarray(point["X"]).argmax(-1), np.asarray(point["E_upper"]).argmax(-1))
                    saved_soft = float(point["contact_primal"])
                readout = hard_readout(hard)
                identity = (local_step, tuple(readout["element_indices"]), tuple(readout["bond_upper_indices"]))
                if identity in seen:
                    continue
                seen.add(identity)
                validate_graph(hard)
                scalar, finite = hard_forward(hard)
                value = float(jax.block_until_ready(scalar))
                audit.require(bool(finite) and math.isfinite(value), "Nonfinite native hard-graph evaluation")
                readout.pop("pairs", None)
                records.append({"local_step": local_step, "source_parent_step": plan["parent_step"],
                                "hard_contact_score": -value, "hard_contact_loss": value,
                                "saved_binding_contact_score": -saved_soft,
                                "binding_forward_mode": forward_mode,
                                "strict_valid_connected": screen.valid_connected(readout),
                                "readout": readout, "snapshot": audit.binding(path)})
            valid = [p for p in records if p["strict_valid_connected"]]
            return {"scope": ("Hard graph at identical100-step grid; equal21 forwards per2k arm; not every-state hard optimum"
                              if plan.get("comparison") == "binding_forward" else
                              "Hard graph at retained100-step grid plus best-soft-valid snapshot; not every-state hard optimum"),
                    "records": records, "best_valid_hard": max(valid, key=lambda p: p["hard_contact_score"]) if valid else None}

        accelerator = module(adapter / "scripts/full_model_pilot.py", "_paired_pilot_acceleration")
        class ForwardLoss(eqx.Module):
            original: object
            mode: str = eqx.field(static=True)

            def __call__(self, graph, *, key):
                return self.original(binding_graph(graph, self.mode), key=key)

        for name, initial_kind, forward_mode in arm_protocol(plan):
            tick()
            initial = original_initial if initial_kind == "original" else projected_initial
            active_loss = loss if forward_mode == "relaxed" else ForwardLoss(loss, forward_mode)
            arm_output = output / name
            arm_output.mkdir(exist_ok=False)
            acceleration = arm_output / "acceleration"
            acceleration.mkdir()
            started = time.monotonic()
            with accelerator.combined_adapters("protenix-random50k", acceleration) as adapters:
                if plan.get("comparison") == "projection_feedback":
                    arm = feedback_blocks(screen, loss, initial, prepare, spec, FailureGuidanceConfig(**source_config["loss"]),
                                          key, config, arm_output, plan, tick, score_hard,
                                          feedback=(name == "project_feedback"))
                    arm.update(arm=name, acceleration=adapters)
                    result["arms"].append(arm)
                    persist()
                    continue
                summary = screen.run_loop(active_loss, initial, prepare, spec, FailureGuidanceConfig(**source_config["loss"]),
                                          key, config, arm_output / "study", steps=plan["updates"], tick=tick)
            audit.require(summary["status"] == "completed_exploratory_updates" and summary["applied_updates"] == plan["updates"],
                          "Original paired runner did not complete its declared updates")
            arm = {"arm": name, "status": summary["status"], "applied_updates": summary["applied_updates"],
                   "initialization": initial_kind, "binding_forward_mode": forward_mode,
                   "saved_probability_scope": "Relaxed probabilities for chemistry; binding mode explicitly recorded",
                   "source_parent_step": plan["parent_step"], "local_start": 0, "local_end": plan["updates"],
                   "elapsed_seconds": time.monotonic() - started, "strict_valid_states": summary["strict_valid_states"],
                   "valid_unique_smiles": summary["valid_unique_smiles"], "report": audit.binding(arm_output / "study/report.json"),
                   "acceleration": adapters, "best_binding_valid": summary["best_valid_contact"],
                   "hard_archive": hard_archive(arm_output / "study", forward_mode)}
            if forward_mode != "relaxed":
                audit.write(arm_output / "score-scope.json", {
                    "binding_forward_mode": forward_mode,
                    "contact_values": "Actual raw-argmax native contact; hard graphs can still be invalid",
                    "probability_gradient": "Biased identity estimator followed by original softmax pullback",
                    "chemistry": "Original relaxed loss with detached step-fixed targets",
                    "inherited_best_valid_scope_label_applicable": False,
                    "canonical_report_adapter_supported": False})
            result["arms"].append(arm)
            persist()
        result["status"] = "completed_paired_exploratory_refinement"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        if frozen is not None:
            after = {"model_digest": array_digest(model), "context_digest": array_digest(context)}
            result["frozen_after"] = after
            result["frozen_unchanged"] = after == frozen
            if after != frozen:
                result.update(status="failed", frozen_error="Frozen model/context changed")
        try:
            result["inputs_after"] = [audit.binding(audit.bound(r)) for r in inputs(plan, row)]
            result["inputs_unchanged"] = result["inputs_after"] == before_inputs
        except BaseException as error:
            result.update(status="failed", input_error=repr(error), inputs_unchanged=False)
        result["elapsed_seconds"] = time.monotonic() - begun
        persist()
    audit.require(result["status"] == "completed_paired_exploratory_refinement", "Paired preservation check failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--startup-only", action="store_true")
    args = parser.parse_args()
    plan = validate_plan(audit.read(args.plan))
    _, _, config = parent_metadata(plan)
    if args.startup_only:
        receipt = {"status": "metadata_startup_passed_no_models", "configuration": config,
                   "paired_arms": arm_protocol(plan), "parent_step": plan["parent_step"]}
        if "confirmation_pilot" in plan:
            budget = confirmation_budget(plan, config)
            storage = module(audit.bound(plan["storage_module"]), "_confirmation_storage")
            inventory = storage.own_output(Path(plan["confirmation_pilot"]["path"]).parent)
            scale = plan["updates"] / 2000 * 1.25
            budget.update(pilot_inventory=inventory,
                          forecast_bytes=math.ceil(inventory["actual_bytes"] * scale) + 64*1024**2,
                          forecast_files=math.ceil(inventory["actual_files"] * scale) + 128)
            audit.require(budget["forecast_bytes"] <= plan["output_cap_bytes"] and
                          budget["forecast_files"] <= plan["file_cap"], "Measured confirmation output does not fit cap")
            receipt["confirmation_budget"] = budget
        print(json.dumps(receipt))
        return
    result = run(plan, args.plan)
    print(json.dumps({"status": result["status"], "applied_updates_per_arm": [a["applied_updates"] for a in result["arms"]]}))


if __name__ == "__main__":
    main()
