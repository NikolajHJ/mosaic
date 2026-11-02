"""Bounded, config-driven Protenix chemistry comparisons on fresh outputs.

Metadata imports are portable. Numerical imports and all evaluations occur only
inside an allocated GPU run. Frozen scorers, original FP32 placement, preparation,
repair loss and accelerated kernels are reused; this module owns the explicitly
new schedule/temperature/phase experiments, never historical continuations.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

VERSION = "protenix-chemistry-screen-v1"
SPARSE = {"mode": "sparse_random", "extra_edge_probability": .02,
          "present_probability": .8, "absent_probability": .98,
          "single_bond_probability": .8}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def binding(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Ordinary bound input required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def bound(row):
    require(binding(row["path"]) == row, "Input identity changed: " + row["path"])
    return Path(row["path"])


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def validate_screen(value):
    require(type(value) is dict and set(value) <= {
        "version", "updates", "init_seed", "initialization", "temperature", "chemistry",
        "binding", "severity_progress", "neighbor_environment", "structural_repair", "sampling_interval", "sampling_draws"},
        "Unknown screen option")
    require(value["version"] == VERSION, "Unknown screen version")
    require(type(value["updates"]) is int and 1 <= value["updates"] <= 50000,
            "Bounded update count required")
    require(type(value["init_seed"]) is int and 0 <= value["init_seed"] < 2**32, "Invalid initialization seed")
    require(type(value["temperature"]) in (int, float) and value["temperature"] in (1., 2.), "Declare temperature1 or2")
    init = value["initialization"]
    require(init == SPARSE or init == {"mode": "dense_original"}, "Unknown initialization recipe")
    if init["mode"] == "dense_original":
        require(value["init_seed"] == 17, "Original dense control uses the authenticated seed17 snapshot")
    chemistry = value["chemistry"]
    require(type(chemistry) is dict, "Explicit chemistry policy required")
    mode = chemistry.get("mode")
    if mode == "delayed":
        require(chemistry == {"mode": "delayed", "initial": .001, "final": 100.,
                              "hold_updates": 5000, "total_updates": 50000}, "Changed baseline global50k ramp")
    elif mode == "constant":
        require(set(chemistry) == {"mode", "coefficient"} and type(chemistry["coefficient"]) in (int, float)
                and chemistry["coefficient"] in (1., 10.),
                "Declare constant coefficient1 or10")
    elif mode == "adaptive":
        require(set(chemistry) == {"mode", "ratio_target", "minimum", "maximum", "ema", "initial"},
                "Explicit bounded adaptive policy required")
        require(all(type(chemistry[k]) in (int, float) and math.isfinite(chemistry[k]) for k in chemistry if k != "mode"),
                "Finite adaptive fields required")
        require(chemistry["ratio_target"] > 0 and 0 < chemistry["minimum"] <= chemistry["initial"] <= chemistry["maximum"] <= 100
                and 0 <= chemistry["ema"] < 1, "Invalid adaptive bounds")
    else:
        raise ValueError("Unknown chemistry schedule")
    phase = value["binding"]
    require(phase == {"mode": "always"} or phase == {"mode": "validity_first", "max_chemistry_steps": 5000, "ramp_steps": 2000},
            "Unknown bounded binding phase")
    require(value.get("sampling_interval", 1000) == 1000 and value.get("sampling_draws", 128) == 128,
            "Sampling protocol must remain declared128 draws/1000 steps")
    return value


class Policy:
    """One coefficient/phase decision per accepted source, fixed for its trial."""
    def __init__(self, config):
        self.config = validate_screen(config)
        self.last_state = -1
        self.coefficient = config["chemistry"].get("initial", 1.)
        self.phase_exit = None
        self.phase_exit_reason = None

    def restore(self, state):
        """Restore the decision already made at this accepted source state."""
        require(type(state['step']) is int and 0 < state['step'] < self.config['updates'], 'Invalid resume state')
        keys=('repair_weight','binding_weight','phase_exit_state','phase_exit_reason',
              'unit_binding_gradient_norm','unit_repair_gradient_norm')
        active={k:state[k] for k in keys}
        require(all(math.isfinite(active[k]) and active[k]>=0 for k in
                    ('repair_weight','binding_weight','unit_binding_gradient_norm','unit_repair_gradient_norm')),
                'Nonfinite/negative saved policy')
        require(active['binding_weight']<=1, 'Invalid saved binding weight')
        self.last_state=state['step'];self.coefficient=active['repair_weight']
        self.phase_exit=active['phase_exit_state'];self.phase_exit_reason=active['phase_exit_reason']
        return active

    def source(self, state, raw_valid_connected, binding_norm, chemistry_norm):
        require(state == self.last_state + 1, "Policy advances exactly once per accepted state")
        require(all(math.isfinite(v) and v >= 0 for v in (binding_norm, chemistry_norm)), "Invalid gradient norm")
        c = self.config["chemistry"]
        if c["mode"] == "delayed":
            update = min(state + 1, c["total_updates"])
            coefficient = c["initial"] if update <= c["hold_updates"] else math.exp(
                math.log(c["initial"]) + (update-c["hold_updates"]) / (c["total_updates"]-c["hold_updates"]) *
                (math.log(c["final"])-math.log(c["initial"])))
        elif c["mode"] == "constant":
            coefficient = c["coefficient"]
        else:
            # A genuinely flat unit chemistry component leaves the previous
            # finite bounded coefficient in place; no divide-by-zero/fake force.
            target = self.coefficient if chemistry_norm == 0 else c["ratio_target"] * binding_norm / chemistry_norm
            target = min(c["maximum"], max(c["minimum"], target))
            coefficient = c["ema"] * self.coefficient + (1-c["ema"]) * target
            coefficient = min(c["maximum"], max(c["minimum"], coefficient))
        phase = self.config["binding"]
        native_weight = 1.
        if phase["mode"] == "validity_first":
            if self.phase_exit is None and (raw_valid_connected or state >= phase["max_chemistry_steps"]):
                self.phase_exit = state
                self.phase_exit_reason = "valid_connected_source" if raw_valid_connected else "warmup_cap_without_validity"
            native_weight = 0. if self.phase_exit is None else min(1., (state-self.phase_exit)/phase["ramp_steps"])
        self.last_state, self.coefficient = state, float(coefficient)
        return {"repair_weight": float(coefficient), "binding_weight": float(native_weight),
                "phase_exit_state": self.phase_exit, "phase_exit_reason": self.phase_exit_reason,
                "unit_binding_gradient_norm": float(binding_norm), "unit_repair_gradient_norm": float(chemistry_norm)}


def initialize(config, snapshot, logit_type):
    """Allocated-only PCG64 recipe; no validity filtering, clipping or redraws."""
    import numpy as np
    import jax.numpy as jnp
    if config["initialization"]["mode"] == "dense_original":
        with np.load(snapshot, allow_pickle=False) as saved:
            atoms, bonds = saved["atom_logits"].copy(), saved["bond_logits"].copy()
        return logit_type(jnp.asarray(atoms), jnp.asarray(bonds), config["temperature"])
    rng = np.random.Generator(np.random.PCG64(config["init_seed"]))
    atoms = rng.dirichlet(np.ones(6), size=25).astype(np.float32)
    sequence = rng.integers(0, 25, size=23).tolist()
    degree = [1 + sequence.count(i) for i in range(25)]
    edges = set()
    for vertex in sequence:
        leaf = next(i for i, d in enumerate(degree) if d == 1)
        edges.add(tuple(sorted((leaf, vertex))))
        degree[leaf] -= 1
        degree[vertex] -= 1
    leaves = [i for i, d in enumerate(degree) if d == 1]
    edges.add(tuple(leaves))
    bonds = []
    for i in range(25):
        for j in range(i + 1, 25):
            present = (i, j) in edges or rng.random() < .02
            category = (1 if rng.random() < .8 else 2) if present else 0
            p = np.full(5, .05 if present else .005, dtype=np.float32)
            p[category] = .8 if present else .98
            bonds.append(p)
    # Log conversion uses the same JAX operation as GraphLogits.from_graph.
    return logit_type(jnp.log(jnp.asarray(atoms)), jnp.log(jnp.asarray(np.asarray(bonds))), config["temperature"])


def logits_class():
    import equinox as eqx
    from protenix_initialization import ProtenixReplayLogits
    from mosaic.small_molecule.chemistry.state import GraphLogits
    class ScreenLogits(ProtenixReplayLogits):
        temperature: float = eqx.field(static=True)

        def graph(self):
            if self.temperature == 1.:
                return ProtenixReplayLogits(self.atoms, self.bonds).graph()
            return ProtenixReplayLogits(self.atoms/self.temperature, self.bonds/self.temperature).graph()

        def decoded(self):
            if self.temperature == 1.:
                return GraphLogits(self.atoms, self.bonds).decoded()
            return GraphLogits(self.atoms/self.temperature, self.bonds/self.temperature).decoded()
    return ScreenLogits


def gradient_norm(value):
    import numpy as np
    return math.sqrt(sum(float(np.sum(np.asarray(getattr(value, name), dtype=np.float64)**2))
                         for name in ("atoms", "bonds")))


def evaluate_units(loss, logits, prepared, environment, spec, loss_config, key, native=None):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.optimizers import evaluate_graph, _nonfinite_paths
    from mosaic.small_molecule.chemistry.loss import failure_aware_repair_loss
    graph = logits.graph()
    if native is None:
        native = evaluate_graph(loss, graph, key)
    else:
        require(np.array_equal(np.asarray(jax.random.key_data(native.key)), np.asarray(jax.random.key_data(key))) and
                all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in
                    ((native.graph.X, graph.X), (native.graph.E_upper, graph.E_upper))), "Native reuse changed graph/key")
    prepared.verify(spec)
    (repair, terms), chemistry_gradient = eqx.filter_value_and_grad(
        lambda value: failure_aware_repair_loss(value.decoded(), prepared, spec, loss_config), has_aux=True)(logits)
    neighbor, neighbor_gradient = 0., type(logits)(jnp.zeros_like(logits.atoms), jnp.zeros_like(logits.bonds), logits.temperature)
    if environment is not None:
        from mosaic.small_molecule.chemistry.environment import neighbor_environment_loss
        # Existing neighbor law explicitly requires FP64 tail arithmetic. Scope
        # only this chemical component; native scorer/global producer stayFP32.
        require(not jax.config.x64_enabled, "Native production must remainFP32")
        with jax.enable_x64(True):
            neighbor, neighbor_gradient = eqx.filter_value_and_grad(
                lambda value: neighbor_environment_loss(value.decoded(), environment, spec)["weighted_loss"])(logits)
        require(not jax.config.x64_enabled, "Private neighbor precision scope leaked")
        neighbor_gradient = type(logits)(neighbor_gradient.atoms.astype(jnp.float32),
                                        neighbor_gradient.bonds.astype(jnp.float32), logits.temperature)
    def pull(p, g):
        return p * (g-jnp.sum(p*g, axis=-1, keepdims=True)) / logits.temperature
    binding_gradient = type(logits)(pull(graph.X, native.gradients.X), pull(graph.E_upper, native.gradients.E_upper), logits.temperature)
    require(not _nonfinite_paths((native, repair, terms, chemistry_gradient, neighbor, neighbor_gradient, binding_gradient)),
            "Nonfinite required scalar/gradient path")
    prepared.verify(spec)
    if environment is not None:
        environment.verify(spec)
    return {"logits": logits, "native": native, "repair": float(repair), "neighbor": float(neighbor),
            "terms": {k: float(v) for k, v in terms.items()}, "binding_gradient": binding_gradient,
            "chemistry_gradient": chemistry_gradient, "neighbor_gradient": neighbor_gradient}


def weighted(units, policy):
    import jax.numpy as jnp
    import numpy as np
    logits = units["logits"]
    gradient = type(logits)(*(
        (policy["binding_weight"]*getattr(units["binding_gradient"], name) +
         policy["repair_weight"]*getattr(units["chemistry_gradient"], name) +
         getattr(units["neighbor_gradient"], name)).astype(jnp.float32)
        for name in ("atoms", "bonds")), logits.temperature)
    total = policy["binding_weight"]*float(units["native"].value) + policy["repair_weight"]*units["repair"] + units["neighbor"]
    require(math.isfinite(total) and all(np.isfinite(np.asarray(getattr(gradient, name))).all()
            for name in ("atoms", "bonds")), "Nonfinite weighted objective/gradient")
    return float(total), gradient


def valid_connected(readout):
    return readout.get("sanitized") is True and readout.get("fragment_count") == 1


def sample_validity(logits, seed, state, output):
    """Independent diagnostic draws; every invalid draw is retained, never repaired."""
    import numpy as np
    import jax.numpy as jnp
    from mosaic.small_molecule.state import RelaxedGraph
    from mosaic.small_molecule.io import hard_readout
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([230017, seed, state])))
    graph = logits.graph()
    samples = []
    for probabilities in (graph.X, graph.E_upper):
        cumulative = np.cumsum(np.asarray(probabilities, dtype=np.float64), axis=-1)
        uniforms = rng.random((128, cumulative.shape[0], 1))*cumulative[None, :, -1:]
        samples.append(np.sum(uniforms >= cumulative[None], axis=-1).astype(np.int16))
    path = Path(output)/f"samples-{state:05d}.npz"
    with path.open("wb") as stream:
        np.savez_compressed(stream, atom_indices=samples[0], bond_indices=samples[1])
    sanitized = strict = 0
    unique, failures = set(), {}
    for atoms, bonds in zip(*samples):
        sampled = RelaxedGraph(jnp.asarray(np.eye(6, dtype=np.float32)[atoms]),
                               jnp.asarray(np.eye(5, dtype=np.float32)[bonds]))
        readout = hard_readout(sampled)
        sanitized += int(readout["sanitized"])
        strict += int(valid_connected(readout))
        if valid_connected(readout):
            unique.add(readout["canonical_smiles"])
        else:
            reason = "disconnected" if readout.get("sanitized") else str(readout.get("conversion_error", "unknown")).split(":")[0]
            failures[reason] = failures.get(reason, 0)+1
    return {"step": state, "draws": 128, "sanitized": sanitized, "strict_valid_connected": strict,
            "unique_valid_smiles": sorted(unique), "failure_categories": failures,
            "raw_samples": binding(path), "native_model_key_changed": False,
            "sampling": "Independent PCG64; FP64 inverseCDF scaled by represented row sum; no rejection/repair",
            "binding_scores_evaluated_for_samples": False}


def restore_teacher(record, spec):
    import jax.numpy as jnp
    from mosaic.small_molecule.chemistry.preparation import PreparedFailureGuidance
    repair=record['repair'];extras={'arrays','identity','preparation_seconds','search_seconds'}
    prepared=PreparedFailureGuidance({k:jnp.asarray(v) for k,v in repair['arrays'].items()},
        {k:v for k,v in repair.items() if k not in extras},repair['identity'],repair['preparation_seconds'],
        tuple((v['original'],v['additional']) for v in repair.get('search_seconds',[])))
    prepared.verify(spec)
    environment=None
    if record['neighbor'] is not None:
        from mosaic.small_molecule.chemistry.environment import PreparedNeighborEnvironment
        n=record['neighbor'];environment=PreparedNeighborEnvironment(n['context'],n['receipt'],n['identity'])
        environment.verify(spec)
    return prepared,environment


def run_loop(loss, initial, prepare, spec, loss_config, key, config, output, *, steps=None, tick=None, resume=None):
    """One finite fixed proposal; source teacher and coefficients fixed per step."""
    import jax
    import numpy as np
    from mosaic.small_molecule.io import hard_readout
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"snapshots").mkdir()
    (output/"samples").mkdir()
    (output/"references").mkdir()
    policy = Policy(config)
    start=0 if resume is None else resume['offset']
    steps = config["updates"]-start if steps is None else steps
    require(type(steps) is int and 1 <= steps and start+steps <= config["updates"], "Invalid actual loop budget")
    summary = {"version": VERSION, "status": "running", "requested_updates": steps,
               "applied_updates": 0, "configuration": config, "derivative_qualified": False,
               "state_file": "states.jsonl", "updates_file": "updates.jsonl", "matrix_stride": 20,
               "all_scalar_losses_retained": True, "saved_results_changed": False,
               "best_contact": None, "best_valid_contact": None, "strict_valid_states": 0,
               "invalid_states": 0, "valid_unique_smiles": [], "atom_identity_changes": 0, "bond_identity_changes": 0}
    if resume is not None:
        summary.update(global_start=start,global_target=config['updates'],resume=resume,parent_prefix_unchanged=True)
    unique = set()
    previous_readout = None
    started = time.monotonic()
    last_sampling_seconds = 0.

    def prepare_step(logits, state):
        result = prepare(logits, state)
        environment = None
        if config.get("neighbor_environment") is not None:
            from mosaic.small_molecule.chemistry.environment import prepare_neighbor_environment
            environment = prepare_neighbor_environment(logits.decoded(), spec, config["neighbor_environment"])
        return result, environment

    def save_point(state, units, prepared, environment, active):
        nonlocal previous_readout, last_sampling_seconds
        last_sampling_seconds = 0.
        graph, logits = units["logits"].graph(), units["logits"]
        readout = hard_readout(graph)
        readout.pop("pairs", None)  # Schema already fixes this redundant300-pair list.
        is_valid = valid_connected(readout)
        summary["strict_valid_states"] += int(is_valid)
        summary["invalid_states"] += int(not is_valid)
        if is_valid:
            unique.add(readout["canonical_smiles"])
        if previous_readout is not None:
            summary["atom_identity_changes"] += sum(a != b for a, b in zip(readout["element_indices"], previous_readout["element_indices"]))
            summary["bond_identity_changes"] += sum(a != b for a, b in zip(readout["bond_upper_indices"], previous_readout["bond_upper_indices"]))
        previous_readout = readout
        total, gradient = weighted(units, active)
        native = float(units["native"].value)
        best = summary["best_contact"] is None or native < summary["best_contact"]["loss"]
        best_valid = is_valid and (summary["best_valid_contact"] is None or native < summary["best_valid_contact"]["loss"])
        arrays = {"X": np.asarray(graph.X), "E_upper": np.asarray(graph.E_upper),
                  "atom_logits": np.asarray(logits.atoms), "bond_logits": np.asarray(logits.bonds),
                  "gradient_atom_logits": np.asarray(gradient.atoms), "gradient_bond_logits": np.asarray(gradient.bonds),
                  "contact_gradient_X": np.asarray(units["native"].gradients.X),
                  "contact_gradient_E_upper": np.asarray(units["native"].gradients.E_upper),
                  "key_data": np.asarray(jax.random.key_data(key)), "contact_primal": np.asarray(native),
                  "temperature": np.asarray(logits.temperature, dtype=np.float32), "step": np.asarray(state)}
        def snapshot(path):
            temporary = path.with_name(path.name+".partial")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            temporary.replace(path)
            return binding(path)
        latest = snapshot(output/"latest-state.npz")
        retained = state % 20 == 0 or state == start+steps or state == start
        selected = snapshot(output/"snapshots"/f"step-{state:05d}.npz") if retained else None
        if retained:
            import gzip
            with gzip.open(output/"references"/f"step-{state:05d}.json.gz", "wt", encoding="utf-8", compresslevel=3) as stream:
                json.dump({"repair": prepared.record(), "neighbor": None if environment is None else environment.record()},
                          stream, allow_nan=False, separators=(",", ":"))
        for flag, field, filename in ((best, "best_contact", "best-state.npz"), (best_valid, "best_valid_contact", "best-valid-state.npz")):
            if flag:
                summary[field] = {"step": state, "loss": native, "snapshot": snapshot(output/filename),
                                  "readout": readout, "scope": "same saved relaxed graph native score; hard argmax not rescored"}
        rows = prepared.receipt["rows"]
        state_record = {"step": state-start, "global_step": state, "contact_loss": native,
                        "chemistry_loss": units["repair"], "neighbor_environment_weighted_loss": units["neighbor"],
                        "refreshed_surrogate": total, "components": units["terms"], **active,
                        "gradient_block_norms": {name: {"native": gradient_norm_block(units["binding_gradient"], name),
                            "unit_repair": gradient_norm_block(units["chemistry_gradient"], name),
                            "neighbor": gradient_norm_block(units["neighbor_gradient"], name)} for name in ("atoms", "bonds")},
                        "preparation_identity": prepared.identity,
                        "preparation_seconds": prepared.preparation_seconds,
                        "target_bond_edits": int(np.count_nonzero(prepared.arrays["changed"])),
                        "neighbor_preparation_identity": None if environment is None else environment.identity,
                        "chemistry_preparation": [{"route": r["route"], "raw_native_valid": r["base"]["raw_valid"],
                            "original_status": r["base"]["status"], "component_count": r["component_count"],
                            "structural_status": r.get("structural", {}).get("status"),
                            "structural_checks": r.get("structural", {}).get("checks", 0),
                            "beam_search_performed": r.get("beam_search_performed", True),
                            "unsupported_local_atoms": r["unsupported_local_atoms"]} for r in rows],
                        "readout": readout, "strict_valid_connected": is_valid,
                        "snapshot_retained": retained, "snapshot": None if selected is None else selected["path"],
                        "snapshot_sha256": None if selected is None else selected["sha256"], "latest_snapshot": latest,
                        "temperature": logits.temperature}
        append(output/"states.jsonl", state_record)
        summary["last_state"] = state_record
        summary["valid_unique_smiles"] = sorted(unique)
        write(output/"latest-state.json", {"step": state, "snapshot": latest, "configuration": config})
        if state % 1000 == 0 or state == start+steps:
            sampling_started = time.monotonic()
            sample_record = sample_validity(logits, config["init_seed"], state, output/"samples")
            append(output/"sampled-validity.jsonl", sample_record)
            summary["last_sampled_validity"] = sample_record
            last_sampling_seconds = time.monotonic()-sampling_started
        return readout

    try:
        if resume is None:
            prepared, environment = prepare_step(initial, 0)
        else:
            import gzip
            with gzip.open(bound(resume['reference']),'rt',encoding='utf8') as stream:
                prepared,environment=restore_teacher(json.load(stream),spec)
            require(prepared.identity==resume['state']['preparation_identity'],'Saved teacher differs')
            require((None if environment is None else environment.identity)==resume['state']['neighbor_preparation_identity'],
                    'Saved neighbor context differs')
        units = evaluate_units(loss, initial, prepared, environment, spec, loss_config, key)
        readout = hard_readout(initial.graph())
        active = (policy.source(0, valid_connected(readout), gradient_norm(units["binding_gradient"]), gradient_norm(units["chemistry_gradient"]))
                  if resume is None else policy.restore(resume['state']))
        save_point(start, units, prepared, environment, active)
        summary["initial_state"] = summary["last_state"]
        for step in range(start,start+steps):
            update_started = time.monotonic()
            if tick:
                tick(step)
            before, gradient = weighted(units, active)
            logits = units["logits"]
            proposal = type(logits)(logits.atoms-.3*gradient.atoms, logits.bonds-.3*gradient.bonds, logits.temperature)
            # Retain the exact proposed input even if native execution fails.
            with (output/"current-trial.npz").open("wb") as stream:
                np.savez_compressed(stream, atom_logits=np.asarray(proposal.atoms), bond_logits=np.asarray(proposal.bonds), step=step)
            candidate = evaluate_units(loss, proposal, prepared, environment, spec, loss_config, key)
            after, _ = weighted(candidate, active)
            record = {"step": step-start, "global_step": step, "preparation_identity": prepared.identity,
                      "fixed_before": before, "fixed_after": after, "fixed_increased": after > before,
                      "contact_before": float(units["native"].value), "contact_after": float(candidate["native"].value),
                      "rate": .3, "candidate_count": 1, **active,
                      "no_op": bool(all(np.array_equal(np.asarray(getattr(logits, n)), np.asarray(getattr(proposal, n))) for n in ("atoms", "bonds")))}
            prepared, environment = prepare_step(proposal, step+1)
            units = evaluate_units(loss, proposal, prepared, environment, spec, loss_config, key, native=candidate["native"])
            readout = hard_readout(proposal.graph())
            active = policy.source(step+1, valid_connected(readout), gradient_norm(units["binding_gradient"]), gradient_norm(units["chemistry_gradient"]))
            save_point(step+1, units, prepared, environment, active)
            summary["applied_updates"] = step+1-start
            record["elapsed_seconds"] = time.monotonic()-update_started
            record["sampling_seconds"] = last_sampling_seconds
            record["optimization_seconds"] = max(0., record["elapsed_seconds"]-last_sampling_seconds)
            append(output/"updates.jsonl", record)
            if (step+1) % 20 == 0 or step+1 == start+steps:
                print(json.dumps({"update": step+1, "of": steps, "contact": float(units["native"].value),
                    "repair": units["repair"], "lambda": active["repair_weight"], "binding_weight": active["binding_weight"],
                    "valid": valid_connected(readout)}, separators=(",", ":")), flush=True)
                write(output/"report.json", summary)
        summary["status"] = "completed_exploratory_updates"
    except BaseException as error:
        summary.update(status="failed", error=repr(error))
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic()-started
        summary["phase_exit_state"] = policy.phase_exit
        summary["phase_exit_reason"] = policy.phase_exit_reason
        write(output/"report.json", summary)
    return summary


def append(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, allow_nan=False, separators=(",", ":"))+"\n")


def gradient_norm_block(gradient, name):
    import numpy as np
    return float(np.linalg.norm(np.asarray(getattr(gradient, name), dtype=np.float64)))


def main(argv=None):
    setup_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--smoke-updates", type=int)
    parser.add_argument("--internal-seconds", type=int)
    args = parser.parse_args(argv)
    plan = read(args.plan)
    row = plan["arms"][args.arm]
    config = validate_screen(row["screen"])
    require(row["rate"] == .3 and row["model_key"] == 17 and row["matrix_stride"] == 20 and row["loss_stride"] == 1,
            "Changed shared LR/model key/retention")
    require(row.get("composition_weight", 0) == 0 and not row.get("margin", False), "Composition/margin must remain off")
    require(not row.get("logit_noise"), "Noise not requested in this comparison")
    for name in ("config", "snapshot", "reference", "pocket", "context_ready"):
        bound(row[name])
    if row.get('resume') is not None:
        for name in ('snapshot','reference'):bound(row['resume'][name])
        require(row['resume']['state']['step']==row['resume']['offset'],'Resume step differs')
        Policy(config).restore(row['resume']['state'])
    code_source, original = Path(plan["code_source_root"]), Path(row["science_source"]["path"])
    require(binding(original/"source_manifest.json")["sha256"] == row["science_source"]["manifest_sha256"], "Original science provenance differs")
    sys.path[:0] = [str(code_source/"src"), str(original/"scripts")]
    if args.startup_only:
        print(json.dumps({"status": "metadata_startup_passed_no_models", "arm": args.arm,
                          "configuration": config, "new_library_first": str(code_source/"src")}))
        return
    from gpu_policy import selected, validate_hardware
    gpu = selected(row)
    require(os.environ.get("LSB_JOBID", "").isdigit(), "Allocated supported GPU required")
    require(os.environ["LSB_QUEUE"] == gpu["queue"], "Declared queue differs")
    import csv
    import fcntl
    import io
    import subprocess
    routing = module(Path(plan["routing_source"])/"scripts/cuda_ownership.py", "_chemistry_screen_cuda")
    visibility = routing.resolve_visibility(os.environ)
    os.environ["CUDA_VISIBLE_DEVICES"] = visibility["selected_cuda_visible_devices"]
    observed = subprocess.run(["nvidia-smi", "--id="+os.environ["CUDA_VISIBLE_DEVICES"], "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True, timeout=30)
    devices = list(csv.reader(io.StringIO(observed.stdout)))
    require(len(devices) == 1, "Exactly one GPU required")
    hardware = validate_hardware(devices[0][0].strip(), int(devices[0][1]), gpu)
    root = Path(plan["root"])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/(("probe-" if args.smoke_updates is not None else "run-")+args.arm+".writer.lock")).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = root/"probe"/args.arm/"study" if args.smoke_updates is not None else root/args.arm/"payload"/"study"
    require(not output.exists(), "Fresh output only")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {"version": VERSION, "status": "running", "arm": args.arm, "job_id": os.environ["LSB_JOBID"],
               "configuration": config, "gpu_hardware": hardware, "original_results_changed": False,
               "derivative_qualified": False, "warnings": ["Exploratory finite updates; no binding or derivative certification"]}
    import jax
    import numpy as np
    require(not jax.config.x64_enabled, "Native Protenix must remainFP32")
    jax.config.update("jax_default_matmul_precision", "highest")
    from mosaic.small_molecule.protenix import FrozenProtenixTiny
    from mosaic.small_molecule.losses import NativeContactLoss
    from mosaic.small_molecule.io import load_context
    from mosaic.small_molecule.pocket import map_protenix_pocket
    from mosaic.small_molecule.gradient_checks import array_digest
    from mosaic.small_molecule.chemistry.state import mosaic_spec
    from mosaic.small_molecule.chemistry.loss import FailureGuidanceConfig
    from mosaic.small_molecule.chemistry.preparation import prepare_failure_guidance
    from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig
    from mosaic.small_molecule.chemistry._reference.repair_progress import ProgressSearchConfig
    source_config = read(row["config"]["path"])
    ready = read(row["context_ready"]["path"])
    context = load_context(Path(row["context_ready"]["path"]).parent/"context", ready["context_files"])
    model = FrozenProtenixTiny.load(cache_dir=Path("/work3/s234818/mosaic/cache/mosaic/protenix"))
    frozen = {"model_digest": array_digest(model), "context_digest": array_digest(context)}
    require(frozen == row["frozen_identity"], "Frozen model/context differs")
    summary["frozen_before"] = frozen
    loss = model.build_loss(context=context, loss=NativeContactLoss(8., map_protenix_pocket(context, read(row["pocket"]["path"]))))
    logits_type = logits_class()
    initial = initialize(config, row["snapshot"]["path"], logits_type)
    key = jax.random.key(17)
    with np.load(row["snapshot"]["path"], allow_pickle=False) as original_point:
        require(np.array_equal(original_point["key_data"], np.asarray(jax.random.key_data(key))), "Original snapshot model key differs")
        if config["initialization"]["mode"] == "dense_original":
            require(all(np.array_equal(original_point[f], np.asarray(getattr(initial, n))) for f, n in
                        (("atom_logits", "atoms"), ("bond_logits", "bonds"))), "Original dense logits changed")
    resume=row.get('resume')
    if resume is not None:
        require(resume['state']['step']==resume['offset'] and 0<resume['offset']<config['updates'],'Resume offset differs')
        require(resume['state']['temperature']==config['temperature'],'Resume temperature differs')
        with np.load(bound(resume['snapshot']),allow_pickle=False) as point:
            require(int(point['step'])==resume['offset'],'Resume matrix step differs')
            require(np.array_equal(point['key_data'],np.asarray(jax.random.key_data(key))),'Resume key differs')
            import jax.numpy as jnp
            initial=logits_type(jnp.asarray(point['atom_logits']),jnp.asarray(point['bond_logits']),config['temperature'])
    spec = mosaic_spec(25)
    search = ProbabilityRepairConfig(**source_config["search"])
    progress = ProgressSearchConfig(**source_config["progress_search"])
    def prepare(logits, state):
        return prepare_failure_guidance(logits.decoded(), spec, update=state, search=search, progress_search=progress,
                                        severity_progress=config.get("severity_progress"),
                                        structural_repair=config.get("structural_repair"))
    begun, next_check = time.monotonic(), time.monotonic()
    internal_seconds = row["internal_seconds"] if args.internal_seconds is None else args.internal_seconds
    require(type(internal_seconds) is int and 1 <= internal_seconds <= row["internal_seconds"], "Invalid internal deadline override")
    summary["setup_elapsed_seconds"] = begun-setup_started
    summary["experimental_mixture"] = "Unit component gradients mixed afterAD; mathematical sum, no claim of old weighted-AD bitidentity"
    def tick(state):
        nonlocal next_check
        require(time.monotonic()-setup_started < internal_seconds, "Own runtime budget reached; saved prefix retained")
        if time.monotonic() >= next_check:
            helper = module(plan["capacity_module"], "_chemistry_screen_capacity")
            write(output.parent/"capacity-latest.json", helper.capacity(plan, "periodic"))
            current_bytes = sum(p.stat().st_size for p in output.parent.rglob("*") if p.is_file())
            require(current_bytes <= row["output_cap_bytes"], "Own output cap exceeded")
            next_check = time.monotonic()+300
    try:
        adapter = Path(row.get("gpu_adapter_source", row["adapter_source"]["path"]))
        sys.path.insert(0, str(adapter/"scripts"))
        os.environ["PYTHONPATH"] = os.pathsep.join([str(code_source/"src"), str(original/"scripts"), str(adapter/"scripts"), os.environ.get("PYTHONPATH", "")])
        pilot = module(adapter/"scripts/full_model_pilot.py", "_chemistry_screen_acceleration")
        acceleration = output.parent/"acceleration"
        acceleration.mkdir(exist_ok=False)
        with pilot.combined_adapters("protenix-random50k", acceleration):
            start=0 if resume is None else resume['offset']
            actual_steps = args.smoke_updates if args.smoke_updates is not None else config["updates"]-start
            require(1 <= actual_steps <= config["updates"], "Invalid pilot budget")
            result = run_loop(loss, initial, prepare, spec, FailureGuidanceConfig(**source_config["loss"]), key, config, output,
                              steps=actual_steps, tick=tick,resume=resume)
        summary.update(status=result["status"], applied_updates=result["applied_updates"],
                       actual_steps=actual_steps, full_budget_complete=start+result["applied_updates"] == config["updates"],
                       probe_not_adopted=args.smoke_updates is not None, report_path=str(output/"report.json"),
                       elapsed_seconds=time.monotonic()-begun, best_contact=result["best_contact"],
                       best_valid_contact=result["best_valid_contact"], strict_valid_states=result["strict_valid_states"])
        if resume is not None:
            summary.update(global_start=start,global_target=config['updates'],global_end=start+result['applied_updates'],resume=resume)
    except BaseException as error:
        summary.update(status="failed", error=repr(error))
        raise
    finally:
        after = {"model_digest": array_digest(model), "context_digest": array_digest(context)}
        summary["frozen_after"] = after
        for name in ("config", "snapshot", "reference", "pocket", "context_ready"):
            bound(row[name])
        require(after == frozen, "Frozen model/context changed")
        write(output.parent/"summary.json", summary)


if __name__ == "__main__":
    main()
