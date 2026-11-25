"""Test score-guided atom births/deaths on the unmasked hard VAE decode."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from rdkit import Chem
import torch

from latentmol.chemistry import inspect_graph
from latentmol.evaluation.evaluate import load_model
from latentmol.multimodel.external import file_sha256, tensor_tree_sha256
from latentmol.protenix_occupancy.budget_native import ReferenceGeometryError, score_topology
from latentmol.protenix_occupancy.native import load_frozen_tiny
from latentmol.protenix_occupancy.one_atom import optional_log_odds
from latentmol.protenix_occupancy.prepare import _bind_native
from latentmol.protenix_occupancy.raw_occupancy_summary import summarize
from latentmol.protenix_occupancy.study import write_json
from latentmol.scoring import require_frozen_eval


def toggle_slot(decoded, spec, slot, present):
    """Harden normally and intervene on just one occupancy decision."""
    if decoded.atoms.shape[0] != 1 or not 0 <= slot < spec.max_nodes:
        raise ValueError("A toggle requires one graph and an in-range decoder slot")
    graph = decoded.harden(spec)
    atoms, charges, bonds = graph.atoms[0], graph.charges[0], graph.bonds[0]
    if present:
        atoms[slot] = decoded.atoms[0, slot, 1:].argmax() + 1
        charges[slot] = decoded.charges[0, slot].argmax()
        incident = decoded.symmetric_bond_logits[0, slot].argmax(-1)
        incident = incident.masked_fill(atoms == 0, 0)
        incident[slot] = 0
    else:
        atoms[slot] = 0
        charges[slot] = spec.formal_charges.index(0)
        incident = torch.zeros_like(bonds[slot])
    bonds[slot, :] = bonds[:, slot] = incident
    return graph


def occupancy_objective(decoded, baseline_loss, alternatives):
    """Sum local one-toggle expected loss changes about the hard decode."""
    result = decoded.atoms.new_tensor(baseline_loss, dtype=torch.float64)
    for slot, present, loss in alternatives:
        log_odds = optional_log_odds(decoded, slot)
        toggle_probability = (log_odds if present else -log_odds).sigmoid()
        result = result + (loss - baseline_loss) * toggle_probability
    return result


def observe(graph, spec):
    checked = inspect_graph(graph.atoms[0], graph.charges[0], graph.bonds[0], spec)
    return dict(smiles=checked.smiles, failure=checked.failure_reason,
                atom_count=int(graph.atoms.count_nonzero()))


def decode(model, point):
    decoded = model.decode(point)
    if any(not getattr(decoded, key).isfinite().all() for key in ("atoms", "charges", "bonds")):
        raise ValueError("Nonfinite VAE logits")
    return decoded


def probe_rows(decoded, spec, scorer, seed, candidate_limit):
    """Try the nearest hard occupancy boundaries, independently for add/drop."""
    atoms = decoded.atoms[0].detach().double()
    margin = atoms[:, 1:].amax(-1) - atoms[:, 0]
    occupancy = atoms.softmax(-1)[:, 1:].sum(-1)
    candidates = margin.abs().argsort(stable=True).tolist()
    for present in (True, False):
        tested = 0
        for slot in candidates:
            if not 0.02 <= occupancy[slot] <= 0.98 or bool(atoms[slot].argmax() == 0) != present:
                continue
            row = observe(toggle_slot(decoded, spec, slot, present), spec)
            row.update(slot=slot, present=present, occupancy=float(occupancy[slot]),
                       hard_margin=float(margin[slot]))
            tested += 1
            if row["smiles"] is not None:
                row.update(scorer(row, seed))
            yield row
            if row.get("loss") is not None or tested >= candidate_limit:
                break


def occupancy_step(model, point, anchor, scorer, config, sign=1):
    """Take one bounded occupancy update, accepting the exact decoded trial."""
    seed = config["geometry_seed"]
    point = point.detach().requires_grad_()
    decoded = decode(model, point)
    raw_graph = decoded.harden(model.spec)
    base = observe(raw_graph, model.spec)
    base.update(scorer(base, seed))
    with torch.no_grad():
        probes = list(probe_rows(decoded, model.spec, scorer, seed, config["probe_limit"]))
    scored = [(row["slot"], row["present"], row["loss"]) for row in probes
              if row.get("loss") is not None]
    record = dict(baseline=base, probes=probes, trials=[])
    if not scored:
        return point.detach(), dict(record, status="no_valid_native_probe")
    objective = occupancy_objective(decoded, base["loss"], scored)
    if not objective.isfinite() or not objective.requires_grad:
        raise ValueError("Nonfinite or detached occupancy objective")
    gradient, = torch.autograd.grad(objective, point)
    norm = float(gradient.norm())
    if not math.isfinite(norm):
        raise ValueError("Nonfinite latent gradient")
    record.update(local_objective=float(objective.detach()), gradient_norm=norm)
    if norm == 0:
        return point.detach(), dict(record, status="flat_gradient")
    return _trial_step(model, point, anchor, scorer, config, raw_graph, record,
                       gradient, norm, sign)


def _trial_step(model, point, anchor, scorer, config, raw_graph, record,
                direction, norm, sign=1):
    """Use the same bounded line search for gradient and random directions."""
    base, probes = record["baseline"], record["probes"]
    with torch.no_grad():
        for scale in config["trial_steps"]:
            delta = point - sign * scale * direction / norm - anchor
            radius = config["latent_radius"]
            proposal = anchor + delta * (radius / delta.norm().clamp_min(radius))
            if torch.equal(proposal, point):
                record["trials"].append(dict(scale=scale, status="identical_latent"))
                continue
            hard = decode(model, proposal).harden(model.spec)
            trial = observe(hard, model.spec)
            trial.update(scale=scale, atom_delta=trial["atom_count"] - base["atom_count"],
                         occupancy_flips=int(((hard.atoms != 0) != (raw_graph.atoms != 0)).sum()),
                         matched_probe=next((row["slot"] for row in probes
                             if trial["smiles"] is not None and row["smiles"] == trial["smiles"]), None))
            record["trials"].append(trial)
            if trial["smiles"] is None:
                continue
            trial.update(scorer(trial, config["geometry_seed"]))
            if trial["loss"] is not None and trial["loss"] <= base["loss"] + 1e-6:
                return proposal.detach(), dict(record, status="accepted")
    return point.detach(), dict(record, status="no_acceptable_trial")


def trajectory(model, anchor, scorer, config, sign):
    """Move on raw occupancy gradients and judge exact decoded trial molecules."""
    point = anchor
    rows = []
    for step in range(config["updates"]):
        point, record = occupancy_step(model, point, anchor, scorer, config, sign)
        record["step"] = step
        rows.append(record)
        if record["status"] != "accepted":
            break
    with torch.no_grad():
        final = observe(decode(model, point).harden(model.spec), model.spec)
    final.update(scorer(final, config["geometry_seed"]))
    return dict(final=final, updates=rows)


class NativeCallBudgetExhausted(RuntimeError):
    """A cache miss would exceed the native scorer call budget."""


class ProtenixOracle:
    """Cache only exact chemistry at the same geometry seed and pocket."""

    def __init__(self, native, pocket, output, max_calls):
        self.native, self.pocket, self.output = native, pocket, output
        self.cache, self.calls, self.max_calls = {}, 0, max_calls

    def __call__(self, row, seed):
        smiles = row["smiles"]
        if smiles is None:
            raise ValueError("Native scorer requires a valid exact decoded molecule")
        key = (smiles, seed)
        if key in self.cache:
            cached = self.cache[key]
            if cached["loss"] is not None and cached["native_input_atom_count"] != row["atom_count"]:
                raise ValueError("Cached native molecule has a different atom count")
            return cached
        if self.calls >= self.max_calls:
            raise NativeCallBudgetExhausted("Native score budget exhausted")
        mol = Chem.MolFromSmiles(smiles)
        if (mol is None or mol.GetNumAtoms() != row["atom_count"]
                or Chem.MolToSmiles(mol, isomericSmiles=False) != smiles):
            raise ValueError("Native preparation would change the hard molecule")
        digest = hashlib.sha256(smiles.encode()).hexdigest()[:20]
        self.calls += 1
        try:
            loss, detail = score_topology(self.native, mol, range(mol.GetNumAtoms()),
                self.pocket["protein_sequence"], [r["sequence_position"] for r in self.pocket["residues"]],
                geometry_seed=seed, mol_path=self.output / f"{seed}-{digest}.mol",
                geometry_max_iterations=64)
        except ReferenceGeometryError:
            result = dict(loss=None, native_failure="reference_geometry")
        else:
            if not math.isfinite(loss) or detail["heavy_atoms"] != row["atom_count"]:
                raise ValueError("Native scorer returned a nonfinite loss or changed atom count")
            result = dict(loss=loss, native_failure=None, native_input_smiles=smiles,
                          native_input_atom_count=detail["heavy_atoms"], native=detail)
        self.cache[key] = result
        return result


def _guided_call_budget(start):
    """Count unique optimization calls, including unsuccessful native attempts."""
    arm = start["arms"]["gradient"]
    rows = [start["initial"], arm["final"]]
    for update in arm["updates"]:
        rows.extend((update["baseline"], *update["probes"], *update["trials"]))
    return len({row["smiles"] for row in rows
                if row.get("smiles") is not None and "native_failure" in row})


def random_trajectory(model, anchor, oracle, config, generator, max_directions):
    """Spend a per-start native call budget on isotropic latent proposals."""
    point = anchor
    updates = []
    stop = "direction_budget"
    with torch.no_grad():
        for step in range(max_directions):
            if oracle.calls >= oracle.max_calls:
                stop = "native_budget"
                break
            raw_graph = decode(model, point).harden(model.spec)
            baseline = observe(raw_graph, model.spec)
            baseline.update(oracle(baseline, config["geometry_seed"]))
            record = dict(step=step, baseline=baseline, trials=[], probes=[])
            direction = torch.randn(point.shape, dtype=point.dtype, generator=generator)
            norm = direction.norm()
            if not norm.isfinite() or norm == 0:
                raise ValueError("Invalid random latent direction")
            try:
                point, record = _trial_step(model, point, anchor, oracle, config, raw_graph,
                                            record, direction.to(point.device), norm.item())
            except NativeCallBudgetExhausted:
                record["status"] = stop = "native_budget"
                record["trials"][-1]["status"] = "native_budget"
            updates.append(record)
            if stop == "native_budget":
                break
        final = observe(decode(model, point).harden(model.spec), model.spec)
        final.update(oracle(final, config["geometry_seed"]))
    if oracle.calls == oracle.max_calls:
        stop = "native_budget"
    return dict(final=final, updates=updates, stop=stop, search_calls=oracle.calls,
                search_call_budget=oracle.max_calls)


def _random_control(control_path, config, output, job_id, model, points, oracle, checkpoint_sha,
                    fresh_seeds):
    options = json.loads(control_path.read_text(encoding="utf-8"))
    reference_path = control_path.parent / options["reference"]
    if file_sha256(reference_path) != options["reference_sha256"]:
        raise ValueError("Guided reference identity differs")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    references = reference["starts"]
    draws = {row["draw"] for row in references}
    if (reference["status"] != "completed" or reference["checkpoint_sha256"] != checkpoint_sha
            or reference["fresh_geometry_seeds"] != fresh_seeds
            or len(draws) != len(references) or len(draws) != config["starts"]
            or any(type(draw) is not int or not 0 <= draw < len(points) for draw in draws)
            or type(options["max_directions"]) is not int or options["max_directions"] < 1
            or type(options["seed"]) is not int or not 0 <= options["seed"] < 2**63 - len(points)):
        raise ValueError("Incompatible guided reference or random control configuration")
    budgets = [_guided_call_budget(row) for row in references]
    if min(budgets) < 1 or sum(budgets) + 2 * len(draws) * len(fresh_seeds) > oracle.max_calls:
        raise ValueError("Invalid matched native score budgets")

    starts = []
    for saved, budget in zip(references, budgets):
        draw = saved["draw"]
        anchor = points[draw][None].to(next(model.parameters()).device)
        with torch.no_grad():
            initial = observe(decode(model, anchor).harden(model.spec), model.spec)
        if (initial["smiles"] is None or any(initial[key] != saved["initial"][key]
                                            for key in ("smiles", "atom_count"))):
            raise ValueError(f"Replayed draw {draw} differs from the guided starting molecule")
        mol_dir = output / f"draw-{draw:03d}"
        mol_dir.mkdir()
        scorer = ProtenixOracle(oracle.native, oracle.pocket, mol_dir, budget)
        initial.update(scorer(initial, config["geometry_seed"]))
        if initial["loss"] is None:
            raise ValueError(f"Cannot score the guided starting molecule for draw {draw}")
        generator = torch.Generator(device="cpu").manual_seed(options["seed"] + draw)
        random_arm = random_trajectory(model, anchor, scorer, config, generator,
                                       options["max_directions"])
        # Fresh calls are reserved separately, with the same exact-identity cache.
        scorer.max_calls = budget + 2 * len(fresh_seeds)
        replayed_fresh = {str(seed): scorer(initial, seed) for seed in fresh_seeds}
        random_arm["fresh_by_seed"] = {str(seed): scorer(random_arm["final"], seed)
                                       for seed in fresh_seeds}
        random_arm["fresh"] = random_arm["fresh_by_seed"][str(fresh_seeds[0])]
        record = dict(draw=draw, initial=initial, initial_fresh=saved["initial_fresh"],
                      initial_fresh_by_seed=saved["initial_fresh_by_seed"],
                      replayed_initial_fresh_by_seed=replayed_fresh,
                      arms=dict(gradient=saved["arms"]["gradient"], random_search=random_arm),
                      native_attempts=scorer.calls)
        starts.append(record)
        write_json(output / f"draw-{draw:03d}.json", record)
    return dict(status="completed", job_id=job_id, starts=starts, starts_requested=config["starts"],
                native_attempts=sum(row["native_attempts"] for row in starts),
                search_calls=sum(row["arms"]["random_search"]["search_calls"] for row in starts),
                search_call_budget=sum(budgets), max_directions=options["max_directions"],
                random_seed=options["seed"], reference_job_id=reference["job_id"],
                reference_sha256=options["reference_sha256"], fresh_geometry_seeds=fresh_seeds,
                summary=summarize(starts, bootstrap_seed=config["seed"], comparator="random_search"),
                checkpoint_sha256=checkpoint_sha, count_policy=reference["count_policy"],
                loss=reference["loss"])


def run(config_path, output, job_id, random_control=None):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fresh_seeds = (config["fresh_geometry_seeds"] if "fresh_geometry_seeds" in config
                   else [config["fresh_geometry_seed"]])
    assets = json.loads((config_path.parent / config["assets"]).read_text(encoding="utf-8"))
    checkpoint = json.loads((config_path.parent / assets["assets_config"]).read_text(encoding="utf-8"))
    pocket_path = Path(assets["pocket"]["path"])
    if (file_sha256(pocket_path) != assets["pocket"]["sha256"]
            or file_sha256(checkpoint["checkpoint"]) != checkpoint["checkpoint_sha256"]):
        raise ValueError("Pocket or VAE checkpoint identity differs")
    if (any(type(config[name]) is not int or config[name] < 1 for name in
            ("starts", "start_draws", "updates", "probe_limit", "max_native_calls"))
            or not math.isfinite(config["latent_radius"]) or config["latent_radius"] <= 0
            or not config["trial_steps"]
            or any(not math.isfinite(scale) or scale <= 0 for scale in config["trial_steps"])):
        raise ValueError("Invalid bounded experiment configuration")
    geometry_seeds = [config["geometry_seed"], *fresh_seeds]
    if (not fresh_seeds or any(type(seed) is not int or not 0 <= seed < 2**31 - 1 for seed in geometry_seeds)
            or len(geometry_seeds) != len(set(geometry_seeds))):
        raise ValueError("Require distinct supported optimization and fresh geometry seeds")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model, _ = load_model(Path(checkpoint["checkpoint"]), "cuda")
    model.requires_grad_(False)
    require_frozen_eval(model)
    model_state = tensor_tree_sha256(model.state_dict())
    _bind_native(Path(checkpoint["native"]["source_root"]), checkpoint["native"]["source_sha256"],
                 Path(assets["ccd_root"]))
    native = load_frozen_tiny(**checkpoint["native"])
    native_state = native.snapshot_state()
    oracle = ProtenixOracle(native, json.loads(pocket_path.read_text()), output, config["max_native_calls"])
    if proof := os.environ.get("DBMOL_LAUNCH_READY"):
        write_json(Path(proof), dict(status="models_and_inputs_ready", job_id=job_id,
                                    nonce=os.environ["DBMOL_LAUNCH_NONCE"]))

    random = torch.Generator(device="cpu").manual_seed(config["seed"])
    points = torch.randn((config["start_draws"], model.config.latent), generator=random)
    if random_control is not None:
        result = _random_control(random_control, config, output, job_id, model, points,
                                 oracle, checkpoint["checkpoint_sha256"], fresh_seeds)
        require_frozen_eval(model)
        if tensor_tree_sha256(model.state_dict()) != model_state:
            raise RuntimeError("Frozen VAE state changed")
        native.assert_unchanged(native_state)
        write_json(output / "result.json", result)
        return
    starts, skipped, used = [], [], set()
    # Reserve enough uncached calls for a complete matched start, including
    # the acquisition probes, both arms' trials and all fresh endpoints.
    reserve = (1 + 2 * config["probe_limit"] + 2 * config["updates"] *
               (1 + 2 * config["probe_limit"] + len(config["trial_steps"])) + 3 * len(fresh_seeds))
    budget_limited = False
    for index, point in enumerate(points):
        if "fresh_geometry_seeds" in config and oracle.calls + reserve > config["max_native_calls"]:
            budget_limited = True
            break
        anchor = point[None].cuda()
        with torch.no_grad():
            decoded = decode(model, anchor)
            base = observe(decoded.harden(model.spec), model.spec)
            if base["smiles"] is None or base["smiles"] in used:
                skipped.append(dict(draw=index, reason=base["failure"] or "duplicate"))
                continue
            probes = list(probe_rows(decoded, model.spec, oracle, config["geometry_seed"], config["probe_limit"]))
            if {r["present"] for r in probes if r.get("loss") is not None} != {False, True}:
                skipped.append(dict(draw=index, reason="no_matched_add_drop", probes=probes))
                continue
            base.update(oracle(base, config["geometry_seed"]))
            if base["loss"] is None:
                skipped.append(dict(draw=index, reason=base["native_failure"]))
                continue
        used.add(base["smiles"])
        arms = {name: trajectory(model, anchor, oracle, config, sign)
                for name, sign in (("gradient", 1), ("reversed_gradient", -1))}
        for arm in arms.values():
            arm["fresh_by_seed"] = {str(seed): oracle(arm["final"], seed) for seed in fresh_seeds}
            arm["fresh"] = arm["fresh_by_seed"][str(fresh_seeds[0])]
        initial_fresh = {str(seed): oracle(base, seed) for seed in fresh_seeds}
        record = dict(draw=index, initial=base, initial_fresh=initial_fresh[str(fresh_seeds[0])],
                      initial_fresh_by_seed=initial_fresh, arms=arms)
        starts.append(record)
        write_json(output / f"draw-{index:03d}.json", record)
        if len(starts) == config["starts"]:
            break
    require_frozen_eval(model)
    if tensor_tree_sha256(model.state_dict()) != model_state:
        raise RuntimeError("Frozen VAE state changed")
    native.assert_unchanged(native_state)
    write_json(output / "result.json", dict(status="completed", job_id=job_id, starts=starts,
        starts_requested=config["starts"], skipped=skipped, native_attempts=oracle.calls,
        fresh_geometry_seeds=fresh_seeds, budget_limited=budget_limited,
        summary=summarize(starts, bootstrap_seed=config["seed"]),
        checkpoint_sha256=checkpoint["checkpoint_sha256"],
        count_policy="unmodified_decoder_argmax", loss="mean_negative_log_pocket_contact_all_ligand_atoms"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-control", type=Path, help="Compare random search to an archived guided arm")
    args = parser.parse_args(argv)
    job_id = os.environ.get("LSB_JOBID", "")
    if not job_id.isdigit() or not torch.cuda.is_available():
        raise RuntimeError("Run native Protenix inside an allocated GPU job")
    run(args.config, args.output, job_id, args.random_control)


if __name__ == "__main__":
    main()
