"""Branch from scored valid molecules with categorical perturbations and latent SGD."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch
from torch.nn import functional as F

from latentmol.chemistry import inspect_graph
from latentmol.design.latent_protenix import (
    LatentDesignConfig, RawBlocks, _json, _model_identity, choose_active,
    evaluate_point, hard_from_graph, latent_step, schedules,
)
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance.mosaic_protenix import MosaicProtenixScorer
from latentmol.representation import DecoderOutput, HardGraphBatch


def mixed_logits(labels, categories, generator):
    """Represent exactly half parent mass and half Dirichlet noise on alternatives."""
    if categories < 3:
        raise ValueError("Need at least two alternatives to independently perturb a row")
    parent = F.one_hot(labels, categories).bool()
    noise = torch.empty(parent.shape, dtype=torch.float32).exponential_(generator=generator)
    noise.masked_fill_(parent, 0)
    mixture = 0.5 * (parent + noise / noise.sum(-1, keepdim=True))
    return mixture.log()


@dataclass(frozen=True)
class RestartOffset:
    delta: DecoderOutput
    fade_updates: int

    def __call__(self, decoded, step):
        if step >= self.fade_updates:
            return decoded
        scale = 1 - step / self.fade_updates
        return DecoderOutput(
            decoded.atoms + scale * self.delta.atoms,
            decoded.charges + scale * self.delta.charges,
            decoded.bonds + scale * self.delta.bonds,
        )

    @classmethod
    def from_parent(cls, decoder, latent, hard, active, *, seed, temperature, fade_updates):
        if not torch.equal(hard.atoms[0] != 0, active) or fade_updates < 1:
            raise ValueError("Restart parent has incompatible occupied slots or fade")
        generator = torch.Generator().manual_seed(seed)
        spec = decoder.spec
        pair = torch.triu(active[:, None] & active[None, :], diagonal=1)
        left, right = pair.nonzero(as_tuple=True)
        with torch.no_grad():
            base = decoder.decode(latent)
            atoms = base.atoms.clone()
            charges = base.charges.clone()
            bonds = base.symmetric_bond_logits.clone()
            atoms[:, active, 1:] = temperature * mixed_logits(
                hard.atoms[:, active] - 1, len(spec.atomic_numbers) - 1, generator)
            charges[:, active] = temperature * mixed_logits(
                hard.charges[:, active], len(spec.formal_charges), generator)
            edges = temperature * mixed_logits(
                hard.bonds[:, left, right], len(spec.bond_names), generator)
            bonds[:, left, right] = bonds[:, right, left] = edges
            offset = cls(DecoderOutput(atoms - base.atoms, charges - base.charges,
                                      bonds - base.symmetric_bond_logits), fade_updates)
            graph = graph_from_decoder(offset(base, 0), spec, temperature=temperature, fixed_active=active)
            for name, mask in (("atoms", active), ("charges", active), ("bonds", pair)):
                values = getattr(graph, name)[:, mask]
                labels = getattr(hard, name)[:, mask]
                selected = values.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                if (not torch.isfinite(values).all() or not torch.allclose(
                        selected, torch.full_like(selected, 0.5), atol=2e-6, rtol=0)
                        or not torch.equal(values.argmax(-1), labels)):
                    raise ValueError("Restart probability mixture lost its parent mass or argmax")
        return offset


@dataclass
class Candidate:
    smiles: str
    latent: torch.Tensor
    hard: HardGraphBatch
    relaxed_loss: float
    branch: str
    step: int

    def record(self, hard_loss):
        return dict(smiles=self.smiles, fresh_loss=hard_loss, relaxed_loss=self.relaxed_loss,
                    branch=self.branch, step=self.step, latent=self.latent.tolist(),
                    labels={name: getattr(self.hard, name)[0].tolist()
                            for name in ("atoms", "charges", "bonds")})


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_branch(decoder, scorer, config, output, latent, active, *, parent=None,
               perturbation_seed=None, fade_updates=25, shortlist_limit=16, scored=(), ready=None,
               proposal=None, select_shortlist=None):
    output.mkdir()
    if proposal is None:
        offset = None if parent is None else RestartOffset.from_parent(
            decoder, latent, parent.hard, active, seed=perturbation_seed,
            temperature=schedules(config, 0)["temperature"], fade_updates=fade_updates)
        diagnostics = {}
    else:
        latent, offset, diagnostics = proposal
    write_json(output / "start.json", dict(parent=None if parent is None else parent.smiles,
               perturbation_seed=perturbation_seed, fade_updates=fade_updates if offset else 0,
               proposal=diagnostics))
    raw = RawBlocks(output / "raw-blocks", decoder.spec.max_nodes, 100)
    noise = torch.Generator().manual_seed(config.noise_seed)
    candidates = {}
    valid_states = bounded_steps = 0
    try:
        with (output / "metrics.jsonl").open("x", encoding="utf-8", buffering=1) as metrics:
            for step in range(config.updates + 1):
                evaluation = evaluate_point(decoder, scorer, latent, active, config, step, logit_offset=offset)
                if step == 0 and ready is not None:
                    _json(ready, dict(status="models_and_inputs_ready",
                        nonce=os.environ.get("DBMOL_LAUNCH_NONCE"), job_id=os.environ.get("LSB_JOBID")))
                hard = hard_from_graph(evaluation["graph"])
                checked = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], decoder.spec)
                smiles = checked.smiles
                valid = bool(smiles and "." not in smiles)
                binding = float(evaluation["binding"])
                raw.add(step, latent, hard)
                if valid:
                    valid_states += 1
                    if smiles not in candidates or binding < candidates[smiles].relaxed_loss:
                        candidates[smiles] = Candidate(smiles, latent.detach().clone(), hard, binding,
                                                      f"{output.parent.name}/{output.name}", step)
                update = None
                if step < config.updates:
                    latent, update = latent_step(latent, evaluation["binding_gradient"],
                                                 evaluation["gradient"], config, step, noise)
                    bounded_steps += update["bound_factor"] < 1
                metrics.write(json.dumps(dict(step=step, smiles=smiles, raw_valid=valid,
                    failure=checked.failure_reason, relaxed_loss=binding, chemistry_loss=float(evaluation["chemistry"]),
                    repair_status=evaluation["repair"]["status"], chemistry_weight=evaluation["schedule"]["chemistry_weight"],
                    gradient_norms=evaluation["gradient_norms"], update=update), allow_nan=False) + "\n")
    finally:
        raw.flush()
    ranked = sorted(candidates.values(), key=lambda row: (row.relaxed_loss, row.smiles))
    new = [row for row in ranked if row.smiles not in scored]
    shortlist = (new[:shortlist_limit] if select_shortlist is None
                 else select_shortlist(new, shortlist_limit))
    write_json(output / "summary.json", dict(updates=config.updates, valid_states=valid_states,
        distinct_valid=len(candidates), unscored_distinct_valid=len(new),
        fresh_shortlist=[row.smiles for row in shortlist], bounded_steps=bounded_steps))
    return shortlist


def run_campaign(decoder, scorer_factory, fresh_score, design, output, *, cycles, elite_count,
                 branches_per_elite, fade_updates, shortlist_limit, perturbation_seed,
                 restart_policy="all-row"):
    from latentmol.design.protenix_restart_proposals import POLICIES, propose_start

    if not (cycles > 0 and elite_count > 0 and branches_per_elite > 0
            and 0 < fade_updates < design.updates and shortlist_limit >= elite_count):
        raise ValueError("Invalid restart campaign budget")
    if restart_policy not in POLICIES:
        raise ValueError(f"Unknown restart policy: {restart_policy}")
    output.mkdir(parents=True)
    decoder.eval().requires_grad_(False)
    if any(p.grad is not None for p in decoder.parameters()):
        raise ValueError("Frozen decoder carries preexisting gradients")
    identity = _model_identity(decoder)
    generator = torch.Generator().manual_seed(design.seed)
    latent = torch.randn((1, decoder.config.latent), generator=generator)
    with torch.no_grad():
        active = choose_active(decoder.decode(latent), decoder.spec, design.occupied_atoms)
    write_json(output / "run.json", dict(design=asdict(design), cycles=cycles, elite_count=elite_count,
        branches_per_elite=branches_per_elite, fade_updates=fade_updates, shortlist_limit=shortlist_limit,
        perturbation_seed=perturbation_seed, restart_policy=restart_policy,
        decoder_sha256=identity, graph_spec=decoder.spec.to_dict(),
        active_slots=active.nonzero().flatten().tolist(), job_id=os.environ.get("LSB_JOBID")))
    scored = set()
    archive = {}
    summaries = []
    for cycle in range(1, cycles + 1):
        parents = [None] if cycle == 1 else [row for row, _ in archive.values()
                                             for _ in range(branches_per_elite)]
        if not parents:
            break
        root = output / f"cycle-{cycle:02d}"
        root.mkdir()
        pending = {}
        scorer = scorer_factory(decoder.spec, active, root)
        try:
            for index, parent in enumerate(parents):
                seed = perturbation_seed + cycle * 10000 + index
                branch_design = replace(design, noise_seed=design.noise_seed + cycle * 10000 + index)
                branch = root / f"branch-{index + 1:02d}"
                proposal = None
                if parent is not None and restart_policy != "all-row":
                    proposal = propose_start(decoder, parent, active, policy=restart_policy, seed=seed,
                                             temperature=schedules(branch_design, 0)["temperature"],
                                             fade_updates=fade_updates)
                shortlist = run_branch(decoder, scorer, branch_design, branch,
                    latent if parent is None else parent.latent, active, parent=parent,
                    perturbation_seed=None if parent is None else seed, fade_updates=fade_updates,
                    shortlist_limit=shortlist_limit, scored=scored | pending.keys(), proposal=proposal,
                    ready=os.environ.get("DBMOL_LAUNCH_READY") if cycle == 1 else None)
                pending.update((row.smiles, row) for row in shortlist)
                print(f"cycle={cycle} branch={index + 1} updates={design.updates} shortlist={len(shortlist)}", flush=True)
        finally:
            scorer.close()
        if pending:
            results = fresh_score(root, list(pending.values()))
            if len(results) != len(pending) or {row["smiles"] for row in results} != pending.keys():
                raise ValueError("Fresh scoring returned a different candidate set")
            for row in results:
                loss = row["loss"]
                if loss is not None:
                    if not math.isfinite(loss):
                        raise ValueError("Nonfinite fresh contact loss")
                    archive[row["smiles"]] = (pending[row["smiles"]], loss)
            scored.update(pending)
        archive = dict(sorted(archive.items(), key=lambda item: (item[1][1], item[0]))[:elite_count])
        entries = [row.record(loss) for row, loss in archive.values()]
        write_json(root / "archive.json", dict(cycle=cycle, branches=len(parents),
            fresh_attempts=len(pending), active_slots=active.nonzero().flatten().tolist(),
            native_identity=getattr(scorer, "identity", None), elites=entries))
        summaries.append(dict(cycle=cycle, branches=len(parents), fresh_attempts=len(pending),
            elites=[dict(smiles=row["smiles"], fresh_loss=row["fresh_loss"]) for row in entries]))
    if _model_identity(decoder) != identity or any(p.grad is not None for p in decoder.parameters()):
        raise ValueError("Frozen decoder parameters or gradients changed")
    result = dict(status="completed" if len(summaries) == cycles else "no_scored_parent",
                  cycles=summaries, unique_fresh_attempts=len(scored), frozen_decoder_unchanged=True)
    write_json(output / "result.json", result)
    return result


def restart_args(description, argv):
    parser = argparse.ArgumentParser(description=description)
    for name in ("config", "native-config", "fresh-config", "fresh-deps", "fresh-data",
                 "checkpoint", "output", "plan"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("checkpoint-sha256", "mosaic-python", "arm", "plan-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("LSB_JOBID", "").isdigit():
        raise RuntimeError("Native restart campaign requires an allocated GPU")
    torch.set_num_threads(4)
    if file_sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Frozen decoder checkpoint SHA differs")
    return args


def native_factory(args):
    native_config = json.loads(args.native_config.read_text())

    def factory(spec, active, root):
        return MosaicProtenixScorer(spec, active,
            command=[args.mosaic_python, "-m", "latentmol.guidance.mosaic_protenix_worker"],
            config=native_config, stderr_path=root / "native.stderr.log")

    return factory


def fresh_score_records(args, root, records):
    write_json(root / "fresh-candidates.json", dict(candidates=records))
    env = os.environ | {"PYTHONPATH": os.pathsep.join((os.environ["PYTHONPATH"], str(args.fresh_deps))),
                        "PROTENIX_ROOT_DIR": str(args.fresh_data), "LAYERNORM_TYPE": "torch"}
    with (root / "fresh.log").open("x", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-m", "latentmol.design.temperature_fresh",
                        "--config", str(args.fresh_config), "--run", str(root)],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((root / "fresh" / "summary.json").read_text())["candidates"]


def main(argv=None):
    args = restart_args(__doc__, argv)
    config = json.loads(args.config.read_text())
    design = LatentDesignConfig(**(config["design"] | config["arms"][args.arm]))
    decoder, _ = load_model(args.checkpoint, "cpu")

    def fresh_score(root, candidates):
        return fresh_score_records(args, root, [
            dict(smiles=row.smiles, start=root.name, arm=row.branch, role="edit", step=row.step)
            for row in candidates])

    return run_campaign(decoder, native_factory(args), fresh_score, design, args.output, **config["campaign"])


if __name__ == "__main__":
    main()
