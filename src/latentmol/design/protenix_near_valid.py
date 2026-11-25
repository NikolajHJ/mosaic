"""Replay the invalid and exact-valid starts for the same saved restart slot."""

from dataclasses import asdict, dataclass, replace
from functools import partial
import json
import math
import os
from pathlib import Path

import torch

from latentmol.chemistry import inspect_graph
from latentmol.design.latent_protenix import LatentDesignConfig, _model_identity
from latentmol.design.protenix_restart_selection import audit_molecule, select_candidates
from latentmol.design.protenix_restart_study import (
    matched_parent_shortlists, paired_budget_readout, score_shortlists,
)
from latentmol.design.protenix_restarts import Candidate, fresh_score_records, native_factory, restart_args, write_json
from latentmol.design.protenix_search_branches import Branch, execute_branches
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.representation import HardGraphBatch


@dataclass(frozen=True, kw_only=True)
class PairedBranch(Branch):
    arm: str
    parent_index: int


def budget(config, arm):
    waves = config["tasks"][arm]["waves"]
    updates, fresh = config["design"]["updates"], config["fresh_per_arm"]
    if (config["study"] != "near_valid" or not waves
            or any(type(wave) is not int or wave < 2 for wave in waves)
            or len(set(waves)) != len(waves)
            or type(updates) is not int or not 1 <= updates <= 500
            or type(fresh) is not int or not 1 <= fresh <= 16
            or len(waves) * 2 * updates > 10_000):
        raise ValueError("Invalid near-valid replay budget or wave selection")
    return dict(updates=2 * len(waves) * updates, fresh_attempts=2 * len(waves) * fresh,
                baseline_attempts=2 * len(waves))


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_point(decoder, saved, occupied_atoms):
    spec = decoder.spec
    labels = saved["labels"]
    hard = HardGraphBatch(*(torch.as_tensor(labels[name]).unsqueeze(0)
                            for name in ("atoms", "charges", "bonds")))
    for values, size, shape in (
        (hard.atoms, len(spec.atomic_numbers), (1, spec.max_nodes)),
        (hard.charges, len(spec.formal_charges), (1, spec.max_nodes)),
        (hard.bonds, len(spec.bond_names), (1, spec.max_nodes, spec.max_nodes)),
    ):
        if values.dtype != torch.long or values.shape != shape or (values < 0).any() or (values >= size).any():
            raise ValueError("Malformed categorical labels in saved restart point")
    latent = torch.as_tensor(saved["latent"], dtype=torch.float32)
    if latent.shape != (1, decoder.config.latent) or not latent.isfinite().all():
        raise ValueError("Saved restart latent has the wrong shape or nonfinite coordinates")
    slots = saved["active_slots"]
    if (len(slots) != occupied_atoms or len(set(slots)) != occupied_atoms
            or any(type(slot) is not int or not 0 <= slot < spec.max_nodes for slot in slots)):
        raise ValueError("Saved restart mask differs from the source design")
    active = torch.zeros(spec.max_nodes, dtype=torch.bool)
    active[slots] = True
    checked = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], spec)
    if not torch.equal(hard.atoms[0] != 0, active) or checked.smiles != saved["smiles"]:
        raise ValueError("Saved graph labels disagree with the recorded mask or molecule")
    if not math.isfinite(saved["relaxed_loss"]):
        raise ValueError("Saved restart point has nonfinite relaxed loss")
    candidate = Candidate(saved["smiles"], latent, hard, saved["relaxed_loss"], saved["branch"], saved["step"])
    return SearchPoint(candidate, active, saved["fresh_loss"], tuple(saved["ancestry"]),
                       saved["repair_edits"], saved["repair_smiles"], saved["repair_target_sha256"])


def load_pairs(decoder, source, config, arm, checkpoint_sha256):
    run = _read(source / "run.json")
    if (run["runner"] != "restart_search" or str(run["job_id"]) != str(config["source_job_id"])
            or run["checkpoint_sha256"] != checkpoint_sha256
            or run["decoder_sha256"] != _model_identity(decoder)
            or run["graph_spec"] != decoder.spec.to_dict()):
        raise ValueError("Source restart job, checkpoint, decoder, or graph schema differs")
    design = replace(LatentDesignConfig(**run["design"]), updates=config["design"]["updates"])
    search = run["search"]
    repair_limit = search["repair_edit_limit"]
    if (design.noise_initial != 0 or search["children"] < 2
            or search["invalid_children"] != 1 or search["exact_children"] < 1):
        raise ValueError("Require a deterministic source with the first exact slot eligible for invalid replacement")
    branches, pairs = [], []
    for index, wave in enumerate(config["tasks"][arm]["waves"]):
        phase = source / f"cycle-{wave:02d}"
        plan, archive = _read(phase / "wave-plan.json"), _read(phase / "archive.json")
        if (plan["wave"] != wave or archive["cycle"] != wave
                or not plan["invalid_selected"] or not archive["invalid_selected"]):
            raise ValueError(f"Source wave {wave} did not complete an invalid restart")
        original, sibling = plan["branches"][:2]
        exact = archive["restart_parents"][0]
        if (original["name"] != "branch-01" or original["policy"] != "raw-invalid"
                or sibling["name"] != "branch-02" or sibling["source"] != exact):
            raise ValueError(f"Cannot verify the exact continuation replaced in wave {wave}")
        invalid, valid = (_load_point(decoder, saved, design.occupied_atoms) for saved in (original["source"], exact))
        if (invalid.candidate.smiles is not None or invalid.fresh_loss is not None
                or not invalid.repair_smiles or type(invalid.repair_edits) is not int
                or not 1 <= invalid.repair_edits <= repair_limit or not invalid.repair_target_sha256
                or not valid.candidate.smiles or valid.fresh_loss is None or not math.isfinite(valid.fresh_loss)):
            raise ValueError(f"Missing valid continuation or bounded repair witness in wave {wave}")
        for name, policy, point in (("near-valid", "raw-invalid", invalid), ("exact-valid", "none", valid)):
            branches.append(PairedBranch(f"parent-{index:02d}-{name}-00", point, original["seed"],
                policy=policy, noise_seed=design.noise_seed + len(branches), arm=name, parent_index=index))
        pairs.append(dict(source_wave=wave, replaced_branch=original["name"], witness_smiles=invalid.repair_smiles,
            repair_edits=invalid.repair_edits, repair_target_sha256=invalid.repair_target_sha256,
            invalid_source=dict(branch=invalid.candidate.branch, step=invalid.candidate.step),
            exact_source=dict(branch=valid.candidate.branch, step=valid.candidate.step,
                smiles=valid.candidate.smiles, archived_fresh_loss=valid.fresh_loss)))
    return design, repair_limit, branches, pairs


def run_paired(decoder, scorer_factory, fresh_score, config, arm, output, checkpoint_sha256):
    declared = budget(config, arm)
    source = Path(config["source_run"]).resolve()
    output = Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("Near-valid replay output must be separate from the live source")
    decoder.eval().requires_grad_(False)
    identity = _model_identity(decoder)
    if any(parameter.grad is not None for parameter in decoder.parameters()):
        raise ValueError("Frozen decoder carries preexisting gradients")
    design, repair_limit, branches, pairs = load_pairs(decoder, source, config, arm, checkpoint_sha256)
    output.mkdir(parents=True)
    write_json(output / "run.json", dict(arm=arm, budget=declared, design=asdict(design), pairs=pairs,
        source_run=str(source), source_job_id=config["source_job_id"], decoder_sha256=identity,
        checkpoint_sha256=checkpoint_sha256, job_id=os.environ.get("LSB_JOBID"),
        graph_spec=decoder.spec.to_dict()))
    paired = output / "paired"
    paired.mkdir()
    scorer = scorer_factory(decoder.spec, branches[0].parent.active, paired)
    try:
        results = execute_branches(decoder, scorer, design, paired, branches, batch_size=1,
            cpu_workers=1, overlap=False, repair_limit=repair_limit, invalid_limit=0,
            ready=os.environ.get("DBMOL_LAUNCH_READY"))
    finally:
        scorer.close()
    baseline_root = output / "baselines"
    baseline_root.mkdir()
    baselines = {}
    baseline_records = {}
    for index, pair in enumerate(pairs):
        for role, smiles, step in (
            ("witness", pair["witness_smiles"], pair["invalid_source"]["step"]),
            ("exact-start", pair["exact_source"]["smiles"], pair["exact_source"]["step"]),
        ):
            baseline_records[f"parent-{index:02d}-{role}"] = {"mixed": [dict(smiles=smiles, step=step)]}
    baseline_attempts = score_shortlists(fresh_score, baseline_root, baseline_records, baselines)

    limit = config["fresh_per_arm"]
    caches = {arm: dict(baselines) for arm in ("near-valid", "exact-valid")}
    reserves = {}
    for result in results:
        ranked = [candidate for candidate in result.valid if candidate.smiles not in baselines]
        preferred = select_candidates(ranked, limit, False)["mixed"]
        chosen = {candidate.smiles for candidate in preferred}
        reserves[result.branch.name] = {"mixed": preferred + [row for row in ranked if row.smiles not in chosen]}
    selected = matched_parent_shortlists(branches, reserves, caches, limit)
    attempts_by_arm = {}
    for name, cache in caches.items():
        root = paired / f"fresh-{name}"
        root.mkdir()
        records = {branch.name: {"mixed": [
            dict(smiles=row.smiles, step=row.step, relaxed_loss=row.relaxed_loss)
            for row in selected[branch.name]["mixed"]]} for branch in branches if branch.arm == name}
        attempts_by_arm[name] = score_shortlists(fresh_score, root, records, cache)
    readouts = []
    for result in results:
        branch = result.branch
        candidates = [dict(smiles=row.smiles, step=row.step, relaxed_loss=row.relaxed_loss,
            **caches[branch.arm][row.smiles]) for row in selected[branch.name]["mixed"]]
        readouts.append(dict(name=branch.name, arm=branch.arm, parent_index=branch.parent_index,
            summary=result.summary, candidates={"mixed": candidates},
            best_new_fresh_loss=min((row["loss"] for row in candidates if row["loss"] is not None), default=None)))
    matched = paired_budget_readout(readouts, updates_per_parent=design.updates, attempts_per_parent=limit)
    if (sum(row["updates"] for row in matched["by_arm"].values()) != declared["updates"]
            or attempts_by_arm != {name: row["fresh_attempts"] for name, row in matched["by_arm"].items()}):
        raise ValueError("Near-valid replay did not meet its matched budget")
    for index, pair in enumerate(pairs):
        pair["fresh_baselines"] = dict(witness=baselines[pair["witness_smiles"]],
                                      exact_start=baselines[pair["exact_source"]["smiles"]])
        pair["arms"] = {row["arm"]: row for row in readouts if row["parent_index"] == index}
    if _model_identity(decoder) != identity or any(parameter.grad is not None for parameter in decoder.parameters()):
        raise ValueError("Frozen decoder parameters or gradients changed")
    write_json(paired / "batch.json", dict(branches=readouts, native_identity=scorer.identity,
        optimization_updates=declared["updates"], new_fresh_attempts=sum(attempts_by_arm.values())))
    result = dict(status="completed", budget=declared, matched_budget=matched, baseline_attempts=baseline_attempts,
                  descendant_attempts=sum(attempts_by_arm.values()), pairs=pairs, frozen_decoder_unchanged=True)
    write_json(output / "result.json", result)
    return result


def main(argv=None):
    args = restart_args(__doc__, argv)
    config = _read(args.config)
    if (file_sha256(args.plan) != args.plan_sha256
            or budget(config, args.arm)["updates"] > _read(args.plan)["planned_updates"][args.arm]):
        raise ValueError("Near-valid replay exceeds or differs from its staged launch plan")
    per_call_limit = max(len(config["tasks"][args.arm]["waves"]) * config["fresh_per_arm"],
                         budget(config, args.arm)["baseline_attempts"])
    if per_call_limit > _read(args.fresh_config)["candidate_limit"]:
        raise ValueError("Near-valid replay exceeds the fresh-score call limit")
    decoder, _ = load_model(args.checkpoint, "cpu")

    def audited_score(root, records):
        return fresh_score_records(args, root, [row | {"chemistry_audit": audit_molecule(row["smiles"])}
                                                for row in records])

    return run_paired(decoder, native_factory(args), audited_score,
                      config, args.arm, args.output, args.checkpoint_sha256)


if __name__ == "__main__":
    main()
