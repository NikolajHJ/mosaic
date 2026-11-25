"""Matched short Protenix branches with opt-in frozen synthesis guidance."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
from time import perf_counter
from types import SimpleNamespace

import torch

from latentmol.chemistry import inspect_graph
from latentmol.design.latent_protenix import LatentDesignConfig, _model_identity, hard_from_graph
from latentmol.design.protenix_restart_selection import max_pairwise_similarity, select_candidates
from latentmol.design.protenix_restart_study import matched_parent_shortlists, score_shortlists
from latentmol.design.protenix_restarts import Candidate, write_json
from latentmol.design.protenix_search_branches import Branch, execute_branches
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.experiments import file_sha256
from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance.synthesis import SynthesisGuidance, load_models
from latentmol.representation import HardGraphBatch


def pilot_variants(config, task):
    """Choose related comparisons from one shared config without copied runners."""
    variants = config.get("variants", {})
    selected = config["tasks"][task].get("variants", list(variants))
    if (not isinstance(selected, list) or len(selected) != len(set(selected))
            or set(selected) - variants.keys()):
        raise ValueError("Unknown or duplicate synthesis task variant")
    return {name: variants[name] for name in selected}


def pilot_assets(config, task):
    assets = config.get("synthesis_assets", {})
    names = config["tasks"][task].get("models", list(assets))
    if len(names) != len(set(names)) or set(names) - assets.keys():
        raise ValueError("Unknown or duplicate task synthesis model")
    return {name: assets[name] for name in names}


def resolved_variant(variant, wave=1):
    variant = dict(variant)
    if schedule := variant.pop("wave_schedule", None):
        if not 1 <= wave <= len(schedule):
            raise ValueError("Synthesis wave schedule is too short")
        variant.update(schedule[wave - 1])
    return variant


def guidance_factory(models, variant, anchor=None, wave=1):
    variant = resolved_variant(variant, wave)
    ratios = variant.get("ratios", {})
    if isinstance(anchor, dict) and variant.get("anchor_ratio", 0):
        anchor = anchor[variant.get("anchor_library", "low_sa")]
    bank = ({"model": anchor, "ratio": variant["anchor_ratio"]}
            if variant.get("anchor_ratio", 0) else None)
    if bank and anchor is None:
        raise ValueError("Latent-anchor variant is missing its library")
    if not any(ratios.values()) and bank is None:
        return None
    kwargs = {key: variant[key] for key in (
        "controller", "schedule", "sharpness", "project_binding") if key in variant}
    return lambda _branch: SynthesisGuidance(models, ratios,
        delay_updates=variant.get("delay_updates", 0), **kwargs, **({"anchor": bank} if bank else {}))


def close_models(models):
    for model in models.values():
        if callable(getattr(model, "close", None)):
            model.close()


def load_anchor_libraries(config, parents, checkpoint_sha256):
    """Load only declared frozen priors, calibrated once on the shared founders."""
    from latentmol.guidance.synthesis_anchor import LatentLibraryCritic

    if libraries := config.get("anchor_libraries"):
        slots = torch.where(parents[0].active)[0].tolist()
        if any(not torch.equal(p.active, parents[0].active) for p in parents):
            raise ValueError("Anchor calibration requires a shared founder slot mask")
        references = torch.cat([p.candidate.latent.detach().cpu() for p in parents])
        return {name: LatentLibraryCritic.from_asset(asset, checkpoint_sha256=checkpoint_sha256,
                    active_slots=slots, reference_points=references)
                for name, asset in libraries.items()}
    if not config.get("anchor_bank"):
        return None
    anchor = LatentLibraryCritic.from_asset(config["anchor_bank"], checkpoint_sha256=checkpoint_sha256)
    if config.get("anchor_control_bank"):
        return dict(low_sa=anchor, unfiltered=LatentLibraryCritic.from_asset(
            config["anchor_control_bank"], checkpoint_sha256=checkpoint_sha256))
    return anchor


def _select_audited_identities(scored_smiles, limit, *, similarity, sa_ceilings):
    """Greedy relaxed/fresh loss rank within successive audited SA tiers."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from latentmol.evaluation.synthesis_accessibility import audited_sa_score

    # The historical founder and archive-screen panels used non-chiral Morgan-2.
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    eligible = []
    for smiles, loss in scored_smiles.items():
        if not math.isfinite(loss):
            raise ValueError("Nonfinite binding loss in synthesis candidate ranking")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError("Expected a valid, connected hard molecule in the synthesis archive")
        sa = audited_sa_score(mol)
        tier = next((i for i, ceiling in enumerate(sa_ceilings) if sa <= ceiling), None)
        if tier is not None:
            eligible.append((tier, loss, smiles, generator.GetFingerprint(mol)))
    selected, fingerprints = [], []
    for _, _, smiles, fingerprint in sorted(eligible, key=lambda row: row[:3]):
        if any(value > similarity for value in DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints)):
            continue
        selected.append(smiles)
        fingerprints.append(fingerprint)
        if len(selected) == limit:
            break
    return selected


def select_near_miss_founders(points, count):
    """Select distinct, scored, replayable panel rows for a BRD4 exploitation run."""
    if type(count) is not int or count < 1:
        raise ValueError("Expected a positive founder count")
    by_smiles = {}
    for row in points:
        loss = row["fresh_loss"]
        if loss is None:
            continue
        if not isinstance(loss, (int, float)) or isinstance(loss, bool) or not math.isfinite(loss):
            raise ValueError("Nonfinite or malformed fresh founder loss")
        if row["smiles"]:
            existing = by_smiles.get(row["smiles"])
            if existing is None or loss < existing["fresh_loss"]:
                by_smiles[row["smiles"]] = row
    by_smiles = {s: r for s, r in by_smiles.items() if 2.3 < r["fresh_loss"] <= 2.4
                 and all(k in r for k in ("latent", "labels", "active_slots"))}
    chosen = _select_audited_identities({s: p["fresh_loss"] for s, p in by_smiles.items()},
                                         count, similarity=.60, sa_ceilings=(5.,))
    if len(chosen) != count:
        raise ValueError(f"Only {len(chosen)} of {count} distinct near-miss founders available. "
                         "Supply more freshly scored replayable parents or lower parent_count.")
    return [by_smiles[smiles] for smiles in chosen]


def synthesis_shortlist(candidates, limit, policy="binding", sa_mode="rdkit"):
    """Allocate fresh attempts using the configured hard-molecule shortlist."""
    if policy == "audited-binding-diverse":
        if sa_mode != "pre_smoothing":
            raise ValueError("The audited shortlist requires pre_smoothing SA")
        by_smiles = {}
        for candidate in sorted(candidates, key=lambda c: (c.relaxed_loss, c.step, c.smiles)):
            by_smiles.setdefault(candidate.smiles, candidate)
        chosen = _select_audited_identities({s: c.relaxed_loss for s, c in by_smiles.items()},
                                             limit, similarity=.65, sa_ceilings=(5., 6.))
        return [by_smiles[smiles] for smiles in chosen]
    if policy not in {"binding", "binding-sa"}:
        raise ValueError("Unknown synthesis shortlist policy")
    standard = select_candidates(candidates, limit, False)["mixed"]
    if policy == "binding" or len(standard) <= 1:
        return standard
    binding_count = (limit + 1) // 2
    chosen = standard[:binding_count]
    seen = {c.smiles for c in chosen}
    chemistry = {c.smiles: molecule_readout(c.smiles, sa_mode) if sa_mode != "rdkit"
                 else molecule_readout(c.smiles) for c in candidates}
    easier = sorted((c for c in candidates if c.smiles not in seen),
                    key=lambda c: (chemistry[c.smiles].get("sa_score", float("inf")),
                                   c.relaxed_loss, c.step, c.smiles))
    return chosen + easier[:max(0, limit - len(chosen))]


def joint_quality_metrics(records, exclude=()):
    """Detached, distinct, successful-score yield at declared comparison cutoffs."""
    from rdkit import DataStructs
    from latentmol.design.protenix_restart_selection import _morgan_fingerprints

    unique = {}
    for row in records:
        if row.get("loss") is None or row.get("smiles") in exclude or not row.get("chemistry", {}).get("valid"):
            continue
        old = unique.get(row["smiles"])
        if old is None or row["loss"] < old["loss"]:
            unique[row["smiles"]] = row
    fingerprints = dict(_morgan_fingerprints(list(unique))) if unique else {}
    result = {}
    for sa in (4., 5., 6.):
        for loss in (2.2, 2.3, 2.4):
            good = sorted((r for r in unique.values() if r["chemistry"]["sa_score"] <= sa and r["loss"] <= loss),
                          key=lambda r: (r["loss"], r["smiles"]))
            selected = []
            for row in good:
                fp = fingerprints[row["smiles"]]
                if not selected or max(DataStructs.BulkTanimotoSimilarity(fp, selected)) <= .6:
                    selected.append(fp)
            result[f"sa<={sa:g},loss<={loss:g}"] = dict(distinct=len(good), diverse=len(selected),
                best_sa=min((r["chemistry"]["sa_score"] for r in good), default=None),
                best_loss=min((r["loss"] for r in good), default=None))
    return result


def resolve_pilot(config, task):
    """Validate a bounded panel without loading models or remote assets."""
    if config.get("study") != "synthesis_pilot" or task not in config.get("tasks", {}):
        raise ValueError("Expected a synthesis_pilot config and known task")
    if "tacogfn" in config:
        from latentmol.design.protenix_tacogfn_study import resolve_pair
        return resolve_pair(config, task)
    design = LatentDesignConfig.for_restart(config["design"])
    parents = config.get("parent_count", 3)
    fresh = config.get("fresh_per_parent", 6)
    variants = pilot_variants(config, task)
    waves = config.get("waves", 1)
    seed = config["tasks"][task].get("seed")
    near_miss = config.get("founder_policy") == "audited-near-miss"
    if config.get("founder_policy") not in {None, "audited-near-miss"} or (
            near_miss and (config.get("founder_groups") is not None or type(waves) is not int or waves < 2
                          or config.get("sa_mode") != "pre_smoothing")):
        raise ValueError("Audited near-miss founders require a shared multi-wave pre_smoothing panel")
    if (type(parents) is not int or not 1 <= parents <= 8 or type(fresh) is not int
            or not 1 <= fresh <= 16 or not (1 if near_miss else 2) <= len(variants) <= 12
            or type(waves) is not int or not 1 <= waves <= 8
            or type(seed) is not int or not 0 <= seed < 2**63 - 10000
            or design.updates > 1000 or design.noise_initial != 0
            or config.get("restart_policy", "latent") not in {"latent", "none"}):
        raise ValueError("Invalid bounded synthesis pilot budget, seed or restart policy")
    scoring_policy = config.get("fresh_scoring_policy", "equal_allowance" if waves > 1 else "matched_minimum")
    if scoring_policy not in {"equal_allowance", "matched_minimum"} or (waves > 1 and scoring_policy != "equal_allowance"):
        raise ValueError("Invalid synthesis fresh scoring policy")
    assets = pilot_assets(config, task)
    if not assets or set(assets) - {"synfrag", "dfr", "gasa", "defog"}:
        raise ValueError("Unknown frozen synthesis/prior model")
    for asset in [config.get("parent_panel", {}), *assets.values(),
                  *([config["anchor_bank"]] if config.get("anchor_bank") else []),
                  *([config["anchor_control_bank"]] if config.get("anchor_control_bank") else []),
                  *config.get("anchor_libraries", {}).values()]:
        if (not isinstance(asset.get("path"), str) or not asset["path"]
                or not re.fullmatch(r"[0-9a-f]{64}", asset.get("sha256", ""))):
            raise ValueError("Pilot inputs require paths and SHA-256 identities")
    if not any(not row.get("ratios") and not row.get("anchor_ratio") for row in variants.values()) and not config.get("guided_controls", False):
        raise ValueError("Include an unguided baseline or explicitly paired guided controls")
    for name, variant in variants.items():
        independent = variant.get("independent_final_slot", False)
        if type(independent) is not bool or (independent and (
                waves < 2 or config.get("recovery_wave", waves) != waves or
                resolved_variant(variant, waves).get("anchor_ratio", 0))):
            raise ValueError("The independent final slot requires a final recovery wave without slot-aligned attraction")
        for wave in range(1, waves + 1):
            row = resolved_variant(variant, wave)
            if row.get("independent_final_slot", False) is not independent:
                raise ValueError("Specify independent_final_slot at the variant level")
            ratios = row.get("ratios", {})
            delay = row.get("delay_updates", 0)
            if (not re.fullmatch(r"[a-z][a-z0-9-]*", name) or not isinstance(ratios, dict)
                    or set(ratios) - assets.keys() or type(delay) is not int
                    or not 0 <= delay < design.updates
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) or v < 0 for v in ratios.values())):
                raise ValueError(f"Invalid synthesis variant: {name}")
            if row.get("anchor_ratio", 0) and config.get("anchor_libraries"):
                if row.get("anchor_library", "low_sa") not in config["anchor_libraries"]:
                    raise ValueError(f"Unknown anchor library for {name}")
            shortlist = row.get("shortlist_policy", config.get("shortlist_policy", "binding"))
            if shortlist not in {"binding", "binding-sa", "audited-binding-diverse"} or (
                    shortlist == "audited-binding-diverse" and config.get("sa_mode") != "pre_smoothing"):
                raise ValueError("Unknown synthesis shortlist policy")
    groups = config.get("founder_groups")
    if groups is not None:
        if (waves < 2 or not isinstance(groups, dict) or set(groups) != set(variants)
                or any(not isinstance(indices, list) or len(indices) != parents
                       or any(type(i) is not int for i in indices) for indices in groups.values())
                or sorted(i for indices in groups.values() for i in indices) != list(range(parents * len(groups)))):
            raise ValueError("Founder comparison needs disjoint complete panels of equal size")
    if config.get("sa_mode", "rdkit") not in {"rdkit", "pre_smoothing"} or (
            config.get("sa_mode", "rdkit") != "rdkit" and waves == 1):
        raise ValueError("Unsupported synthesis SA mode for this run")
    branches = parents * len(variants) * waves
    updates = branches * design.updates
    if updates > 32_000:
        raise ValueError("A synthesis pilot panel exceeds 32,000 updates")
    founders_per_arm = parents if near_miss or groups else 0
    founder_attempts = founders_per_arm * (len(variants) if groups else 1)
    budget = dict(updates=updates, branches=branches, relaxed_calls=updates + branches,
                  updates_per_arm=parents * waves * design.updates,
                  fresh_attempts=branches * fresh + founder_attempts,
                  fresh_attempts_per_arm=parents * waves * fresh + founders_per_arm)
    if "confirm_top_hits" in config:
        cap = config["confirm_top_hits"]
        if not near_miss or type(cap) is not int or not 0 <= cap <= 16:
            raise ValueError("Expected 0–16 confirmation candidates per audited near-miss arm")
        budget["confirmation_fresh_attempts"] = cap * 2 * len(variants)
    return design, budget


@torch.no_grad()
def load_pilot_parents(decoder, config, design, checkpoint_sha256):
    """Replay detached archived latents; never re-encode or repair their graphs."""
    panel = config["parent_panel"]
    path = Path(panel["path"])
    if file_sha256(path) != panel["sha256"]:
        raise ValueError("Synthesis parent panel checksum changed")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("version") != 1 or data.get("checkpoint_sha256") != checkpoint_sha256
            or data.get("decoder_sha256") != _model_identity(decoder)
            or data.get("graph_spec") != decoder.spec.to_dict()):
        raise ValueError("Synthesis parent panel belongs to another frozen model/schema")
    count = config.get("parent_count", 3) * (len(config.get("founder_groups") or {}) or 1)
    rows = data.get("points", [])
    if config.get("founder_policy") == "audited-near-miss":
        rows = select_near_miss_founders(rows, count)
    if len(rows) < count:
        raise ValueError("Synthesis parent panel has too few points")
    result, records, seen = [], [], set()
    spec = decoder.spec
    n = spec.max_nodes
    for row in rows[:count]:
        latent = torch.tensor(row["latent"], dtype=torch.float32)
        slots = row["active_slots"]
        if (latent.shape != (1, decoder.config.latent) or not bool(latent.isfinite().all())
                or len(slots) != design.occupied_atoms or len(set(slots)) != len(slots)
                or any(type(i) is not int or not 0 <= i < n for i in slots)
                or not isinstance(row["smiles"], str) or row["smiles"] in seen
                or (not (config.get("founder_groups") and row["fresh_loss"] is None)
                    and (not isinstance(row["fresh_loss"], (int, float)) or not math.isfinite(row["fresh_loss"])))):
            raise ValueError("Malformed or duplicate synthesis founder")
        tensors = [torch.as_tensor(row["labels"][key]) for key in ("atoms", "charges", "bonds")]
        if any(value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
               for value in tensors):
            raise ValueError("Synthesis founder hard labels must be integers")
        hard = HardGraphBatch(*(value.long()[None] for value in tensors))
        if (hard.atoms.shape != (1, n) or hard.charges.shape != (1, n)
                or hard.bonds.shape != (1, n, n)):
            raise ValueError("Synthesis founder hard-label shapes differ")
        for value, categories in ((hard.atoms, len(spec.atomic_numbers)),
                                   (hard.charges, len(spec.formal_charges)),
                                   (hard.bonds, len(spec.bond_names))):
            if bool(((value < 0) | (value >= categories)).any()):
                raise ValueError("Synthesis founder category outside graph vocabulary")
        active = torch.zeros(n, dtype=torch.bool)
        active[slots] = True
        if not torch.equal(hard.atoms[0] != 0, active):
            raise ValueError("Synthesis founder occupancy differs from active slots")
        graph = graph_from_decoder(decoder.decode(latent), spec, fixed_active=active,
                                   temperature=design.temperature_initial)
        if any(not bool(getattr(graph, key).isfinite().all()) for key in ("atoms", "charges", "bonds")):
            raise ValueError("Nonfinite replayed synthesis founder")
        replay = hard_from_graph(graph)
        checked = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], spec)
        if (checked.smiles != row["smiles"]
                or any(not torch.equal(getattr(hard, key), getattr(replay, key))
                       for key in ("atoms", "charges", "bonds"))):
            raise ValueError("Synthesis founder latent does not replay its scored graph")
        candidate = Candidate(row["smiles"], latent, hard, row["relaxed_loss"], row["branch"], row["step"])
        result.append(SearchPoint(candidate, active, row["fresh_loss"], tuple(row.get("ancestry", ()))))
        records.append(row)
        seen.add(row["smiles"])
    return result, records


def molecule_readout(smiles, sa_mode="rdkit"):
    """Independent hard-chemistry measurements, strictly outside autograd."""
    if smiles is None:
        return dict(valid=False, failure="invalid_terminal_graph")
    from rdkit import Chem
    from rdkit.Chem import QED, rdMolDescriptors
    from rdkit.Contrib.SA_Score import sascorer

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return dict(valid=False, failure="invalid_or_disconnected_smiles")
    sa = float(sascorer.calculateScore(mol))
    extra = {}
    if sa_mode == "pre_smoothing":
        from latentmol.evaluation.synthesis_accessibility import audited_sa_score
        extra = dict(rdkit_sa_score=sa, sa_mode=sa_mode)
        sa = audited_sa_score(mol)
    elif sa_mode != "rdkit":
        raise ValueError("Unknown SA readout mode")
    return dict(valid=True, sa_score=sa, **extra, qed=float(QED.qed(mol)),
                rings=int(rdMolDescriptors.CalcNumRings(mol)), heavy_atoms=mol.GetNumHeavyAtoms(),
                formal_charge=Chem.GetFormalCharge(mol))


@torch.no_grad()
def _candidate_readout(candidate, active, models, spec, sa_mode="rdkit"):
    row = dict(smiles=candidate.smiles, step=candidate.step, branch=candidate.branch,
               relaxed_loss=candidate.relaxed_loss, chemistry=molecule_readout(candidate.smiles, sa_mode))
    if not row["chemistry"]["valid"]:
        return row
    graph = candidate.hard.to_relaxed(spec)
    row["hard_adapter_scores"] = {}
    row["native_hard_scores"] = {}
    from rdkit import Chem
    mol = Chem.MolFromSmiles(candidate.smiles)
    for name, model in models.items():
        score = model(graph, active)
        fields = {key: float(score[key].reshape(-1)[0]) for key in ("difficulty", "raw_score")}
        if any(not math.isfinite(v) for v in fields.values()):
            raise ValueError("Nonfinite posthoc synthesis score")
        row["hard_adapter_scores"][name] = fields
        if hasattr(model, "reference_score"):
            native = model.reference_score(mol)
            if isinstance(native, dict):
                native_fields = {key: float(torch.as_tensor(native[key]).reshape(-1)[0])
                                 for key in ("difficulty", "raw_score")}
            else:
                difficulty = float(native)
                native_fields = dict(difficulty=difficulty,
                                     raw_score=1 - difficulty if name == "synfrag" else difficulty)
            if any(not math.isfinite(v) for v in native_fields.values()):
                raise ValueError("Nonfinite native-feature hard synthesis score")
            row["native_hard_scores"][name] = native_fields
    return row


def run_pilot(decoder, make_scorer, fresh_score, config, task, output, checkpoint_sha256):
    """Optimize matched starts with each guide; match new fresh attempts by parent."""
    if "tacogfn" in config:
        from latentmol.design.protenix_tacogfn_study import run_pair
        return run_pair(decoder, make_scorer, fresh_score, config, task, output, checkpoint_sha256)
    if config.get("waves", 1) > 1:
        from latentmol.design.protenix_synthesis_waves import run_waves
        return run_waves(decoder, make_scorer, fresh_score, config, task, output, checkpoint_sha256)
    started = perf_counter()
    design, budget = resolve_pilot(config, task)
    scoring_policy = config.get("fresh_scoring_policy", "matched_minimum")
    decoder.eval().requires_grad_(False)
    if any(p.grad is not None for p in decoder.parameters()):
        raise ValueError("Frozen decoder carries preexisting gradients")
    identity = _model_identity(decoder)
    parents, parent_records = load_pilot_parents(decoder, config, design, checkpoint_sha256)
    models = load_models(decoder.spec, pilot_assets(config, task))
    try:
        anchor = load_anchor_libraries(config, parents, checkpoint_sha256)
        anchor_identities = ({name: _model_identity(model) for name, model in
                              (anchor.items() if isinstance(anchor, dict) else {"low_sa": anchor}.items())}
                             if anchor is not None else {})
        model_identities = {name: _model_identity(model) for name, model in models.items()}
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        root = output / "cycle-01"
        root.mkdir()
        variants = pilot_variants(config, task)
        seed = config["tasks"][task]["seed"]
        limit = config.get("fresh_per_parent", 6)
        write_json(output / "run.json", dict(runner="synthesis_pilot", task=task, budget=budget,
            design=asdict(design), variants=variants, parents=parent_records,
            checkpoint_sha256=checkpoint_sha256, decoder_sha256=identity,
            graph_spec=decoder.spec.to_dict(), synthesis_assets=pilot_assets(config, task),
            fresh_scoring_policy=scoring_policy,
            synthesis_model_identities=model_identities, seed=seed, job_id=os.environ.get("LSB_JOBID"),
            anchor_libraries=config.get("anchor_libraries"), anchor_identities=anchor_identities,
            anchor_metadata=({k: v.adapter_metadata for k, v in anchor.items()}
                             if isinstance(anchor, dict) else None),
            synthesis_adapters={name: getattr(model, "adapter_metadata", {
                "feature_policy": getattr(model, "feature_policy", "unspecified"),
                "scope": "Frozen weights; approximate continuous extension of hard graph features"})
                               for name, model in models.items()},
            guidance_scope="Frozen-weight experimental relaxed-graph extension; no predictor training",
            hard_score_scope="Released weights with detached reference features; exact original-runtime parity not established"))
        founder_smiles = {parent.candidate.smiles for parent in parents}
        # A first fresh rescore of a founder is charged to each arm's real budget.
        # Excluding historical founders can reduce every arm to zero fresh attempts
        # when one guided trajectory retains only its starting identity.
        caches = {arm: {} for arm in variants}
        reserves, results, grouping, optimization_seconds = {}, {}, [], {}
        scorer_started = perf_counter()
        scorer = make_scorer(decoder.spec, parents[0].active, root)
        scorer_seconds = perf_counter() - scorer_started
        native_identity = getattr(scorer, "identity", None)
        try:
            for arm_index, (arm, variant) in enumerate(variants.items()):
                arm_started = perf_counter()
                branches = [Branch(f"parent-{index:02d}-{arm}-00", parent, seed + 100 * index,
                                   policy=config.get("restart_policy", "latent"))
                            for index, parent in enumerate(parents)]
                factory = guidance_factory(models, variant, anchor)
                rows = execute_branches(decoder, scorer, design, root, branches, batch_size=1,
                    cpu_workers=1, overlap=False, repair_limit=0, invalid_limit=0,
                    guidance_factory=factory,
                    ready=os.environ.get("DBMOL_LAUNCH_READY") if arm_index == 0 else None)
                if len(rows) != len(parents) or any(row.summary["updates"] != design.updates for row in rows):
                    raise ValueError("Mismatched actual synthesis optimization budgets")
                results[arm] = rows
                for index, row in enumerate(rows):
                    eligible = [candidate for candidate in row.valid if candidate.smiles not in caches[arm]]
                    preferred = synthesis_shortlist(eligible, limit,
                        variant.get("shortlist_policy", config.get("shortlist_policy", "binding")))
                    chosen = {candidate.smiles for candidate in preferred}
                    reserves[row.branch.name] = {"mixed": preferred + [c for c in eligible if c.smiles not in chosen]}
                    grouping.append(SimpleNamespace(name=row.branch.name, arm=arm, parent_index=index))
                optimization_seconds[arm] = perf_counter() - arm_started
                print(json.dumps(dict(event="synthesis_arm_optimized", arm=arm,
                                      updates=len(rows) * design.updates, seconds=optimization_seconds[arm])), flush=True)
        finally:
            scorer.close()
        if scoring_policy == "equal_allowance":
            selections = {}
            seen = {arm: set(cache) for arm, cache in caches.items()}
            for branch in grouping:
                choices = []
                for candidate in reserves[branch.name]["mixed"]:
                    if candidate.smiles not in seen[branch.arm]:
                        choices.append(candidate)
                        seen[branch.arm].add(candidate.smiles)
                        if len(choices) == limit:
                            break
                selections[branch.name] = {"mixed": choices}
        else:
            selections = matched_parent_shortlists(grouping, reserves, caches, limit)
        planned = {name: {"mixed": [dict(smiles=c.smiles, step=c.step, relaxed_loss=c.relaxed_loss)
                                    for c in choices["mixed"]]} for name, choices in selections.items()}
        write_json(root / "shortlists.json", planned)
        arms, branch_records = {}, []
        for arm, rows in results.items():
            fresh_root = root / f"fresh-{arm}"
            fresh_root.mkdir()
            fresh_started = perf_counter()
            attempts = score_shortlists(fresh_score, fresh_root,
                {row.branch.name: planned[row.branch.name] for row in rows}, caches[arm])
            if attempts > len(parents) * limit:
                raise ValueError("Synthesis pilot exceeded its fresh scoring allowance")
            fresh_seconds = perf_counter() - fresh_started
            selected_rows = []
            for index, row in enumerate(rows):
                selected = selections[row.branch.name]["mixed"]
                candidates = []
                for candidate in selected:
                    data = _candidate_readout(candidate, row.branch.parent.active, models, decoder.spec)
                    data.update(caches[arm][candidate.smiles])
                    if (data["status"] == "scored") != (data["loss"] is not None):
                        raise ValueError("Inconsistent fresh synthesis score/status")
                    data["is_founder"] = candidate.smiles in founder_smiles
                    candidates.append(data)
                endpoint = (_candidate_readout(row.terminal, row.branch.parent.active, models, decoder.spec)
                            if row.terminal is not None else None)
                selected_rows.extend(candidates)
                branch_records.append(dict(name=row.branch.name, arm=arm, parent_index=index,
                    seed=row.branch.seed, parent_smiles=row.branch.parent.candidate.smiles,
                    parent_fresh_loss=row.branch.parent.fresh_loss, summary=row.summary,
                    fresh_available=len(reserves[row.branch.name]["mixed"]),
                    fresh_attempts=len(candidates), fresh_shortfall=limit - len(candidates),
                    candidates={"mixed": candidates}, endpoint=endpoint))
            scored = [row for row in selected_rows if row["loss"] is not None]
            losses = sorted(row["loss"] for row in scored)
            novel = [row for row in scored if not row["is_founder"]]
            best_new = min(losses) if losses else None
            all_known = [*losses, *(parent.fresh_loss for parent in parents)]
            chemistries = [row["chemistry"] for row in selected_rows if row["chemistry"]["valid"]]
            arms[arm] = dict(updates=len(rows) * design.updates,
                joint_quality=joint_quality_metrics(selected_rows, founder_smiles),
                relaxed_calls=len(rows) * (design.updates + 1), fresh_attempts=attempts,
                fresh_attempt_allowance=len(parents) * limit, fresh_successes=len(scored),
                fresh_failures=attempts - len(scored), fresh_shortfall=len(parents) * limit - attempts,
                best_new_fresh_loss=best_new, best_fresh_loss=min(all_known),
                best_novel_fresh_loss=min((row["loss"] for row in novel), default=None),
                fresh_novel_successes=len(novel), founder_rescore_successes=len(scored) - len(novel),
                mean_top_three_new_fresh_loss=sum(losses[:3]) / min(3, len(losses)) if losses else None,
                best_new_fresh_loss_by_sa_ceiling={str(ceiling): min(
                    (r["loss"] for r in scored if r["chemistry"].get("sa_score", float("inf")) <= ceiling),
                    default=None) for ceiling in (4, 5, 6)},
                selected_mean_sa=sum(c["sa_score"] for c in chemistries) / len(chemistries) if chemistries else None,
                selected_mean_qed=sum(c["qed"] for c in chemistries) / len(chemistries) if chemistries else None,
                selected_mean_rings=sum(c["rings"] for c in chemistries) / len(chemistries) if chemistries else None,
                trajectory_valid_fraction=sum(row.summary.get("valid_states", 0) for row in rows)
                    / (len(rows) * (design.updates + 1)),
                selected_max_similarity=max_pairwise_similarity([row["smiles"] for row in selected_rows]),
                optimization_seconds=optimization_seconds[arm], fresh_seconds=fresh_seconds)
        if scoring_policy == "matched_minimum" and len({row["fresh_attempts"] for row in arms.values()}) != 1:
            raise ValueError("Mismatched actual fresh scoring budgets")
        initial_readouts = [_candidate_readout(p.candidate, p.active, models, decoder.spec)
                            | dict(fresh_loss=p.fresh_loss) for p in parents]
        if (_model_identity(decoder) != identity or any(p.grad is not None for p in decoder.parameters())
                or any(_model_identity(model) != model_identities[name]
                       or any(p.grad is not None for p in model.parameters()) for name, model in models.items())):
            raise ValueError("Frozen decoder or synthesis model changed")
        if anchor is not None and any(_model_identity(model) != anchor_identities[name]
                for name, model in (anchor.items() if isinstance(anchor, dict) else {"low_sa": anchor}.items())):
            raise ValueError("Frozen attraction library changed")
        if file_sha256(Path(config["parent_panel"]["path"])) != config["parent_panel"]["sha256"]:
            raise ValueError("Synthesis founder input changed during the pilot")
        write_json(root / "batch.json", dict(optimization_updates=budget["updates"],
            new_fresh_attempts=sum(a["fresh_attempts"] for a in arms.values()),
            native_identity=native_identity, branches=branch_records))
        result = dict(status="completed", planned=budget, arms=arms,
            fresh_scoring_policy=scoring_policy,
            initial_parents=initial_readouts,
            scorer_start_seconds=scorer_seconds, elapsed_seconds=perf_counter() - started,
            frozen_decoder_unchanged=True, frozen_synthesis_models_unchanged=True,
            interpretation=("Equal independent maximum fresh allowances; actual attempts can differ. "
                            if scoring_policy == "equal_allowance" else "Matched actual fresh attempts. ")
                           + "Budgeted attempts include first founder rescores; is_founder marks novelty. "
                           "SA/QED/rings are independent hard measurements. Hard adapter scores are approximate model "
                           "extensions; reference-feature scores do not establish exact original-runtime parity.")
        write_json(output / "result.json", result)
        return result
    finally:
        close_models(models)
