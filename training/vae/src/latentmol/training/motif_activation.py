"""One-time, paired activation check for a new epoch-48 motif recipe."""

import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
import torch

from latentmol.datasets.data import collate_graphs
from latentmol.experiments import file_sha256, write_json
from latentmol.guidance.motif_completion import AMIDE
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.sampling import sample_hard_graph
from latentmol.training.combined_continuation import _train
from latentmol.training.occupancy_monitor import _inspect


TRIAL_UPDATES = 512
PRIOR_COUNT = 4096
RECON_COUNT = 256


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def policies(config):
    candidate = dict(config["functional_groups"])
    gate = candidate.pop("activation_gate")
    if (set(gate) != {"version", "seed", "control"} or gate["version"] not in (1, 2)
            or type(gate["seed"]) is not int or not 0 <= gate["seed"] < 2**63 - 1):
        raise ValueError("Unsupported motif activation gate configuration")
    control = gate["control"]
    if not isinstance(control, dict) or not control or "activation_gate" in control or control == candidate:
        raise ValueError("Require distinct control and candidate motif policies")
    if gate["version"] == 2 and (not candidate.get("ring_size", {}).get("weight")
            or control != {k: v for k, v in candidate.items() if k != "ring_size"}):
        raise ValueError("The matched ring comparison must differ only in its ring-size objective")
    return gate["seed"], {"control": control, "candidate": candidate}


def paired_decision(control, candidate):
    counts = {}
    for name in ("valid", "aromatic", "amide", "exact"):
        left, right = control[name], candidate[name]
        n = RECON_COUNT if name == "exact" else PRIOR_COUNT
        if len(left) != n or len(right) != n:
            raise ValueError(f"Incomplete matched {name} panel")
        gains = sum(bool(b) and not a for a,b in zip(left,right))
        losses = sum(bool(a) and not b for a,b in zip(left,right))
        delta = gains - losses
        # Standard error in *count* units for the mean of paired {-1,0,+1} differences.
        se = math.sqrt(max(0., (n * (gains + losses) - delta**2) / (n - 1)))
        counts[name] = dict(n=n, control=sum(left), candidate=sum(right), gains=gains, losses=losses,
                            net=delta, paired_se_count=se)
    a = counts["aromatic"]
    checks = dict(aromatic_minimum_four=a["net"] >= 4, aromatic_above_one_paired_se=a["net"] > a["paired_se_count"],
                  valid_drop_within_one_percent=counts["valid"]["net"] >= -.01 * PRIOR_COUNT,
                  amide_drop_within_half_percent=counts["amide"]["net"] >= -.005 * PRIOR_COUNT,
                  exact_drop_at_most_two=counts["exact"]["net"] >= -2)
    return ("candidate" if all(checks.values()) else "control"), counts, checks


class PanelTimeLimit(Exception):
    """The allocation needs its closing checkpoint reserve."""


@torch.inference_mode()
def evaluate_panel(state, spec, validation, quality, seed, output, device, deadline):
    """Use the production holdout rows and fresh common prior/sampling RNGs in both arms."""
    model = GraphAutoencoder(spec, ModelConfig(**state["model_config"])).to(device).eval()
    model.load_state_dict(state["model_state"], strict=True)
    if quality["validation_points"] != RECON_COUNT or validation.split != "val":
        raise ValueError("Activation reconstruction panel requires the existing 256 validation rows")
    start = validation.manifest["provenance"]["source_counts"]["val"]
    rows = random.Random(quality["seed"]).sample(range(start, len(validation)), RECON_COUNT)
    z = torch.randn(PRIOR_COUNT, model.config.latent, generator=torch.Generator().manual_seed(seed))
    sampler = torch.Generator(device=device).manual_seed(seed + 1)
    scores = {k: [] for k in ("valid", "aromatic", "amide", "exact")}
    check_rings = quality.get("ring_activation", False)
    if check_rings:
        scores.update({k: [] for k in ("triangle", "ring8plus", "any_ring", "occupied_atoms", "scaffold")})
    step = quality["batch_size"]
    def time_check():
        if deadline is not None and time.monotonic() >= deadline - 120:
            raise PanelTimeLimit
    with rdBase.BlockLogs(), gzip.open(output / "paired-panel.jsonl.gz", "xt", encoding="utf-8") as stream:
        for chunk in z.split(step):
            time_check()
            decoded = model.decode(chunk.to(device))
            if not all(t.isfinite().all() for t in (decoded.atoms, decoded.charges, decoded.bonds)):
                raise FloatingPointError("Nonfinite categorical evaluation logits")
            sampled = sample_hard_graph(decoded, spec, generator=sampler)
            atom_counts = (sampled.atoms != 0).sum(-1).tolist()
            for item, atom_count in zip(_inspect(sampled, spec), atom_counts, strict=True):
                mol = Chem.MolFromSmiles(item.smiles) if item.smiles else None
                flags = dict(valid=mol is not None, aromatic=bool(mol and rdMolDescriptors.CalcNumAromaticRings(mol)),
                             amide=bool(mol and mol.HasSubstructMatch(AMIDE)))
                failure = "smiles_roundtrip" if item.smiles and mol is None else item.failure_reason
                ring_sizes = set() if mol is None else {len(ring) for ring in mol.GetRingInfo().AtomRings()}
                ring8 = None if mol is None else any(n >= 8 for n in ring_sizes)
                topology = {}
                if check_rings:
                    topology = dict(triangle=3 in ring_sizes, any_ring=bool(ring_sizes),
                                    scaffold=MurckoScaffoldSmiles(mol=mol) if ring_sizes else "")
                    for key, value in dict(topology, occupied_atoms=atom_count, ring8plus=bool(ring8)).items():
                        scores[key].append(value)
                stream.write(json.dumps(dict(prior=len(scores["valid"]), smiles=item.smiles, failure_reason=failure,
                                             occupied_atoms=atom_count, ring8plus=ring8, **topology, **flags)) + "\n")
                for key, value in flags.items():
                    scores[key].append(value)
        for k in range(0, len(rows), step):
            time_check()
            ids = rows[k:k + step]
            graph = collate_graphs([validation[i] for i in ids]).to(device)
            mu, _ = model.encoder(graph, from_canonical_cache=True)
            decoded = model.decode(mu)
            if not all(t.isfinite().all() for t in (decoded.atoms, decoded.charges, decoded.bonds)):
                raise FloatingPointError("Nonfinite reconstruction evaluation logits")
            for index, item in zip(ids, _inspect(decoded.harden(spec), spec), strict=True):
                exact = item.smiles == validation.smiles[index]
                scores["exact"].append(exact)
                stream.write(json.dumps(dict(validation_row=index, smiles=item.smiles, exact=exact)) + "\n")
    return scores


def _load_endpoint(report, directory, trial_identity):
    receipt = report["latest"]
    path = Path(receipt["checkpoint"])
    if (path != directory / f"checkpoint-{report['updates']:09d}.pt" or file_sha256(path) != receipt["sha256"]
            or receipt["updates"] != report["updates"]):
        raise ValueError("Activation trial returned an incompatible checkpoint receipt")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state["config_sha256"] != trial_identity
            or any(state["continuation"][k] != receipt[k] or receipt[k] != report[k]
                   for k in ("updates", "row", "epoch", "real_points"))):
        raise ValueError("Activation trial checkpoint identity or cursor changed")
    return state


def _promote_endpoint(state, receipt, output, identity, metadata):
    """Supply a root checkpoint if the winning trial used up the available allocation."""
    path = output / f"checkpoint-{receipt['updates']:09d}.pt"
    if path.exists() or (output / "latest.json").exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(".tmp")
    with tmp.open("xb") as stream:
        torch.save(dict(state, run_id=output.name, config_sha256=identity, experiment_state=metadata), stream)
    tmp.replace(path)
    latest = dict(receipt, checkpoint=str(path), sha256=file_sha256(path))
    write_json(output / "latest.json", latest)
    return latest


def _join_accepted_history(chosen_dir, output):
    """Only the selected arm's updates belong in root training and quality histories."""
    for name in ("training.jsonl", "quality-panel.jsonl"):
        before, after = chosen_dir / name, output / name
        if not before.exists():
            continue
        merged = after.with_suffix(".merge-tmp")
        with before.open("rb") as old, after.open("rb") as new, merged.open("xb") as dest:
            shutil.copyfileobj(old, dest)
            if name == "quality-panel.jsonl":
                new.readline()  # the previous endpoint was also the production start
            shutil.copyfileobj(new, dest)
        merged.replace(after)


def run_activation(config, identity, data, assets, output, device, size, resume, max_updates, deadline, microbatch,
                   *, previous_policy, input_receipt, train=_train, panel=evaluate_panel, **kwargs):
    """Dispatch one matched gate or continue the previously adopted arm from its checkpoint."""
    seed, arms = policies(config)
    if resume is None or max_updates is not None or not input_receipt:
        raise ValueError("Motif activation is available only on a resumed full-epoch segment")
    old_update = resume["continuation"]["updates"]
    prior_state = resume.get("experiment_state", {}).get("motif_activation")
    def run(arm, run_id, directory, state, limit, metadata=None):
        effective = dict(config, functional_groups=arms[arm])
        return train(effective, run_id, data, assets, directory, device, size, state, limit, deadline, microbatch,
                     experiment_state=metadata, **kwargs)
    if previous_policy is None or "generated_roundtrip" in config:
        metadata = {"motif_activation": prior_state}
        selection_protocol = identity
        if "generated_roundtrip" in config:
            # The ring comparison has already chosen a policy. Keep its decision and source
            # protocol intact, including when the control won, across the recipe change.
            if previous_policy is not None:
                selection_protocol = resume["config_sha256"]
                if (previous_policy != config["functional_groups"] or selection_protocol == identity
                        or config["functional_groups"]["activation_gate"]["version"] != 2):
                    raise ValueError("Round-trip training must inherit the selected predecessor ring policy")
                handoff = {"source_protocol": selection_protocol, "target_protocol": identity}
            else:
                handoff = resume.get("experiment_state", {}).get("roundtrip_handoff", {})
                selection_protocol = handoff.get("source_protocol")
                if (not selection_protocol or handoff.get("target_protocol") != identity
                        or resume["config_sha256"] != identity):
                    raise ValueError("Missing authenticated round-trip recipe handoff")
            if input_receipt["updates"] != old_update:
                raise ValueError("Round-trip predecessor cursor mismatch")
            metadata["roundtrip_handoff"] = handoff
        if (not prior_state or prior_state.get("protocol") != selection_protocol or prior_state.get("winner") not in arms
                or prior_state["policy_sha256"] != _digest(arms[prior_state["winner"]])
                or not 0 < prior_state["adopted_at_update"] <= old_update):
            raise ValueError("Resume lacks the authenticated selection for this activation protocol")
        continued = run(prior_state["winner"], identity, output, resume, None, metadata)
        continued.update(metadata)
        write_json(output / "summary.json", continued)
        return continued
    version = config["functional_groups"]["activation_gate"]["version"]
    if input_receipt["updates"] != old_update:
        raise ValueError("Activation predecessor cursor mismatch")
    if version == 1 and (previous_policy != arms["control"] or prior_state is not None):
        raise ValueError("First gate must fork the original, authenticated control policy and cursor")
    if version == 2 and prior_state is not None:
        # A pending worker can start during a cutover. Only its authenticated,
        # previously adopted arm can feed this new comparison at the next boundary.
        _, previous_arms = policies({"functional_groups": previous_policy})
        if (prior_state["protocol"] != resume["config_sha256"]
                or prior_state["policy_sha256"] != _digest(previous_arms[prior_state["winner"]])):
            raise ValueError("Prior selection differs from the predecessor checkpoint")
    started = time.monotonic()
    gate_dir = output / "activation"
    gate_dir.mkdir()
    reports, states, scores = {}, {}, {}
    decision, reason, counts, checks = "control", "allocation_budget", {}, {}
    for arm in ("control", "candidate"):
        if arm == "candidate" and deadline is not None and time.monotonic() >= deadline - 120:
            break
        directory = gate_dir / arm
        directory.mkdir()
        trial_identity = _digest([identity, "activation-trial", arm])
        result = run(arm, trial_identity, directory, resume, TRIAL_UPDATES)
        if result["updates"] != old_update + result["segment_updates"]:
            raise ValueError("Activation trial did not run from the matched cursor")
        reports[arm] = dict(latest=result["latest"], segment_updates=result["segment_updates"],
                            status=result["status"], stop_reason=result["stop_reason"])
        if result["segment_updates"]:
            states[arm] = _load_endpoint(result, directory, trial_identity)
        if result["segment_updates"] != TRIAL_UPDATES:
            reason = result["stop_reason"]
            break
    if all(reports.get(a, {}).get("segment_updates") == TRIAL_UPDATES for a in arms):
        try:
            for arm in arms:
                quality = dict(config["quality_panel"], ring_activation=version == 2)
                scores[arm] = panel(states[arm], assets[0].spec, kwargs["validation"], quality,
                                    seed, gate_dir / arm, device, deadline)
                raw_panel = gate_dir / arm / "paired-panel.jsonl.gz"
                if raw_panel.exists():
                    reports[arm]["paired_panel_sha256"] = file_sha256(raw_panel)
            if version == 2:
                from latentmol.training.ring_activation import ring_decision
                decision, counts, checks = ring_decision(scores["control"], scores["candidate"], PRIOR_COUNT, RECON_COUNT)
            else:
                decision, counts, checks = paired_decision(scores["control"], scores["candidate"])
            reason = "all_checks_passed" if decision == "candidate" else "paired_checks_failed"
        except PanelTimeLimit:
            reason = "allocation_budget_during_panel"
    if "control" not in states:
        raise RuntimeError("Allocation ended before the first control update could be checkpointed")
    winner = states[decision]
    adopted = reports[decision]["latest"]
    evidence = dict(protocol=identity, predecessor=input_receipt, seed=seed, holdout_seed=config["quality_panel"]["seed"],
                    target_updates=TRIAL_UPDATES,
                    prior_count=PRIOR_COUNT, validation_count=RECON_COUNT, arms=reports,
                    paired=counts, checks=checks, winner=decision, reason=reason)
    write_json(gate_dir / "decision.json", evidence)
    stamp = dict(protocol=identity, winner=decision, policy_sha256=_digest(arms[decision]),
                 adopted_at_update=adopted["updates"], decision_sha256=file_sha256(gate_dir / "decision.json"),
                 decision_file=str(gate_dir / "decision.json"))
    metadata = {"motif_activation": stamp}
    result = run(decision, identity, output, winner, None, metadata)
    if result["latest"] is None:
        result["latest"] = _promote_endpoint(winner, adopted, output, identity, metadata)
    _join_accepted_history(gate_dir / decision, output)
    result.update(segment_updates=result["updates"] - old_update, elapsed_seconds=time.monotonic() - started,
                  motif_activation=stamp, discarded_trial_updates=sum(r["segment_updates"] for a,r in reports.items()
                                                                      if a != decision))
    write_json(output / "summary.json", result)
    return result
