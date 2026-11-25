"""Compare count-budget arms over a small, enumerated native topology family."""

import argparse
from dataclasses import asdict
from itertools import product
import json
import os
from pathlib import Path

from rdkit import Chem
import torch

from latentmol.chemistry import encode_smiles
from latentmol.design.latent import count_band_penalty
from latentmol.evaluation.evaluate import load_model
from latentmol.multimodel.config import AtomBudget
from latentmol.multimodel.external import file_sha256, tensor_tree_sha256
from latentmol.protenix_occupancy.budget_native import score_topology
from latentmol.protenix_occupancy.native import load_frozen_tiny
from latentmol.protenix_occupancy.one_atom import hard_observation, labels, optional_slot
from latentmol.protenix_occupancy.prepare import _bind_native
from latentmol.protenix_occupancy.study import write_json
from latentmol.scoring import require_frozen_eval


def topology_family(start_smiles, fluorine_sites, spec):
    """Keep the core order stable across one chlorine and three fluorine choices."""
    base = Chem.MolFromSmiles(start_smiles)
    if base is None or base.GetNumAtoms() != 25 or len(Chem.GetMolFrags(base)) != 1:
        raise ValueError("Require one connected 25-heavy-atom start")
    chlorines = [atom.GetIdx() for atom in base.GetAtoms() if atom.GetAtomicNum() == 17]
    if len(chlorines) != 1 or base.GetAtomWithIdx(chlorines[0]).GetDegree() != 1:
        raise ValueError("The starting optional atom must be a unique terminal chlorine")
    chlorine = chlorines[0]
    if len(fluorine_sites) != 3 or len(set(fluorine_sites)) != 3:
        raise ValueError("Require three distinct addition sites")
    for site in fluorine_sites:
        if not 0 <= site < 25:
            raise ValueError("Addition site outside the shared ligand")
        atom = base.GetAtomWithIdx(site)
        if atom.GetAtomicNum() != 6 or not atom.GetTotalNumHs():
            raise ValueError("Addition site must be a carbon with a substitutable hydrogen")

    family = {}
    for bits in product((False, True), repeat=4):
        ligand = Chem.RWMol(base)
        for enabled, site in zip(bits[1:], fluorine_sites):
            if enabled:
                index = ligand.AddAtom(Chem.Atom(9))
                ligand.AddBond(site, index, Chem.BondType.SINGLE)
        if not bits[0]:
            ligand.RemoveAtom(chlorine)
        ligand = ligand.GetMol()
        Chem.SanitizeMol(ligand)
        retained = tuple(i - int(not bits[0] and i > chlorine)
                         for i in range(25) if i != chlorine)
        smiles = encode_smiles(Chem.MolToSmiles(ligand), spec)["smiles"]
        if ligand.GetNumAtoms() != len(retained) + sum(bits):
            raise ValueError("Topology count differs from the declared occupancy state")
        family[bits] = (ligand, retained, smiles)
    if len({entry[2] for entry in family.values()}) != len(family):
        raise ValueError("Optional choices must produce distinct native ligands")
    return family


def topology_probabilities(q, masks):
    """Exact product-Bernoulli probabilities over the declared support."""
    return torch.where(masks, q, 1 - q).prod(-1)


def occupancy_logits(decoded, slots):
    logits = decoded.atoms[0, slots].double()
    return logits[:, 1:].logsumexp(-1) - logits[:, 0]


def optimize(model, anchor, slots, offset, family, losses, config, weight):
    masks = torch.tensor(list(family), device=anchor.device)
    core = len(next(iter(family.values()))[1])
    point = anchor
    trace = []
    for step in range(config["updates"] + 1):
        point = point.detach().requires_grad_()
        decoded = model.decode(point)
        if any(not getattr(decoded, name).isfinite().all() for name in ("atoms", "charges", "bonds")):
            raise RuntimeError("Nonfinite decoder output")
        raw_logits = occupancy_logits(decoded, slots)
        q = (raw_logits + offset).sigmoid()
        probabilities = topology_probabilities(q, masks)
        binding = (probabilities * losses).sum()
        expected = core + q.sum()
        count = weight * count_band_penalty(expected, config["target"], config["tolerance"])
        objective = binding + count
        if not objective.isfinite() or not objective.requires_grad:
            raise RuntimeError("Nonfinite or detached latent objective")
        binding_gradient, = torch.autograd.grad(binding, point, retain_graph=True)
        gradient, = torch.autograd.grad(objective, point)
        if not gradient.isfinite().all() or not binding_gradient.isfinite().all():
            raise RuntimeError("Nonfinite latent gradient")

        with torch.no_grad():
            bits = tuple((q >= 0.5).tolist())
            raw = hard_observation(decoded, model.spec, int(slots[0]))
            raw_expected = float(decoded.atoms.double().softmax(-1)[..., 1:].sum())
            norm = float(gradient.norm())
            trace.append(dict(step=step, optional_occupancies=q.tolist(),
                expected_atoms=float(expected), raw_expected_atoms=raw_expected, raw=raw,
                probability_28_atoms=float(q.prod()),
                hard_topology=list(bits), hard_atoms=core + sum(bits), hard_smiles=family[bits][2],
                binding_loss=float(binding), weighted_count_loss=float(count), objective=float(objective),
                binding_gradient_norm=float(binding_gradient.norm()),
                weighted_count_gradient_norm=float((gradient - binding_gradient).norm()),
                displacement=float((point - anchor).norm())))
            if step < config["updates"] and norm > 0:
                delta = point - config["latent_step"] * gradient / norm - anchor
                radius = config["latent_radius"]
                point = anchor + delta * (radius / delta.norm().clamp_min(radius))
    return trace


def run(config_path, output, job_id):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    budget = AtomBudget(**config["atom_budget"])
    if budget.target != 25 or budget.weight <= 0 or budget.outside_weight != 0:
        raise ValueError("This comparison requires a 25-atom count-only budget")
    if (type(config["updates"]) is not int or config["updates"] < 1 or
            not all(0 < config[key] < float("inf") for key in ("latent_step", "latent_radius"))):
        raise ValueError("Require positive bounded latent update settings")
    gate = config["pilot"]
    if file_sha256(gate["result"]) != gate["sha256"]:
        raise ValueError("One-atom pilot result identity differs")
    receipt = json.loads(Path(gate["result"]).read_text(encoding="utf-8"))
    if (receipt["job_id"] != gate["job_id"] or receipt["status"] != "completed"
            or receipt["supports_budget_followup"] is not True):
        raise ValueError("The one-atom pilot has not passed its follow-up gate")
    assets = json.loads((config_path.parent / config["assets_config"]).read_text(encoding="utf-8"))
    if file_sha256(assets["checkpoint"]) != assets["checkpoint_sha256"]:
        raise ValueError("Epoch-45 checkpoint identity differs")
    pocket_path = Path(config["pocket"]["path"])
    if file_sha256(pocket_path) != config["pocket"]["sha256"]:
        raise ValueError("BRD4 pocket definition identity differs")
    pocket = json.loads(pocket_path.read_text(encoding="utf-8"))
    if pocket["pdb_id"] != "3MXF":
        raise ValueError("Require the shared BRD4/3MXF pocket")
    pocket_residues = tuple(row["sequence_position"] for row in pocket["residues"])

    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model, _ = load_model(Path(assets["checkpoint"]), "cuda")
    model.requires_grad_(False)
    require_frozen_eval(model)
    model_state = tensor_tree_sha256(model.state_dict())
    family = topology_family(config["start_smiles"], config["fluorine_sites"], model.spec)
    start_bits = (True, False, False, False)
    _, _, start_smiles = family[start_bits]
    starting_labels = labels(start_smiles, model.spec, "cuda")
    chlorine_slot = optional_slot(starting_labels, model.spec, 17, family[(False,) * 4][2])
    slots = torch.tensor([chlorine_slot, 25, 26, 27], device="cuda")
    with torch.no_grad():
        anchor, _ = model.encode(starting_labels.to_relaxed(model.spec))
        decoded = model.decode(anchor)
        start_raw = hard_observation(decoded, model.spec, chlorine_slot)
        hard_start = decoded.harden(model.spec)
        if (start_raw["smiles"] != start_smiles or start_raw["atom_count"] != budget.target
                or any(not torch.equal(getattr(hard_start, key), getattr(starting_labels, key))
                       for key in ("atoms", "charges", "bonds"))):
            raise ValueError("Raw VAE decode does not reproduce the shared 25-atom start and its mapped slots")
        q_start = anchor.new_tensor([0.8, 0.2, 0.2, 0.2], dtype=torch.float64)
        offset = torch.logit(q_start) - occupancy_logits(decoded, slots)
        if not anchor.isfinite().all() or not offset.isfinite().all():
            raise ValueError("Nonfinite latent start or occupancy calibration")
    _bind_native(Path(assets["native"]["source_root"]), assets["native"]["source_sha256"],
                 Path(config["ccd_root"]))
    native = load_frozen_tiny(**assets["native"])
    native_state = native.snapshot_state()
    if proof := os.environ.get("DBMOL_LAUNCH_READY"):
        write_json(Path(proof), dict(status="models_and_inputs_ready", job_id=job_id,
                                    nonce=os.environ["DBMOL_LAUNCH_NONCE"]))

    def score(bits, stage):
        mol, retained, smiles = family[bits]
        bit_label = "".join(str(int(bit)) for bit in bits)
        loss, detail = score_topology(native, mol, retained, pocket["protein_sequence"],
            pocket_residues, geometry_seed=config["geometry_seeds"][stage],
            mol_path=output / f"{stage}-{bit_label}.mol")
        return dict(topology=list(bits), heavy_atoms=mol.GetNumAtoms(), smiles=smiles,
                    loss=loss, details=detail)

    panel = [score(bits, "optimization") for bits in family]
    write_json(output / "panel.json", dict(topologies=panel))
    scores = torch.tensor([row["loss"] for row in panel], device="cuda", dtype=torch.float64)
    settings = {**config, "target": budget.target, "tolerance": budget.tolerance}
    traces = {name: optimize(model, anchor, slots, offset, family, scores, settings, weight)
              for name, weight in (("off", 0.0), ("on", budget.weight))}
    fresh = {}
    by_smiles = {entry[2]: bits for bits, entry in family.items()}

    def fresh_score(bits):
        if bits not in fresh:
            fresh[bits] = score(bits, "fresh")
        return fresh[bits]["loss"]

    for name, trace in traces.items():
        initial = trace[0]
        if tuple(initial["hard_topology"]) != start_bits or initial["weighted_count_loss"] != 0:
            raise RuntimeError("Comparison arms must start from the same unpenalized 25-atom topology")
        write_json(output / f"{name}.json", dict(trajectory=trace))
    baseline_loss = fresh_score(start_bits)
    arms = {}
    for name, trace in traces.items():
        final = trace[-1]
        raw_bits = by_smiles.get(final["raw"]["smiles"])
        arms[name] = dict(max_expected_atoms=max(row["expected_atoms"] for row in trace),
            max_hard_atoms=max(row["hard_atoms"] for row in trace),
            max_probability_28_atoms=max(row["probability_28_atoms"] for row in trace),
            count_penalty_active_updates=sum(row["weighted_count_loss"] > 0 for row in trace[:-1]),
            final=final, fresh_hard_loss=fresh_score(tuple(final["hard_topology"])),
            fresh_raw_loss=fresh_score(raw_bits) if raw_bits is not None else None)
    require_frozen_eval(model)
    if tensor_tree_sha256(model.state_dict()) != model_state:
        raise RuntimeError("Frozen VAE state changed")
    native.assert_unchanged(native_state)
    write_json(output / "result.json", dict(status="completed", job_id=job_id,
        method="exact_expectation_over_16_declared_native_topologies", pilot=gate,
        checkpoint_sha256=assets["checkpoint_sha256"], budget=asdict(budget), start_raw=start_raw,
        supported_counts=[24, 25, 26, 27, 28], slots=slots.tolist(), logit_offsets=offset.tolist(),
        fresh_start_loss=baseline_loss, fresh_scores=list(fresh.values()), arms=arms,
        off_entered_penalty_region=any(abs(row["expected_atoms"] - budget.target) > budget.tolerance
                                      for row in traces["off"])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    job_id = os.environ.get("LSB_JOBID", "")
    if not job_id.isdigit() or not torch.cuda.is_available():
        raise RuntimeError("Run native Protenix inside an allocated GPU job")
    run(args.config, args.output, job_id)


if __name__ == "__main__":
    main()
