"""Measure which edits a whole-graph neighbor target would overwrite."""

import argparse
from collections import Counter
import json
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
import torch

from latentmol.chemistry import inspect_graph
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256


def valid_prior_centers(model, device, draws, seed):
    """Keep every valid center from a fixed sequence, regardless of size or re-encoding."""
    generator = torch.Generator().manual_seed(seed)
    parents, failures = [], Counter()
    for i in range(draws):
        z = torch.randn(1, model.config.latent, generator=generator)
        graph = model.decode(z.to(device)).harden(model.spec)
        hard = inspect_graph(graph.atoms[0].cpu(), graph.charges[0].cpu(), graph.bonds[0].cpu(), model.spec)
        if hard.smiles:
            parents.append(dict(name=f"draw-{i}", smiles=hard.smiles, states={"original_random": {"latent": z[0].tolist()}}))
        else:
            failures[hard.failure_reason] += 1
    return parents, failures


@torch.inference_mode()
def inspect_neighbors(checkpoint, panel_path, *, device, seed, prior_draws=None):
    checkpoint_sha = file_sha256(checkpoint)
    model, state = load_model(checkpoint, device)
    if state["continuation"]["updates"] != 597604:
        raise ValueError("Incompatible training update")
    model.eval().requires_grad_(False)
    if panel_path:
        panel = json.loads(panel_path.read_bytes())
        if checkpoint_sha != panel["checkpoint_sha256"] or panel["accepted"] != panel["requested"]:
            raise ValueError("Incomplete matched panel or checkpoint mismatch")
        parents = panel["parents"]
        selection = {"panel_sha256": file_sha256(panel_path)}
    else:
        if prior_draws is None or prior_draws < 1:
            raise ValueError("Specify a positive number of independent prior draws")
        parents, failures = valid_prior_centers(model, device, prior_draws, seed + 1)
        selection = dict(prior_seed=seed + 1, prior_draws=prior_draws, rejected_centers=failures)
    generator = torch.Generator().manual_seed(seed)
    rows, counts = [], {}
    with rdBase.BlockLogs():
        for parent in parents:
            molecule = Chem.MolFromSmiles(parent["smiles"])
            parent_core = MurckoScaffoldSmiles(mol=molecule)
            # Share three isotropic unit vectors across all starts and radii for this parent.
            directions = torch.randn(3, model.config.latent, generator=generator)
            directions = torch.nn.functional.normalize(directions).to(device)
            for arm, data in parent["states"].items():
                z = torch.tensor(data["latent"], device=device)
                for radius in (0., .1, 1., 2.):
                    offsets = directions if radius else directions[:1]
                    decoded = model.decode(z[None] + radius * offsets)
                    if any(not x.isfinite().all() for x in (decoded.atoms, decoded.charges, decoded.bonds)):
                        raise FloatingPointError("Nonfinite neighbor decoder output")
                    graph = decoded.harden(model.spec)
                    for i, labels in enumerate(zip(graph.atoms.cpu(), graph.charges.cpu(), graph.bonds.cpu(), strict=True)):
                        hard = inspect_graph(*labels, model.spec)
                        mol = Chem.MolFromSmiles(hard.smiles) if hard.smiles else None
                        category = "invalid"
                        if mol is not None:
                            category = ("same_parent" if hard.smiles == parent["smiles"]
                                        else "changed_same_scaffold" if parent_core and MurckoScaffoldSmiles(mol=mol) == parent_core
                                        else "other_valid_edit")
                        if radius == 0 and category != "same_parent":
                            raise ValueError("Evaluation device did not reproduce a matched parent")
                        rows.append(dict(parent=parent["name"], arm=arm, radius=radius, direction=i,
                                         category=category, smiles=hard.smiles, failure=hard.failure_reason))
                        counts.setdefault(arm, {}).setdefault(str(radius), Counter())[category] += 1
    return dict(seed=seed, device=device, checkpoint_sha256=checkpoint_sha, centers=len(parents),
                **selection, directions_per_parent=3, rows=rows, counts=counts)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--checkpoint", type=Path, required=True)
    cohort = cli.add_mutually_exclusive_group(required=True)
    cohort.add_argument("--panel", type=Path)
    cohort.add_argument("--prior-draws", type=int)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    args = cli.parse_args()
    torch.set_num_threads(2)
    result = inspect_neighbors(args.checkpoint, args.panel, device=args.device, seed=2026112299,
                               prior_draws=args.prior_draws)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result["counts"], indent=2))
