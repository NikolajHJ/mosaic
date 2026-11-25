"""Local small-batch replay using a cached earlier parent, reference and canonical PubChem rows."""

import argparse
import json
from pathlib import Path
import time

import torch

from latentmol.chemistry import encode_canonical_smiles
from latentmol.datasets.data import collate_graphs
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import write_json
from latentmol.guidance.valence_projection import projection_targets
from latentmol.train import state_digest
from latentmol.training.projected_continuation import _student, projected_update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "reference", "panel", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(3)
    parent, payload = load_model(args.checkpoint, "cpu")
    reference, _ = load_model(args.reference, "cpu", parent.spec)
    reference.requires_grad_(False)
    before = state_digest(reference.state_dict())
    config = json.loads(args.config.read_bytes())
    rows = json.loads(args.panel.read_bytes())["rows"][:128]
    graphs = [encode_canonical_smiles(row[2], parent.spec) for row in rows]
    hard = collate_graphs([{k: torch.as_tensor(g[v]).long() for k, v in
        (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))} for g in graphs])
    i, j = torch.triu_indices(parent.spec.max_nodes, parent.spec.max_nodes, offset=1)
    bonds = torch.nn.functional.one_hot(hard.bonds[:, i, j], len(parent.spec.bond_names))
    raw, target, stats = projection_targets(hard.atoms.numpy(), hard.charges.numpy(), bonds.float().numpy(),
                                           parent.spec, config["projection"]["max_edits"])
    assert not (raw != target).any(), stats
    hard = type(hard)(hard.atoms[:32], hard.charges[:32], hard.bonds[:32])
    arms = {}
    for count in (0, 3):
        model, optimizer, training, generators = _student(parent, payload, None, 32, "cpu", None,
            count=count, seed=config["extension_seed"], atom_order=config["atom_order"], atom_refinement_dropout=.05)
        assert training.loss.atom_order == "graph"
        model.eval()
        fixed = torch.randn(4, model.config.latent, generator=torch.Generator().manual_seed(99))
        with torch.no_grad():
            decoded = model.decode(fixed)
            heads = [decoded.atoms, decoded.charges, decoded.bonds]
        if not count:
            baseline = heads
        else:
            assert all(torch.equal(x, y) for x, y in zip(baseline, heads))
        model.train()
        started = time.monotonic()
        result = projected_update(config, model, reference, optimizer, hard, training, None, None, generators,
                                  payload["continuation"]["updates"], 48, 16)
        arms[str(4 + count)] = dict(result=result, seconds=time.monotonic()-started,
            parameters=sum(p.numel() for p in model.parameters()),
            gates=[[float(b.atom_gate.detach()), float(b.pair_gate.detach())] for b in model.decoder.extra_rounds])
    assert state_digest(reference.state_dict()) == before
    assert arms["4"]["result"]["projection_counts"] == arms["7"]["result"]["projection_counts"]
    report = dict(known_valid_checked=len(rows), known_valid_edited=0, initial_logits_identical=True,
                  source_checkpoint=str(args.checkpoint), source_update=payload["continuation"]["updates"],
                  reference_unchanged=True, atom_order=training.loss.atom_order, arms=arms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
