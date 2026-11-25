import json

import pytest
import torch

from latentmol.design.boltz_direct_graph import DirectGraphLogits, carbonyl_loss, motif_and_triangles, run_child
from latentmol.design.latent_protenix import hard_from_graph
from latentmol.guidance.boltz_search import hard_graph_record
from latentmol.representation import GraphSpec, HardGraphBatch


def chain():
    spec = GraphSpec(max_nodes=6)
    # Empty slots interleaved to exercise the scorer's fixed pose order.
    slots = [0, 2, 3, 5]
    atoms = torch.zeros((1, 6), dtype=torch.long)
    atoms[0, slots] = spec.atomic_numbers.index(6)
    charges = torch.full_like(atoms, spec.formal_charges.index(0))
    bonds = torch.zeros((1, 6, 6), dtype=torch.long)
    for left, right in zip(slots[:-1], slots[1:], strict=True):
        bonds[0, left, right] = bonds[0, right, left] = spec.bond_names.index("SINGLE")
    return spec, HardGraphBatch(atoms, charges, bonds)


def test_direct_logits_preserve_labels_slots_and_pair_symmetry():
    spec, hard = chain()
    model = DirectGraphLogits(hard, spec, 17)
    assert model.start_digest() == DirectGraphLogits(hard, spec, 17).start_digest()
    assert model.start_digest() != DirectGraphLogits(hard, spec, 18).start_digest()
    _, graph = model()
    assert hard_graph_record(hard_from_graph(graph)) == hard_graph_record(hard)
    assert torch.equal(graph.bonds, graph.bonds.transpose(1, 2))
    assert torch.equal(graph.occupancy[0].bool(), model.active)
    assert model.bond_logits.shape == (6, len(spec.bond_names))
    carbonyl, triangles = motif_and_triangles(graph, spec, model.active)
    atom_grad, bond_grad = torch.autograd.grad(carbonyl_loss(carbonyl), (model.atom_logits, model.bond_logits),
                                                retain_graph=True)
    assert atom_grad.norm() > .1 and bond_grad.norm() > .1
    assert torch.autograd.grad(triangles, model.bond_logits)[0].norm() > 0
    assert carbonyl_loss(torch.tensor(.6)) == 0
    triangle = HardGraphBatch(hard.atoms, hard.charges, hard.bonds.clone())
    triangle.bonds[0, 0, 3] = triangle.bonds[0, 3, 0] = 1
    assert motif_and_triangles(triangle.to_relaxed(spec), spec, model.active)[1].item() == 1


class FlatBinding:
    def score_at(self, graph, active):
        return (graph.atoms.sum() + graph.charges.sum() + graph.bonds.sum()) * 0., {}

    def score_valid_records(self, root, records):
        return [dict(record, loss=-1.) for record in records]


def test_direct_updates_create_distinct_outputs_log_gradients_and_exclude_founder(tmp_path):
    spec, hard = chain()
    kwargs = dict(seed=19, lr=.12, motif_weight=1., triangle_weight=0., fresh_budget=2)
    output = tmp_path / "arm"
    result = run_child(FlatBinding(), spec, hard, output, updates=25, **kwargs)
    rows = list(map(json.loads, (output / "metrics.jsonl").read_text().splitlines()))
    assert len(rows) == 26 and rows[0]["smiles"] == "CCCC"
    assert rows[-1]["carbonyl_probability"] > 10 * rows[0]["carbonyl_probability"]
    assert result["updates"] == 25 and result["relaxed_observations"] == 26
    assert set(rows[0]["binding_gradient_norms"]) == {"atoms", "charges", "bonds"}
    assert result["distinct_novel_valid"] > 0
    assert 0 < result["fresh_successes"] == result["fresh_attempts"] <= 2
    assert all(r["smiles"] != "CCCC" for r in result["top20"])
    with pytest.raises(FileExistsError):
        run_child(FlatBinding(), spec, hard, output, updates=1, **kwargs)


def test_native_path_must_reach_every_direct_parameter(tmp_path):
    class BrokenBinding:
        def score_at(self, graph, active):
            return graph.atoms.sum(), {}

    spec, hard = chain()
    with pytest.raises(ValueError, match="Detached native gradient"):
        run_child(BrokenBinding(), spec, hard, tmp_path / "broken", seed=13, lr=.1, updates=1,
                  fresh_budget=0, motif_weight=1., triangle_weight=.25)
