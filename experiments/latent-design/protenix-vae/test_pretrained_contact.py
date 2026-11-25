import json

import pytest
import torch

from latentmol.representation import GraphSpec, RelaxedGraph
from pretrained_contact import combined_summary, trajectory


class ToyDecoder:
    def decode(self, z, state, temperature):
        # Two fixed occupied slots; the first element can flip from C to O.
        p = torch.sigmoid(z[0, 0] / temperature)
        first = torch.stack((p * 0, 1 - p, p))
        atoms = torch.stack((first, z.new_tensor([0., 1., 0.])))[None]
        bonds = torch.nn.functional.one_hot(torch.tensor([[[0, 1], [1, 0]]]), 5).float()
        return RelaxedGraph(atoms, z.new_ones((1, 2, 1)), bonds)


def test_lr60_direction_endpoints_fresh_edit_and_replayed_control(tmp_path):
    def scorer(graph):
        return -graph.atoms[0, 0, 2], {}

    spec = GraphSpec(max_nodes=2, atomic_numbers=(0, 6, 8), formal_charges=(0,))
    cfg = dict(updates=2, learning_rate=60., temperature=1., edit_limit=3)
    start = torch.tensor([[-1., 0., 0.]])
    model = ToyDecoder()
    stats, chosen, lengths = trajectory(model, scorer, start, None, spec, cfg, tmp_path / "gradient")
    assert stats["native_evaluations"] == 3
    assert stats["relaxed_contact_gain"] > 0
    assert lengths[0] == pytest.approx(60 * torch.sigmoid(start[0, 0]).item() * (1-torch.sigmoid(start[0, 0]).item()))
    assert [r["smiles"] for r in chosen] == ["CC", "CO"]
    random, _, realized = trajectory(model, scorer, start, None, spec, cfg, tmp_path / "random",
                                     random_lengths=lengths, seed=17)
    assert random["native_evaluations"] == stats["native_evaluations"]
    assert realized == pytest.approx(lengths, abs=2e-6)
    points = [json.loads(line)["latent"] for line in (tmp_path / "random" / "metrics.jsonl").read_text().splitlines()]
    assert points[0] == start[0].tolist() and points[1][1] != 0
    assert torch.equal(start, torch.tensor([[-1., 0., 0.]]))
    with pytest.raises(FileExistsError):
        trajectory(model, scorer, start, None, spec, cfg, tmp_path / "gradient")


def test_fresh_summary_separates_the_source_from_a_misreconstructed_start():
    panel = {"starts": [dict(name="a", exact_reconstruction=False, arms={"gradient": {}})]}
    rows = [dict(start="a", arm="encoded_input", role="initial", smiles="CC", loss=1., status="scored"),
            dict(start="a", arm="gradient", role="initial", smiles="CF", loss=2., status="scored"),
            dict(start="a", arm="gradient", role="first_distinct_valid", smiles="CO", loss=1.5,
                 loss_delta_from_valid_start=-.5, status="scored")]
    summary = combined_summary(panel, dict(unique_candidates=3, candidates=rows))
    arm = summary["starts"][0]["arms"]["gradient"]
    assert summary["exact_reconstructions"] == 0
    assert arm["best_fresh_loss_delta_vs_decoded_start"] == -.5
    assert arm["best_fresh_loss_delta_vs_encoded_input"] == .5
    assert arm["distinct_fresh_improvements_vs_encoded_input"] == 0
