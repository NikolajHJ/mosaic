import json
from dataclasses import replace

import numpy as np
import torch

from latentmol.chemistry import inspect_graph
from latentmol.design import boltz_allocation_pilot as allocation
from latentmol.design.boltz_generations import run_generations
from latentmol.design.boltz_search_methods import archive_starts
from latentmol.design.latent_protenix import LatentDesignConfig, hard_from_graph
from latentmol.design.protenix_restarts import Candidate
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.graph_policy import graph_from_decoder
from latentmol.representation import DecoderOutput, GraphSpec


class FounderDecoder:
    """A two-carbon founder whose first atom also has appreciable nitrogen probability."""

    def __init__(self):
        self.spec = GraphSpec(max_nodes=3)
        self.paired_starts_finished = False

    def decode(self, z):
        assert not self.paired_starts_finished, "VAE called during optimization or a later restart"
        spec = self.spec
        atoms = z.new_zeros((1, 3, len(spec.atomic_numbers)))
        atoms[0, [0, 2], spec.atomic_numbers.index(6)] = 8
        atoms[0, 0, spec.atomic_numbers.index(7)] = 7.8 + 0.1 * z[0, 0]
        charges = z.new_zeros((1, 3, len(spec.formal_charges)))
        charges[:, :, spec.formal_charges.index(0)] = 7 + z[0, 1]
        bonds = z.new_zeros((1, 3, 3, len(spec.bond_names)))
        bonds[0, 0, 2, 1] = bonds[0, 2, 0, 1] = 8 + z[0, 2]
        return DecoderOutput(atoms, charges, bonds)


class ProbeScorer:
    def __init__(self, decoder):
        self.decoder = decoder
        self.graphs = []

    def score_at(self, graph, active):
        self.decoder.paired_starts_finished = True
        assert torch.equal(graph.occupancy[0].bool(), active)
        assert torch.equal(graph.bonds, graph.bonds.transpose(1, 2))
        self.graphs.append(tuple(x.detach().clone() for x in (graph.atoms, graph.charges, graph.bonds)))
        binding = -4 * graph.atoms[0, 0, 2] + .1 * graph.charges[0, 2, 0] + .1 * graph.bonds[0, 0, 2, 0]
        return binding, {}

    def score_valid_records(self, path, rows):
        return [dict(row, loss=-1.) for row in rows]


def test_graph_generations_pair_starts_and_replay_new_elite_without_vae(tmp_path, monkeypatch):
    from latentmol.design.boltz_graph_restarts import DirectGraphCoordinates

    monkeypatch.setitem(allocation.POLICY_LENGTHS, "fixed75", 2)
    decoder = FounderDecoder()
    design = replace(LatentDesignConfig.for_restart({}), occupied_atoms=2, learning_rate=1.,
                     max_step_norm=.25, temperature_initial=1., temperature_final=1.,
                     chemistry_initial=1., chemistry_final=1., valence_excess_weight=1.)
    active = torch.tensor([True, False, True])
    z = torch.zeros((1, 3))
    hard = hard_from_graph(graph_from_decoder(decoder.decode(z), decoder.spec, fixed_active=active))
    founder = SearchPoint(Candidate("CC", z, hard, 0., "founder", 0), active, 0., ("founder",))
    state = dict(archive={"CC": founder}, ineligible=set())
    starts, _ = archive_starts(decoder, design, state, 2, 91, parent_count=1)
    paired = [graph_from_decoder(decoder.decode(p.candidate.latent), decoder.spec, fixed_active=active) for p in starts]

    scorer = ProbeScorer(decoder)
    result = run_generations(decoder, scorer, design, state, tmp_path, policy="fixed75", seed=91,
                             generations=2, updates_per_generation=4, fresh_per_generation=2,
                             parents=1, children_per_parent=2, optimization_space="relaxed_graph_logits")

    assert result["updates"] == 8
    assert result["relaxed_observations"] == len(scorer.graphs) == 12
    assert 0 < result["fresh_successes"] == result["fresh_attempts"] <= result["fresh_budget"] == 4
    assert "CC" not in [r["smiles"] for r in result["top20"]]
    for graph, scored in zip(paired, (scorer.graphs[0], scorer.graphs[3]), strict=True):
        for expected, actual in zip((graph.atoms, graph.charges, graph.bonds), scored, strict=True):
            torch.testing.assert_close(expected, actual)

    first, second = sorted((tmp_path / "generations").iterdir())
    new_parent = json.loads((second / "starts.json").read_text())["elites"][0]
    assert new_parent["smiles"] == "CN" and "generation-01" in new_parent["source"]
    for wave in (first, second):
        elite, = torch.load(wave / "elites.pt", weights_only=True)
        assert "latent" not in elite
        coords = DirectGraphCoordinates(decoder.spec, elite["active"])
        replay = graph_from_decoder(coords.decode(elite["graph_logits"]), decoder.spec, fixed_active=elite["active"])
        hard = hard_from_graph(replay)
        assert inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], decoder.spec).smiles == elite["smiles"]
        for component in ("atoms", "charges", "bonds"):
            assert torch.equal(getattr(hard, component), elite[component])
        files = list(wave.rglob("states-*.npz"))
        assert len(files) == 2
        for path in files:
            with np.load(path) as raw:
                assert "latent" not in raw
                logits = raw["graph_logits"]
                assert logits.shape[0] == 3 and np.isfinite(logits).all()
                assert 0 < np.linalg.norm(logits[1] - logits[0]) <= .25001
            first_step = json.loads((path.parent.parent / "metrics.jsonl").read_text().splitlines()[0])
            channel_norms = first_step["gradient_norms"]["direct_binding_channels"]
            assert set(channel_norms) == {"atoms", "charges", "bonds"}
            assert all(np.isfinite(v) and v > 0 for v in channel_norms.values())
