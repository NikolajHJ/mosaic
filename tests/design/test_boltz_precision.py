"""Paired precision initialization and common-FP32 ranking without native assets."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from latentmol.chemistry import encode_smiles, inspect_graph
from latentmol.design import boltz_speed as speed
from latentmol.design.latent_protenix import LatentDesignConfig, RawBlocks, hard_from_graph
from latentmol.design.protenix_restarts import Candidate
from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance.boltz_search import hard_pose_key
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch


def hard(smiles, spec):
    encoded = encode_smiles(smiles, spec)
    return HardGraphBatch(*(torch.as_tensor(encoded[key], dtype=torch.long)[None]
                            for key in ("atom_types", "charges", "bond_types")))


def write_trajectory(root, sequence, spec, *, seconds=1., own_loss=99.):
    root.mkdir(parents=True)
    blocks = RawBlocks(root/"raw-blocks", spec.max_nodes, 100)
    metrics = []
    for step, graph in enumerate(sequence):
        checked = inspect_graph(graph.atoms[0], graph.charges[0], graph.bonds[0], spec)
        blocks.add(step, torch.tensor([[float(step), 0.]]), graph)
        metrics.append(dict(step=step, smiles=checked.smiles, raw_valid=checked.smiles is not None,
                            relaxed_loss=own_loss, chemistry_loss=0., timing=dict(iteration_seconds=seconds)))
    blocks.flush()
    (root/"metrics.jsonl").write_text("".join(json.dumps(row)+"\n" for row in metrics))
    result = speed.collect_precision_states(root, spec, sequence[0].atoms[0] != 0, len(sequence)-1)
    result.pop("initial_latent")
    result.pop("terminal_latent")
    result.update(initial_point_sha256="same-point", initial_latent_sha256="same-latent", active_sha256="same-mask",
        initial_hard_key=result["observations"][0]["hard_graph_key"],
        initial_smiles=result["observations"][0]["smiles"], parent_state_point_sha256="same-parent",
        child_seed=17, measured_seconds=seconds*len(sequence), seconds_per_update=seconds*len(sequence)/(len(sequence)-1),
        execution=dict(autocast="none"))
    return result


def test_precision_budget_has_400_updates_402_observations_and_402_shared_score_cap():
    config = dict(children=1, updates=200, child_seed=17, checkpoints=[0, 50, 100, 200],
        cases=dict(fp32=dict(execution=dict(autocast="none")), bf16=dict(execution=dict(autocast="bfloat16_blocks"))),
        tasks=dict(seed_a=dict(case_order=["bf16", "fp32"], child_seed=31)))
    budget = speed.precision_budget(config, "seed_a")
    assert budget["updates"] == 400 and budget["relaxed_observations"] == budget["fresh_attempts"] == 402
    assert budget["case_order"] == ["bf16", "fp32"] and budget["child_seed"] == 31
    for field, value in (("children", 2), ("updates", 199)):
        wrong = deepcopy(config)
        wrong[field] = value
        with pytest.raises(ValueError):
            speed.precision_budget(wrong, "seed_a")
    config["cases"]["fp32"]["execution"]["tf32"] = True
    with pytest.raises(ValueError, match="FP32 reference"):
        speed.precision_budget(config, "seed_a")


def test_every_raw_pose_is_scored_in_fp32_without_future_or_other_arm_leakage(tmp_path):
    spec = GraphSpec(max_nodes=3)
    a, b, c = (hard(smiles, spec) for smiles in ("CCC", "CCO", "CCN"))
    permutation = torch.tensor([2, 1, 0])
    b_reordered = HardGraphBatch(b.atoms[:, permutation], b.charges[:, permutation],
                                b.bonds[:, permutation][:, :, permutation])
    fp32 = write_trajectory(tmp_path/"fp32", [a]*50+[b]*50+[c]*101, spec, seconds=2., own_loss=-999.)
    bf16 = write_trajectory(tmp_path/"bf16", [a]*50+[c]*100+[b_reordered]*51, spec, seconds=1., own_loss=999.)
    arms = dict(fp32=fp32, bf16=bf16)
    union = speed.precision_union(arms)
    assert len(union) == 4  # same CCO identity in two pose-slot arrangements
    losses = {"CCC": 2., "CCN": 1.5}
    scores = [dict(row, loss=(1. if row["hard_graph"] == fp32["candidates"][1]["hard_graph"] else .5)
                   if row["smiles"] == "CCO" else losses[row["smiles"]]) for row in union]
    result = speed.summarize_precision_pair(arms, scores, [0, 50, 100, 200])
    fa, ba = result["arms"]["fp32"], result["arms"]["bf16"]
    assert fa["by_update"][1]["best_loss"] == 1.
    assert ba["by_update"][1]["best_loss"] == 1.5  # cannot use future CCO
    assert fa["by_update"][2]["mean_top_three_loss"] == 1.5
    assert ba["by_update"][2]["mean_top_three_loss"] is None  # cannot borrow FP32's CCO
    assert fa["final"]["best_loss"] == 1. and ba["final"]["best_loss"] == .5
    assert fa["terminal"]["loss"] == 1.5 and ba["terminal"]["loss"] == .5
    assert fa["final"]["mean_top_twenty_loss"] is None and fa["final"]["top_twenty_count"] == 3
    assert fa["fixed_time_proxy"]["last_observed_update"] == 99
    assert ba["fixed_time_proxy"]["last_observed_update"] == 200
    assert result["intersection_pose_graphs"] == 2 and result["cross_arm_duplicate_calls_saved"] == 2
    # Counterbalancing execution order cannot change the common score reduction.
    reverse = speed.summarize_precision_pair(dict(bf16=bf16, fp32=fp32), scores, [0, 50, 100, 200])
    assert reverse["arms"] == result["arms"]
    changed = deepcopy(arms)
    changed["bf16"]["initial_point_sha256"] = "wrong"
    with pytest.raises(ValueError, match="same saved parent"):
        speed.precision_union(changed)


def test_no_valid_molecules_is_a_recorded_empty_result_and_invalid_terminal_keeps_history(tmp_path):
    spec = GraphSpec(max_nodes=3)
    valid = hard("CCC", spec)
    invalid = HardGraphBatch(valid.atoms.clone(), valid.charges.clone(), torch.zeros_like(valid.bonds))
    empty_a = write_trajectory(tmp_path/"empty-a", [invalid]*201, spec)
    empty_b = write_trajectory(tmp_path/"empty-b", [invalid]*201, spec)
    result = speed.summarize_precision_pair(dict(fp32=empty_a, bf16=empty_b), [], [0, 50, 100, 200])
    assert result["common_fp32_pose_graphs"] == 0
    assert result["arms"]["bf16"]["final"]["best_loss"] is None
    assert result["arms"]["bf16"]["final"]["raw_valid_states"] == 0
    history_a = write_trajectory(tmp_path/"history-a", [valid]+[invalid]*200, spec)
    history_b = write_trajectory(tmp_path/"history-b", [valid]+[invalid]*200, spec)
    union = speed.precision_union(dict(fp32=history_a, bf16=history_b))
    result = speed.summarize_precision_pair(dict(fp32=history_a, bf16=history_b),
        [dict(row, loss=1.) for row in union], [0, 50, 100, 200])
    assert result["arms"]["fp32"]["final"]["best_loss"] == 1.
    assert result["arms"]["fp32"]["terminal"]["raw_valid"] is False
    assert result["arms"]["fp32"]["terminal"]["loss"] is None


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spec = GraphSpec(max_nodes=3)
        self.config = SimpleNamespace(latent=2)
        self.weight = torch.nn.Parameter(torch.tensor(.001), requires_grad=False)

    def decode(self, point):
        atoms = torch.full((1, 3, 10), -5.)
        atoms[..., 1] = 5.
        atoms[..., 3] += point[:, :1]*self.weight
        charges = torch.full((1, 3, 5), -5.)
        charges[..., 2] = 5.
        charges[..., 1] += point[:, 1:]*self.weight
        bonds = torch.full((1, 3, 3, 5), -5.)
        bonds[..., 0] = 5.
        bonds[:, 0, 1, 1] = bonds[:, 1, 0, 1] = 10.
        bonds[:, 1, 2, 1] = bonds[:, 2, 1, 1] = 10.
        return DecoderOutput(atoms, charges, bonds)


class MockScorer:
    def score_at(self, graph, active):
        value = (graph.atoms*torch.arange(10.)).sum() + (graph.charges*torch.arange(5.)).sum()
        return value, {}


def test_real_branch_engine_replays_identical_initialization_in_fp32_fp32_checker(tmp_path, monkeypatch):
    import latentmol.design.protenix_restart_proposals as proposals

    decoder = TinyDecoder().eval()
    latent = torch.zeros(1, 2)
    active = torch.ones(3, dtype=torch.bool)
    initial = hard_from_graph(graph_from_decoder(decoder.decode(latent), decoder.spec, fixed_active=active))
    candidate = Candidate("CCC", latent, initial, 1., "saved-parent", 17)
    calls = []

    def initialize(_decoder, parent, mask, *, seed, **kwargs):
        calls.append(seed)
        offset = torch.randn(parent.latent.shape, generator=torch.Generator().manual_seed(seed))*.01
        return parent.latent+offset, None, dict(seed=seed)

    monkeypatch.setattr(proposals, "propose_start", initialize)
    design = LatentDesignConfig.for_restart(dict(updates=2, schedule_horizon=2, occupied_atoms=3,
        learning_rate=1., max_step_norm=.25, chemistry_initial=1., chemistry_final=1., chemistry_hold=0,
        chemistry_ramp_end=2, temperature_initial=1., temperature_final=1., noise_initial=0., noise_zero_at=2))
    first = speed.run_precision_child(decoder, MockScorer(), dict(active=active), candidate, design, 73, tmp_path/"a")
    second = speed.run_precision_child(decoder, MockScorer(), dict(active=active), candidate, design, 73, tmp_path/"b")
    assert calls == [73, 73]  # no second perturbation inside execute_branches
    assert first["initial_point_sha256"] == second["initial_point_sha256"]
    assert first["initial_latent_sha256"] == second["initial_latent_sha256"]
    assert first["initial_hard_graph"] == second["initial_hard_graph"]
    assert first["terminal_latent"] == second["terminal_latent"]
    assert first["updates"] == 2 and len(first["observations"]) == 3
    first["execution"] = second["execution"] = dict(autocast="none")
    union = speed.precision_union(dict(fp32=first, fp32_repeat=second))
    result = speed.summarize_precision_pair(dict(fp32=first, fp32_repeat=second),
        [dict(row, loss=1.) for row in union], [0, 1, 2])
    assert result["arms"]["fp32"]["by_update"] == result["arms"]["fp32_repeat"]["by_update"]
