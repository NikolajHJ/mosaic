"""Bounded-witness guidance engineering fixtures; execute only on DTU HPC."""

import copy

import pytest
import torch
from torch.nn import functional as F

from latentmol.chemistry import inspect_graph
from latentmol.probability_repair import ProbabilityRepairConfig, measure_probability_edits, probability_identity
from latentmol.quality import graph_identity
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.search_repair_guidance import SearchRepairTeacher


def _decoded(spec, kinds=("invalid",)):
    batch, n = len(kinds), spec.max_nodes
    atoms = torch.full((batch, n, len(spec.atomic_numbers)), -12., dtype=torch.float64)
    charges = torch.full((batch, n, len(spec.formal_charges)), -12., dtype=torch.float64)
    bonds = torch.full((batch, n, n, len(spec.bond_names)), -12., dtype=torch.float64)
    atoms[..., 0] = 0
    charges[..., spec.formal_charges.index(0)] = 0
    bonds[..., 0] = 0
    for row, kind in enumerate(kinds):
        if kind == "empty":
            continue
        atoms[row, :2, 0] = -12
        atoms[row, 0, spec.atomic_numbers.index(8 if kind == "invalid" else 6)] = 0
        atoms[row, 1, spec.atomic_numbers.index(6)] = 0
        label = spec.bond_names.index("TRIPLE" if kind == "invalid" else "SINGLE")
        bonds[row, 0, 1, 0] = bonds[row, 1, 0, 0] = -12
        bonds[row, 0, 1, label] = bonds[row, 1, 0, label] = 0
    return DecoderOutput(*(value.requires_grad_() for value in (atoms, charges, bonds)))


def _teacher(spec, count=1):
    panel = torch.arange(count, dtype=torch.float64)[:, None]
    return SearchRepairTeacher(panel, spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=16, beam_width=2),
                               generator=torch.Generator().manual_seed(941))


def _repair_result(raw, probabilities, spec, config, *, status="witness_found"):
    inspected = inspect_graph(*raw, spec)
    result = {"policy": config.policy(), "raw_graph_id": graph_identity(*raw, spec),
              "probability_sha256": probability_identity(probabilities, spec),
              "raw_valid": inspected.smiles is not None, "checks": 0,
              "checks_by_depth": [], "status": status}
    if inspected.smiles is not None:
        result["status"] = "already_valid"
        return result
    if status != "witness_found":
        return result
    fixed = tuple(value.clone() for value in raw)
    fixed[2][0, 1] = fixed[2][1, 0] = spec.bond_names.index("DOUBLE")
    witness_inspected = inspect_graph(*fixed, spec)
    assert witness_inspected.smiles is not None
    witness = {**measure_probability_edits(raw, fixed, probabilities, spec, config),
               "graph_id": graph_identity(*fixed, spec), "smiles": witness_inspected.smiles,
               "labels": {name: value.tolist() for name, value in zip(("atoms", "charges", "bonds"), fixed)}}
    return {**result, "checks": 3, "checks_by_depth": [{"depth": 1, "checks": 3, "capped": False}],
            "weighted_witness": witness, "minimum_edit_witness": witness}


def test_refresh_is_detached_fixed_quota_no_rng_and_full_witness_receipt(tmp_path, monkeypatch):
    spec = GraphSpec(max_nodes=3)
    teacher = _teacher(spec, 2)
    calls, returned = [], []
    def repair(raw, probabilities, spec, config):
        assert not torch.is_grad_enabled()
        assert all(not value.requires_grad for value in probabilities)
        result = _repair_result(raw, probabilities, spec, config)
        calls.append(raw)
        returned.append(result)
        return result
    monkeypatch.setattr("latentmol.search_repair_guidance.search_probability_repair", repair)
    generator_state = teacher.generator.get_state()
    receipt = teacher.refresh(_decoded(spec, ("invalid", "valid")), update=0, directory=tmp_path)
    assert len(calls) == 2 and receipt["attempted_candidates"] == 2
    assert receipt["accepted_targets"] == 1
    assert receipt["guidance_status_counts"] == {"accepted": 1, "skipped_already_valid": 1}
    assert receipt["search_checks"] == 3 and not receipt["rng_consumed"]
    assert receipt["explicit_raw_inspection_calls"] == 2
    assert receipt["explicit_witness_inspection_calls"] == 1
    assert torch.equal(generator_state, teacher.generator.get_state())
    assert (tmp_path / "search-refresh-00000000.json").is_file()
    target = receipt["attempts"][0]["target"]
    assert target["decisions"] == 3 + 2 + 1
    assert target["edit_count"] == 1
    assert target["labels"]["atoms"][2] == 0
    # Neither the returned receipt nor the external search dict owns cached
    # target tensors: mutating those copies cannot change held guidance.
    receipt["attempts"][0]["target"]["labels"]["atoms"][0] = 0
    returned[0]["weighted_witness"]["labels"]["atoms"][0] = 0
    assert teacher.objective(_decoded(spec, ("invalid", "valid")))["eligible"] == 1


def test_unconditional_nll_matches_full_categorical_formula_and_gradients(monkeypatch):
    spec = GraphSpec(max_nodes=3)
    teacher = _teacher(spec)
    monkeypatch.setattr("latentmol.search_repair_guidance.search_probability_repair", _repair_result)
    refresh = teacher.refresh(_decoded(spec), update=0)
    target = refresh["attempts"][0]["target"]
    generator = torch.Generator().manual_seed(937)
    values = (torch.randn(1, 3, 10, generator=generator, dtype=torch.float64, requires_grad=True),
              torch.randn(1, 3, 5, generator=generator, dtype=torch.float64, requires_grad=True),
              torch.randn(1, 3, 3, 5, generator=generator, dtype=torch.float64, requires_grad=True))
    output = DecoderOutput(*values)
    actual = teacher.objective(output)["nll"]
    atoms, charges, bonds = (torch.tensor(target["labels"][name], dtype=torch.long) for name in ("atoms", "charges", "bonds"))
    active = atoms != 0
    atom_term = F.cross_entropy(output.atoms[0], atoms, reduction="sum")
    charge_term = F.cross_entropy(output.charges[0, active], charges[active], reduction="sum")
    pairs = (active[:, None] & active[None, :]).triu(1)
    bond_term = F.cross_entropy(output.symmetric_bond_logits[0, pairs], bonds[pairs], reduction="sum")
    exact = (atom_term + charge_term + bond_term) / target["decisions"]
    torch.testing.assert_close(actual, exact)
    actual_gradients = torch.autograd.grad(actual, values, retain_graph=True)
    expected_gradients = torch.autograd.grad(exact, values)
    for observed, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(observed, expected)
    # The padded NO_ATOM decision contributes: this is not a conditioned loss.
    assert actual_gradients[0][0, 2, 0] < 0
    assert torch.count_nonzero(actual_gradients[1][0, 2]) == 0
    assert torch.count_nonzero(actual_gradients[2][0, 2]) == 0


def test_extremely_unlikely_targets_keep_finite_nonzero_gradients_without_floors(monkeypatch):
    spec = GraphSpec(max_nodes=3)
    teacher = _teacher(spec)
    monkeypatch.setattr("latentmol.search_repair_guidance.search_probability_repair", _repair_result)
    receipt = teacher.refresh(_decoded(spec), update=0)
    labels = receipt["attempts"][0]["target"]["labels"]
    output = _decoded(spec)
    with torch.no_grad():
        for slot, label in enumerate(labels["atoms"]):
            output.atoms[0, slot, label] = -1000
        for slot in (0, 1):
            output.charges[0, slot, labels["charges"][slot]] = -1000
        bond = labels["bonds"][0][1]
        output.bonds[0, 0, 1, bond] = output.bonds[0, 1, 0, bond] = -1000
    objective = teacher.objective(output)["total"]
    assert torch.isfinite(objective) and objective > 100
    gradients = torch.autograd.grad(objective, (output.atoms, output.charges, output.bonds))
    assert all(torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients)


def test_failed_refresh_clears_stale_targets_and_returns_explicit_zero_gradients(monkeypatch):
    spec = GraphSpec(max_nodes=3)
    teacher = _teacher(spec)
    monkeypatch.setattr("latentmol.search_repair_guidance.search_probability_repair", _repair_result)
    assert teacher.refresh(_decoded(spec), update=0)["accepted_targets"] == 1
    monkeypatch.setattr("latentmol.search_repair_guidance.search_probability_repair",
                        lambda *args: _repair_result(*args, status="censored"))
    receipt = teacher.refresh(_decoded(spec), update=1)
    assert receipt["accepted_targets"] == 0 and receipt["cleared_old_targets"]
    output = _decoded(spec)
    observed = teacher.objective(output)
    assert observed["status"] == "unavailable_no_valid_witness" and observed["total"] == 0
    assert observed["eligible_only_nll"] is None
    gradients = torch.autograd.grad(observed["total"], (output.atoms, output.charges, output.bonds))
    assert all(torch.count_nonzero(value) == 0 for value in gradients)


def test_raw_occupancy_can_change_next_refresh_but_each_witness_retains_current_slots(monkeypatch):
    spec = GraphSpec(max_nodes=3)
    teacher = _teacher(spec)
    monkeypatch.setattr("latentmol.search_repair_guidance.search_probability_repair", _repair_result)
    first = teacher.refresh(_decoded(spec), update=0)
    assert first["raw_atom_counts"] == [2]
    def unsupported(*args):
        return _repair_result(*args, status="unsupported")
    monkeypatch.setattr("latentmol.search_repair_guidance.search_probability_repair", unsupported)
    second = teacher.refresh(_decoded(spec, ("empty",)), update=1)
    assert second["raw_atom_counts"] == [0]
    assert second["changed_occupancy_latent_ids"] == [0]
    assert second["atom_count_changes_from_previous_refresh"] == [-2]
    assert second["guidance_status_counts"] == {"unsupported": 1}
    assert not teacher.objective(_decoded(spec))["available_mask"].any()


def test_batch_ids_and_full_batch_mean_retain_unavailable_zero_weight(monkeypatch):
    spec = GraphSpec(max_nodes=3)
    teacher = _teacher(spec, 3)
    monkeypatch.setattr("latentmol.search_repair_guidance.search_probability_repair", _repair_result)
    teacher.refresh(_decoded(spec, ("invalid", "valid", "invalid")), update=0)
    output = _decoded(spec, ("invalid", "valid"))
    result = teacher.objective(output, latent_ids=[2, 1])
    assert result["eligible"] == 1 and result["attempted"] == 2
    torch.testing.assert_close(result["total"], result["eligible_only_nll"] / 2)
    assert result["available_mask"].tolist() == [True, False]
    with pytest.raises(ValueError, match="explicit teacher latent IDs"):
        teacher.objective(output)
    with pytest.raises(ValueError, match="disagree"):
        teacher.objective(output, latent_ids=[2, 8])


@pytest.mark.parametrize("mutation", ["occupancy", "invalid", "cost", "probability", "boolean", "slot_permutation"])
def test_bad_witness_scope_or_provenance_is_retained_as_rejected(monkeypatch, mutation):
    spec = GraphSpec(max_nodes=3)
    teacher = _teacher(spec)
    def wrong(*args):
        result = copy.deepcopy(_repair_result(*args))
        witness = result["weighted_witness"]
        if mutation == "occupancy":
            witness["labels"]["atoms"][2] = spec.atomic_numbers.index(6)
        elif mutation == "invalid":
            witness["labels"]["bonds"][0][1] = witness["labels"]["bonds"][1][0] = spec.bond_names.index("TRIPLE")
        elif mutation == "cost":
            witness["weighted_cost"] += 1
        elif mutation == "probability":
            result["probability_sha256"] = "a" * 64
        elif mutation == "boolean":
            witness["labels"]["atoms"][0] = True
        else:
            witness["labels"]["atoms"] = [0, witness["labels"]["atoms"][0], witness["labels"]["atoms"][1]]
        return result
    monkeypatch.setattr("latentmol.search_repair_guidance.search_probability_repair", wrong)
    receipt = teacher.refresh(_decoded(spec), update=0)
    assert receipt["accepted_targets"] == 0
    assert receipt["guidance_status_counts"] == {"rejected_or_error": 1}
    assert "error" in receipt["attempts"][0]
    assert teacher.objective(_decoded(spec))["total"] == 0


def test_held_target_and_panel_mutation_fail_closed_and_refresh_receipts_are_exclusive(tmp_path, monkeypatch):
    spec = GraphSpec(max_nodes=3)
    teacher = _teacher(spec)
    monkeypatch.setattr("latentmol.search_repair_guidance.search_probability_repair", _repair_result)
    teacher.refresh(_decoded(spec), update=0, directory=tmp_path)
    with pytest.raises(ValueError, match="increase monotonically"):
        teacher.refresh(_decoded(spec), update=0, directory=tmp_path)
    teacher._targets[0].graph.atoms[0, 0] = 0
    with pytest.raises(ValueError, match="identity changed"):
        teacher.objective(_decoded(spec))
    fresh = _teacher(spec)
    snapshot = fresh.panel
    snapshot[0, 0] = 99
    assert fresh.panel[0, 0] == 0
    fresh._panel[0, 0] = 99
    with pytest.raises(ValueError, match="mutated"):
        fresh.objective(_decoded(spec))


def test_actual_bounded_search_guidance_records_budget_without_a_success_claim():
    spec = GraphSpec(max_nodes=3)
    teacher = _teacher(spec)
    receipt = teacher.refresh(_decoded(spec), update=0)
    assert receipt["attempted_candidates"] == 1
    assert receipt["search_checks"] <= 16
    assert receipt["accepted_targets"] in (0, 1)
    assert not receipt["validity_guarantee"]
    objective = teacher.objective(_decoded(spec))
    assert torch.isfinite(objective["total"])


def test_unavailable_half_precision_zero_does_not_overflow_full_logit_sums():
    spec = GraphSpec(max_nodes=50)
    teacher = _teacher(spec)
    values = (torch.full((1, 50, 10), 1000., dtype=torch.float16, requires_grad=True),
              torch.full((1, 50, 5), 1000., dtype=torch.float16, requires_grad=True),
              torch.full((1, 50, 50, 5), 1000., dtype=torch.float16, requires_grad=True))
    result = teacher.objective(DecoderOutput(*values))
    assert result["total"] == 0 and torch.isfinite(result["total"])
    gradients = torch.autograd.grad(result["total"], values)
    assert all(torch.count_nonzero(value) == 0 for value in gradients)
