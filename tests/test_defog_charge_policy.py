"""HPC-only fixtures for the explicit lossy charge policy; native proof separate.

Authored without local numerical execution. These mocked sampling transitions
test contracts, not DeFoG repair efficacy. charge_policy_diagnostic compares the
actual pinned builder separately, before any optional-mode optimization.
"""

from copy import deepcopy

import pytest
import torch

from latentmol.defog.candidate_repair import _validated_request, repair_request, request_digest
from latentmol.defog.charge_policy_diagnostic import parity_cases, molecule_record
from latentmol.defog.codec import (PROJECT_CHARGE_POLICY, from_guacamol_native_charges,
    native_charge_molecule, projection_receipt, to_guacamol_hard)
from latentmol.repair_guidance.config import RepairGuidanceConfig
from latentmol.repair_guidance.loss import FrozenRepairTarget, target_score_at
from latentmol.repair_guidance.provider import DeFoGRepairProvider, RepairProviderContractError
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from test_defog_candidate_repair import _candidate, _request, _MockNative, _State
from test_repair_guidance_provider import _config


class _EchoNative(_MockNative):
    """Same-shape, unchanged-label return solely for transport engineering."""
    def sample_p_zs_given_zt(self, t, s, x, e, y, node_mask):
        self.calls += 1
        self.model(x)
        self.native_inputs.append((x.clone(), e.clone(), y.clone()))
        edges = torch.nn.functional.one_hot(e.argmax(-1), e.shape[-1]).type_as(e)
        return _State(x.clone(), edges, y), _State(x.argmax(-1).type_as(y), edges.argmax(-1).type_as(y), y)


def _projected_request(spec, graph):
    return _request(spec, graph, version=2, charge_policy=PROJECT_CHARGE_POLICY,
        occupied_slots=(graph.atoms[0] != 0).nonzero(as_tuple=False).flatten().tolist(),
        initialization="direct_candidate_no_corruption", sampling={})


def _repair(request):
    model = _EchoNative()
    response = repair_request(request, model_loader=lambda *_: model, asset_verifier=lambda *_: None)
    return response, model


def _provider(tmp_path):
    raw = _config(tmp_path).record()
    raw["provider"]["charge_policy"] = PROJECT_CHARGE_POLICY
    return DeFoGRepairProvider(RepairGuidanceConfig.from_mapping(raw, tmp_path), "a" * 64)


def _charged_star():
    spec = GraphSpec(max_nodes=7)
    slots = (0, 2, 3, 5, 6)
    atoms = torch.zeros((1, 7), dtype=torch.long)
    charges = torch.full_like(atoms, spec.formal_charges.index(0))
    bonds = torch.zeros((1, 7, 7), dtype=torch.long)
    atoms[0, list(slots)] = 1
    atoms[0, 0] = spec.atomic_numbers.index(7)
    charges[0, 0] = spec.formal_charges.index(1)
    for slot in slots[1:]:
        bonds[0, 0, slot] = bonds[0, slot, 0] = 1
    return spec, HardGraphBatch(atoms, charges, bonds)


@pytest.mark.parametrize("charge", [-2, -1, 1, 2])
def test_charge_only_alternatives_share_native_projection_but_not_candidate_identity(charge):
    spec, neutral = _candidate()
    _, charged = _candidate(charge=charge)
    before = [v.clone() for v in (charged.atoms, charged.charges, charged.bonds)]
    plain = to_guacamol_hard(neutral, spec)
    view = to_guacamol_hard(charged, spec, charge_policy=PROJECT_CHARGE_POLICY)
    assert torch.equal(view.x, plain.x) and torch.equal(view.e, plain.e)
    assert torch.equal(view.node_mask, plain.node_mask) and view.occupied_slots == (0, 3)
    one, two = projection_receipt(neutral, spec, plain), projection_receipt(charged, spec, view)
    assert one["native_projection_sha256"] == two["native_projection_sha256"]
    assert two["canonical_input_formal_charges"] == [charge, charge]
    assert two["nonzero_charge_slots"] == [0, 3] and not two["canonical_candidate_modified"]
    assert _projected_request(spec, charged)["request_sha256"] != _projected_request(spec, neutral)["request_sha256"]
    for value, original in zip((charged.atoms, charged.charges, charged.bonds), before):
        assert torch.equal(value, original)
    with pytest.raises(ValueError, match="Formal charges"):
        to_guacamol_hard(charged, spec)


@pytest.mark.parametrize("category", [-1, 5])
def test_projection_does_not_hide_invalid_charge_categories(category):
    spec, graph = _candidate()
    graph.charges[0, 0] = category
    with pytest.raises(ValueError, match="charge category"):
        to_guacamol_hard(graph, spec, charge_policy=PROJECT_CHARGE_POLICY)


def test_legacy_config_and_transport_do_not_gain_optional_fields(tmp_path):
    config = _config(tmp_path)
    assert "charge_policy" not in config.record()["provider"]
    assert RepairGuidanceConfig.from_mapping(config.record(), tmp_path).record() == config.record()
    spec, graph = _candidate()
    request = _request(spec, graph)
    original = deepcopy(request)
    strict, _ = _repair(request)
    projected, _ = _repair(_projected_request(spec, graph))
    assert request == original and request["version"] == strict["version"] == 1
    assert strict["provenance"]["charged_input_policy"] == "reject"
    assert "input_feature_loss" not in strict and "output_charge_inference" not in strict
    assert "target_sha256" not in strict and strict["gates"]["neutral_charge_subset"]
    assert strict["native_input_labels"] == projected["native_input_labels"]
    assert strict["native_output_labels"] == projected["native_output_labels"]
    assert strict["target"] == projected["target"]


@pytest.mark.parametrize("version,policy", [(1, PROJECT_CHARGE_POLICY), (2, None), (2, "neutralize")])
def test_explicit_policy_requires_its_own_transport_version(version, policy):
    spec, graph = _candidate()
    request = _request(spec, graph, version=version)
    if policy is not None:
        request["charge_policy"] = policy
    request["request_sha256"] = request_digest(request)
    with pytest.raises(ValueError, match="Charge projection"):
        _validated_request(request)


def test_charged_native_target_preserves_slots_and_parent_recomputes_inference(tmp_path):
    spec, graph = _charged_star()
    request = _projected_request(spec, graph)
    response, model = _repair(request)
    assert response["status"] == "accepted", response
    assert model.calls > 0 and response["version"] == 2
    assert response["output_charge_inference"]["formal_charges"] == [1, 0, 0, 0, 0]
    assert response["occupied_slots"] == [0, 2, 3, 5, 6]
    target = _provider(tmp_path)._verify_response(response, request, spec, graph, graph.atoms[0] != 0, 0)
    assert torch.equal(target.charges, graph.charges)
    assert torch.equal(target.atoms, graph.atoms) and torch.equal(target.bonds, graph.bonds)


@pytest.mark.parametrize("mutation", [
    lambda row: row["input_feature_loss"].update(native_projection_sha256="0" * 64),
    lambda row: row["input_feature_loss"].update(canonical_input_formal_charges=[0] * 5),
    lambda row: row["output_charge_inference"].update(formal_charges=[0] * 5),
    lambda row: row["output_charge_inference"].update(neural_charge_prediction=True),
    lambda row: row["target"]["charges"][0].__setitem__(0, 2),
    lambda row: row.update(target_sha256="0" * 64),
    lambda row: row["native_output_labels"]["atoms"][0].__setitem__(0, 0),
])
def test_charge_blind_roundtrip_cannot_hide_forged_output_charges(tmp_path, mutation):
    spec, graph = _charged_star()
    request = _projected_request(spec, graph)
    response, _ = _repair(request)
    mutation(response)
    with pytest.raises(RepairProviderContractError):
        _provider(tmp_path)._verify_response(response, request, spec, graph, graph.atoms[0] != 0, 0)


def test_negative_candidate_charge_is_disclosed_not_claimed_reconstructed():
    spec, graph = _candidate(charge=-1)
    response, _ = _repair(_projected_request(spec, graph))
    assert response["status"] == "accepted", response
    assert response["input_feature_loss"]["canonical_input_formal_charges"] == [-1, -1]
    assert response["output_charge_inference"]["formal_charges"] == [0, 0]
    assert response["edits"]["charge_changes"] == 2
    assert graph.charges[0, 0] == spec.formal_charges.index(-1)


def test_native_charge_rule_cases_include_rejected_full_graphs_without_correction():
    rows = {}
    for name, x, e, expected in parity_cases():
        record = molecule_record(native_charge_molecule(x, e))
        assert record["formal_charges"] == expected
        rows[name] = record
    assert len(rows) == 11
    assert not rows["invalid_carbon_not_corrected"]["sanitized"]
    assert rows["disconnected_not_cropped"]["connected_components"] == 2


@pytest.mark.parametrize("element", [4, 10, 11])
def test_native_unsupported_elements_remain_rejected(element):
    with pytest.raises(ValueError, match="unsupported"):
        from_guacamol_native_charges(torch.tensor([[element, 0]]),
            torch.tensor([[[0, 1], [1, 0]]]), GraphSpec(), (0, 3))


def test_inferred_charge_reference_has_decoder_gradient_without_a_charge_model():
    spec = GraphSpec(max_nodes=5)
    _, x, e, _ = next(row for row in parity_cases() if row[0] == "native_N_plus")
    graph, _ = from_guacamol_native_charges(x, e, spec, tuple(range(5)))
    active = torch.ones(5, dtype=torch.bool)
    target = FrozenRepairTarget.create(graph, active, refresh_step=0, receipt_path="fixture")
    class Decoder:
        def __init__(self):
            self.spec = spec
        def decode(self, point):
            selected = torch.zeros((1, 5, 5), dtype=point.dtype)
            selected[0, 0, spec.formal_charges.index(1)] = 1
            return DecoderOutput(torch.zeros((1, 5, 10), dtype=point.dtype), point * selected,
                                 torch.zeros((1, 5, 5, 5), dtype=point.dtype))
    score, gradient = target_score_at(Decoder(), torch.zeros((1, 1), dtype=torch.float64), active, target, gradients=True)
    assert gradient.item() == pytest.approx(-.8)
    assert score["target_sha256"] == target.identity and not target.graph.charges.requires_grad


def test_disconnected_native_output_is_not_cropped_into_an_accepted_reference():
    spec, graph = _candidate(charge=1)
    graph.bonds.zero_()
    response, _ = _repair(_projected_request(spec, graph))
    assert response["status"] == "rejected" and response["target"] is None
    assert response["reason"] == "output_chemical_invalid"
    assert response["output_inspection"]["failure_reason"] == "disconnected"
    assert response["native_output_labels"]["atoms"] == [[0, 0]]


def test_full_sanitization_cannot_supply_undeclared_nitro_charge_or_bond_corrections(tmp_path):
    # Native per-bond rule does not assign +1 for N valence5. Full RDKit cleanup
    # may turn neutral N(=O)=O into N+/O-; the optional target must reject that
    # later rewrite instead of claiming those charges were native-rule outputs.
    x = torch.tensor([[0, 1, 2, 2]])
    e = torch.zeros((1, 4, 4), dtype=torch.long)
    for i, j, bond in ((0, 1, 1), (1, 2, 2), (1, 3, 2)):
        e[0, i, j] = e[0, j, i] = bond
    target, receipt = from_guacamol_native_charges(x, e, GraphSpec(), (0, 1, 2, 3))
    assert receipt["formal_charges"] == [0, 0, 0, 0]
    assert receipt["sanitization_identity"]["passed"] is False
    assert receipt["sanitization_identity"]["reason"] == "sanitization_changed_declared_labels"
    assert bool((target.charges == GraphSpec().formal_charges.index(0)).all())
    assert torch.equal(target.bonds[:, :4, :4], e)
    # Exercise the worker and independent provider too: a projected X/E
    # roundtrip alone cannot authorize cleanup-derived chemical identities.
    spec = GraphSpec()
    request = _projected_request(spec, target)
    response, _ = _repair(request)
    assert response["status"] == "rejected" and response["target"] is None
    assert response["reason"] == "output_sanitization_changes_or_rejects_declared_labels"
    assert response["gates"]["output_identity_preserved"] is False
    provider = _provider(tmp_path)
    active = target.atoms[0] != 0
    assert provider._verify_response(response, request, spec, target, active, 0) is None
    forged = deepcopy(response)
    forged.update(status="accepted", target=forged["proposed_target"])
    forged["gates"].update(output_identity_preserved=True, chemical_validity=True)
    forged["output_charge_inference"]["sanitization_identity"]["passed"] = True
    with pytest.raises(RepairProviderContractError, match="independent native-rule"):
        provider._verify_response(forged, request, spec, target, active, 0)
