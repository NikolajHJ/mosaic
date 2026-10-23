"""HPC-only engineering fixtures for candidate-start mapping and bookkeeping.

Mock transitions establish the wrapper's contract, not native DeFoG efficacy.
This suite has not been run when authored; the native checkpoint requires its
own forward/restart/parity/chemistry gate in an immutable HPC release.
"""

from types import SimpleNamespace

import pytest
import torch

from latentmol.defog.candidate_repair import (
    NativeTransitionContractError, _validated_request, graph_record, native_time_grid,
    repair_request, request_digest, validate_native_transition,
)
from latentmol.defog.codec import from_guacamol_labels, to_guacamol_hard
from latentmol.defog.interface_smoke import DEFOG_COMMIT, GUACAMOL_SHA256
from latentmol.representation import GraphSpec, HardGraphBatch


def _candidate(*, numbers=(6, 6), charge=0):
    spec = GraphSpec(max_nodes=5)
    atoms = torch.zeros((1, 5), dtype=torch.long)
    charges = torch.full_like(atoms, spec.formal_charges.index(0))
    bonds = torch.zeros((1, 5, 5), dtype=torch.long)
    for slot, number in zip((0, 3), numbers):
        atoms[0, slot] = spec.atomic_numbers.index(number)
        charges[0, slot] = spec.formal_charges.index(charge)
    bonds[0, 0, 3] = bonds[0, 3, 0] = 1
    return spec, HardGraphBatch(atoms, charges, bonds)


def _request(spec, graph, **changes):
    value = {"version": 1, "spec": spec.to_dict(), "candidate": graph_record(graph),
             "occupied_slots": [0, 3], "seed": 17, "t0": 0.8, "steps": 10,
             "source": {"path": "/test/defog", "commit": DEFOG_COMMIT},
             "checkpoint": {"path": "/test/guacamol.ckpt", "sha256": GUACAMOL_SHA256},
             "adapter_source_sha256": "a" * 64, "device": "cpu"}
    value.update(changes)
    value["request_sha256"] = request_digest(value)
    return value


class _State:
    def __init__(self, x, e, y):
        self.X, self.E, self.y = x, e, y

    def mask(self, _mask, collapse=False):
        return (_State(self.X.argmax(-1), self.E.argmax(-1), self.y)
                if collapse else self)


class _MockNative(torch.nn.Module):
    """Minimal native transition shape; never stands in for pretrained weights."""
    def __init__(self, *, output_labels=(0, 1), fail_on=None):
        super().__init__()
        self.model = torch.nn.Identity()
        self.frozen_fixture_weight = torch.nn.Parameter(torch.tensor(1.0))
        self.cfg = SimpleNamespace(sample=SimpleNamespace(
            sample_steps=1000, time_distortion="identity", eta=7.0, omega=0.2,
            rdb="general", rdb_crit="max_marginal"))
        self.time_distorter = SimpleNamespace(sample_ft=lambda t, d:
            2 * t - t**2 if d == "polydec" else t)
        self.rate_matrix_designer = SimpleNamespace()
        self.noise_dist = SimpleNamespace(ignore_virtual_classes=lambda x, e, y: (x, e, y))
        self.conditional = False
        self.output_dims = {"X": 12, "E": 5, "y": 0}
        self.node_dist = SimpleNamespace(prob=torch.ones(6))
        self.output_labels = output_labels
        self.fail_on = fail_on
        self.calls = 0
        self.corruption_calls = 0
        self.native_inputs = []

    def apply_noise(self, x, e, y, mask, t):
        self.corruption_calls += 1
        noisy = x.clone()
        noisy[0, 0] = torch.nn.functional.one_hot(torch.tensor(1), 12).float()
        return {"X_t": noisy, "E_t": e.clone(), "y_t": y}

    def sample_p_zs_given_zt(self, t, s, x, e, y, node_mask):
        self.calls += 1
        self.model(x)  # The wrapper's hook counts actual native-network calls.
        if self.calls == self.fail_on:
            raise RuntimeError("deliberate native transition failure")
        self.native_inputs.append((x.clone(), e.clone(), y.clone()))
        labels = torch.tensor([self.output_labels], dtype=torch.long, device=x.device)
        out_x = torch.nn.functional.one_hot(labels, 12).float()
        # Mirror the pinned native return, not the clean-input codec: sampling
        # reflects a strict upper triangle (diagonal label0), then one-hots ALL
        # edge labels. PlaceHolder.mask does not erase the NONE diagonal.
        out_e = torch.nn.functional.one_hot(e.argmax(-1), e.shape[-1]).type_as(e)
        state = _State(out_x, out_e, y)
        return state, _State(out_x.argmax(-1).type_as(y), out_e.argmax(-1).type_as(y), y)


def _repair(request, model):
    return repair_request(request, model_loader=lambda *_: model,
                          asset_verifier=lambda *_: None)


def test_hard_native_roundtrip_preserves_noncontiguous_slots():
    spec, graph = _candidate(numbers=(7, 8))
    mapped = to_guacamol_hard(graph, spec)
    restored = from_guacamol_labels(mapped.x.argmax(-1), mapped.e.argmax(-1), spec,
                                    mapped.occupied_slots)
    for name in ("atoms", "charges", "bonds"):
        assert torch.equal(getattr(restored, name), getattr(graph, name))
    assert mapped.occupied_slots == (0, 3)


@pytest.mark.parametrize("unsupported", [4, 10, 11])  # B, Se, Si in GuacaMol.
def test_native_unsupported_emission_is_not_relabelled(unsupported):
    spec, graph = _candidate()
    mapped = to_guacamol_hard(graph, spec)
    labels = mapped.x.argmax(-1)
    labels[0, 0] = unsupported
    with pytest.raises(ValueError, match="unsupported"):
        from_guacamol_labels(labels, mapped.e.argmax(-1), spec, mapped.occupied_slots)


@pytest.mark.parametrize("mutate", [
    lambda value: value.update(seed=True),
    lambda value: value.update(steps=1),
    lambda value: value.update(t0=float("nan")),
    lambda value: value.update(occupied_slots=[3, 0]),
    lambda value: value.update(initialization="unconditional_noise"),
    lambda value: value.update(sampling={"eta": -1}),
    lambda value: value["checkpoint"].update(sha256="b" * 64),
])
def test_contract_rejects_invalid_settings(mutate):
    spec, graph = _candidate()
    request = _request(spec, graph)
    mutate(request)
    # Invalid NaN JSON is rejected by hashing itself, before numerical execution.
    with pytest.raises(ValueError):
        request["request_sha256"] = request_digest(request)
        _validated_request(request)


def test_transport_hash_includes_candidate_and_configuration():
    spec, graph = _candidate()
    request = _request(spec, graph)
    request["candidate"]["atoms"][0][0] = 2
    with pytest.raises(ValueError, match="SHA256"):
        _validated_request(request)


def test_charged_attempt_is_retained_without_any_model_call():
    spec, graph = _candidate(charge=1)
    model = _MockNative()
    result = _repair(_request(spec, graph), model)
    assert result["status"] == "rejected"
    assert result["reason"] == "input_representation_unsupported"
    assert result["target"] is None
    assert result["cost"]["native_forward_calls"] == 0
    assert model.calls == 0


def test_accepted_target_keeps_slots_and_records_checkpoint_settings_unchanged():
    spec, graph = _candidate()
    model = _MockNative()
    result = _repair(_request(spec, graph), model)
    assert result["status"] == "accepted", result
    assert result["occupied_slots"] == [0, 3]
    assert result["native_order_preserved"]
    assert result["target"]["atoms"][0][3] == spec.atomic_numbers.index(7)
    assert result["target"]["atoms"][0][1] == 0
    assert result["edits"] == {"atom_changes": 1, "bond_changes": 0, "charge_changes": 0}
    assert result["cost"]["native_transitions"] == 2
    assert result["cost"]["native_forward_calls"] == 2
    assert result["cost"]["effective_sampling"]["eta"] == 7.0
    assert result["cost"]["effective_sampling"]["time_distortion"] == "identity"
    assert result["provenance"]["frozen_state_unchanged"]
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_invalid_candidate_is_not_pre_sanitized_or_excluded():
    spec, graph = _candidate(numbers=(8, 8))
    graph.bonds[0, 0, 3] = graph.bonds[0, 3, 0] = 3  # Neutral O#O is invalid.
    model = _MockNative(output_labels=(0, 0))
    result = _repair(_request(spec, graph), model)
    assert result["input_inspection"]["smiles"] is None
    assert model.calls == 2
    # C#C is valid; native labels reach the reverse mapping without SMILES reindex.
    assert result["status"] == "accepted", result
    assert result["provenance"]["input_sanitization"] == "none"


def test_invalid_output_and_unsupported_outputs_keep_cost_and_native_labels():
    spec, graph = _candidate()
    graph.bonds[0, 0, 3] = graph.bonds[0, 3, 0] = 3
    invalid = _repair(_request(spec, graph), _MockNative(output_labels=(2, 2)))
    unsupported = _repair(_request(spec, graph), _MockNative(output_labels=(4, 0)))
    assert invalid["reason"] == "output_chemical_invalid"
    assert invalid["proposed_target"] is not None
    assert unsupported["reason"] == "output_representation_unsupported"
    for result in (invalid, unsupported):
        assert result["status"] == "rejected"
        assert result["target"] is None
        assert result["native_output_labels"] is not None
        assert result["cost"]["native_forward_calls"] == 2


def test_partial_native_failure_keeps_attempted_completed_and_forward_counts():
    spec, graph = _candidate()
    result = _repair(_request(spec, graph), _MockNative(fail_on=2))
    assert result["status"] == "error"
    assert result["target"] is None
    assert result["cost"]["attempted_native_transitions"] == 2
    assert result["cost"]["native_transitions"] == 1
    assert result["cost"]["native_forward_calls"] == 2
    assert len(result["cost"]["trajectory"]) == 1


def test_explicit_corruption_calls_native_apply_noise_once_and_records_changes():
    spec, graph = _candidate()
    model = _MockNative()
    result = _repair(_request(spec, graph, initialization="checkpoint_forward_corruption"), model)
    assert result["status"] == "accepted", result
    assert model.corruption_calls == 1
    assert result["cost"]["candidate_to_initialized_edits"]["atom_changes"] == 1
    assert result["cost"]["initialization"] == "checkpoint_forward_corruption"


def test_distorted_model_time_is_distinct_from_grid_tail_fraction():
    model = _MockNative()
    grid = native_time_grid(model, steps=1000, distortion="polydec", device="cpu")
    first = int((grid[:-1] >= 0.8).nonzero(as_tuple=False)[0, 0])
    assert first == 553
    assert 1000 - first == 447
    assert 0.8 <= float(grid[first]) < 0.801


def _native_return_views():
    """The exact pinned return convention, independently of model weights."""
    spec, graph = _candidate()
    mapped = to_guacamol_hard(graph, spec)
    # Native F.one_hot(sampled_s.E) includes diagonal category0. The clean
    # input's diagonal vectors remain zero; no bridge mutation is needed.
    edge_labels = mapped.e.argmax(-1)
    one_hot_edges = torch.nn.functional.one_hot(edge_labels, 5).type_as(mapped.e)
    y = mapped.x.new_zeros((1, 0))
    sampled = _State(mapped.x.clone(), one_hot_edges, y)
    discrete = _State(mapped.x.argmax(-1).type_as(y), edge_labels.type_as(y), y.clone())
    return mapped, sampled, discrete


def test_native_none_diagonal_reproduces_original_guard_false_rejection_without_mutation():
    mapped, sampled, discrete = _native_return_views()
    diagonal = torch.arange(mapped.e.shape[1])
    assert not bool(mapped.e[:, diagonal, diagonal, :].any())
    # This exact old guard rejects EVERY native result, even an unchanged valid
    # categorical graph. Self edges are NONE, not physical chemical bonds.
    assert bool(sampled.E[:, diagonal, diagonal, :].any())
    assert bool((sampled.E[:, diagonal, diagonal, 0] == 1).all())
    before = [value.clone() for value in (sampled.X, sampled.E, discrete.X, discrete.E, mapped.e)]
    record = validate_native_transition(sampled, discrete, mapped)
    assert record["passed"] and not record["failed_checks"]
    assert record["checks"]["sampled_E_diagonal_NONE_one_hot"]
    assert record["checks"]["discrete_E_matches_one_hot"]
    assert record["observed_shapes"]["sampled_E"] == [1, 2, 2, 5]
    assert record["observed_dtypes"]["discrete_E"] == "torch.float32"
    for original, value in zip(before, (sampled.X, sampled.E, discrete.X, discrete.E, mapped.e)):
        assert torch.equal(value, original)


@pytest.mark.parametrize("failure", [
    "sampled_X_shape", "sampled_E_shape", "discrete_X_shape", "discrete_E_shape", "sampled_y_shape",
    "sampled_X_dtype", "discrete_E_dtype", "sampled_X_finite", "sampled_E_finite", "discrete_X_finite",
    "sampled_X_one_hot", "sampled_E_one_hot", "sampled_E_symmetric", "sampled_E_diagonal_NONE_one_hot",
    "discrete_X_integer_categories", "discrete_X_category_range", "discrete_X_matches_one_hot",
    "discrete_E_matches_one_hot", "discrete_E_diagonal_NONE",
])
def test_native_contract_rejects_specific_schema_numeric_and_label_drift(failure):
    mapped, sampled, discrete = _native_return_views()
    if failure == "sampled_X_shape": sampled.X = sampled.X[:, :1]
    elif failure == "sampled_E_shape": sampled.E = sampled.E[..., :4]
    elif failure == "discrete_X_shape": discrete.X = discrete.X[..., None]
    elif failure == "discrete_E_shape": discrete.E = discrete.E[:, :1]
    elif failure == "sampled_y_shape": sampled.y = sampled.y.new_zeros((1, 1))
    elif failure == "sampled_X_dtype": sampled.X = sampled.X.double()
    elif failure == "discrete_E_dtype": discrete.E = discrete.E.long()
    elif failure == "sampled_X_finite": sampled.X[0, 0, 0] = float("nan")
    elif failure == "sampled_E_finite": sampled.E[0, 0, 1, 1] = float("inf")
    elif failure == "discrete_X_finite": discrete.X[0, 0] = float("nan")
    elif failure == "sampled_X_one_hot": sampled.X[0, 0, 0], sampled.X[0, 0, 1] = .25, .75
    elif failure == "sampled_E_one_hot": sampled.E[0, 0, 1, 0], sampled.E[0, 0, 1, 1] = .25, .75
    elif failure == "sampled_E_symmetric":
        sampled.E[0, 0, 1] = torch.nn.functional.one_hot(torch.tensor(2), 5).float()
    elif failure == "sampled_E_diagonal_NONE_one_hot":
        sampled.E[0, 0, 0] = 0  # The old mock convention must no longer pass.
    elif failure == "discrete_X_integer_categories": discrete.X[0, 0] = .5
    elif failure == "discrete_X_category_range": discrete.X[0, 0] = 12
    elif failure == "discrete_X_matches_one_hot": discrete.X[0, 0] = 1
    elif failure == "discrete_E_matches_one_hot": discrete.E[0, 0, 1] = discrete.E[0, 1, 0] = 2
    else: discrete.E[0, 0, 0] = 1
    with pytest.raises(NativeTransitionContractError, match=failure) as caught:
        validate_native_transition(sampled, discrete, mapped)
    assert caught.value.receipt["passed"] is False
    assert failure in caught.value.receipt["failed_checks"]


def test_native_none_diagonal_is_passed_unchanged_into_next_transition_and_final_mapping():
    spec, graph = _candidate()
    model = _MockNative()
    result = _repair(_request(spec, graph), model)
    assert result["status"] == "accepted", result
    assert len(model.native_inputs) == 2
    diagonal = torch.arange(2)
    first_edges = model.native_inputs[0][1]
    second_edges = model.native_inputs[1][1]
    assert not bool(first_edges[:, diagonal, diagonal, :].any())
    assert bool((second_edges[:, diagonal, diagonal, 0] == 1).all())
    assert not bool(second_edges[:, diagonal, diagonal, 1:].any())
    assert result["native_output_labels"]["bonds"][0][0][0] == 0
    assert result["native_output_labels"]["bonds"][0][1][1] == 0
    assert result["target"]["bonds"][0][0][0] == result["target"]["bonds"][0][3][3] == 0
    assert result["cost"]["last_native_transition_contract"]["passed"]
    assert result["provenance"]["frozen_state_unchanged"]


def test_failed_second_return_contract_preserves_counters_specific_receipt_and_frozen_state():
    class BadSecond(_MockNative):
        def sample_p_zs_given_zt(self, *args):
            sampled, discrete = super().sample_p_zs_given_zt(*args)
            if self.calls == 2:
                sampled.E[0, 0, 0] = 0
            return sampled, discrete
    spec, graph = _candidate()
    model = BadSecond()
    result = _repair(_request(spec, graph), model)
    assert result["status"] == "error" and result["target"] is None
    assert result["error_type"] == "NativeTransitionContractError"
    assert result["failure_stage"] == "native_continuation"
    assert "sampled_E_diagonal_NONE_one_hot" in result["detail"]
    assert result["cost"]["attempted_native_transitions"] == 2
    assert result["cost"]["native_transitions"] == 1
    assert result["cost"]["native_forward_calls"] == 2
    assert len(result["cost"]["trajectory"]) == 1
    assert result["cost"]["last_native_transition_contract"]["passed"] is False
    assert result["cost"]["last_native_transition_contract"]["grid_index"] == 10
    assert result["provenance"]["frozen_state_unchanged"]
    assert not model.model._forward_pre_hooks


def test_fixed_native_sampling_seed_replays_all_transition_draws_and_labels():
    class RandomNative(_MockNative):
        def sample_p_zs_given_zt(self, *args):
            self.output_labels = tuple(torch.randint(0, 2, (2,)).tolist())
            return super().sample_p_zs_given_zt(*args)
    spec, graph = _candidate()
    request = _request(spec, graph)
    left_model, right_model = RandomNative(), RandomNative()
    left, right = _repair(request, left_model), _repair(request, right_model)
    assert left["status"] == right["status"] == "accepted"
    assert left["target"] == right["target"]
    assert left["native_output_labels"] == right["native_output_labels"]
    assert left["cost"]["trajectory"] == right["cost"]["trajectory"]
    assert left["cost"]["native_forward_calls"] == right["cost"]["native_forward_calls"] == 2
    for left_input, right_input in zip(left_model.native_inputs, right_model.native_inputs):
        assert all(torch.equal(a, b) for a, b in zip(left_input, right_input))
