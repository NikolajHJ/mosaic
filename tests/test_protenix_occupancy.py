"""Allocated-HPC arithmetic fixtures for the Protenix occupancy boundary.

These tests use tiny explicit tensors, not native Protenix, checkpoint weights,
prepared molecules or GPU inference. Passing them establishes neither native
parity nor useful variable-occupancy gradients. Do not run numerically locally.
"""

import math
import inspect
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from latentmol.multimodel.external import tensor_tree_sha256
from latentmol.protenix_occupancy import native as native_loader
from latentmol.protenix_occupancy.study import (
    compare_native_repeat,
    compare_outputs,
    compare_parity_outputs,
    derivative_agreement,
    floating_element_features,
    inject_occupancy,
    reduce_distogram,
)


class EvalRecycleFixture(torch.nn.Module):
    """Source-visible tiny stand-in for the audited native recycle guard."""

    def __init__(self):
        super().__init__()
        self.train_confidence_only = False
        self.events = []
        self.eval()

    def get_pairformer_output(
        self, input_feature_dict: dict[str, torch.Tensor], N_cycle: int,
        inplace_safe: bool = False, chunk_size: int | None = None,
        mc_dropout: bool = False, mc_dropout_rate: float = 0.4,
    ) -> tuple[torch.Tensor, ...]:
        self.events.append(("input", self.training, torch.is_grad_enabled()))
        s_inputs = input_feature_dict["input"] * 2
        s = torch.zeros_like(s_inputs)
        z = torch.zeros_like(s_inputs)
        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(
                self.training
                and (not self.train_confidence_only)
                and cycle_no == (N_cycle - 1)
            ):
                self.events.append(("recycle", cycle_no, self.training, torch.is_grad_enabled()))
                z = z + s_inputs
                s = s + 3 * z
                self.events.append(("output", cycle_no))
        return s_inputs, s, z


class MissingRecycleGuardFixture:
    def get_pairformer_output(self, input_feature_dict, N_cycle):
        return input_feature_dict, N_cycle


class WrongRecycleGuardFixture:
    def get_pairformer_output(self, input_feature_dict, N_cycle):
        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(self.training):
                input_feature_dict = input_feature_dict + 1
        return input_feature_dict


class MultipleRecycleGuardFixture:
    def get_pairformer_output(self, input_feature_dict, N_cycle):
        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(
                self.training and (not self.train_confidence_only)
                and cycle_no == (N_cycle - 1)
            ):
                input_feature_dict = input_feature_dict + 1
            with torch.set_grad_enabled(
                self.training and (not self.train_confidence_only)
                and cycle_no == (N_cycle - 1)
            ):
                input_feature_dict = input_feature_dict + 2
        return input_feature_dict


class UnrelatedContextFixture:
    def get_pairformer_output(self, input_feature_dict, N_cycle):
        with torch.no_grad():
            for cycle_no in range(N_cycle):
                with torch.set_grad_enabled(
                    self.training and (not self.train_confidence_only)
                    and cycle_no == (N_cycle - 1)
                ):
                    input_feature_dict = input_feature_dict + 1
        return input_feature_dict


class DecoratedRecycleFixture(EvalRecycleFixture):
    @torch.no_grad()
    def get_pairformer_output(self, input_feature_dict, N_cycle):
        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(
                self.training and (not self.train_confidence_only)
                and cycle_no == (N_cycle - 1)
            ):
                input_feature_dict = input_feature_dict + 1
        return input_feature_dict


def test_attached_native_ast_preserves_signature_tuple_order_and_eval_mode():
    model = EvalRecycleFixture()
    original_method = model.get_pairformer_output.__func__
    attached = native_loader._make_attached_pairformer(model.get_pairformer_output)
    assert inspect.signature(attached) == inspect.signature(model.get_pairformer_output)
    value = torch.tensor([2.], requires_grad=True)
    s_inputs, s, z = attached({"input": value}, N_cycle=1)
    assert model.get_pairformer_output.__func__ is original_method
    assert not model.training
    assert model.events == [("input", False, True), ("recycle", 0, False, True), ("output", 0)]
    torch.testing.assert_close(s_inputs, torch.tensor([4.]), rtol=0, atol=0)
    torch.testing.assert_close(s, torch.tensor([12.]), rtol=0, atol=0)
    torch.testing.assert_close(z, torch.tensor([4.]), rtol=0, atol=0)
    gradient, = torch.autograd.grad(s_inputs.sum() + s.sum() + z.sum(), value)
    torch.testing.assert_close(gradient, torch.tensor([10.]), rtol=0, atol=0)
    metadata = attached._protenix_ast_metadata
    assert metadata["version"] == "native-eval-grad-context-v1"
    assert metadata["removed_grad_contexts"] == 1


def test_attached_native_ast_obeys_callers_no_grad_and_restores_grad_mode():
    model = EvalRecycleFixture()
    attached = native_loader._make_attached_pairformer(model.get_pairformer_output)
    value = torch.tensor([2.], requires_grad=True)
    with torch.no_grad():
        result = attached({"input": value}, N_cycle=1)
        assert not torch.is_grad_enabled()
        assert all(not output.requires_grad for output in result)
    assert torch.is_grad_enabled()
    assert model.events == [("input", False, False), ("recycle", 0, False, False), ("output", 0)]


def test_untouched_native_fixture_still_disables_recycle_gradients_in_eval():
    model = EvalRecycleFixture()
    value = torch.tensor([2.], requires_grad=True)
    s_inputs, s, z = model.get_pairformer_output({"input": value}, N_cycle=1)
    assert s_inputs.requires_grad
    assert not s.requires_grad and not z.requires_grad
    assert model.events == [("input", False, True), ("recycle", 0, False, False), ("output", 0)]
    assert torch.is_grad_enabled()


@pytest.mark.parametrize("fixture_type", [MissingRecycleGuardFixture,
    WrongRecycleGuardFixture, MultipleRecycleGuardFixture, UnrelatedContextFixture,
    DecoratedRecycleFixture])
def test_attached_native_ast_rejects_unreviewed_context_structure(fixture_type):
    with pytest.raises(ValueError):
        native_loader._make_attached_pairformer(fixture_type().get_pairformer_output)


LEGACY_ESM_KEY = "input_embedder.linear_esm.weight"


class DormantCheckpointModel(torch.nn.Module):
    """Small active model with the pinned Tiny input dimensions, no ESM path."""

    def __init__(self):
        super().__init__()
        self.configs = SimpleNamespace(
            esm=SimpleNamespace(enable=False, embedding_dim=2560), c_s_inputs=449)
        self.input_embedder = torch.nn.Module()
        self.input_embedder.c_token = 384
        self.input_embedder.esm_configs = {"enable": False, "embedding_dim": 2560}
        self.input_embedder.active = torch.nn.Parameter(torch.tensor([2.]))

    def forward(self, value):
        if self.input_embedder.esm_configs["enable"]:
            raise AssertionError("The Tiny-default ESM path must remain disabled")
        return value * self.input_embedder.active


@pytest.fixture
def dormant_checkpoint(monkeypatch):
    """Replace only the official layer constructor; no native model is imported."""
    original_import = native_loader.importlib.import_module
    factory_calls = []

    def linear_factory(in_features, out_features):
        factory_calls.append((in_features, out_features))
        return torch.nn.Linear(in_features, out_features, bias=False)

    def import_module(name, *args, **kwargs):
        if name == "protenix.model.modules.primitives":
            return SimpleNamespace(LinearNoBias=linear_factory)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(native_loader.importlib, "import_module", import_module)
    model = DormantCheckpointModel()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    state[LEGACY_ESM_KEY] = torch.zeros((449, 2560), dtype=torch.float32)
    return model, state, factory_calls


def test_dormant_esm_compatibility_preserves_state_and_active_forward(dormant_checkpoint):
    model, state, factory_calls = dormant_checkpoint
    state_digest = tensor_tree_sha256(state)
    original_names = tuple(state)
    original_tensors = {name: id(value) for name, value in state.items()}
    baseline = model(torch.tensor([3.])).detach().clone()

    metadata = native_loader._register_dormant_checkpoint_esm(model, state)
    # The complete, unmodified checkpoint still has to satisfy strict loading.
    incompatible = model.load_state_dict(state, strict=True)

    assert incompatible.missing_keys == incompatible.unexpected_keys == []
    assert factory_calls == [(2560, 449)]
    assert model.configs.esm.enable is False
    assert model.input_embedder.esm_configs["enable"] is False
    assert tuple(state) == original_names
    assert {name: id(value) for name, value in state.items()} == original_tensors
    assert tensor_tree_sha256(state) == state_digest
    assert LEGACY_ESM_KEY in model.state_dict()
    torch.testing.assert_close(model(torch.tensor([3.])), baseline, rtol=0, atol=0)
    assert metadata["version"] == "tiny-default-dormant-esm-v1"
    assert metadata["parameter_name"] == LEGACY_ESM_KEY
    assert tuple(metadata["shape"]) == (449, 2560)
    assert metadata["all_zero"] is True
    assert metadata["esm_enabled"] is False
    assert metadata["strict_load"] is True
    assert len(metadata["tensor_sha256"]) == 64


@pytest.mark.parametrize("mutation", [
    "enabled_config", "enabled_embedder", "existing_module", "wrong_token_dim",
    "wrong_embedding_dim", "missing_legacy", "unknown_extra", "missing_active",
    "wrong_shape", "wrong_dtype", "nonzero", "nan", "infinity", "sparse",
])
def test_dormant_esm_compatibility_fails_closed(dormant_checkpoint, mutation):
    model, state, factory_calls = dormant_checkpoint
    if mutation == "enabled_config":
        model.configs.esm.enable = True
    elif mutation == "enabled_embedder":
        model.input_embedder.esm_configs["enable"] = True
    elif mutation == "existing_module":
        model.input_embedder.linear_esm = torch.nn.Linear(2560, 449, bias=False)
    elif mutation == "wrong_token_dim":
        model.input_embedder.c_token = 383
    elif mutation == "wrong_embedding_dim":
        model.input_embedder.esm_configs["embedding_dim"] = 2559
        model.configs.esm.embedding_dim = 2559
    elif mutation == "missing_legacy":
        del state[LEGACY_ESM_KEY]
    elif mutation == "unknown_extra":
        state["unrecognized.weight"] = torch.zeros(1)
    elif mutation == "missing_active":
        del state["input_embedder.active"]
    elif mutation == "wrong_shape":
        state[LEGACY_ESM_KEY] = torch.zeros((448, 2560))
    elif mutation == "wrong_dtype":
        state[LEGACY_ESM_KEY] = state[LEGACY_ESM_KEY].double()
    elif mutation == "nonzero":
        state[LEGACY_ESM_KEY][0, 0] = 1e-30
    elif mutation == "nan":
        state[LEGACY_ESM_KEY][0, 0] = float("nan")
    elif mutation == "infinity":
        state[LEGACY_ESM_KEY][0, 0] = float("inf")
    elif mutation == "sparse":
        state[LEGACY_ESM_KEY] = state[LEGACY_ESM_KEY].to_sparse()

    with pytest.raises((ValueError, TypeError)):
        native_loader._register_dormant_checkpoint_esm(model, state)
    assert factory_calls == []


def test_dormant_module_does_not_relax_later_active_strict_loading(dormant_checkpoint):
    model, state, _ = dormant_checkpoint
    native_loader._register_dormant_checkpoint_esm(model, state)
    del state["input_embedder.active"]
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(state, strict=True)


def native_features(dtype=torch.int64, *, noncontiguous=False):
    """Official integer one-hot rows, optionally varied for boundary checks."""
    element = F.one_hot(torch.tensor([5, 5, 7]), 128).to(dtype=dtype)
    if noncontiguous:
        element = element.transpose(0, 1).contiguous().transpose(0, 1)
    return {
        "ref_element": element,
        "ref_charge": torch.tensor([0., 0., 0.]),
        "ref_mask": torch.tensor([1., 1., 1.]),
        "ref_pos": torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]),
        "token_bonds": torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.]]),
        "atom_to_token_idx": torch.arange(3),
        "nested_cache": {"indices": torch.tensor([1, 2])},
    }


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32, torch.float64])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_floating_element_boundary_preserves_values_and_builds_valid_probe(
        dtype, noncontiguous, requires_grad):
    features = native_features(dtype, noncontiguous=noncontiguous)
    before = tensor_tree_sha256(features)
    prepared = floating_element_features(features, requires_grad=requires_grad)
    element = prepared["ref_element"]

    assert prepared is not features
    assert prepared.keys() == features.keys()
    assert element is not features["ref_element"]
    assert element.data_ptr() != features["ref_element"].data_ptr()
    assert element.dtype == torch.float32
    assert element.device == features["ref_element"].device
    assert element.is_contiguous() and element.is_leaf
    assert element.requires_grad is requires_grad
    torch.testing.assert_close(element, features["ref_element"].float(), rtol=0, atol=0)
    for field in features.keys() - {"ref_element"}:
        assert prepared[field] is features[field]
    assert tensor_tree_sha256(features) == before
    if requires_grad:
        gradient, = torch.autograd.grad(element.square().sum(), element)
        torch.testing.assert_close(gradient, 2 * features["ref_element"].float(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32, torch.float64])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_occupied_control_and_zero_element_arm_share_fp32_values_and_layout(
        dtype, noncontiguous):
    features = native_features(dtype, noncontiguous=noncontiguous)
    before = tensor_tree_sha256(features)
    control = inject_occupancy(features, 2, torch.tensor(1.), "C")
    zero_element = inject_occupancy(features, 2, torch.tensor(1.), "Z")
    c_element, z_element = control["ref_element"], zero_element["ref_element"]

    assert c_element.dtype == z_element.dtype == torch.float32
    assert c_element.stride() == z_element.stride() == (128, 1)
    assert c_element.data_ptr() != z_element.data_ptr()
    assert c_element.is_contiguous() and z_element.is_contiguous()
    torch.testing.assert_close(c_element, z_element, rtol=0, atol=0)
    torch.testing.assert_close(c_element, features["ref_element"].float(), rtol=0, atol=0)
    for field in features.keys() - {"ref_element"}:
        assert control[field] is zero_element[field] is features[field]
    assert tensor_tree_sha256(features) == before


@pytest.mark.parametrize("q_value", [0., .2, .5, 1.])
def test_zero_element_condition_retains_unnormalized_occupancy_mass(q_value):
    features = native_features()
    before = tensor_tree_sha256(features)
    adapted = inject_occupancy(features, 2, torch.tensor(q_value), "Z")

    assert adapted is not features
    assert adapted["ref_element"] is not features["ref_element"]
    assert adapted["ref_element"][2].sum().item() == pytest.approx(q_value)
    assert adapted["ref_element"][2, 7].item() == pytest.approx(q_value)
    assert torch.count_nonzero(adapted["ref_element"][2]).item() == int(q_value > 0)
    torch.testing.assert_close(adapted["ref_element"][:2], features["ref_element"][:2].float())
    # This arm changes element mass only. ref_mask remains native reference
    # validity; bonds and all other feature routes are deliberately clamped.
    for field in features.keys() - {"ref_element"}:
        assert adapted[field] is features[field]
    assert tensor_tree_sha256(features) == before


def test_no_atom_largest_still_has_nonzero_joint_softmax_gradient():
    features = native_features()
    # Real oxygen occupies 20% of the slot even though NO_ATOM wins argmax.
    logits = torch.tensor([math.log(4.), 0.], dtype=torch.float64, requires_grad=True)
    probabilities = logits.softmax(-1)
    q = probabilities[1]
    adapted = inject_occupancy(features, 2, q, "Z")
    loss = 3. * adapted["ref_element"][2, 7]
    gradient, = torch.autograd.grad(loss, logits)

    assert probabilities.argmax().item() == 0
    assert adapted["ref_element"][2, 7].item() == pytest.approx(.2)
    torch.testing.assert_close(gradient, torch.tensor([-.48, .48], dtype=torch.float64))
    assert torch.isfinite(gradient).all()


def test_explicit_absence_logit_has_opposite_sign_to_real_element_logit():
    features = native_features()
    absence_logit = torch.tensor(math.log(4.), dtype=torch.float64, requires_grad=True)
    q = (-absence_logit).sigmoid()
    loss = inject_occupancy(features, 2, q, "Z")["ref_element"][2, 7]
    gradient, = torch.autograd.grad(loss, absence_logit)

    assert q.item() == pytest.approx(.2)
    assert gradient.item() == pytest.approx(-.16)
    assert derivative_agreement(gradient.item(), -.16)["passed"]


@pytest.mark.parametrize("q_value", [0., .2, .5, 1.])
def test_clamped_control_discards_q_including_zero_mass_endpoint(q_value):
    features = native_features()
    before = tensor_tree_sha256(features)
    q = torch.tensor(q_value, requires_grad=True)
    adapted = inject_occupancy(features, 2, q, "C")

    assert adapted is not features
    assert adapted.keys() == features.keys()
    assert adapted["ref_element"] is not features["ref_element"]
    assert adapted["ref_element"].dtype == torch.float32
    assert adapted["ref_element"].is_contiguous()
    torch.testing.assert_close(adapted["ref_element"], features["ref_element"].float(), rtol=0, atol=0)
    assert all(adapted[field] is features[field] for field in features.keys() - {"ref_element"})
    assert adapted["ref_element"][2, 7].item() == 1.
    # The caller may add this analytical zero to obtain an autograd zero
    # instead of treating an intentionally unused q as a missing gradient.
    scalar = adapted["ref_element"].sum() + 0. * q
    gradient, = torch.autograd.grad(scalar, q)
    assert gradient.item() == 0.
    assert tensor_tree_sha256(features) == before


@pytest.mark.parametrize("q_value", [-.001, 1.001, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("condition", ["C", "Z"])
def test_occupancy_conditions_reject_invalid_q_even_for_clamped_control(q_value, condition):
    with pytest.raises(ValueError, match="finite|Occupancy"):
        inject_occupancy(native_features(), 2, torch.tensor(q_value), condition)


@pytest.mark.parametrize("q", [0.5, torch.tensor(1), torch.tensor(True), torch.tensor([.1, .2])])
def test_occupancy_requires_one_floating_tensor(q):
    with pytest.raises(ValueError, match="floating scalar"):
        inject_occupancy(native_features(), 2, q, "Z")


@pytest.mark.parametrize("optional_atom", [-1, 3, True, 1.5])
def test_optional_atom_must_be_an_actual_integer_native_row(optional_atom):
    with pytest.raises(ValueError, match="Optional atom"):
        inject_occupancy(native_features(), optional_atom, torch.tensor(.5), "Z")


def test_feature_shape_and_native_normalization_are_checked():
    features = native_features()
    features["ref_element"] = features["ref_element"][None]
    with pytest.raises(ValueError, match="ref_element"):
        inject_occupancy(features, 2, torch.tensor(.5), "Z")
    features = native_features()
    features["ref_element"] = features["ref_element"].float()
    features["ref_element"][2] *= .5
    with pytest.raises(ValueError, match="normalized hard reference"):
        inject_occupancy(features, 2, torch.tensor(.5), "Z")


def test_full_propagation_cannot_be_faked_by_element_scaling():
    with pytest.raises(NotImplementedError, match="audited occupancy propagation"):
        inject_occupancy(native_features(), 2, torch.tensor(.5), "G")


def test_uniform_distogram_is_one_for_fixed_pair_sets():
    logits = torch.zeros(5, 5, 4)
    for pairs in (torch.tensor([[0, 2]]), torch.tensor([[0, 2], [1, 2]])):
        target_bins = torch.zeros(pairs.shape[0], dtype=torch.long)
        assert reduce_distogram(logits, pairs, target_bins).item() == pytest.approx(1.)


def test_readout_has_fixed_pair_denominator_and_ignores_optional_atom_logits():
    # Native target probabilities .5 and .25 give normalized CE .75 exactly.
    logits = torch.zeros(4, 4, 4)
    logits[0, 2] = torch.tensor([.5, .25, .125, .125]).log()
    logits[1, 2] = torch.tensor([.5, .25, .125, .125]).log()
    pairs = torch.tensor([[0, 2], [1, 2]])
    targets = torch.tensor([0, 1])
    original_pairs, original_targets = pairs.clone(), targets.clone()
    baseline = reduce_distogram(logits, pairs, targets)
    assert baseline.item() == pytest.approx(.75)

    changed = logits.detach().clone()
    # Atom/token 3 is optional and excluded from this common-core readout.
    changed[3] = torch.tensor([100., -100., 50., -50.])
    changed[:, 3] = torch.tensor([-100., 100., -50., 50.])
    changed.requires_grad_(True)
    loss = reduce_distogram(changed, pairs, targets)
    torch.testing.assert_close(loss, baseline)
    gradient, = torch.autograd.grad(loss, changed)
    assert not gradient[3].any()
    assert not gradient[:, 3].any()
    assert gradient[0, 2].abs().sum() > 0
    assert gradient[1, 2].abs().sum() > 0
    assert torch.equal(pairs, original_pairs)
    assert torch.equal(targets, original_targets)


@pytest.mark.parametrize("pairs", [
    torch.tensor([[0, 2], [0, 2]]),
    torch.tensor([[0, 2], [2, 0]]),
    torch.tensor([[1, 1]]),
])
def test_readout_rejects_duplicate_unordered_and_self_pairs(pairs):
    with pytest.raises(ValueError):
        reduce_distogram(torch.zeros(3, 3, 4), pairs, torch.zeros(pairs.shape[0], dtype=torch.long))


@pytest.mark.parametrize("shape", [(3, 3), (3, 4, 4), (3, 3, 1)])
def test_readout_rejects_invalid_native_logit_shapes(shape):
    with pytest.raises(ValueError, match="distogram"):
        reduce_distogram(torch.zeros(shape), torch.tensor([[0, 2]]), torch.tensor([0]))


@pytest.mark.parametrize("pairs, targets", [
    (torch.empty(0, 2, dtype=torch.long), torch.empty(0, dtype=torch.long)),
    (torch.tensor([0, 2]), torch.tensor([0])),
    (torch.tensor([[0., 2.]]), torch.tensor([0])),
    (torch.tensor([[-1, 2]]), torch.tensor([0])),
    (torch.tensor([[0, 3]]), torch.tensor([0])),
    (torch.tensor([[0, 2]]), torch.tensor([0.])),
    (torch.tensor([[0, 2]]), torch.tensor([[0]])),
    (torch.tensor([[0, 2]]), torch.tensor([-1])),
    (torch.tensor([[0, 2]]), torch.tensor([4])),
])
def test_readout_rejects_malformed_pair_or_target_contract(pairs, targets):
    with pytest.raises(ValueError):
        reduce_distogram(torch.zeros(3, 3, 4), pairs, targets)


def output_fixture():
    return {"logits": torch.zeros(2, 4), "probabilities": torch.full((2, 4), .25),
            "loss": torch.tensor(1.)}


@pytest.mark.parametrize("comparison, version", [
    (compare_native_repeat, "native-repeat-loss-tv-v1"),
    (compare_parity_outputs, "native-output-loss-tv-v1"),
])
def test_distribution_parity_raw_logit_tolerance_is_diagnostic_only(comparison, version):
    reference = output_fixture()
    actual = output_fixture()
    # A common logit offset leaves the distribution and fixed CE unchanged.
    # Use an exact binary fraction, above the original allclose threshold.
    actual["logits"] += 1 / 1024
    strict = compare_outputs(actual, reference)
    repeat = comparison(actual, reference)

    assert strict["logits"]["finite"] and not strict["logits"]["passed"]
    assert not strict["passed"]
    assert repeat["version"] == version
    assert repeat["logits_used_as_gate"] is False
    assert repeat["strict_parity_passed"] is False
    assert repeat["logits"] == strict["logits"]
    assert repeat["loss"]["passed"]
    assert repeat["maximum_pair_total_variation"] == 0.
    assert repeat["passed"]
    # The historical strict comparator remains available for diagnostics.
    assert not compare_outputs(actual, reference)["passed"]


@pytest.mark.parametrize("comparison", [compare_native_repeat, compare_parity_outputs])
@pytest.mark.parametrize("field", ["logits", "probabilities", "loss"])
def test_distribution_parity_still_rejects_nonfinite_outputs(comparison, field):
    actual = output_fixture()
    actual[field] = torch.full_like(actual[field], float("nan"))
    assert not comparison(actual, output_fixture())["passed"]


@pytest.mark.parametrize("comparison", [compare_native_repeat, compare_parity_outputs])
def test_distribution_parity_keeps_fixed_loss_tolerance_as_hard_gate(comparison):
    actual = output_fixture()
    actual["loss"] += 1 / 1024
    repeat = comparison(actual, output_fixture())
    assert repeat["logits"]["finite"] and repeat["logits"]["passed"]
    assert not repeat["loss"]["passed"]
    assert repeat["maximum_pair_total_variation"] == 0.
    assert not repeat["passed"]


@pytest.mark.parametrize("comparison", [compare_native_repeat, compare_parity_outputs])
def test_distribution_parity_keeps_probability_total_variation_as_hard_gate(comparison):
    actual = output_fixture()
    actual["probabilities"][0, 0] += 1 / 1024
    actual["probabilities"][0, 1] -= 1 / 1024
    repeat = comparison(actual, output_fixture())
    assert repeat["logits"]["finite"] and repeat["loss"]["passed"]
    assert repeat["maximum_pair_total_variation"] == pytest.approx(1 / 1024)
    assert not repeat["passed"]


@pytest.mark.parametrize("comparison", [compare_native_repeat, compare_parity_outputs])
@pytest.mark.parametrize("field, shape", [
    ("logits", (1, 4)), ("probabilities", (1, 4)), ("loss", (1,)),
    ("logits", (0, 4)), ("probabilities", (0, 4)),
])
def test_distribution_parity_rejects_broadcastable_or_empty_shapes(comparison, field, shape):
    actual = output_fixture()
    actual[field] = torch.zeros(shape)
    with pytest.raises(ValueError):
        comparison(actual, output_fixture())


def test_output_comparison_requires_logits_and_probabilities_not_only_equal_loss():
    reference = output_fixture()
    assert compare_outputs(output_fixture(), reference)["passed"]

    actual = output_fixture()
    actual["logits"][0, 0] = .01
    assert not compare_outputs(actual, reference)["passed"]
    actual = output_fixture()
    # Exact binary fraction avoids making this fixture a decimal-rounding test.
    actual["probabilities"][0, 0] += 1 / 1024
    actual["probabilities"][0, 1] -= 1 / 1024
    comparison = compare_outputs(actual, reference)
    assert comparison["logits"]["passed"]
    assert comparison["loss"]["passed"]
    assert comparison["maximum_pair_total_variation"] == pytest.approx(1 / 1024)
    assert not comparison["passed"]


@pytest.mark.parametrize("field", ["logits", "probabilities", "loss"])
def test_output_comparison_rejects_nonfinite_observations(field):
    actual = output_fixture()
    actual[field] = torch.full_like(actual[field], float("nan"))
    assert not compare_outputs(actual, output_fixture())["passed"]


@pytest.mark.parametrize("field, shape", [
    ("logits", (1, 4)), ("probabilities", (1, 4)), ("loss", (1,)),
    ("logits", (0, 4)), ("probabilities", (0, 4)),
])
def test_output_comparison_rejects_broadcastable_or_empty_shapes(field, shape):
    actual = output_fixture()
    actual[field] = torch.zeros(shape)
    with pytest.raises(ValueError):
        compare_outputs(actual, output_fixture())


@pytest.mark.parametrize("autograd, fd, passed", [
    (1., 1.04, True), (1., 1.2, False), (-1., 1., False),
    (0., 0., True), (0., .5e-6, True), (0., 2e-6, False),
    (float("nan"), 1., False), (1., float("inf"), False),
])
def test_derivative_comparison_checks_signed_scaled_finite_difference(autograd, fd, passed):
    comparison = derivative_agreement(autograd, fd)
    assert comparison["passed"] is passed
    if not math.isfinite(autograd) or not math.isfinite(fd):
        assert comparison["absolute_error"] is None

