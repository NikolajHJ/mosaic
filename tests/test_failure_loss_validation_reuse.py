"""Allocated-HPC old/new failure-loss parity, including mandatory CUDA cases.

The byte-pinned source124A2 module preserves the old enclosing forward. Its
baseline NLL is rebound to the literal historical oracle in the companion
fixture, so this compares both pre-extraction paths, not two new wrappers.
Only synchronous validation work may differ; losses and VJPs must be exact.
"""
from dataclasses import replace
import os

import pytest
import torch

from latentmol import atom_preserving_chemistry as chemistry
from latentmol import repair_failure_guidance as guidance
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.repair_execution_reference import load_preparation_reference
from latentmol.repair_progress import ProgressSearchConfig
from latentmol.representation import DecoderOutput, GraphSpec
from test_repair_failure_guidance import output_for
from test_repair_loss_validation_reuse import _legacy_repair_loss


COMPONENTS = ("total", "valid_full_nll", "extra_valid_nll", "partial_edited_nll",
              "local_bond_risk", "component_cut_risk", "nonempty_risk")


@pytest.fixture(autouse=True)
def strict_determinism():
    """Exercise the same strict CUDA execution contract as native training."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.fixture(scope="module")
def historical():
    path = os.environ.get("LATENTMOL_FAILURE_REFERENCE_MODULE")
    if not path:
        pytest.fail("Loss parity requires the byte-pinned source124A2 reference")
    module = load_preparation_reference(path,
        "34c1a67cb339233312a1317db39a926b55f378c0eb8ed42c38bde86307fdd184")
    previous = module.repair_loss
    module.repair_loss = _legacy_repair_loss
    try:
        yield module
    finally:
        module.repair_loss = previous


def _device(name):
    if name == "cuda" and not torch.cuda.is_available():
        if os.environ.get("LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES") == "1":
            pytest.fail("Allocated native validation requires CUDA loss-parity fixtures")
        pytest.skip("CUDA cases run in the mandatory allocated native stage")
    return name


def _fixture(depth):
    spec = GraphSpec(max_nodes=8)
    rows = [output_for(atoms, edges, spec=spec)[1] for atoms, edges in (
        ([9, 9], [(0, 1, "TRIPLE")]),
        ([6, 6], [(0, 1, "SINGLE")]),
        ([9] * 4, [(0, 1, "TRIPLE"), (2, 3, "TRIPLE")]),
        ([], []),
        # Both fluorines need separate edits: outside the old one-edit budget,
        # but a connected F-C-F witness exists for the deeper search branch.
        ([6, 9, 9], [(0, 1, "TRIPLE"), (0, 2, "TRIPLE")]))]
    output = DecoderOutput(*(torch.cat([getattr(row, field) for row in rows])
                              for field in ("atoms", "charges", "bonds")))
    with chemistry.AtomPreservingRepairTeacher(spec,
            ProbabilityRepairConfig(max_edits=1, checks_per_depth=32)) as teacher:
        base = teacher.build_targets(output, candidate_ids=[71, 72, 73, 74, 75], update=37)
    prepared = guidance.prepare_failure_guidance(output, base, spec,
        progress_search=None if depth is None else ProgressSearchConfig(max_edits=depth))
    assert base.targets[0] is not None and any(row is not None for row in prepared.rows)
    return spec, output, prepared


def _old_prepared(module, prepared):
    rows = tuple(None if row is None else module.FrozenFailureRow(**{
        key: getattr(row, key) for key in guidance.FrozenFailureRow.__dataclass_fields__})
        for row in prepared.rows)
    return module.PreparedFailureGuidance(prepared.base, rows, prepared.schema_sha256,
                                         prepared.identity, prepared.receipt)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("depth", [None, 6])
def test_historical_and_new_forward_have_identical_components_and_all_head_vjps(historical, device, dtype, depth):
    device = _device(device)
    spec, output, prepared = _fixture(depth)
    observed = []
    for module, reference in ((historical, _old_prepared(historical, prepared)), (guidance, prepared)):
        live = DecoderOutput(*(value.detach().to(device=device, dtype=dtype).clone().requires_grad_()
                               for value in (output.atoms, output.charges, output.bonds)))
        result = module.failure_aware_repair_loss(live, reference, spec)
        assert result["valid_full_nll"] > 0 and result["nonempty_risk"] > 0
        if depth == 6:
            assert result["extra_valid_nll"] > 0
        gradients = torch.autograd.grad(result["total"], (live.atoms, live.charges, live.bonds))
        observed.append((result, gradients))
    for component in COMPONENTS:
        assert torch.equal(observed[0][0][component], observed[1][0][component]), component
    for old, new in zip(observed[0][1], observed[1][1]):
        assert torch.equal(old, new)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_historical_and_new_loss_preserve_shared_parameter_vjp(historical, device, dtype):
    device = _device(device)
    spec, output, prepared = _fixture(6)
    observed = []
    for module, reference in ((historical, _old_prepared(historical, prepared)), (guidance, prepared)):
        parameters = torch.linspace(-.1, .1, 9, dtype=dtype, device=device, requires_grad=True)
        heads = []
        for index, value in enumerate((output.atoms, output.charges, output.bonds)):
            value = value.detach().to(device=device, dtype=dtype)
            weights = (torch.arange(value.numel(), device=device, dtype=dtype).reshape(value.shape).remainder(7) + 1) / 10
            heads.append(value + weights * parameters[index]
                + weights.square() * parameters[index + 3].sin() + .1 * weights * parameters[index + 6].square())
        result = module.failure_aware_repair_loss(DecoderOutput(*heads), reference, spec)
        observed.append((result["total"].detach(), torch.autograd.grad(result["total"], parameters)[0]))
    assert torch.equal(observed[0][0], observed[1][0])
    assert torch.equal(observed[0][1], observed[1][1]) and observed[0][1].norm() > 0


@pytest.mark.parametrize("field", ["atoms", "charges", "bonds"])
@pytest.mark.parametrize("storage", ["_graph", "_raw"])
def test_failure_forward_revalidates_current_labels_after_prior_success(field, storage):
    spec, output, prepared = _fixture(None)
    guidance.failure_aware_repair_loss(output, prepared, spec)
    getattr(getattr(prepared.base.targets[0], storage), field).reshape(-1)[0] += 1
    with pytest.raises(ValueError, match="identity changed"):
        guidance.failure_aware_repair_loss(output, prepared, spec)


@pytest.mark.parametrize("defect", ["duplicate_ids", "negative_id", "alignment", "extra_target"])
def test_failure_forward_keeps_strict_prepared_batch_identity_checks(defect):
    spec, output, prepared = _fixture(None)
    changes = {"duplicate_ids": dict(candidate_ids=(71, 71, 73, 74, 75)),
               "negative_id": dict(candidate_ids=(-1, 72, 73, 74, 75)),
               "alignment": dict(candidate_ids=(99, 72, 73, 74, 75)),
               "extra_target": dict(targets=prepared.base.targets + (None,))}[defect]
    prepared = replace(prepared, base=replace(prepared.base, **changes))
    with pytest.raises(ValueError, match="identities"):
        guidance.failure_aware_repair_loss(output, prepared, spec)
