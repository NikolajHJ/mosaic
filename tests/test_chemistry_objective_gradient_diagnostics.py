"""Allocated CPU/CUDA observations against a full-batch weighted-gradient oracle."""
from contextlib import nullcontext
from copy import deepcopy
import math
import os

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from latentmol.chemistry_objective_gradient_diagnostics import ObjectiveGradientDiagnostics
from latentmol.chemistry_optimizer_step import apply_optimizer_step


CONTROL = dict(version="adamw-displacement-scale-v1", factor=.25)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        if os.environ.get("LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES") == "1":
            pytest.fail("Mandatory full-logical-gradient CUDA fixture has no CUDA device")
        pytest.skip("CUDA observation runs in the allocated native gate")
    return request.param


@pytest.fixture(autouse=True)
def restore_fixture_state():
    rng = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.set_rng_state(rng)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        torch.use_deterministic_algorithms(deterministic)


class Model(nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.decoder = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3))
        self.to(device=device, dtype=dtype)
        self.encoder.eval()


def _scalars(model, branch, index, multiplier=1.):
    p = next(model.parameters())
    hard = torch.arange(8, device=p.device, dtype=p.dtype).reshape(2, 4) / 10 + .03 * index
    z = model.encoder(hard) if branch == "real" else hard + .4
    decoded = model.decoder(z)
    # Exercise the caller's RNG stream; the observer must introduce no draws.
    noise = torch.randn((), device=p.device, dtype=p.dtype) * .001
    zero = decoded.sum() * 0
    fraction = .5 if branch == "real" else .25
    terms = dict(reconstruction=((decoded - .3 + noise).square().mean() * fraction
                                 if branch == "real" else zero),
        kl=(z.square().mean() * .001 * fraction if branch == "real" else zero),
        repair=(decoded[:, 1] + .4).square().mean() * (.08 if branch == "real" else .25) * fraction,
        composition=decoded[:, 0].mean() * .002 * fraction,
        occupancy=((decoded[:, 0].sigmoid() - .4).square().mean() * .03 * fraction
                   if branch == "prior" else zero),
        valid_self=(F.cross_entropy(decoded, torch.zeros(2, dtype=torch.long, device=p.device)) * .1 * fraction
                    if branch == "prior" else zero))
    return {key: value * multiplier for key, value in terms.items()}


def _rng():
    return (torch.get_rng_state().clone(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _restore_rng(value):
    torch.set_rng_state(value[0])
    if value[1]:
        torch.cuda.set_rng_state_all(value[1])


def _same(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _same(a, b)
    else:
        assert left == right


def _run(model, optimizer, *, observe, control=None, multiplier=1.):
    observer = ObjectiveGradientDiagnostics(model, real_points=4, prior_points=8, microbatch_size=2)
    with observer if observe else nullcontext():
        optimizer.zero_grad(set_to_none=True)
        for index, branch in enumerate(["real"] * 2 + ["prior"] * 4):
            terms = _scalars(model, branch, index, multiplier)
            if observe:
                observer(branch, terms)
            sum(terms.values()).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        apply_optimizer_step(optimizer, model.parameters(), control)
    return observer.receipt() if observe else None


def _full_batch_oracle(model):
    by_term = {}
    for index, branch in enumerate(["real"] * 2 + ["prior"] * 4):
        for term, scalar in _scalars(model, branch, index).items():
            key = branch + "/" + term
            by_term[key] = scalar if key not in by_term else by_term[key] + scalar
    parameters = tuple(model.decoder.parameters())
    result = {}
    # Differentiate each complete logical scalar once, rather than summing the
    # observer's per-microbatch VJPs. Include an exact full-objective oracle.
    by_term.update({term: sum(value for key, value in by_term.items() if key.endswith("/" + term))
                    for term in ("reconstruction", "kl", "repair", "composition", "occupancy", "valid_self")})
    by_term["full_objective"] = sum(value for key, value in by_term.items() if "/" in key)
    for term, scalar in by_term.items():
        values = torch.autograd.grad(scalar, parameters, retain_graph=True, allow_unused=True)
        result[term] = [torch.zeros_like(p).double() if value is None else value.detach().double()
                        for p, value in zip(parameters, values)]
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("control", [None, CONTROL])
def test_complete_logical_terms_match_global_oracle_and_observation_changes_no_state(device, dtype, control):
    model = Model(device, dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    sum(p.square().mean() for p in model.parameters()).backward()
    optimizer.step()  # Preserve nonzero moments in the compared update.
    start_model, start_optimizer, start_rng = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()), _rng()
    oracle = _full_batch_oracle(model)
    _restore_rng(start_rng)
    _run(model, optimizer, observe=False, control=control)
    expected = (deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()),
                [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()],
                _rng(), [module.training for module in model.modules()])
    model.load_state_dict(start_model)
    optimizer.load_state_dict(start_optimizer)
    _restore_rng(start_rng)
    report = _run(model, optimizer, observe=True, control=control)
    actual = (model.state_dict(), optimizer.state_dict(), [p.grad for p in model.parameters()],
              _rng(), [module.training for module in model.modules()])
    _same(actual, expected)
    assert report["observed_microbatches"] == {"real": 2, "prior": 4}
    assert report["component_sum_matches_native_backward"]
    assert report["observer_model_forwards"] == report["observer_optimizer_steps"] == report["observer_random_samples"] == 0
    deltas = [p.detach().double() - start_model["decoder." + name].double()
              for name, p in model.decoder.named_parameters()]
    tolerance = 2e-5 if dtype == torch.float32 else 1e-9
    for key, values in oracle.items():
        norm = math.sqrt(sum(float(value.square().sum()) for value in values))
        dot = sum(float((value * delta).sum()) for value, delta in zip(values, deltas))
        assert report["metrics"][key]["gradient_norm"] == pytest.approx(norm, rel=tolerance, abs=2e-9)
        assert report["metrics"][key]["gradient_dot_displacement"] == pytest.approx(dot, rel=tolerance, abs=2e-10)
    for a, b in [("reconstruction", "repair"), ("reconstruction", "valid_self"),
                 ("repair", "valid_self"), ("reconstruction", "prior/repair")]:
        dot = sum(float((left * right).sum()) for left, right in zip(oracle[a], oracle[b]))
        denominator = report["metrics"][a]["gradient_norm"] * report["metrics"][b]["gradient_norm"]
        expected_cosine = dot / denominator if denominator else None
        assert report["term_gradient_cosines"][a + "__" + b] == pytest.approx(expected_cosine, rel=tolerance, abs=2e-7)


def test_gradient_reconstruction_observes_native_accumulation_before_clipping(device):
    model = Model(device, torch.float64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    report = _run(model, optimizer, observe=True, multiplier=10000.)
    clipped_decoder_norm = math.sqrt(sum(float(p.grad.double().square().sum()) for p in model.decoder.parameters()))
    assert report["metrics"]["full_objective"]["gradient_norm"] > clipped_decoder_norm
    assert report["component_sum_matches_native_backward"]


@pytest.mark.parametrize("defect", ["missing", "order", "changed_terms", "extra"])
def test_incomplete_or_changed_schedule_cannot_claim_full_logical_gradients(device, defect):
    model = Model(device, torch.float64)
    observer = ObjectiveGradientDiagnostics(model, real_points=4, prior_points=8, microbatch_size=2)
    with pytest.raises(ValueError):
        with observer:
            model.zero_grad(set_to_none=True)
            schedule = ["real"] * 2 + ["prior"] * 4
            if defect == "missing": schedule.pop()
            elif defect == "order": schedule[0] = "prior"
            elif defect == "extra": schedule.append("prior")
            for index, branch in enumerate(schedule):
                terms = _scalars(model, branch, index)
                if defect == "changed_terms" and index == 2:
                    terms.pop("valid_self")
                observer(branch, terms)
                sum(terms.values()).backward()
    with pytest.raises(ValueError):
        observer.receipt()
    assert not observer._handles and not observer._before and not observer._values


def test_callback_preserves_existing_gradients_and_exception_removes_hooks(device):
    model = Model(device, torch.float64)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    references = [p.grad for p in model.parameters()]
    values = [p.grad.clone() for p in model.parameters()]
    observer = ObjectiveGradientDiagnostics(model, real_points=4, prior_points=8, microbatch_size=2)
    with pytest.raises(RuntimeError, match="Injected"):
        with observer:
            observer("real", _scalars(model, "real", 0))
            for p, reference, value in zip(model.parameters(), references, values):
                assert p.grad is reference and torch.equal(p.grad, value)
            raise RuntimeError("Injected caller failure")
    model.decoder(torch.ones(2, 4, device=device, dtype=torch.float64)).sum().backward()
    assert not observer._native and not observer._handles
    with pytest.raises(ValueError):
        observer.receipt()


def test_actual_self_guided_trainer_observation_keeps_native_update_and_all_state_exact(device):
    """Exercise the relaxed real trainer hook, not just the scalar toy caller.

    The ordinary warm update supplies inherited nonzero AdamW moments. Both
    compared updates start from that same full native/fresh RNG snapshot and
    apply the declared quarter displacement with alpha .1 and genuine strict
    valid teacher targets. The unequal final real chunk tests logical weighting.
    """
    from latentmol import chemistry_continuation as continuation
    from latentmol.chemistry_execution_validation import EXCLUDED_TIMINGS, _restore, _snapshot
    from latentmol.chemistry_valid_self_guidance import VERSION as SELF_VERSION
    from test_chemistry_continuation import _hard
    from test_chemistry_valid_self_training import _models, _teacher

    model, reference, config, protocol, composition = _models(device)
    protocol["valid_prior_self_guidance"] = dict(version=SELF_VERSION, weight=.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generators = {"fresh_prior": torch.Generator().manual_seed(901)}
    hard = _hard(model.spec)
    assert len(hard.atoms) == 3 and protocol["prior_ratio"] == 2
    # Populate moments with an actual unobserved native chemistry update.
    with _teacher(model) as teacher:
        continuation.chemistry_update(model, reference, optimizer, hard, config, protocol,
            composition, teacher, generators, 6, 37, 35, microbatch_size=2)
    assert optimizer.state and any(bool(state["exp_avg"].count_nonzero())
                                   for state in optimizer.state.values())
    initial = _snapshot(model, reference, optimizer, generators)
    results = []
    receipt = None
    for observed in (False, True):
        _restore(initial, model, reference, optimizer, generators)
        assert continuation.nested_equal(initial, _snapshot(model, reference, optimizer, generators))
        observer = ObjectiveGradientDiagnostics(model, real_points=3, prior_points=6, microbatch_size=2)
        with _teacher(model) as teacher, observer if observed else nullcontext():
            metrics = continuation.chemistry_update(model, reference, optimizer, hard, config, protocol,
                composition, teacher, generators, 7, 37, 35, microbatch_size=2, step_control=CONTROL,
                **({"component_observer": observer} if observed else {}))
        if observed:
            receipt = observer.receipt()
        results.append(({key: value for key, value in metrics.items() if key not in EXCLUDED_TIMINGS},
                        _snapshot(model, reference, optimizer, generators)))
    assert continuation.nested_equal(results[0], results[1])
    assert results[1][0]["real_points"] == 3 and results[1][0]["fresh_prior_points"] == 6
    assert results[1][0]["valid_prior_self_guidance"]["eligible"] == 6
    assert results[1][0]["valid_prior_self_weighted"] > 0
    assert results[1][0]["optimizer_step_control"]["factor"] == .25
    assert receipt["observed_microbatches"] == {"real": 2, "prior": 3}
    assert receipt["component_sum_matches_native_backward"] is True
    assert receipt["complete_logical_update"] is True
    assert receipt["metrics"]["valid_self"]["gradient_norm"] > 0
    assert receipt["observer_model_forwards"] == receipt["observer_optimizer_steps"] == receipt["observer_random_samples"] == 0
