import pytest
import torch

from latentmol.guidance.synthesis import SynthesisGuidance


class Quadratic(torch.nn.Module):
    def forward(self, graph, active):
        score = graph.square().sum(-1)
        return {"difficulty": score, "raw_score": score}


def test_guidance_calibration_is_per_child_and_fixed_after_first_point():
    model = Quadratic().requires_grad_(False)
    first = SynthesisGuidance({"model": model}, {"model": .5})
    point = torch.tensor([[1., 2.]], requires_grad=True)
    reference = torch.tensor([[3., 4.]])
    gradient, info = first.evaluate(point, None, point, reference, 0)
    assert gradient.norm().item() == pytest.approx(2.5)
    coefficient = info["models"]["model"]["coefficient"]
    later = torch.tensor([[2., 4.]], requires_grad=True)
    gradient, info = first.evaluate(later, None, later, reference * 10, 1)
    assert info["models"]["model"]["coefficient"] == coefficient
    assert gradient.norm().item() == pytest.approx(5.)
    assert info["models"]["model"]["calibration_step"] == 0
    second = SynthesisGuidance({"model": model}, {"model": .5})
    _, info = second.evaluate(later, None, later, reference, 0)
    assert info["models"]["model"]["coefficient"] != coefficient


def test_guidance_delay_and_flat_derivative_are_recorded():
    session = SynthesisGuidance({"model": Quadratic()}, {"model": .1}, delay_updates=2)
    point = torch.zeros((1, 2), requires_grad=True)
    gradient, info = session.evaluate(point, None, point, torch.ones_like(point), 1)
    assert not info["active"] and gradient.count_nonzero() == 0
    _, info = session.evaluate(point, None, point, torch.ones_like(point), 2)
    assert info["models"]["model"]["warning"] == "calibration_pending_flat_gradient"
    point = torch.ones((1, 2), requires_grad=True)
    gradient, info = session.evaluate(point, None, point, torch.ones_like(point), 3)
    assert gradient.norm() > 0
    assert info["models"]["model"]["calibration_step"] == 3


def test_detached_required_guidance_fails_clearly():
    class Detached(Quadratic):
        def forward(self, graph, active):
            result = super().forward(graph, active)
            return {key: value.detach() for key, value in result.items()}
    point = torch.ones((1, 2), requires_grad=True)
    with pytest.raises(ValueError, match="Detached"):
        SynthesisGuidance({"model": Detached()}, {"model": .1}).evaluate(
            point, None, point, torch.ones_like(point), 0)


def test_auxiliary_guidance_changes_update_but_preserves_contact_readout():
    from types import SimpleNamespace
    from latentmol.design.latent_protenix import LatentDesignConfig, PreparedPoint, finish_point

    config = LatentDesignConfig(updates=1, schedule_horizon=1, chemistry_hold=0,
        chemistry_ramp_end=1, noise_initial=0., noise_zero_at=1)

    def evaluate(guidance):
        point = torch.tensor([[1., 2.]], requires_grad=True)
        decoded = SimpleNamespace(atoms=point, charges=point, bonds=point)
        prepared = PreparedPoint(point, None, config, 0, {"chemistry_weight": 1.},
                                 decoded, point)
        return finish_point(SimpleNamespace(spec=None), prepared, (point.square().sum(), {}),
                            target_bundle=(None, {}), guidance=guidance)

    base = evaluate(None)
    guided = evaluate(SynthesisGuidance({"model": Quadratic()}, {"model": .5}))
    assert guided["binding"].item() == base["binding"].item()
    assert torch.equal(guided["binding_gradient"], base["binding_gradient"])
    assert torch.allclose(guided["gradient"], base["gradient"] * 1.5)
    assert "synthesis" in guided and "synthesis" not in base


def test_large_finite_gradients_do_not_overflow_calibration_norm():
    class Large(torch.nn.Module):
        def forward(self, graph, active):
            result = graph.sum(-1) * 1e25
            return {"difficulty": result, "raw_score": result}
    point = torch.ones((1, 2), requires_grad=True)
    reference = torch.ones_like(point)
    gradient, info = SynthesisGuidance({"model": Large()}, {"model": .5}).evaluate(
        point, None, point, reference, 0)
    assert torch.isfinite(gradient).all()
    assert gradient.norm().item() == pytest.approx(.5 * 2**.5)
    assert info["models"]["model"]["coefficient"] > 0


def test_total_online_cap_controls_growth_and_uses_contact_ema():
    guide = SynthesisGuidance({"a": Quadratic(), "b": Quadratic()}, {"a": .5, "b": .5},
                             controller={"max_ratio": .1, "ema_decay": .5})
    first = torch.tensor([[1., 0.]], requires_grad=True)
    gradient, info = guide.evaluate(first, None, first, torch.tensor([[2., 0.]]), 0)
    assert gradient.double().norm() == pytest.approx(.2)
    later = torch.tensor([[100., 0.]], requires_grad=True)
    gradient, info = guide.evaluate(later, None, later, torch.tensor([[4., 0.]]), 1)
    assert info["pre_control_gradient_norm"] == pytest.approx(200.)
    assert info["controller_reference_norm"] == pytest.approx(3.)
    assert gradient.double().norm() == pytest.approx(.3)
    assert info["controller_cap_scale"] < .01


@pytest.mark.parametrize("scale_cap, expected", [(True, .04), (False, .4)])
def test_phase_cap_scales_down_even_when_the_auxiliary_gradient_grows(scale_cap, expected):
    guide = SynthesisGuidance({"model": Quadratic()}, {"model": .2}, controller={
        "max_ratio": .2, "ema_decay": 0., "scale_cap_with_schedule": scale_cap},
        schedule={"kind": "phases", "phases": [{"start": 0, "end": 1, "scale": 1.},
                                                 {"start": 1, "end": 3, "scale": .1}]})
    first = torch.tensor([[1., 0.]], requires_grad=True)
    reference = torch.tensor([[2., 0.]])
    full, _ = guide.evaluate(first, None, first, reference, 0)
    later = torch.tensor([[100., 0.]], requires_grad=True)
    settled, info = guide.evaluate(later, None, later, reference, 1)
    assert full.norm() == pytest.approx(.4)
    assert info["pre_control_gradient_norm"] == pytest.approx(4.)
    assert settled.norm() == pytest.approx(expected)


def test_projection_removes_only_binding_conflict():
    class Linear(torch.nn.Module):
        def forward(self, graph, active):
            value = -graph[..., 0] + graph[..., 1]
            return {"difficulty": value, "raw_score": value}

    point = torch.ones((1, 2), requires_grad=True)
    binding = torch.tensor([[1., 0.]])
    guide = SynthesisGuidance({"model": Linear()}, {"model": 1.}, project_binding=True)
    gradient, info = guide.evaluate(point, None, point, binding, 0)
    assert info["binding_projection_applied"]
    assert info["binding_cosine"] == pytest.approx(-2**-.5)
    assert gradient[0, 0] == pytest.approx(0., abs=1e-7)
    assert gradient[0, 1] == pytest.approx(2**-.5)


def test_phases_postpone_calibration_and_resume_existing_weight():
    guide = SynthesisGuidance({"model": Quadratic()}, {"model": .5}, schedule={
        "kind": "phases", "phases": [{"start": 2, "end": 4, "scale": 1.},
                                      {"start": 5, "end": 7, "scale": .25}]})
    point = torch.ones((1, 2), requires_grad=True)
    reference = torch.ones_like(point)
    zero, meta = guide.evaluate(point, None, point, reference, 0)
    assert not meta["active"] and not guide.coefficients and not zero.any()
    first, meta = guide.evaluate(point, None, point, reference, 2)
    coefficient = meta["models"]["model"]["coefficient"]
    assert meta["models"]["model"]["calibration_step"] == 2
    zero, meta = guide.evaluate(point, None, point, reference, 4)
    assert not meta["active"] and not zero.any()
    resumed, meta = guide.evaluate(point, None, point, reference * 20, 5)
    assert torch.allclose(resumed, first * .25)
    assert guide.coefficients["model"] == coefficient


def test_cosine_schedule_has_bounded_off_peak_and_repeat_points():
    guide = SynthesisGuidance({"model": Quadratic()}, {"model": .5}, schedule={
        "kind": "cosine_wave", "period": 20, "low": 0., "high": 1.})
    point = torch.ones((1, 2), requires_grad=True)
    reference = torch.ones_like(point)
    off, _ = guide.evaluate(point, None, point, reference, 0)
    peak, _ = guide.evaluate(point, None, point, reference, 10)
    repeated_off, _ = guide.evaluate(point, None, point, reference, 20)
    assert not off.any() and not repeated_off.any() and peak.norm() > 0


def test_validity_controller_allows_brief_excursion_and_recovers():
    guide = SynthesisGuidance({"model": Quadratic()}, {"model": .5}, controller={
        "max_ratio": 1., "ema_decay": 0., "invalid_grace": 1})
    assert guide.requires_raw_valid
    point = torch.ones((1, 2), requires_grad=True)
    reference = torch.ones_like(point)
    with pytest.raises(ValueError, match="raw_valid"):
        guide.evaluate(point, None, point, reference, 0)
    first, _ = guide.evaluate(point, None, point, reference, 0, raw_valid=False)
    suppressed, info = guide.evaluate(point, None, point, reference, 1, raw_valid=False)
    recovered, _ = guide.evaluate(point, None, point, reference, 2, raw_valid=True)
    assert first.norm() > 0 and not suppressed.any()
    assert info["invalid_streak"] == 2 and info["validity_scale"] == 0
    assert torch.equal(first, recovered)


def test_invalid_scale_reduces_the_cap_even_for_a_dominating_guide():
    guide = SynthesisGuidance({"model": Quadratic()}, {"model": 1.}, controller={
        "max_ratio": .1, "ema_decay": 0., "invalid_grace": 0, "invalid_scale": .2})
    first = torch.tensor([[1., 0.]], requires_grad=True)
    reference = torch.tensor([[2., 0.]])
    valid, _ = guide.evaluate(first, None, first, reference, 0, raw_valid=True)
    later = torch.tensor([[100., 0.]], requires_grad=True)
    invalid, info = guide.evaluate(later, None, later, reference, 1, raw_valid=False)
    assert info["pre_control_gradient_norm"] == pytest.approx(200.)
    assert valid.norm() == pytest.approx(.2)
    assert invalid.norm() == pytest.approx(.04)
    assert invalid.norm() == pytest.approx(valid.norm().item() * .2)


def test_sharpness_views_are_connected_and_preserve_fixed_support():
    from latentmol.representation import RelaxedGraph

    class Inspect(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.views = []

        def forward(self, graph, active):
            self.views.append(graph)
            value = graph.atoms[..., 1].sum(-1)
            return {"difficulty": value, "raw_score": 1 - value}

    point = torch.tensor([[1., -1.]], requires_grad=True)
    probabilities = point.softmax(-1)
    atoms = torch.cat((torch.zeros((1, 1)), probabilities), -1)[:, None]
    graph = RelaxedGraph(atoms, torch.ones((1, 1, 1)), torch.ones((1, 1, 1, 1)))
    model = Inspect()
    guide = SynthesisGuidance({"model": model}, {"model": .5},
                             sharpness={"temperatures": [1., .5], "reduction": "mean"})
    gradient, info = guide.evaluate(graph, torch.ones(1, dtype=torch.bool), point,
                                    torch.ones_like(point), 0)
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    assert model.views[1].atoms[0, 0, 1] > model.views[0].atoms[0, 0, 1]
    assert model.views[1].atoms[0, 0, 0] == 0
    assert graph.atoms[0, 0, 1] == probabilities[0, 0]
    assert len(info["models"]["model"]["views"]) == 2
    with pytest.raises(ValueError, match="fixed-occupancy"):
        guide.evaluate(graph, None, point, torch.ones_like(point), 1)


def test_latent_anchor_can_run_without_graph_and_has_own_calibration():
    class Anchor(torch.nn.Module):
        def forward_latent(self, point):
            value = (point - point.new_tensor([[0., 1.]])).square().sum(-1)
            return {"difficulty": value, "raw_score": value}

    point = torch.tensor([[1., 1.]], requires_grad=True)
    guide = SynthesisGuidance({}, {}, anchor={"model": Anchor(), "ratio": .1})
    gradient, info = guide.evaluate(None, None, point, torch.tensor([[0., 2.]]), 0)
    assert torch.allclose(gradient, torch.tensor([[.2, 0.]]))
    assert info["models"]["anchor"]["calibration_step"] == 0


@pytest.mark.parametrize("kwargs", [
    {"controller": {"max_ratio": -.1}},
    {"controller": {"max_ratio": .1, "ema_decay": 1.}},
    {"schedule": {"kind": "phases", "phases": [{"start": 2, "end": 1, "scale": 1}]}},
    {"schedule": {"kind": "cosine_wave", "period": 0}},
    {"sharpness": {"temperatures": [0.]}},
])
def test_invalid_exploratory_controls_fail_before_evaluation(kwargs):
    with pytest.raises(ValueError):
        SynthesisGuidance({"model": Quadratic()}, {"model": .1}, **kwargs)
