"""Local-authorized streamed guidance contracts, also reusable on allocated HPC."""
import pytest
import torch

from latentmol.chemical_environment import prepare_neighbor_environment, neighbor_environment_loss
from latentmol.neighbor_training import NeighborEnvironmentGuidance
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch


POLICY = dict(version="common-organic-neighbor-v1", weight=.3)


def _outputs():
    spec = GraphSpec(max_nodes=4)
    atoms = torch.tensor([[spec.atomic_numbers.index(z) for z in row]
                          for row in ([6, 0, 0, 0], [16, 8, 6, 7])])
    charges = torch.full_like(atoms, spec.formal_charges.index(0))
    bonds = torch.zeros((2, 4, 4), dtype=torch.long)
    bonds[1, 0, 1] = bonds[1, 1, 0] = spec.bond_names.index("DOUBLE")
    hard = HardGraphBatch(atoms, charges, bonds).to_relaxed(spec)
    context = DecoderOutput(*(x.double() * 40 - 20 for x in (hard.atoms, hard.charges, hard.bonds)))
    rng = torch.Generator().manual_seed(9843)
    live = DecoderOutput(context.atoms.clone().requires_grad_(), context.charges.clone().requires_grad_(),
                         torch.randn(context.bonds.shape, generator=rng, dtype=torch.double).requires_grad_())
    return spec, context, live


def _slice(output, start, stop):
    return DecoderOutput(*(x[start:stop] for x in (output.atoms, output.charges, output.bonds)))


def test_center_weighted_streaming_matches_full_scalar_and_vjp_for_unequal_sizes():
    spec, context, live = _outputs()
    prepared = prepare_neighbor_environment(context, spec, POLICY)
    full = neighbor_environment_loss(live, prepared, spec)["weighted_loss"]
    expected = torch.autograd.grad(full, (live.atoms, live.charges, live.bonds), retain_graph=True)
    guidance = NeighborEnvironmentGuidance(spec, POLICY)
    guidance.begin_update(update=27)
    for branch in ("real", "prior"):
        for start, stop in ((0, 1), (1, 2)):
            guidance.prepare(_slice(context, start, stop), branch=branch, start=start, stop=stop)
    guidance.finalize_preparation()
    streamed = sum(guidance.loss(_slice(live, start, stop), branch=branch, start=start, stop=stop)
                   for branch in ("real", "prior") for start, stop in ((0, 1), (1, 2)))
    actual = torch.autograd.grad(streamed, (live.atoms, live.charges, live.bonds))
    torch.testing.assert_close(streamed, full, rtol=2e-12, atol=2e-12)
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference, rtol=2e-11, atol=2e-12)
    summary = guidance.summary()
    assert summary["assessed_centers"] == {"real": 5, "prior": 5}
    assert summary["weighted_loss"] == pytest.approx(float(full.detach()), abs=2e-12)
    assert summary["branch_mean_risk"]["real"] == pytest.approx(float(full.detach()) / .3)
    assert actual[2].norm() > 0 and actual[0].count_nonzero() == actual[1].count_nonzero() == 0


def test_guidance_lifecycle_rejects_missing_duplicate_and_changed_context():
    spec, context, live = _outputs()
    g = NeighborEnvironmentGuidance(spec, POLICY)
    with pytest.raises(ValueError, match="nonnegative integer"):
        g.begin_update(update=-1)
    g.begin_update(update=0)
    g.prepare(context, branch="real", start=0, stop=2)
    with pytest.raises(ValueError, match="once"):
        g.prepare(context, branch="real", start=0, stop=2)
    with pytest.raises(ValueError, match="Both real and prior"):
        g.finalize_preparation()
    g.prepare(context, branch="prior", start=0, stop=2)
    g.finalize_preparation()
    with pytest.raises(ValueError, match="Not every"):
        g.summary()
    prepared, _ = g._chunks["real", 0, 2]
    prepared.context["caps"][0][0] = 99
    with pytest.raises(ValueError, match="changed"):
        g.loss(live, branch="real", start=0, stop=2)


def test_guidance_rejects_branch_weight_changes_after_preparation():
    spec, context, _ = _outputs()
    g = NeighborEnvironmentGuidance(spec, POLICY)
    g.begin_update(update=0)
    g.prepare(context, branch="real", start=0, stop=2)
    g.branch_weights["real"] = .5
    with pytest.raises(ValueError, match="configuration changed"):
        g.prepare(context, branch="prior", start=0, stop=2)


@pytest.mark.parametrize("weights", [{"real": 1.}, {"real": True, "prior": 1.},
                                    {"real": 0., "prior": 0.}, {"real": -1., "prior": 2.}])
def test_branch_weight_policy_is_explicit(weights):
    spec, _, _ = _outputs()
    with pytest.raises(ValueError, match="weights"):
        NeighborEnvironmentGuidance(spec, POLICY, weights)


def _joint_update(weight, *, explicit_none=False, microbatch=2):
    # Reuse the existing audited tiny joint-VAE/repair fixture. This covers the
    # actual training hook, not a test implementation of the update itself.
    from test_chemistry_continuation import _tiny_models, _hard, _FixedTeacher
    from latentmol.chemistry_continuation import chemistry_update
    model, reference, config, protocol, composition = _tiny_models()
    hard = _hard(model.spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    kwargs = {}
    if explicit_none:
        kwargs["additive_guidance"] = None
    elif weight is not None:
        kwargs["additive_guidance"] = NeighborEnvironmentGuidance(model.spec, dict(POLICY, weight=weight))
    torch.manual_seed(9301)
    metric = chemistry_update(model, reference, optimizer, hard, config, protocol, composition,
                              _FixedTeacher(model.spec, accepted=False),
                              {"fresh_prior": torch.Generator().manual_seed(9302)},
                              update=7, epoch=34, phase_start_epoch=31, microbatch_size=microbatch,
                              **kwargs)
    return model, optimizer, metric


def test_absent_none_and_zero_weight_preserve_actual_joint_adamw_update():
    from latentmol.chemistry_continuation import nested_equal
    baseline, optimizer, metric = _joint_update(None)
    for weight, explicit_none in ((None, True), (0., False)):
        model, other_optimizer, result = _joint_update(weight, explicit_none=explicit_none)
        assert nested_equal(model.state_dict(), baseline.state_dict())
        assert nested_equal(other_optimizer.state_dict(), optimizer.state_dict())
        assert result["total"] == metric["total"]
        if explicit_none:
            assert "additive_guidance" not in result
        else:
            assert result["additive_guidance"]["weighted_loss"] == 0


def test_nonzero_guidance_reaches_real_encoder_decoder_parameters_and_streams():
    model, _, metric = _joint_update(.3, microbatch=2)
    full, _, full_metric = _joint_update(.3, microbatch=12)
    control, _, _ = _joint_update(None, microbatch=2)
    assert metric["additive_guidance"]["weighted_loss"] > 0
    assert metric["group_gradient_norms"]["encoder"] > 0
    assert metric["group_gradient_norms"]["decoder"] > 0
    assert all(torch.isfinite(p).all() for p in model.parameters())
    # The baseline already has encoder/decoder gradients. Check an actual
    # change in their gradients, rather than mistaking those for new guidance.
    for group in ("encoder", "decoder"):
        observed = dict(getattr(model, group).named_parameters())
        reference = dict(getattr(control, group).named_parameters())
        assert any((p.grad is None) != (reference[name].grad is None)
                   or (p.grad is not None and reference[name].grad is not None
                       and not torch.equal(p.grad, reference[name].grad))
                   for name, p in observed.items())
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, dict(full.named_parameters())[name].grad,
                                   rtol=3e-5, atol=3e-6)
    assert metric["additive_guidance"]["weighted_loss"] == pytest.approx(
        full_metric["additive_guidance"]["weighted_loss"], rel=3e-5, abs=3e-6)
