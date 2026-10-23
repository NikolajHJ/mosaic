"""Allocated-HPC fixtures for the rounding mechanism and head isolation.

These exercise the installed native distogram class without loading checkpoint
weights. Pretrained saved-state and pilot gates are separate requirements.
"""

from dataclasses import replace

import pytest
import torch

from latentmol.boltz.contact_projection import contact_distogram_float64
from latentmol.boltz.dbmol_objective import (
    CONTACT64_ARITHMETIC, CONTACT_HEAD64_ARITHMETIC, contact_loss,
    native_bin_centers, reduce_profile,
)
from test_dbmol_precision import context, primitives
from test_dbmol_objective import profile


def native_head():
    from boltz.model.modules.trunkv2 import DistogramModule
    head = DistogramModule(1, 64).eval().requires_grad_(False)
    with torch.no_grad():
        head.distogram.weight.zero_()
        head.distogram.bias.fill_(2.)
        head.distogram.weight[0, 0] = .001
    return head


def test_widen_before_affine_recovers_signal_lost_before_contact_reduction():
    head = native_head()
    mask = (torch.tensor([True, False]), torch.tensor([False, True]))
    epsilon = 1e-4
    def score(z, widened):
        pair = z.expand(1, 2, 2, 1)
        native = head(pair)
        logits = contact_distogram_float64(head, pair) if widened else native.double()
        return contact_loss(logits, *mask, bin_centers=native_bin_centers(native).double())['dbmol_contact_loss']
    point = torch.zeros(1, requires_grad=True)
    grad, = torch.autograd.grad(score(point, True), point)
    assert grad.abs().item() > 0
    # Native logits have already rounded the +/-2e-7 perturbation at bias 2.
    # Choose an even smaller chord to isolate head rounding deterministically.
    epsilon = 1e-5
    native_fd = float(score(point.detach() + epsilon, False) - score(point.detach() - epsilon, False)) / (2 * epsilon)
    wide_fd = float(score(point.detach() + epsilon, True) - score(point.detach() - epsilon, True)) / (2 * epsilon)
    assert native_fd == 0
    assert wide_fd == pytest.approx(grad.item(), rel=1e-5, abs=1e-9)


def test_native_parameters_and_confidence_primitives_remain_unchanged(tmp_path):
    head = native_head()
    weights = {k: v.clone() for k, v in head.state_dict().items()}
    pair = torch.ones(1, 2, 2, 1, requires_grad=True)
    native = head(pair)
    wide = contact_distogram_float64(head, pair)
    affinity, structure = primitives()
    cfg = profile(tmp_path, objective_arithmetic=CONTACT_HEAD64_ARITHMETIC)
    result = reduce_profile(affinity, structure, native, context(), cfg, contact_distogram=wide)
    assert result['dbmol_contact_loss'].dtype == torch.float64
    for key, value in {**affinity, **structure}.items():
        assert result[key] is value and result[key].dtype == torch.float32
    torch.autograd.grad(result['dbmol_total_loss'], pair)
    assert all(torch.equal(weights[k], v) and v.dtype == torch.float32 for k, v in head.state_dict().items())
    assert head.training is False and all(p.grad is None for p in head.parameters())
    assert profile(tmp_path, objective_arithmetic=CONTACT_HEAD64_ARITHMETIC).record()['objective_arithmetic'] == CONTACT_HEAD64_ARITHMETIC


def test_head_mode_retains_native_bin_partition_and_rejects_missing_bridge(tmp_path):
    head = native_head()
    pair = torch.ones(1, 2, 2, 1)
    native, wide = head(pair), contact_distogram_float64(head, pair)
    affinity, structure = primitives()
    cfg = profile(tmp_path, objective_arithmetic=CONTACT_HEAD64_ARITHMETIC,
                  distance_cutoff=float(native_bin_centers(native)[19]))
    actual = reduce_profile(affinity, structure, native, context(), cfg, contact_distogram=wide)
    expected = contact_loss(wide, torch.tensor([True, False]), torch.tensor([False, True]),
                            cutoff=cfg.distance_cutoff, bin_centers=native_bin_centers(native).double())
    assert torch.equal(actual['dbmol_contact_loss'], expected['dbmol_contact_loss'])
    with pytest.raises(ValueError, match='explicit FP64'):
        reduce_profile(affinity, structure, native, context(), cfg)
    with pytest.raises(ValueError, match='declared'):
        reduce_profile(affinity, structure, native, context(), replace(cfg, objective_arithmetic=CONTACT64_ARITHMETIC), contact_distogram=wide)


@pytest.mark.parametrize('mutation', ['train', 'unfrozen', 'pair_double', 'multiple'])
def test_unsupported_head_policies_fail_closed(mutation):
    head = native_head()
    pair = torch.ones(1, 2, 2, 1)
    if mutation == 'train': head.train()
    if mutation == 'unfrozen': head.requires_grad_(True)
    if mutation == 'pair_double': pair = pair.double()
    if mutation == 'multiple': head.num_distograms = 2
    with pytest.raises(ValueError, match='frozen eval FP32'):
        contact_distogram_float64(head, pair)
