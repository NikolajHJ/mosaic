"""Synthetic checks for the experimental latent gradient comparison."""
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("ipsae_gradient_probe", Path(__file__).with_name("ipsae_gradient_probe.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_comparison_isolates_term_and_reports_effective_weight():
    z = torch.tensor([[1., 2.]], requires_grad=True)
    ipsae = 1. - 3. * z[0, 1]
    old_confidence = 2. * z[0, 0]
    values = {"ligand_ipsae": ipsae, "dbmol_confidence_loss": old_confidence - .3 * ipsae}
    result = probe.compare_gradients(values, z, .3, 2.5)
    assert result["original_confidence_gradient_norm"] == pytest.approx(2.)
    assert result["ipsae_loss_gradient_norm"] == pytest.approx(3.)
    assert result["cosine"] == pytest.approx(0.)
    assert result["effective_added_ipsae_gradient_norm"] == pytest.approx(2.25)
    assert result["effective_added_to_confidence_norm_ratio"] == pytest.approx(.45)
    assert result["original_confidence_loss"] == pytest.approx(2.)


def test_flat_gradient_is_reported_without_inventing_a_direction():
    z = torch.tensor([[1., 2.]], requires_grad=True)
    ipsae = z.sum() * 0
    result = probe.compare_gradients({"ligand_ipsae": ipsae,
        "dbmol_confidence_loss": z.sum() - .3 * ipsae}, z, .3, 2.5)
    assert result["ipsae_loss_gradient_norm"] == 0
    assert result["cosine"] is None
    assert result["zero_gradient"]
