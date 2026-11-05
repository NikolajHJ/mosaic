"""Allocated controls for detached single-atom targets in live preparation."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from mosaic.small_molecule.chemistry import (
    GraphLogits, StructuralRepairConfig, failure_aware_repair_loss, mosaic_spec,
    prepare_failure_guidance,
)
from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig


EXCHANGE = {"atom_edit_penalty": 1.0, "max_substitutions": 4}


def hard_logits(numbers, single_edges):
    spec = mosaic_spec(len(numbers))
    atom_labels = [spec.atomic_numbers.index(z) - 1 for z in numbers]
    atoms = np.full((len(numbers), len(spec.atomic_numbers) - 1), -12, dtype=np.float32)
    atoms[np.arange(len(numbers)), atom_labels] = 0
    pairs = list(zip(*np.triu_indices(len(numbers), 1)))
    bond_labels = [spec.bond_names.index("SINGLE") if ij in single_edges else 0 for ij in pairs]
    bonds = np.full((len(pairs), len(spec.bond_names)), -8, dtype=np.float32)
    bonds[np.arange(len(pairs)), bond_labels] = 0
    return spec, GraphLogits(jnp.asarray(atoms), jnp.asarray(bonds))


def test_route_one_exchange_has_a_live_atom_gradient_and_fixed_target(monkeypatch):
    spec, source = hard_logits((6, 9, 6), {(0, 1), (1, 2)})
    fluorine = spec.atomic_numbers.index(9) - 1
    oxygen = spec.atomic_numbers.index(8) - 1
    logits = GraphLogits(source.atoms.at[1, oxygen].set(-0.2), source.bonds)
    settings = {"structural_repair": StructuralRepairConfig(), "common_organic_scope": True}
    control = prepare_failure_guidance(logits.decoded(), spec, **settings)
    prepared = prepare_failure_guidance(logits.decoded(), spec, atom_exchange=EXCHANGE, **settings)
    assert int(control.arrays["target_atoms"][0, 1]) == fluorine + 1
    assert int(prepared.arrays["route"][0]) == 1
    assert int(prepared.arrays["target_atoms"][0, 1]) == oxygen + 1
    assert not bool(prepared.arrays["changed"].any())
    assert prepared.receipt["rows"][0]["atom_exchange"]["selected"]["slot"] == 1
    identity = prepared.identity

    def reject_search(*args, **kwargs):
        raise AssertionError("The live loss must reuse the frozen repair target")

    monkeypatch.setattr("mosaic.small_molecule.chemistry.atom_exchange.search_atom_exchange", reject_search)

    def live_value(point):
        return failure_aware_repair_loss(point.decoded(), prepared, spec)[0]

    value, gradient = eqx.filter_value_and_grad(live_value)(logits)
    assert np.isfinite(float(value))
    assert np.isfinite(np.asarray(gradient.atoms)).all()
    assert np.isfinite(np.asarray(gradient.bonds)).all()
    assert float(gradient.atoms[1, oxygen]) < 0 < float(gradient.atoms[1, fluorine])
    trial = GraphLogits(logits.atoms.at[1, oxygen].add(1.0), logits.bonds)
    assert float(live_value(trial)) < float(value)
    prepared.verify(spec)
    assert prepared.identity == identity


def test_out_of_scope_native_validity_cannot_bypass_scoped_bond_only_fallback():
    spec, logits = hard_logits((16, *(6,) * 6), {(0, i) for i in range(1, 7)})
    settings = {"structural_repair": StructuralRepairConfig(),
                "search": ProbabilityRepairConfig(max_edits=1, checks_per_depth=4, beam_width=1)}
    legacy = prepare_failure_guidance(logits.decoded(), spec, **settings)
    scoped = prepare_failure_guidance(logits.decoded(), spec, common_organic_scope=True, **settings)
    assert int(legacy.arrays["route"][0]) == 0
    assert int(scoped.arrays["route"][0]) == 4
    row = scoped.receipt["rows"][0]
    assert row["beam_search_performed"]
    assert row["scope_rejections"] == [{"search": "beam", "status": "already_valid"}]
    assert row["base"]["weighted_witness"] is None
    assert not row["base"]["raw_valid"]
    assert scoped.receipt["original_valid_targets"] == scoped.receipt["extra_valid_targets"] == 0
    scoped.verify(spec)
