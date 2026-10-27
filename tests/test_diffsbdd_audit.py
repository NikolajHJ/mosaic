"""Held-out ordering, geometry, common-noise and audit failure contracts.

Run on allocated HPC only. Native model behavior is not inferred from these
synthetic fixtures; the real 200-identity/two-conformer stage remains separate.
"""

from types import SimpleNamespace

import pytest
from rdkit import Chem
import torch
from torch import nn

from latentmol.chemistry import encode_smiles
from latentmol.diffsbdd import audit
from latentmol.diffsbdd.bridge import NativeAtomBoundary
from latentmol.diffsbdd.paired_data import EPOCH39_SHA256, slot_to_canonical
from latentmol.representation import GraphSpec, HardGraphBatch


NATIVE = ("C", "N", "O", "S", "B", "Br", "Cl", "P", "I", "F")


def observations(identities=("A", "B"), *, relative_error=0.02, omit=None):
    rows = []
    for identity in identities:
        for conformer in range(2):
            for level in audit.TIMES:
                if omit == (identity, conformer, level):
                    continue
                rows.append(dict(identity=identity, conformer=conformer, time=level,
                    errors={mode: {component: dict(mse=relative_error**2,
                        target_power=1.0, maximum_absolute_error=relative_error)
                        for component in audit.COMPONENTS} for mode in audit.MODES}))
    return rows


def test_audit_identity_selection_is_first_distinct_stored_order_without_replacements():
    data = dict(split=["train", "audit", "audit", "dev", "audit", "audit"],
                identity=["heldout", "C", "C", "D", "A", "B"])
    assert audit.select_audit_rows(data, 2) == [1, 4]
    assert audit.select_audit_rows(data, 4) == [1, 4, 5]
    assert audit.select_audit_rows(data, 2) == audit.select_audit_rows(data, 2)


def test_two_conformer_equal_identity_metrics_and_thresholds():
    summary = audit.summarize_errors(observations(), 2)
    assert all(summary["checks"].values())
    for mode in audit.MODES:
        for time in ("0", "0.04", "0.1", "0.2"):
            for component in ("coordinates", "chemistry"):
                metric = summary["by_mode"][mode][time][component]
                assert metric["identities"] == metric["complete_two_conformer_identities"] == 2
                assert metric["nrmse"] == pytest.approx(0.02)
                assert metric["p95_per_identity_nrmse"] == pytest.approx(0.02)
    corrupted = audit.summarize_errors(observations(relative_error=0.2), 2)
    assert corrupted["checks"]["0:coordinates:nrmse"] is False
    assert corrupted["checks"]["0:chemistry:p95"] is False


def test_a_missing_or_duplicated_conformer_cannot_pass_population_coverage():
    missing = audit.summarize_errors(observations(omit=("B", 1, 0.1)), 2)
    assert missing["checks"]["0.1:coordinates:coverage"] is False
    assert missing["checks"]["0.1:chemistry:coverage"] is False
    duplicate = observations()
    for row in duplicate:
        row["conformer"] = 0
    assert not audit.summarize_errors(duplicate, 2)["checks"]["0:chemistry:coverage"]


def test_conformer_p95_is_not_hidden_by_averaging_each_identity_first():
    rows = observations()
    for row in rows:
        if row["identity"] == "B" and row["conformer"] == 1:
            for mode in audit.MODES:
                for metric in row["errors"][mode].values():
                    metric.update(mse=0.14**2, maximum_absolute_error=0.14)
    summary = audit.summarize_errors(rows, 2)
    assert summary["checks"]["0:coordinates:p95"] is True
    assert summary["checks"]["0:coordinates:p95_conformer_example"] is False


def test_identity_pooling_does_not_treat_correlated_conformers_as_independent_molecules():
    rows = observations()
    for row in rows:
        if row["identity"] == "B":
            for mode in audit.MODES:
                for component in row["errors"][mode].values():
                    component.update(mse=0.04, target_power=4.0, maximum_absolute_error=0.2)
    metric = audit.summarize_errors(rows, 2)["by_mode"]["learned"]["0"]["coordinates"]
    assert metric["mse"] == pytest.approx((0.02**2 + 0.04) / 2)
    assert metric["target_power"] == 2.5
    assert metric["nrmse"] == pytest.approx(((0.02**2 + 0.04) / 5) ** 0.5)
    assert metric["p95_per_identity_nrmse"] == pytest.approx(0.096)


def test_near_zero_reference_power_keeps_absolute_error_and_explicit_floor():
    rows = observations(identities=("A",))
    for row in rows:
        for mode in audit.MODES:
            row["errors"][mode]["coordinates"] = dict(mse=1e-8, target_power=0.0,
                                                     maximum_absolute_error=1e-4)
    summary = audit.summarize_errors(rows, 1)
    metric = summary["by_mode"]["learned"]["0"]["coordinates"]
    assert metric["tiny_target_identities"] == 1 and metric["maximum_absolute_error"] == 1e-4
    assert metric["nrmse"] == pytest.approx(100.0)
    assert summary["checks"]["0:coordinates:nrmse"] is False


def test_etkdg_conformers_reproduce_the_saved_seed_and_do_not_change_chemistry():
    molecule = Chem.MolFromSmiles("CCOC1CCCCC1")
    before = Chem.MolToSmiles(molecule)
    first = audit.conformer_positions(molecule, 17)
    repeat = audit.conformer_positions(molecule, 17)
    second = audit.conformer_positions(molecule, 23)
    assert first.shape == second.shape == (molecule.GetNumAtoms(), 3)
    assert torch.equal(first, repeat)
    assert bool(torch.isfinite(first).all() & torch.isfinite(second).all())
    assert Chem.MolToSmiles(molecule) == before and molecule.GetNumConformers() == 0


def test_canonical_map_verifies_atoms_and_typed_bonds_including_symmetric_graph():
    spec = GraphSpec(max_nodes=10)
    encoded = encode_smiles("c1ccncc1", spec)
    hard = HardGraphBatch(*(torch.tensor(encoded[key])[None].long()
                           for key in ("atom_types", "charges", "bond_types")))
    data = dict(identity=[encoded["smiles"]], atoms=hard.atoms, charges=hard.charges,
        bonds=hard.bonds, active=hard.atoms != 0,
        slot_to_canonical=torch.tensor([slot_to_canonical(hard, spec, encoded["smiles"])]))
    molecule, slots, mapping = audit._canonical_molecule_and_map(data, 0, spec)
    assert molecule.GetNumAtoms() == len(slots) == len(mapping) == 6
    data["slot_to_canonical"][0, slots[0]] = data["slot_to_canonical"][0, slots[1]]
    with pytest.raises(ValueError, match="cover exactly"):
        audit._canonical_molecule_and_map(data, 0, spec)


def test_native_noising_uses_exact_supplied_common_noise_and_restores_method():
    class Core:
        def sample_gaussian(self, size, device):
            raise AssertionError("Global RNG must never be consumed")

        def gamma(self, time):
            return time

        def noised_representation(self, ligand, pocket, ligand_mask, pocket_mask, gamma):
            noise = self.sample_gaussian(ligand.shape, ligand.device)
            return ligand + noise, pocket.clone(), noise

    core = Core()
    ligand = dict(x=torch.ones(3, 3), one_hot=torch.zeros(3, 10), mask=torch.zeros(3).long())
    pocket = dict(x=torch.ones(4, 3), one_hot=torch.zeros(4, 10), mask=torch.zeros(4).long())
    noise = torch.randn(3, 13, generator=torch.Generator().manual_seed(17))
    first, fixed = audit._native_noise(core, ligand, pocket, torch.tensor([[0.1]]), noise)
    second, repeated = audit._native_noise(core, ligand, pocket, torch.tensor([[0.1]]), noise)
    assert torch.equal(first, second) and torch.equal(fixed, repeated)
    assert "sample_gaussian" not in core.__dict__
    with pytest.raises(RuntimeError, match="unexpected Gaussian"):
        audit._native_noise(core, ligand, pocket, torch.tensor([[0.1]]), noise[:, :3])
    assert "sample_gaussian" not in core.__dict__


def test_frozen_modes_and_gradients_are_checked_in_addition_to_weight_hashes():
    module = nn.Linear(2, 3).eval().requires_grad_(False)
    original = audit._frozen(dict(model=module))
    module.train()
    with pytest.raises(ValueError, match="frozen eval"):
        audit._frozen(dict(model=module))
    assert audit._frozen(dict(model=module), strict=False) != original


@pytest.mark.parametrize("level", (0.0, 0.1))
def test_transition_mean_calls_native_sampler_with_exact_zero_innovation(level):
    class Core:
        T = 500

        def sample_gaussian(self, size, device):
            raise AssertionError("Mean audit cannot consume the global RNG")

        def sample_p_zs_given_zt(self, previous, time, noisy, pocket, ligand_mask, pocket_mask):
            assert torch.allclose(previous, time - 1 / self.T)
            return noisy + self.sample_gaussian(noisy.shape, noisy.device), pocket

        def sample_p_xh_given_z0(self, noisy, pocket, ligand_mask, pocket_mask, batch_size):
            assert batch_size == 1
            innovation = self.sample_gaussian(noisy.shape, noisy.device)
            return noisy[:, :3] + innovation[:, :3], noisy[:, 3:], pocket[:, :3], pocket[:, 3:]

    core = Core()
    noisy, pocket = torch.ones(3, 13), torch.ones(4, 13)
    actual = audit._native_transition_mean(core, noisy, pocket, torch.tensor([[level]]),
        torch.zeros(3).long(), torch.zeros(4).long(), level)
    assert torch.equal(actual, noisy)
    assert "sample_gaussian" not in core.__dict__


def test_less_than_200_distinct_identities_fails_even_with_no_model_forward(tmp_path, monkeypatch):
    vae = nn.Linear(2, 3).eval().requires_grad_(False)
    vae.diffsbdd_checkpoint_identity = dict(epoch=39, sha256=EPOCH39_SHA256)
    core = nn.Module()
    core.dynamics = nn.Module()
    core.dynamics.atom_encoder = nn.Linear(10, 8).eval().requires_grad_(False)
    core.atom_nf = 10
    core.eval().requires_grad_(False)
    boundary = NativeAtomBoundary(core.dynamics.atom_encoder, NATIVE, 4.0, 0.0).eval()
    bridge = nn.Linear(2, 3).eval().requires_grad_(False)
    data = dict(identity=[], split=[], active=torch.empty(0, 50, dtype=torch.bool))
    monkeypatch.setattr(audit, "load_dataset", lambda *args: (data, {}, "a" * 64))
    pocket = dict(x=torch.zeros(3, 3), one_hot=torch.nn.functional.one_hot(torch.zeros(3).long(), 10).float(),
                  size=torch.tensor([3]), mask=torch.zeros(3).long())
    result = audit.run_denoiser_audit(vae, boundary, bridge, "unused", core, pocket, tmp_path / "audit")
    assert result["passed"] is False and result["status"] == "failed_not_passed"
    assert result["checks"]["distinct_identity_coverage"] is False
    assert result["attempted_conformers"] == result["successful_conformers"] == 0
    assert result["frozen_modules_unchanged"] is True
    assert (tmp_path / "audit/audit-result.json").is_file()


@pytest.mark.parametrize("change", (dict(identities=199), dict(conformer_seeds=(17, 29)), dict(max_nrmse=0.1)))
def test_audit_size_seeds_or_thresholds_cannot_be_relaxed_in_place(tmp_path, change):
    with pytest.raises(ValueError, match="new protocol"):
        audit.run_denoiser_audit(None, None, None, None, None, None, tmp_path / "audit", **change)
