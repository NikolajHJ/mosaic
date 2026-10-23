"""HPC-only pipeline tests; fixture outcomes do not establish validity gains."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from latentmol.chemistry import encode_smiles
from latentmol.decoder_finetune import FineTuneConfig, fit_case, make_panel
from latentmol.experiments import file_sha256
from latentmol.losses import LossConfig
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.representation import GraphSpec
from latentmol.representation import DecoderOutput, HardGraphBatch
from latentmol.quality import graph_identity
from latentmol.search_repair_guidance import FrozenSearchRepairTarget, witness_nll
from latentmol.search_decoder_study import SearchPanelGuidance, probe_fixed_teacher_gradient, read_protocol


class TinyDataset:
    def __init__(self, spec):
        self.spec = spec
        encoded = [encode_smiles(value, spec) for value in ("C", "CC", "CO", "CN")]
        self.smiles = [row["smiles"] for row in encoded]
        self.manifest = {"dataset_sha256": "a" * 64, "split_sha256": "b" * 64}
        self.rows = [{name: torch.tensor(row[source], dtype=torch.long) for name, source in
                     (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))} for row in encoded]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_matched_search_hook_real_optimizer_and_budget_receipts(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Allocated CUDA device required")
    spec = GraphSpec(max_nodes=3)
    model_config = ModelConfig(hidden=8, latent=4, layers=1)
    model = GraphAutoencoder(spec, model_config)
    checkpoint = tmp_path / "initial.pt"
    torch.save({"format_version": 1, "graph_spec": spec.to_dict(), "model_config": asdict(model_config),
        "model_state": model.state_dict(), "loss_config": asdict(LossConfig()), "dataset_sha256": "a" * 64,
        "train_config": {"mode": "vae", "beta": .001}, "epoch": 89}, checkpoint)
    data = TinyDataset(spec)
    config = FineTuneConfig(epochs=2, batch_size=2, chemistry_reconstruction_batch=2, prior_batch_size=2)
    panel = make_panel(4, 4, 71000)
    repair = ProbabilityRepairConfig(max_edits=1, checks_per_depth=4, beam_width=1)
    results = []
    for name, weight in (("control", 0.), ("search", 1.)):
        hook = SearchPanelGuidance(panel, spec, repair, name=name, weight=weight, batch_size=2)
        result = fit_case(checkpoint, data, data, tmp_path / "runs", config, seed=17, coefficient=0.,
            train_limit=4, val_limit=4, validation_panel=make_panel(2, 4, 81000), device=device, source_root=tmp_path,
            budget_seconds=120, extra_guidance=hook)
        receipt = result["extra_guidance"]
        assert receipt["updates"] == 4 and receipt["scheduled_points"] == 8
        assert len(receipt["refreshes"]) == 2
        for refresh in receipt["refreshes"]:
            assert refresh["attempted_candidates"] == 4
            assert "attempts" not in refresh
            assert file_sha256(Path(refresh["receipt_path"])) == refresh["receipt_file_sha256"]
        assert receipt["weighted_eligible_guidance_updates"] <= 4
        if weight == 0:
            assert receipt["weighted_eligible_guidance_updates"] == 0
        results.append(result)
    assert results[0]["terminal_rng"] == results[1]["terminal_rng"]
    assert results[0]["encoder_sha256"] == results[1]["encoder_sha256"]


def test_no_witness_probe_is_uninformative_not_verified(tmp_path):
    class EmptyTeacher:
        def objective(self, decoded, **kwargs):
            return {"available_mask": torch.zeros(decoded.atoms.shape[0], dtype=torch.bool)}
    model = GraphAutoencoder(GraphSpec(max_nodes=3), ModelConfig(hidden=8, latent=4, layers=1))
    result = probe_fixed_teacher_gradient(model, EmptyTeacher(), make_panel(2, 4, 71000), tmp_path / "probe")
    assert result["status"] == "uninformative_no_native_witness"
    assert result["gradient_verified"] is False


def test_search_protocol_authentication_and_prespecified_arms(tmp_path):
    source = Path(__file__).resolve().parents[1] / "configs/search_decoder_pilot.json"
    protocol = read_protocol(source, file_sha256(source))
    assert protocol["training_prior_count"] == 256 and protocol["train_limit"] == 16384
    with pytest.raises(ValueError):
        read_protocol(source, "0" * 64)
    protocol["arms"][2]["repair_weight"] = .2
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        read_protocol(changed, file_sha256(changed))


class DroppedComponent(torch.autograd.Function):
    """Deliberately wrong backward: first coordinate correct, second erased."""
    @staticmethod
    def forward(ctx, value):
        return value.sum(-1)

    @staticmethod
    def backward(ctx, derivative):
        return torch.stack((derivative, derivative * 0), -1)


class ProbeModel(torch.nn.Module):
    def __init__(self, spec, *, wrong_backward=False, zero=False):
        super().__init__()
        self.spec, self.wrong_backward, self.zero = spec, wrong_backward, zero
        self.dummy = torch.nn.Parameter(torch.tensor([0.]))

    def decode(self, z):
        value = DroppedComponent.apply(z) if self.wrong_backward else z.sum(-1)
        if self.zero:
            value = value * 0
        base = z.new_zeros(len(z), self.spec.max_nodes, len(self.spec.atomic_numbers))
        mask = torch.zeros_like(base)
        mask[:, 0, self.spec.atomic_numbers.index(6)] = 1
        atoms = base + value[:, None, None] * mask
        return DecoderOutput(atoms, z.new_zeros(len(z), self.spec.max_nodes, len(self.spec.formal_charges)),
            z.new_zeros(len(z), self.spec.max_nodes, self.spec.max_nodes, len(self.spec.bond_names)))


def _fixed_probability_teacher(spec):
    """Verified synthetic CO target; this is not a native-search success claim."""
    encoded = encode_smiles("CO", spec)
    hard = HardGraphBatch(*(torch.tensor(encoded[key], dtype=torch.long)[None]
                           for key in ("atom_types", "charges", "bond_types")))
    digest = graph_identity(hard.atoms[0], hard.charges[0], hard.bonds[0], spec)
    target = FrozenSearchRepairTarget(0, 0, hard, digest, digest, "a" * 64, 1., 1, 6, spec.fingerprint())
    class FixedTeacher:
        def objective(self, decoded, **kwargs):
            return {"total": witness_nll(decoded, target, spec),
                    "available_mask": torch.ones(len(decoded.atoms), dtype=torch.bool)}
    return FixedTeacher()


def test_two_direction_probe_verifies_live_probability_gradient(tmp_path):
    spec = GraphSpec(max_nodes=3)
    result = probe_fixed_teacher_gradient(ProbeModel(spec), _fixed_probability_teacher(spec),
                                         torch.tensor([[.1, .2]]), tmp_path / "correct")
    assert result["gradient_verified"] is True
    assert all(value == "verified_local_plateau" for value in result["direction_status"].values())


def test_independent_probe_detects_wrong_backward_that_passes_its_own_direction(tmp_path):
    spec = GraphSpec(max_nodes=3)
    result = probe_fixed_teacher_gradient(ProbeModel(spec, wrong_backward=True), _fixed_probability_teacher(spec),
                                         torch.tensor([[.1, .2]]), tmp_path / "wrong")
    assert result["direction_status"]["gradient_unit"] == "verified_local_plateau"
    assert result["direction_status"]["seeded_random_unit"] != "verified_local_plateau"
    assert result["gradient_verified"] is False


def test_accepted_target_zero_gradient_remains_uninformative(tmp_path):
    spec = GraphSpec(max_nodes=3)
    result = probe_fixed_teacher_gradient(ProbeModel(spec, zero=True), _fixed_probability_teacher(spec),
                                         torch.tensor([[.1, .2]]), tmp_path / "zero")
    assert result["status"] == "uninformative_zero_native_gradient"
    assert result["gradient_verified"] is False
