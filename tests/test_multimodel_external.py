"""Tiny tensor-only external-adapter checks, not native model validation.

No checkpoint, external package, GPU, training data or upstream forward is used.
These test doubles establish mapping/autograd/frozen-state engineering only.
Real Nesso/Protenix parity and usefulness remain pending HPC experiments.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from latentmol.multimodel.external import NativeBundle, native_source_sha256
from latentmol.multimodel.nesso import NessoBranch
from latentmol.multimodel.protenix import ProtenixBranch
from latentmol.representation import GraphSpec, HardGraphBatch, RelaxedGraph


def make_bundle(model="nesso1"):
    spec = GraphSpec(max_nodes=3)
    seed = HardGraphBatch(torch.tensor([[1, 2, 0]]), torch.full((1, 3), 2),
                          torch.tensor([[[0, 1, 0], [1, 0, 0], [0, 0, 0]]]))
    offset = 0 if model == "nesso1" else 1
    elements = F.one_hot(torch.tensor([6, 6, 6, 7]) - offset, 128).float()
    bonds = torch.zeros(4, 4)
    bonds[2, 3] = bonds[3, 2] = 1
    batched = model == "nesso1"
    features = {"ref_element": elements[None] if batched else elements,
                "ref_charge": torch.zeros(1, 4) if batched else torch.zeros(4),
                "token_bonds": bonds[None, ..., None] if batched else bonds}
    if batched:
        features.update(atom_to_token=torch.eye(4)[None], type_bonds=(bonds * 2).long()[None],
                        token_pad_mask=torch.ones(1, 4), mol_type=torch.tensor([[0, 0, 3, 3]]),
                        affinity_token_mask=torch.tensor([[0., 0., 1., 1.]]),
                        s_esm=torch.zeros(1, 4, 3), asym_id=torch.tensor([[0, 0, 1, 1]]),
                        residue_index=torch.arange(4)[None])
        extra = {"affinity_keep_indices": torch.arange(4)}
    else:
        features.update(atom_to_token_idx=torch.arange(4), relp=torch.zeros(4, 4, 3),
                        is_ligand=torch.tensor([0, 0, 1, 1]), is_protein=torch.tensor([1, 1, 0, 0]))
        mask = torch.zeros(4, 4, dtype=torch.bool)
        mask[0, 2] = mask[1, 3] = True
        extra = dict(distance_pair_mask=mask, distance_target_bins=torch.zeros(4, 4, dtype=torch.long),
                     protein_token_mask=torch.tensor([True, True, False, False]))
    payload = dict(schema_version=1, model=model, spec=spec.to_dict(), features=features,
                   seed=dict(atoms=seed.atoms, charges=seed.charges, bonds=seed.bonds),
                   slot_to_atom=torch.tensor([2, 3, -1]), slot_to_token=torch.tensor([2, 3, -1]),
                   element_index=torch.tensor([-1, *(z - offset for z in spec.atomic_numbers[1:])]),
                   ligand_atom_mask=torch.tensor([False, False, True, True]),
                   ligand_token_mask=torch.tensor([False, False, True, True]),
                   metadata=dict(preparation_command="unit fixture", native_source_version="fake-no-native-model",
                                 seed_identity="unit fixture only", distance_target_source="unit fixture"), extra=extra)
    return NativeBundle(payload, spec=spec, model=model, device="cpu")


def differentiable_seed(bundle):
    graph = bundle.seed.to_relaxed(bundle.spec)
    return RelaxedGraph(*(t.detach().clone().requires_grad_() for t in (graph.atoms, graph.charges, graph.bonds)))


@pytest.mark.parametrize("model", ["nesso1", "protenix_v1"])
def test_soft_native_features_preserve_seed_and_context(model):
    bundle = make_bundle(model)
    bundle.assert_seed_features(charge=True)
    graph = differentiable_seed(bundle)
    features = bundle.soft_features(graph, charge=True)
    score = (features["ref_element"] * torch.arange(128)).sum() + features["ref_charge"].sum() + features["token_bonds"].sum()
    gradients = torch.autograd.grad(score, (graph.atoms, graph.charges, graph.bonds))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    assert torch.equal(bundle.features["ref_element"], make_bundle(model).features["ref_element"])


@pytest.mark.parametrize("channel", ["atoms", "charges", "bonds"])
def test_native_branch_rejects_bad_channel_probabilities(channel):
    bundle = make_bundle()
    graph = bundle.seed.to_relaxed(bundle.spec)
    getattr(graph, channel).flatten()[0] = float("nan")
    assert bundle.assess(graph, ()).status == "unsupported"


def test_native_branch_rejects_changed_occupancy_and_padding_bonds():
    bundle = make_bundle()
    graph = bundle.seed.to_relaxed(bundle.spec)
    graph.atoms[0, 2, 0], graph.atoms[0, 2, 1] = 0, 1
    assert bundle.assess(graph, ()).status == "unsupported"
    graph = bundle.seed.to_relaxed(bundle.spec)
    graph.bonds[0, 0, 2] = graph.bonds[0, 2, 0] = F.one_hot(torch.tensor(1), 5).float()
    assert bundle.assess(graph, ()).status == "unsupported"


def test_native_source_hash_includes_uncommitted_bytes(tmp_path):
    package = tmp_path / "nesso"
    package.mkdir()
    source = package / "model.py"
    source.write_text("x = 1\n")
    before = native_source_sha256(tmp_path, "nesso")
    source.write_text("x = 2\n")
    assert before != native_source_sha256(tmp_path, "nesso")


def test_native_bundle_detects_loaded_context_mutation():
    bundle = make_bundle()
    bundle.assert_unchanged()
    bundle.features["s_esm"][0, 0, 0] = 1
    with pytest.raises(ValueError, match="context was mutated"):
        bundle.assert_unchanged()


class Input(nn.Module):
    def __init__(self, batched):
        super().__init__()
        self.atom_encoder = SimpleNamespace(add_additional_atom_features=True)
        self.batched = batched

    def forward(self, f, **kwargs):
        scalar = f["ref_element"] @ torch.arange(128, dtype=torch.float32) / 128 + f["ref_charge"]
        return scalar[..., None].expand(*scalar.shape, 3)


class KeepPair(nn.Module):
    def forward(self, z, **kwargs):
        return z


class Affinity(nn.Module):
    def forward(self, s_inputs, z, pdistogram, **kwargs):
        value = (s_inputs.mean() + z.mean() + pdistogram[..., 0].mean()).reshape(1, 1)
        return {"affinity_pred_value": value, "affinity_logits_binary": value}


class FakeNesso(nn.Module):
    def __init__(self):
        super().__init__()
        self.affinity_prediction = True
        self.input_embedder = Input(True)
        self.z_init_1 = self.z_init_2 = nn.Identity()
        self.token_bonds = nn.Linear(1, 3)
        self.token_bonds_type = nn.Embedding(7, 3)
        self.z_recycle = self.z_norm = nn.Identity()
        self.esm_module = self.pairformer_module = KeepPair()
        self.distogram_head = nn.Linear(3, 4)
        self.affinity_module = self.affinity_module2 = Affinity()

    def rel_pos(self, f):
        return f["token_bonds"].expand(1, 4, 4, 3) * 0


def test_nesso_fixed_context_path_keeps_input_gradients_and_frozen_parameters():
    bundle = make_bundle()
    branch = NessoBranch(FakeNesso(), bundle, None, recycling_steps=1,
                         bond_ids=torch.tensor([0, 2, 3, 4, 5]))
    graph = differentiable_seed(bundle)
    output = branch(graph)
    gradients = torch.autograd.grad(output["affinity_pred_value"].sum(), (graph.atoms, graph.charges, graph.bonds))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    assert not any(p.requires_grad for p in branch.parameters())
    assert not any(m.training for m in branch.modules())
    assert branch.assess(graph).status == "lossy"


class MSA(nn.Module):
    def forward(self, f, z, s, **kwargs):
        return z + 0.1 * (s[:, None] + s[None, :])


class Pairformer(nn.Module):
    def forward(self, s, z, **kwargs):
        return s, z + 0.2 * (s[:, None] + s[None, :])


class FakeProtenix(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_confidence_only = False
        self.configs = SimpleNamespace(triangle_multiplicative="torch", triangle_attention="torch")
        self.input_embedder = Input(False)
        self.linear_no_bias_sinit = self.linear_no_bias_zinit1 = self.linear_no_bias_zinit2 = nn.Identity()
        self.relative_position_encoding = self.layernorm_z_cycle = self.layernorm_s = nn.Identity()
        self.linear_no_bias_z_cycle = self.linear_no_bias_s = nn.Identity()
        self.linear_no_bias_token_bond = nn.Linear(1, 3)
        self.template_embedder = nn.Identity()
        self.template_embedder.n_blocks = 0
        self.msa_module, self.pairformer_stack = MSA(), Pairformer()
        self.distogram_head = nn.Linear(3, 4)


def test_protenix_distogram_path_has_graph_gradients_and_no_affinity_output():
    bundle = make_bundle("protenix_v1")
    branch = ProtenixBranch(FakeProtenix(), bundle, None, cycles=2)
    graph = differentiable_seed(bundle)
    output = branch(graph)
    assert set(output) == {"distance_consistency_loss", "distogram_entropy"}
    gradients = torch.autograd.grad(output["distance_consistency_loss"].sum(), (graph.atoms, graph.charges, graph.bonds))
    # With identity fake blocks, ligand-to-ligand bonds cannot reach selected
    # ligand-to-protein pairs: this is an intentional zero, not a real-model gate.
    assert all(torch.isfinite(g).all() for g in gradients)
    assert gradients[0].abs().sum() > 0 and gradients[1].abs().sum() > 0
    assert not any(p.requires_grad for p in branch.parameters())
    assert not any(m.training for m in branch.modules())
    uniform = branch._reduce(torch.zeros(4, 4, 4))
    torch.testing.assert_close(uniform["distance_consistency_loss"], torch.ones(1))


def test_protenix_rejects_nonprotein_distance_pairs():
    bundle = make_bundle("protenix_v1")
    bundle.extra["distance_pair_mask"][2, 3] = True
    with pytest.raises(ValueError, match="ligand-to-context"):
        ProtenixBranch(FakeProtenix(), bundle, None, cycles=1)
