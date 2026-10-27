"""Allocated graph/slot identity contracts, including symmetric graph maps."""
from dataclasses import replace

import pytest
import torch
from torch import nn

from latentmol.chemistry import encode_smiles
from latentmol.diffsbdd.codec import FrozenVAECodec, SlotMap, encode_graph, graph_in_native_order
from latentmol.diffsbdd.transport import HandoffRejected, MolecularGraph
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch


SPEC = GraphSpec(max_nodes=5)


def _native(elements=("O", "C", "C"), atom_ids=(11, 23, 7)):
    return MolecularGraph(elements, (0, 0, 0),
                          ((0, 1, 0), (1, 0, 1), (0, 1, 0)), atom_ids)


class ExactEngineeringVAE(nn.Module):
    """Known reconstruction boundary; not a claim about pretrained VAE quality."""

    def __init__(self, identity="CCO", *, occupancy_error=False):
        super().__init__()
        self.spec = SPEC
        self.identity = identity
        self.occupancy_error = occupancy_error
        self.eval()

    def encode(self, graph):
        return torch.zeros(1, 2), torch.zeros(1, 2)

    def decode(self, latent):
        encoded = encode_smiles(self.identity, self.spec)
        fields = []
        for name, count in (("atom_types", len(SPEC.atomic_numbers)),
                            ("charges", len(SPEC.formal_charges)),
                            ("bond_types", len(SPEC.bond_names))):
            value = torch.as_tensor(encoded[name], dtype=torch.long)
            if name == "atom_types" and self.occupancy_error:
                value[-1] = SPEC.atomic_numbers.index(6)
            fields.append((torch.nn.functional.one_hot(value, count).float() * 30 - 15)[None])
        return DecoderOutput(*fields)


@pytest.mark.parametrize("molecule", [_native(), _native(("C", "C", "C"))])
def test_canonical_codec_retains_explicit_native_permutation(molecule):
    encoded = encode_graph(molecule, SPEC)
    assert graph_in_native_order(encoded.hard, SPEC, encoded.slots) == molecule
    assert set(encoded.slots.native_to_slot) == set(encoded.slots.active_slots)
    assert encoded.slots.native_atom_ids == molecule.atom_ids
    assert encoded.slots.record()["symmetry_policy"].startswith("explicit_verified")


def test_symmetric_graph_survives_native_id_permutation():
    source = _native(("C", "C", "C"))
    permutation = (2, 1, 0)
    permuted = MolecularGraph(tuple(source.elements[i] for i in permutation),
                              tuple(source.charges[i] for i in permutation),
                              tuple(tuple(source.bonds[i][j] for j in permutation) for i in permutation),
                              tuple(source.atom_ids[i] for i in permutation))
    encoded = encode_graph(permuted, SPEC)
    assert graph_in_native_order(encoded.hard, SPEC, encoded.slots) == permuted


def test_ordinary_roundtrip_passes_before_fixed_policy_and_preserves_all_bonds():
    molecule = _native()
    codec = FrozenVAECodec(ExactEngineeringVAE(), atom_count=3)
    admission = codec.admit(molecule)
    assert admission.receipt["fixed_occupancy_introduced_after_natural_identity"]
    assert admission.receipt["natural"]["smiles"] == "CCO"
    assert admission.native_graph == molecule
    decoded = codec.decode(admission.latent, admission)
    assert decoded.graph == molecule
    assert decoded.receipt["conditioned"]["smiles"] == "CCO"


def test_natural_occupancy_error_cannot_be_hidden_by_fixed_mask():
    codec = FrozenVAECodec(ExactEngineeringVAE(occupancy_error=True), atom_count=3)
    with pytest.raises(HandoffRejected, match="natural_roundtrip_changed_atom_count") as error:
        codec.admit(_native())
    assert error.value.receipt["natural"]["occupied_slots"] == 4


def test_changed_natural_identity_is_rejected_before_mapping():
    codec = FrozenVAECodec(ExactEngineeringVAE(identity="CCC"), atom_count=3)
    with pytest.raises(HandoffRejected, match="natural_roundtrip_changed_identity"):
        codec.admit(_native())


def test_slot_map_rejects_incomplete_or_duplicate_bijection():
    with pytest.raises(ValueError, match="bijection"):
        SlotMap((0, 1, 2), (0, 0, 2), (11, 23, 7))
    with pytest.raises(ValueError, match="bijection"):
        SlotMap((0, 1, 2), (0, 1), (11, 23, 7))


def test_graph_conversion_rejects_changed_fixed_occupancy():
    encoded = encode_graph(_native(), SPEC)
    hard = encoded.hard
    changed = hard.atoms.clone()
    changed[0, -1] = SPEC.atomic_numbers.index(6)
    with pytest.raises(HandoffRejected, match="occupancy_changed"):
        graph_in_native_order(HardGraphBatch(changed, hard.charges, hard.bonds), SPEC, encoded.slots)
