"""Chemistry/cache contract checks. Run these tests on the configured HPC only."""

import csv
import json

import numpy as np
import pytest
import torch

from latentmol.chemistry import MoleculeRejected, canonicalize_smiles, decode_graph, encode_smiles, inspect_graph
from latentmol.data import GraphDataset, collate_graphs, prepare_dataset
from latentmol.representation import GraphSpec


@pytest.mark.parametrize("smiles", [
    "CCO", "c1ccccc1", "c1ccncc1", "CC(=O)[O-]", "C[N+](C)(C)C", "[NH4+]",
])
def test_roundtrip_aromatic_and_charged_molecules(smiles):
    """Aromatic bonds and formal charges survive graph encoding and sanitization."""
    spec = GraphSpec()
    encoded = encode_smiles(smiles, spec)
    assert decode_graph(encoded["atom_types"], encoded["charges"],
                        encoded["bond_types"], spec) == encoded["smiles"]
    assert encoded["smiles"] == canonicalize_smiles(smiles, spec)
    for key in ("atom_types", "charges", "bond_types"):
        assert encoded[key].dtype == np.int8


def test_canonical_order_and_deliberate_stereo_stripping():
    """Equivalent atom orders and enantiomers have identical model targets."""
    spec = GraphSpec()
    for first, second in [("OCC", "CCO"), ("F[C@H](Cl)Br", "F[C@@H](Cl)Br")]:
        left, right = encode_smiles(first, spec), encode_smiles(second, spec)
        assert left["smiles"] == right["smiles"]
        assert "@" not in left["smiles"]
        for key in ("atom_types", "charges", "bond_types"):
            np.testing.assert_array_equal(left[key], right[key])


@pytest.mark.parametrize(("smiles", "reason"), [
    ("", "empty"), ("not_a_smiles", "invalid_smiles"), ("CC.O", "disconnected"),
    ("[SiH4]", "unsupported_element"), ("[13CH4]", "isotope"), ("[CH3]", "radical"),
    ("c1cc[nH]c1", "representation_roundtrip"),
])
def test_source_rejections_are_explicit(smiles, reason):
    """Unsupported chemistry is counted as rejection rather than silently repaired."""
    with pytest.raises(MoleculeRejected) as error:
        encode_smiles(smiles, GraphSpec())
    assert error.value.reason == reason


def test_source_size_limit():
    """The fixed maximum capacity is enforced before padding arrays are created."""
    with pytest.raises(MoleculeRejected, match="too_many_atoms"):
        encode_smiles("CCO", GraphSpec(max_nodes=2))


def test_decode_padding_permutation_and_tensor_inputs():
    """NO_ATOM slots may occur anywhere and never create bonds or fragments."""
    spec = GraphSpec(max_nodes=8)
    encoded = encode_smiles("CCO", spec)
    atoms, charges, bonds = (encoded[key].copy() for key in ("atom_types", "charges", "bond_types"))
    charges[-1] = 127  # Unused charge categories do not describe a real atom.
    bonds[0, -1] = bonds[-1, 0] = spec.bond_names.index("TRIPLE")
    permutation = np.roll(np.arange(spec.max_nodes), 3)
    result = decode_graph(torch.tensor(atoms[permutation], dtype=torch.long),
                          torch.tensor(charges[permutation], dtype=torch.long),
                          torch.tensor(bonds[np.ix_(permutation, permutation)], dtype=torch.long), spec)
    assert result == encoded["smiles"]


def test_decode_rejects_empty_disconnected_and_asymmetric_graphs():
    """Decoding rejects invalid graphs without selecting a largest fragment."""
    spec = GraphSpec(max_nodes=8)
    encoded = encode_smiles("CC", spec)
    atoms, charges, bonds = (encoded[key].copy() for key in ("atom_types", "charges", "bond_types"))
    assert decode_graph(np.zeros_like(atoms), charges, bonds, spec) is None
    disconnected = np.zeros_like(bonds)
    assert decode_graph(atoms, charges, disconnected, spec) is None
    asymmetric = bonds.copy()
    asymmetric[0, 1] = spec.bond_names.index("NONE")
    assert decode_graph(atoms, charges, asymmetric, spec) is None
    assert decode_graph(atoms.astype(float), charges, bonds, spec) is None


def test_decode_rejects_impossible_valence():
    """A neutral nitrogen with five single bonds is rejected without charge repair."""
    spec = GraphSpec(max_nodes=6)
    atoms = np.full(6, spec.atomic_numbers.index(6), dtype=np.int8)
    atoms[0] = spec.atomic_numbers.index(7)
    charges = np.full(6, spec.formal_charges.index(0), dtype=np.int8)
    bonds = np.zeros((6, 6), dtype=np.int8)
    bonds[0, 1:] = bonds[1:, 0] = spec.bond_names.index("SINGLE")
    assert decode_graph(atoms, charges, bonds, spec) is None
    result = inspect_graph(atoms, charges, bonds, spec)
    assert result.smiles is None
    assert result.failure_reason == "sanitization"
    assert result.detail == "AtomValenceException"


def test_decode_diagnostics_preserve_validity_and_first_failure():
    """Diagnostics distinguish empty/fragmented graphs without modifying them."""
    spec = GraphSpec(max_nodes=8)
    encoded = encode_smiles("CC", spec)
    atoms, charges, bonds = (encoded[key] for key in ("atom_types", "charges", "bond_types"))
    valid = inspect_graph(atoms, charges, bonds, spec)
    assert valid.smiles == "CC" and valid.failure_reason is None
    assert inspect_graph(np.zeros_like(atoms), charges, bonds, spec).failure_reason == "empty"
    fragmented = inspect_graph(atoms, charges, np.zeros_like(bonds), spec)
    assert fragmented.smiles is None and fragmented.failure_reason == "disconnected"
    assert inspect_graph(atoms.astype(float), charges, bonds, spec).failure_reason == "input_or_construction"


def _write_source(path, rows):
    """Create a small caller-owned CSV fixture without network acquisition."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["smiles", "source_id"])
        writer.writerows((smiles, index) for index, smiles in enumerate(rows))


def test_dedup_precedes_seeded_subset_and_split(tmp_path):
    """Row order and duplicate syntax cannot leak the same target across splits."""
    rows = ["C" * length for length in range(1, 21)] + [
        "CCO", "OCC", "F[C@H](Cl)Br", "F[C@@H](Cl)Br", "CC.O", "not_a_smiles", "",
    ]
    source_a, source_b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write_source(source_a, rows)
    _write_source(source_b, list(reversed(rows)))
    cache_a, cache_b = tmp_path / "cache_a", tmp_path / "cache_b"
    manifest_a = prepare_dataset(source_a, cache_a, limit=12, seed=17)
    manifest_b = prepare_dataset(source_b, cache_b, limit=12, seed=17)
    assert manifest_a["counts"]["source_rows"] == 27
    assert manifest_a["counts"]["unique_accepted_before_limit"] == 22
    assert manifest_a["counts"]["duplicate_rows"] == 2
    assert manifest_a["counts"]["rejected_rows"] == 3
    assert manifest_a["counts"]["total"] == 12
    assert manifest_a["counts"]["selected_target_graphs_roundtrip_valid"] == 12
    assert manifest_a["raw_sha256"] != manifest_b["raw_sha256"]
    assert manifest_a["dataset_sha256"] == manifest_b["dataset_sha256"]
    assert manifest_a["split_sha256"] == manifest_b["split_sha256"]
    datasets = [GraphDataset(cache_a, split) for split in ("train", "val", "test")]
    assert [len(dataset) for dataset in datasets] == [9, 1, 2]
    all_smiles = [smiles for dataset in datasets for smiles in dataset.smiles]
    assert len(all_smiles) == len(set(all_smiles)) == 12
    batch = collate_graphs([datasets[0][0], datasets[0][1]])
    assert batch.atoms.shape == (2, GraphSpec().max_nodes)
    assert batch.atoms.dtype == batch.charges.dtype == batch.bonds.dtype == torch.long
    with np.load(cache_a / "dataset.npz", allow_pickle=False) as archive:
        assert archive["smiles"].dtype.kind == "U"
        assert archive["split"].dtype == np.int8
    with pytest.raises(FileExistsError):
        prepare_dataset(source_a, cache_a)


def test_smi_input_and_integrity_checks(tmp_path):
    """The optional .smi reader supports names, and cache tampering is detected."""
    source = tmp_path / "molecules.smi"
    source.write_text("# locally supplied molecules\nCCO ethanol\nOCC duplicate\nC methane\n",
                      encoding="utf-8")
    cache = tmp_path / "cache"
    manifest = prepare_dataset(source, cache)
    assert manifest["counts"]["total"] == 2
    assert manifest["counts"]["duplicate_rows"] == 1
    manifest["dataset_sha256"] = "0" * 64
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        GraphDataset(cache, "train")
