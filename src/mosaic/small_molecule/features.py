"""Prepare native fixed metadata once, then inject the shared graph with JAX.

RDKit and native hard-ligand import happen only in ``prepare_context``. That
function is preprocessing and belongs on an allocated HPC node. ``inject_graph``
uses neither discrete molecule conversion nor host arrays/identity lookups.
"""

import hashlib
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

from .contracts import PreparedContext, validate_context
from .native_hook import (
    GEOMETRY_FEATURE,
    HOOK_VERSION,
    UPSTREAM_REVISION,
    file_sha256,
    verify_native_source,
)
from .state import BOND_STATES, ELEMENTS, RelaxedGraph, bond_existence, graph_from_hard

# Native element indices are atomic number minus one, audited in
# protenix/data/constants.py:get_all_elems and Featurizer.elem_onehot_encoded.
NATIVE_ELEMENT_INDICES = (5, 6, 7, 15, 8, 16)
MODEL_ALIAS = "protenix_tiny_default_v0.5.0"
ASSET_REVISION = "81a257b61a82a17c41a7c20638460c04c29d97ad"
ASSET_SHA256 = {
    "clusters-by-entity-40.txt": "4828f38172e9f6cc5b6c552c23012d5f1d73178c178a31482a1363991a16d3cf",
    "components.v20240608.cif": "7240b17369ccfbbcc86e2d02dc8c9db59f46c32e0420f58889c6c121c60bfef0",
    "components.v20240608.cif.rdkit_mol.pkl": "2d6caced2d26c62015115a1d0a50f4106755a300e0c2a2d2b5c101c9038dcbcd",
    MODEL_ALIAS
    + ".eqx": "19f5f7af00cda104e8b059289e7e641cecc03309d7e960e6e53d82ad6e3f1f57",
    MODEL_ALIAS
    + ".skeleton.pkl": "d2192a547d471ed132e1b051b496baf5e9eb60dca1519ae12a78de124808aa1e",
}


def native_cache_dir(cache_dir: Path | str | None = None) -> Path:
    """An explicit path denotes the native asset directory, not its parent."""
    if cache_dir is None:
        from mosaic.cache import cache_dir as mosaic_cache_dir

        return mosaic_cache_dir() / "protenix"
    return Path(cache_dir).expanduser().resolve()


def bootstrap_native_assets(cache_dir=None, *, include_model=False) -> dict[str, str]:
    """Authenticate staged files before CCD imports; never fetch missing assets.

    Native CCD captures paths at module import. A process which imported it from
    another cache must fail explicitly instead of silently mixing asset sources.
    Model construction also verifies the serialized skeleton before unpickling.
    """
    verify_native_source()
    cache = native_cache_dir(cache_dir)
    names = tuple(ASSET_SHA256) if include_model else tuple(ASSET_SHA256)[:3]
    for name in names:
        path = cache / name
        if not path.is_file():
            raise FileNotFoundError(f"Pre-stage the pinned native asset on HPC: {path}")
        actual = file_sha256(path)
        if actual != ASSET_SHA256[name]:
            raise RuntimeError(f"Native asset checksum mismatch: {path}: {actual}")
    # Keep this explicit: the cluster file is NOT the CIF path.
    paths = {
        "ccd_components_file": str(cache / "components.v20240608.cif"),
        "ccd_components_rdkit_mol_file": str(
            cache / "components.v20240608.cif.rdkit_mol.pkl"
        ),
        "pdb_cluster_file": str(cache / "clusters-by-entity-40.txt"),
    }
    loaded_ccd = sys.modules.get("protenix.data.ccd")
    if loaded_ccd is not None and (
        Path(loaded_ccd.COMPONENTS_FILE).resolve() != Path(paths["ccd_components_file"])
        or Path(loaded_ccd.RKDIT_MOL_PKL).resolve()
        != Path(paths["ccd_components_rdkit_mol_file"])
    ):
        raise RuntimeError(
            "CCD was already imported with different assets; use a fresh process"
        )
    from protenix.configs.configs_data import data_configs

    for name, path in paths.items():
        if name not in data_configs:
            raise RuntimeError(f"Unaudited native asset configuration field: {name}")
        data_configs[name] = path
    return paths


def _feature_hash(features) -> str:
    """Stable host digest includes dtype, shape and every native feature byte."""
    digest = hashlib.sha256()
    for name, value in sorted(features.items()):
        if isinstance(value, dict):
            digest.update(name.encode())
            digest.update(_feature_hash(value).encode())
        else:
            array = np.ascontiguousarray(value)
            digest.update(json.dumps([name, str(array.dtype), array.shape]).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def _jax_features(features):
    # The native featurizer produces numeric arrays; preserve integer metadata
    # while using float32 for learned feature channels, as native inference does.
    float_channels = {
        "ref_element",
        "ref_atom_name_chars",
        "restype",
        "profile",
        "token_bonds",
    }
    return {
        name: jnp.asarray(value, dtype=jnp.float32)
        if name in float_channels or np.asarray(value).dtype.kind == "f"
        else jnp.asarray(value)
        for name, value in features.items()
    }


def _paired_hard_graph(mol) -> RelaxedGraph:
    """Recover orders from the paired hard Mol, never native's orderless bonds."""
    from rdkit import Chem

    n_atoms = mol.GetNumAtoms()
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    if n_atoms < 2 or n_atoms > 1000:
        raise ValueError("Stage 0 fixtures require 2..1000 occupied heavy-atom slots")
    if any(element not in ELEMENTS for element in elements):
        raise ValueError(
            f"Unsupported elements: {sorted(set(elements) - set(ELEMENTS))}"
        )
    if any(
        atom.GetFormalCharge() != 0 or atom.GetIsotope() != 0 for atom in mol.GetAtoms()
    ):
        raise ValueError("V0 requires neutral, non-isotopic atoms")
    pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]
    pair_lookup = {pair: i for i, pair in enumerate(pairs)}
    states = np.zeros(len(pairs), dtype=np.int32)
    categories = {
        Chem.BondType.SINGLE: "SINGLE",
        Chem.BondType.DOUBLE: "DOUBLE",
        Chem.BondType.TRIPLE: "TRIPLE",
        Chem.BondType.AROMATIC: "AROMATIC",
    }
    for bond in mol.GetBonds():
        category = categories.get(bond.GetBondType())
        if category is None:
            raise ValueError(f"Unsupported hard bond category: {bond.GetBondType()}")
        pair = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        states[pair_lookup[pair]] = BOND_STATES.index(category)
    return graph_from_hard([ELEMENTS.index(e) for e in elements], states)


def prepare_context(
    smiles: str = "CC(=O)Nc1ccccc1",
    protein_chains=(),
    *,
    cache_dir=None,
    input_name="stage0",
    identities: dict[str, str] | None = None,
) -> tuple[PreparedContext, RelaxedGraph]:
    """Import one neutral ligand and native proteins outside JIT/optimization.

    Protein chains may be strings (no MSA/templates) or Mosaic ``TargetChain``
    instances explicitly using ``use_msa=False`` and no template. Ligand slots
    follow native importer heavy-atom order. Native import uses a generated hard
    conformer; its complete features are sealed by hash. Conformer regeneration
    is not promised identical, so restart uses the saved prepared context.
    """
    paths = bootstrap_native_assets(cache_dir)
    from rdkit import Chem
    from protenix.data.featurizer import Featurizer
    from protenix.data.json_to_feature import SampleDictToFeatures
    from protenix.data.tokenizer import AtomArrayTokenizer

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("The hard fixture SMILES could not be parsed")
    mol = Chem.RemoveHs(mol)
    graph = _paired_hard_graph(mol)
    sequences = []
    for chain in protein_chains:
        sequence = chain if isinstance(chain, str) else chain.sequence
        if not isinstance(chain, str):
            if chain.use_msa or chain.template_chain is not None:
                raise ValueError(
                    "Stage 0 preparation requires no MSA search or templates"
                )
            from mosaic.structure_prediction import PolymerType

            if chain.polymer_type != PolymerType.PROTEIN:
                raise ValueError(
                    "V0 context supports standard proteins and one ligand only"
                )
        if not sequence or any(
            letter not in "ACDEFGHIKLMNPQRSTVWY" for letter in sequence
        ):
            raise ValueError(
                "V0 protein context requires a nonempty standard amino-acid sequence"
            )
        sequences.append({"proteinChain": {"sequence": sequence, "count": 1}})
    sequences.append({"ligand": {"ligand": smiles, "count": 1}})
    sample = {"name": input_name, "sequences": sequences}
    importer = SampleDictToFeatures(sample)
    atom_array = importer.get_atom_array()
    tokens = AtomArrayTokenizer(atom_array).get_token_array()
    # Disable only arbitrary reference augmentation. This native hard baseline
    # retains the actual importer conformer and truthful native frame inference.
    native = Featurizer(
        tokens, atom_array, ref_pos_augment=False
    ).get_all_input_features()
    native["distogram_rep_atom_mask"] = np.asarray(
        atom_array.distogram_rep_atom_mask, dtype=np.int64
    )
    # Native query-only tokenization assigns restype to nonprotein tokens too;
    # profile=one-hot restype and deletion=0 reproduce its one-query convention.
    token_count = native["restype"].shape[0]
    native.update(
        profile=native["restype"].astype(np.float32),
        deletion_mean=np.zeros(token_count, dtype=np.float32),
        msa=native["restype"].argmax(-1).astype(np.int32)[None, :],
        has_deletion=np.zeros((1, token_count), dtype=np.float32),
        deletion_value=np.zeros((1, token_count), dtype=np.float32),
    )
    ligand_atoms = np.flatnonzero(native["is_ligand"]).astype(int)
    ligand_tokens = native["atom_to_token_idx"][ligand_atoms].astype(int)
    protein_tokens = np.unique(
        native["atom_to_token_idx"][np.flatnonzero(native["is_protein"])]
    ).astype(int)
    if (
        ligand_atoms.size != graph.X.shape[0]
        or np.unique(ligand_tokens).size != ligand_atoms.size
    ):
        raise ValueError(
            "The native importer did not preserve one token per ligand heavy atom"
        )
    imported_elements = np.asarray(atom_array.element)[ligand_atoms].tolist()
    if imported_elements != [atom.GetSymbol().upper() for atom in mol.GetAtoms()]:
        raise ValueError(
            "Native importer and paired hard molecule disagree on atom order"
        )
    if np.any(native["ref_charge"][ligand_atoms] != 0):
        raise ValueError("Native ligand reference charge is not neutral")
    if np.any(native["ref_mask"][ligand_atoms] != 1):
        raise ValueError(
            "Native ligand identity conditioning requires present reference slots"
        )
    # Native BondList drops order; check only existence against the paired Mol.
    expected_bonds = np.zeros((ligand_atoms.size, ligand_atoms.size), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        expected_bonds[i, j] = expected_bonds[j, i] = 1
    if not np.array_equal(
        native["token_bonds"][np.ix_(ligand_tokens, ligand_tokens)], expected_bonds
    ):
        raise ValueError(
            "Native ligand bond existence disagrees with the paired hard molecule"
        )
    if protein_tokens.size and np.any(
        native["token_bonds"][np.ix_(ligand_tokens, protein_tokens)]
    ):
        raise ValueError(
            "Covalent protein-ligand links are outside the V0 graph contract"
        )

    names = tuple(f"Q{i:03d}" for i in range(ligand_atoms.size))
    base = {name: np.array(value, copy=True) for name, value in native.items()}
    base["ref_pos"][ligand_atoms] = 0
    base["ref_charge"][ligand_atoms] = 0
    # Static slot names must not reveal the initial mutable element (C1/N1/etc).
    base["ref_atom_name_chars"][ligand_atoms] = Featurizer.ref_atom_name_chars_encoded(
        names
    )
    base["has_frame"][ligand_tokens] = 0
    base["frame_atom_index"][ligand_tokens] = -1
    geometry = np.ones(len(atom_array), dtype=np.float32)
    geometry[ligand_atoms] = 0
    input_identity = {
        "smiles": smiles,
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "protein_sequences": [
            next(iter(entry.values()))["sequence"] for entry in sequences[:-1]
        ],
        "ligand_atom_indices": ligand_atoms.tolist(),
        "ligand_token_indices": ligand_tokens.tolist(),
        "hydrogens": "native AddHs followed by heavy-atom removal; no optimized H slots",
    }
    # Authenticate the actual resident model inputs after dtype normalization,
    # not only the NumPy importer's pre-cast integer one-hot representation.
    prepared_native = _jax_features(native)
    prepared_base = _jax_features(base)
    binding = dict(identities or {})
    protected = {
        "upstream_revision": UPSTREAM_REVISION,
        "hook_version": HOOK_VERSION,
        "asset_revision": ASSET_REVISION,
        "input_sha256": hashlib.sha256(
            json.dumps(input_identity, sort_keys=True).encode()
        ).hexdigest(),
        "native_features_sha256": _feature_hash(prepared_native),
        "base_features_sha256": _feature_hash(prepared_base),
        "native_precast_features_sha256": _feature_hash(native),
        "model_alias": MODEL_ALIAS,
    }
    if any(
        name in binding and binding[name] != value for name, value in protected.items()
    ):
        raise ValueError(
            "Provided identities disagree with authenticated prepared inputs"
        )
    binding.update(protected)
    binding["asset_paths_sha256"] = hashlib.sha256(
        json.dumps(paths, sort_keys=True).encode()
    ).hexdigest()
    context = PreparedContext(
        base_features=prepared_base,
        ligand_atom_indices=tuple(ligand_atoms.tolist()),
        ligand_token_indices=tuple(ligand_tokens.tolist()),
        protein_token_indices=tuple(protein_tokens.tolist()),
        reference_geometry_mask=jnp.asarray(geometry),
        atom_names=names,
        geometry_mode="off",
        identities=binding,
        native_features=prepared_native,
    )
    validate_context(context)
    return context, graph


def inject_graph(context: PreparedContext, graph: RelaxedGraph):
    """Pure differentiable graph -> native feature update; no mutable caches.

    Element probabilities enter the native 128-element linear feature channel.
    Only NO_BOND versus total bond existence is observed by Protenix; exchanges
    among the four positive-order states are deliberately a null direction.
    """
    if context.geometry_mode != "off":
        raise ValueError(f"Unsupported relaxed geometry mode: {context.geometry_mode}")
    n_atoms = context.n_atoms
    if graph.X.shape != (n_atoms, len(ELEMENTS)) or graph.E_upper.shape != (
        n_atoms * (n_atoms - 1) // 2,
        len(BOND_STATES),
    ):
        raise ValueError("Graph shape disagrees with the prepared native slot contract")
    atom_indices = jnp.asarray(context.ligand_atom_indices, dtype=jnp.int32)
    token_indices = jnp.asarray(context.ligand_token_indices, dtype=jnp.int32)
    # This is a pure channel copy, not a learned linear projection. A one-hot
    # matrix product can use reduced-precision GPU matmul (TF32) and round the
    # graph probabilities even when both arrays are float32. Indexed scatter
    # preserves the supplied values exactly and has the same linear derivative.
    ligand_elements = (
        jnp.zeros((n_atoms, 128), dtype=graph.X.dtype)
        .at[:, jnp.asarray(NATIVE_ELEMENT_INDICES, dtype=jnp.int32)]
        .set(graph.X)
    )
    # Native hard one-hot channels may be integer. Scatter into an explicitly
    # floating array: writing a relaxed distribution to integer storage would
    # silently truncate it and destroy the atom-probability derivative.
    elements = (
        jnp.asarray(context.base_features["ref_element"], dtype=graph.X.dtype)
        .at[atom_indices]
        .set(ligand_elements)
    )
    # Reconstruct both orientations from the single stored unordered pair. Each
    # pair's reverse-mode gradient therefore includes both native matrix entries.
    existence = bond_existence(graph)
    bonds = (
        jnp.asarray(context.base_features["token_bonds"], dtype=graph.E_upper.dtype)
        .at[token_indices[:, None], token_indices[None, :]]
        .set(existence)
    )
    # Reassert zero placeholder even if a test substitutes an unused conformer.
    positions = context.base_features["ref_pos"].at[atom_indices].set(0)
    return context.base_features | {
        "ref_element": elements,
        "token_bonds": bonds,
        "ref_pos": positions,
        GEOMETRY_FEATURE: context.reference_geometry_mask,
    }
