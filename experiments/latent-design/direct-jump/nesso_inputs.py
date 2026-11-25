"""Native Nesso-1 inference features for the fixed BRD4 construct and neutral 25-slot ligands."""
import hashlib
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from study import BOND_TYPES, ELEMENTS, PAIR_I, PAIR_J


# The same 3MXF BRD4 BD1 chain sequence supplied to our Boltz2 baseline.
BRD4_SEQUENCE = ("SMNPPPPETSNPNKPKRQTNQLQYLLRVVLKTLWKHQFAWPFQQPVDAVKLNLPDYYKIIKTPMDMGTI"
                 "KKRLENNYYWNAQECIQDFNTMFTNCYIYNKPGDDIVLMAEALEKLFLQKINELPTEE")


class ConformerUnavailable(ValueError):
    """Embedding the given discrete ligand failed with both native ETKDG proposals."""


def prepare_esm(esm_dir, sequence=BRD4_SEQUENCE, *, model_name="facebook/esm2_t33_650M_UR50D",
                cache=None, device="cuda"):
    """Generate the same last-layer ESM features as the released native CLI, once per sequence."""
    from nesso.data.esm import extract_esm_embedding, setup_esm_model
    from nesso.data.yaml_input import esm_keys
    from safetensors.torch import load_file, save_file

    folder = Path(esm_dir)
    folder.mkdir(parents=True, exist_ok=True)
    key, = esm_keys({0: sequence})
    path = folder / f"{key}.safetensors"
    if path.exists():
        emb = load_file(str(path))["embeddings"]
    else:
        model, tokenizer = setup_esm_model(str(model_name), torch.device(device), cache_dir=cache)
        emb = extract_esm_embedding(sequence, model, tokenizer)
    if emb.shape != (1, len(sequence) + 2, 1280) or not torch.isfinite(emb).all():
        raise ValueError("Invalid sequence-conditioned ESM-2 feature tensor")
    if not path.exists():
        save_file({"embeddings": emb.contiguous()}, str(path))
    return path


def canonical_graph(smiles, graph_spec=None):
    """Project RDKit graph classes in native SMILES atom order onto the direct design vocabulary."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() != 25 or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Expected one sanitized, connected 25-atom ligand")
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    mol = Chem.MolFromSmiles(canonical)
    elements = ELEMENTS if graph_spec is None else graph_spec.atomic_numbers[1:]
    charges = (0,) if graph_spec is None else graph_spec.formal_charges
    if any(a.GetFormalCharge() not in charges or a.GetAtomicNum() not in elements for a in mol.GetAtoms()):
        raise ValueError("Ligand element or formal charge is outside the configured vocabulary")
    atoms = np.eye(len(elements), dtype=np.float32)[[elements.index(a.GetAtomicNum()) for a in mol.GetAtoms()]]
    matrix = np.zeros((25, 25), dtype=np.int64)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        matrix[i, j] = matrix[j, i] = BOND_TYPES.index(bond.GetBondType())
    bonds = np.eye(len(BOND_TYPES), dtype=np.float32)[matrix[PAIR_I, PAIR_J]]
    return canonical, mol, atoms, bonds


class NessoInputs:
    def __init__(self, root, ccd, esm_dir, *, protein_sequence=BRD4_SEQUENCE, device="cuda", graph_spec=None):
        from nesso.data.inference import load_standard_aa_mols
        from nesso.data.featurizer import NessoFeaturizer

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sequence, self.device = protein_sequence, device
        self.graph_spec = graph_spec
        self.standard = load_standard_aa_mols(Path(ccd))
        self.featurizer = NessoFeaturizer(esm_emb_dir=Path(esm_dir), esm_emb_dim=1280, esm_num_layers=33)

    def featurize(self, smiles):
        """Build a fresh seeded heavy-atom conformer, native protein/ligand features and slot map."""
        from nesso.data import const
        from nesso.data.inference import InferenceDataset, inference_collate
        from nesso.data.types import Manifest
        from nesso.data.yaml_input import parse_schema

        canonical, mol, atoms, bonds = canonical_graph(smiles, self.graph_spec)
        tag = hashlib.sha256(canonical.encode()).hexdigest()
        explicit = Chem.AddHs(mol)
        # A fixed numerical seed per identity makes each fresh conformer independent of job order.
        seed = int(tag[:8], 16) % (2**31 - 1)
        params = AllChem.ETKDGv3()
        params.randomSeed, params.numThreads = seed, 1
        if AllChem.EmbedMolecule(explicit, params) < 0:
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(explicit, params) < 0:
                raise ConformerUnavailable(f"ETKDG failed for {canonical}")
        conformer = Chem.RemoveHs(explicit, sanitize=False)
        if Chem.MolToSmiles(conformer, isomericSmiles=True) != canonical:
            raise ValueError("Preparing the Nesso reference conformer changed the ligand identity")
        with TemporaryDirectory(dir=self.root) as temp:
            folder = Path(temp)
            conformer_path = folder / "input.pkl"
            with conformer_path.open("xb") as handle:
                pickle.dump(conformer, handle)
            schema = {"version": 1, "sequences": [
                {"protein": {"id": "A", "sequence": self.sequence}},
                {"ligand": {"id": "B", "smiles": canonical, "conformer": str(conformer_path)}}],
                "properties": [{"affinity": {"binder": "B"}}]}
            struct, record, _, _ = parse_schema(schema, folder, ccd_dict=self.standard, record_id=tag)
            (folder / "structures").mkdir()
            struct.dump(folder / "structures" / f"{record.id}.npz")
            ds = InferenceDataset(Manifest([record]), folder, self.featurizer, folder)
            # Read the official CCD once per allocation instead of reloading it for each candidate.
            ds._standard_aa_mols = self.standard
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(1701)
                row = ds[0]
            if "exception" in row:
                raise ValueError(f"Native Nesso featurization failed for {canonical}")
            batch = {k: v for k, v in inference_collate([row]).items() if torch.is_tensor(v)}

        binder_tokens = batch["affinity_token_mask"][0].bool()
        if (int(binder_tokens.sum()) != 25 or
                not torch.all(batch["mol_type"][0, binder_tokens] == const.chain_type_ids["NONPOLYMER"])):
            raise ValueError("Unexpected ligand-token mapping in native Nesso input")
        incidence = batch["atom_to_token"][0, :, binder_tokens]
        slot_to_atom = torch.nonzero(incidence.sum(-1), as_tuple=True)[0]
        if not torch.equal(incidence[slot_to_atom], torch.eye(25, dtype=incidence.dtype)):
            raise ValueError("Expected one heavy atom per Nesso ligand token in input order")
        elements = batch["ref_element"][0, slot_to_atom].argmax(-1).numpy()
        if not np.array_equal(elements, [a.GetAtomicNum() for a in mol.GetAtoms()]):
            raise ValueError("RDKit/Nesso element correspondence disagrees")
        if self.graph_spec is not None:
            native_charge = batch["ref_charge"][0, slot_to_atom].reshape(-1).numpy()
            if not np.array_equal(native_charge, [a.GetFormalCharge() for a in mol.GetAtoms()]):
                raise ValueError("RDKit/Nesso formal charge correspondence disagrees")
        native_tokens = torch.nonzero(binder_tokens, as_tuple=True)[0]
        native_types = batch["type_bonds"][0, native_tokens[:, None], native_tokens[None, :]]
        bond_ids = np.asarray([0] + [1 + const.bond_type_ids[b.name] for b in BOND_TYPES[1:]])
        if not np.array_equal(native_types.numpy()[PAIR_I, PAIR_J], bond_ids[bonds.argmax(-1)]):
            raise ValueError("RDKit/Nesso bond correspondence disagrees")
        return {k: v.to(self.device) for k, v in batch.items()}, slot_to_atom.to(self.device), atoms, bonds
