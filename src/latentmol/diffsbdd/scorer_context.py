"""Fresh native Boltz features for a supplied ligand graph and receptor-frame pose.

This experiment-owned path parses every new graph with the official schema,
attaches the supplied bound coordinates to that fresh structure, then uses the
ordinary native tokenizer, affinity cropper and Boltz2 featurizer. The native
reference conformers remain separate from the supplied bound pose. No existing
JQ1/carbon-chain context or validator is changed or used as a feature template.
All chemistry/native execution belongs on allocated HPC.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
import torch

from latentmol.chemistry import decode_graph
from latentmol.data import sha256_file
from latentmol.boltz.contracts import PreparedComplex
from latentmol.boltz.dbmol_model import validate_frame_context
from latentmol.boltz.prepare import installed_boltz_provenance
from latentmol.boltz.provenance import context_digest
from .paired_data import assess_graph, slot_to_canonical, write_json


CONTEXT_VERSION = "diffsbdd-supplied-pose-native-context-v1"
AMINO_ACIDS = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G",
                   HIS="H", ILE="I", LEU="L", LYS="K", MET="M", PHE="F", PRO="P", SER="S",
                   THR="T", TRP="W", TYR="Y", VAL="V")


@dataclass(frozen=True)
class SuppliedPoseContext:
    prepared: PreparedComplex
    pocket_residues: tuple[dict, ...]
    receipt: dict

    def to(self, device):
        return SuppliedPoseContext(self.prepared.to(device), self.pocket_residues, self.receipt)


def normalize_author_residues(residues, protein_chain="A"):
    result = []
    for residue in residues:
        if (set(residue) != {"chain", "residue_number", "insertion_code"}
                or residue["chain"] != protein_chain or type(residue["residue_number"]) is not int
                or not isinstance(residue["insertion_code"], str) or len(residue["insertion_code"]) > 1):
            raise ValueError("Pocket requires explicit chain/author residue number/insertion code")
        result.append((residue["chain"], residue["residue_number"], residue["insertion_code"]))
    if not result or len(set(result)) != len(result):
        raise ValueError("Author pocket residues must be nonempty and unique")
    return tuple(sorted(result))


def read_receptor_pdb(path, *, protein_chain="A"):
    """Read one crystal model in file order, retaining actual atom coordinates.

    Select highest-occupancy alternate atoms with a deterministic lexical tie;
    record this policy. Modified polymer residues and ambiguous duplicate atom
    records fail. Waters, other chains and nonpolymer ligands are not receptor
    atoms. Missing atoms are kept unresolved by the fresh native structure.
    """
    residues, model_count = {}, 0
    for line in Path(path).read_text(encoding="ascii").splitlines():
        if line.startswith("MODEL "):
            model_count += 1
            if model_count > 1:
                raise ValueError("Supplied-pose preparation requires an explicitly selected single PDB model")
        kind = line[:6].strip()
        if kind not in ("ATOM", "HETATM") or len(line) < 54 or line[21:22] != protein_chain:
            continue
        name = line[17:20].strip()
        if name not in AMINO_ACIDS:
            if kind == "ATOM" or name in {"MSE", "SEC", "PYL"}:
                raise ValueError("Noncanonical polymer residue is outside the sealed receptor mapping")
            continue
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        atom_name = line[12:16].strip()
        if element in {"H", "D"} or atom_name.lstrip("0123456789").startswith("H"):
            continue
        key = (protein_chain, int(line[22:26]), line[26:27].strip())
        record = residues.setdefault(key, dict(name=name, atoms={}, alternate_count=0))
        if record["name"] != name:
            raise ValueError("One author residue has multiple chemical identities")
        altloc = line[16:17].strip()
        occupancy = float(line[54:60].strip() or "0") if len(line) >= 60 else 0.0
        coordinate = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float32)
        if not np.isfinite(coordinate).all() or not np.isfinite(occupancy):
            raise ValueError("Nonfinite crystal receptor coordinate/occupancy")
        candidate = dict(coordinates=coordinate, occupancy=occupancy, altloc=altloc, element=element)
        old = record["atoms"].get(atom_name)
        if old is not None:
            if old["altloc"] == altloc:
                raise ValueError("Duplicate crystal receptor atom identity")
            record["alternate_count"] += 1
        if old is None or (-occupancy, altloc) < (-old["occupancy"], old["altloc"]):
            record["atoms"][atom_name] = candidate
    if not residues:
        raise ValueError("No canonical receptor residues in the bound author chain")
    sequence = "".join(AMINO_ACIDS[row["name"]] for row in residues.values())
    return dict(sequence=sequence, residues=residues,
                alternate_policy="highest occupancy; lexical altloc tie, no coordinate averaging",
                model_policy="one explicitly selected crystal model")


def msa_query(path):
    query, started = [], False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if started:
                break
            started = True
        elif started:
            query.append(line.strip())
    result = "".join(query)
    if not result or any(character not in "ACDEFGHIKLMNPQRSTVWY" for character in result):
        raise ValueError("Require an explicit canonical ungapped uppercase A3M query")
    return result


def _native_backend():
    """Deferred official imports; no local numerical import is needed to inspect."""
    from boltz.data import const
    from boltz.data.crop.affinity import AffinityCropper
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.module.inferencev2 import collate
    from boltz.data.mol import load_canonicals, load_molecules
    from boltz.data.parse.a3m import parse_a3m
    from boltz.data.parse.schema import parse_boltz_schema
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    from boltz.data.types import Input
    return dict(const=const, AffinityCropper=AffinityCropper, Boltz2Featurizer=Boltz2Featurizer,
        collate=collate, load_canonicals=load_canonicals, load_molecules=load_molecules,
        parse_a3m=parse_a3m, parse_boltz_schema=parse_boltz_schema, Boltz2Tokenizer=Boltz2Tokenizer, Input=Input)


def _verify_native_molecule(molecule, identity, hard, spec):
    """Native affinity standardization may run; identity changes are rejected."""
    molecule = Chem.RemoveHs(Chem.Mol(molecule))
    Chem.RemoveStereochemistry(molecule)
    native_identity = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    if native_identity != identity:
        raise ValueError("Official native parser standardized the candidate to a different molecular identity")
    canonical = Chem.MolFromSmiles(identity)
    mapping = molecule.GetSubstructMatch(canonical, useChirality=False)
    active = int((hard.atoms != 0).sum())
    if len(mapping) != active or molecule.GetNumAtoms() != active or len(set(mapping)) != active:
        raise ValueError("Native parser lost/added ligand atoms or has incomplete canonical correspondence")
    slot_map = slot_to_canonical(hard, spec, identity)
    name_to_slot = {}
    for slot, canonical_index in enumerate(slot_map):
        if canonical_index < 0:
            continue
        atom = molecule.GetAtomWithIdx(mapping[canonical_index])
        if not atom.HasProp("name"):
            raise ValueError("Native parser molecule lacks ordinary atom name metadata")
        name = atom.GetProp("name")
        if name in name_to_slot:
            raise ValueError("Native ligand names are not unique")
        name_to_slot[name] = slot
    return molecule, name_to_slot


def _bind_structure(target, receptor, name_to_slot, slot_coordinates, protein_chain):
    structure = target.structure
    protein = [row for row in structure.chains if str(row["name"]) == protein_chain]
    ligand = [row for row in structure.chains if str(row["name"]) == "L"]
    if len(protein) != 1 or len(ligand) != 1 or len(structure.chains) != 2:
        raise ValueError("Require one exact receptor chain and one candidate ligand chain")
    chain = protein[0]
    author_rows = list(receptor["residues"].items())
    if int(chain["res_num"]) != len(author_rows):
        raise ValueError("Native protein parser changed the observed receptor sequence length")
    author_to_native, atom_receipts = {}, []
    for offset, (author, row) in enumerate(author_rows):
        native_residue = structure.residues[int(chain["res_idx"]) + offset]
        if str(native_residue["name"]) != row["name"]:
            raise ValueError("Native residue sequence/name differs from authenticated crystal chain")
        key = (int(chain["asym_id"]), int(native_residue["res_idx"]))
        author_to_native[author] = key
        native_residue["is_present"] = "CA" in row["atoms"]
        start, stop = int(native_residue["atom_idx"]), int(native_residue["atom_idx"] + native_residue["atom_num"])
        for index in range(start, stop):
            atom = structure.atoms[index]
            name = str(atom["name"])
            source = row["atoms"].get(name)
            coordinate = source["coordinates"] if source else np.zeros(3, dtype=np.float32)
            atom["coords"], atom["is_present"] = coordinate, source is not None
            structure.coords["coords"][index] = coordinate
            atom_receipts.append(dict(author=list(author), atom_name=name, resolved=source is not None,
                altloc=None if source is None else source["altloc"]))
    ligand_chain = ligand[0]
    start, stop = int(ligand_chain["atom_idx"]), int(ligand_chain["atom_idx"] + ligand_chain["atom_num"])
    seen = set()
    for index in range(start, stop):
        atom = structure.atoms[index]
        name = str(atom["name"])
        if name not in name_to_slot or name in seen:
            raise ValueError("Native ligand structure atom/name coverage differs from the new chemical graph")
        seen.add(name)
        coordinate = slot_coordinates[name_to_slot[name]]
        atom["coords"], atom["is_present"] = coordinate, True
        structure.coords["coords"][index] = coordinate
    if seen != set(name_to_slot):
        raise ValueError("Native structure does not preserve every supplied ligand atom")
    return author_to_native, atom_receipts


def prepare_supplied_pose(hard, spec, slot_coordinates, *, receptor_pdb, msa_path, molecule_dir,
                          expected_pdb_sha256, expected_msa_sha256, pocket_author_residues,
                          output, protein_chain="A", seed=17, required_atom_count=25,
                          persist_context=True):
    """Build and save a NEW official context; fail rather than redock/substitute.

    ``slot_coordinates`` has [GraphSpec.max_nodes,3] rows in the receptor's
    original PDB frame. Pocket dictionaries explicitly name chain, residue_number
    and insertion_code. The expected hashes bind externally prepared assets.
    A3M query must match the observed crystal polymer sequence exactly.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(version=CONTEXT_VERSION, status="preparing_not_passed", passed=False,
        reference_conformer_policy="fresh official parser conformer; native featurizer ref_pos, never supplied bound pose",
        bound_pose_policy="supplied receptor-frame ligand coordinates and authenticated crystal receptor atoms",
        context_reused_from_another_graph=False, seed=seed)
    try:
        receptor_pdb, msa_path, molecule_dir = Path(receptor_pdb), Path(msa_path), Path(molecule_dir)
        if (sha256_file(receptor_pdb) != expected_pdb_sha256 or sha256_file(msa_path) != expected_msa_sha256):
            raise ValueError("Authenticated receptor/MSA asset identity mismatch")
        report.update(installed_boltz_provenance())
        author_pocket = normalize_author_residues(pocket_author_residues, protein_chain)
        receptor = read_receptor_pdb(receptor_pdb, protein_chain=protein_chain)
        if msa_query(msa_path) != receptor["sequence"]:
            raise ValueError("MSA query differs from observed crystal chain; no silent truncation or sequence alignment")
        assessment = assess_graph(hard, spec, heavy_atoms=required_atom_count)
        if not assessment["eligible"]:
            raise ValueError("Supplied graph outside the pilot cohort: " + str(assessment["reason"]))
        hard = hard.to("cpu")
        if any(value.dtype != torch.long for value in (hard.atoms, hard.charges, hard.bonds)):
            raise ValueError("Supplied hard graph labels must be categorical long tensors")
        slot_coordinates = torch.as_tensor(slot_coordinates).detach().cpu().float()
        active = hard.atoms[0] != 0
        if slot_coordinates.shape != (spec.max_nodes, 3) or not bool(slot_coordinates[active].isfinite().all()):
            raise ValueError("Supplied pose needs one finite receptor-frame coordinate per occupied VAE slot")
        slot_array = slot_coordinates.numpy()
        torch.save(dict(atoms=hard.atoms.to(torch.uint8), charges=hard.charges.to(torch.uint8),
            bonds=hard.bonds.to(torch.uint8), slot_coordinates=slot_coordinates,
            graph_spec=spec.to_dict()), output / "supplied-input.pt")
        report["compact_supplied_input_sha256"] = sha256_file(output / "supplied-input.pt")
        identity = decode_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], spec)
        native = _native_backend()
        molecules = native["load_canonicals"](molecule_dir)
        schema = {"version": 1, "sequences": [
            {"protein": {"id": protein_chain, "sequence": receptor["sequence"], "msa": str(msa_path.resolve())}},
            {"ligand": {"id": "L", "smiles": identity}}], "properties": [{"affinity": {"binder": "L"}}]}
        # Native reference-conformer generation and featurizer augmentation use
        # an experiment-owned stream. They are outside sampler/hallucination RNG.
        rdBase.SeedRandomNumberGenerator(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            target = native["parse_boltz_schema"]("diffsbdd_supplied_pose", schema, molecules,
                mol_dir=molecule_dir, boltz_2=True)
            ligand_molecules = list((target.extra_mols or {}).items())
            if len(ligand_molecules) != 1:
                raise ValueError("Official parser must emit exactly one new candidate ligand molecule")
            residue_name, ligand_molecule = ligand_molecules[0]
            _, name_to_slot = _verify_native_molecule(ligand_molecule, identity, hard, spec)
            author_to_native, atom_receipts = _bind_structure(target, receptor, name_to_slot,
                                                            slot_array, protein_chain)
            if any(key not in author_to_native for key in author_pocket):
                raise ValueError("Frozen author pocket residue missing from the authenticated receptor")
            pocket = tuple(dict(asym_id=asym, residue_index=residue) for asym, residue in
                           sorted(author_to_native[key] for key in author_pocket))
            chain_ids = [int(row["asym_id"]) for row in target.structure.chains if str(row["name"]) == protein_chain]
            msa = native["parse_a3m"](msa_path, taxonomy=None, max_seqs=native["const"].max_msa_seqs)
            inp = native["Input"](structure=target.structure, msa={chain_ids[0]: msa}, record=target.record,
                residue_constraints=target.residue_constraints, templates=target.templates, extra_mols=target.extra_mols)
            tokenized = native["Boltz2Tokenizer"]().tokenize(inp)
            tokenized = native["AffinityCropper"]().crop(tokenized, max_tokens=256, max_atoms=2048)
            for residue in pocket:
                selected = ((tokenized.tokens["asym_id"] == residue["asym_id"])
                            & (tokenized.tokens["res_idx"] == residue["residue_index"]))
                if int(selected.sum()) != 1 or not bool(tokenized.tokens["resolved_mask"][selected].all()):
                    raise ValueError("Native affinity crop removed or failed to resolve a frozen biological pocket residue")
            molecules.update(target.extra_mols or {})
            molecules.update(native["load_molecules"](molecule_dir,
                set(tokenized.tokens["res_name"]) - set(molecules)))
            features = native["Boltz2Featurizer"]().process(tokenized, molecules=molecules,
                random=np.random.default_rng(seed), training=False, max_atoms=None, max_tokens=None,
                max_seqs=native["const"].max_msa_seqs, pad_to_max_seqs=False, single_sequence_prop=0.0,
                compute_frames=True, compute_constraint_features=True, override_method="other", compute_affinity=True)
        batch = {key: value.detach().clone() for key, value in native["collate"]([features]).items()
                 if isinstance(value, torch.Tensor)}
        original_msa_rows = batch["msa"].shape[1]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        rows = torch.cat((torch.zeros(1, dtype=torch.long),
                          torch.randperm(original_msa_rows - 1, generator=generator)[:127] + 1))
        for key in ("msa", "has_deletion", "deletion_value", "msa_paired", "msa_mask"):
            batch[key] = batch[key].index_select(1, rows)
        slot_atom = torch.full((spec.max_nodes,), -1, dtype=torch.long)
        slot_token = torch.full_like(slot_atom, -1)
        raw_order, mapping_receipts, atom_row = [], [], 0
        for token_row, token in enumerate(tokenized.tokens):
            start, stop = int(token["atom_idx"]), int(token["atom_idx"] + token["atom_num"])
            raw_order.extend(tokenized.structure.coords["coords"][start:stop])
            if bool(token["affinity_mask"]):
                if int(token["atom_num"]) != 1 or str(token["res_name"]) != residue_name:
                    raise ValueError("Native ligand must retain exactly one ordinary atom per affinity token")
                name = str(tokenized.structure.atoms[start]["name"])
                slot = name_to_slot[name]
                if slot_atom[slot] != -1:
                    raise ValueError("Duplicate native ligand slot after cropping")
                slot_atom[slot], slot_token[slot] = atom_row, token_row
                mapping_receipts.append(dict(slot=slot, atom_name=name, native_atom=atom_row, native_token=token_row))
            atom_row += int(token["atom_num"])
        raw_order = torch.tensor(np.asarray(raw_order), dtype=batch["coords"].dtype)
        centered = batch["coords"].reshape(-1, 3)
        resolved = batch["atom_resolved_mask"].reshape(-1).bool()
        if raw_order.shape != centered.shape or not bool(resolved.any()):
            raise ValueError("Native featurizer atom coordinate order/shape cannot be authenticated")
        translation = (raw_order[resolved] - centered[resolved]).mean(0)
        torch.testing.assert_close(centered[resolved] + translation, raw_order[resolved], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(centered[slot_atom[active]] + translation, slot_coordinates[active], atol=2e-5, rtol=2e-5)
        metadata = {**report, "seed_smiles": identity, "slot_map": sorted(mapping_receipts, key=lambda row: row["slot"]),
            "receptor_pdb_sha256": expected_pdb_sha256, "msa_sha256": expected_msa_sha256,
            "target_sequence_sha256": hashlib.sha256(receptor["sequence"].encode()).hexdigest(),
            "author_pocket": [list(key) for key in author_pocket], "native_pocket": list(pocket),
            "receptor_frame_to_centered_translation": translation.tolist(),
            "original_msa_rows": original_msa_rows, "fixed_msa_row_indices": rows.tolist(),
            "alternate_atom_policy": receptor["alternate_policy"], "receptor_atom_map": atom_receipts,
            "pose_policy": "supplied ligand and crystal receptor, jointly centered by unchanged native featurizer",
            "occupancy_policy": "fixed complete ligand map; no atom birth/death",
            "fixed_features": "all native chemistry auxiliaries, ref_pos, names, frames, MSA, pose and crop fixed inside a block"}
        prepared = PreparedComplex(batch, batch["coords"].detach().clone(), slot_atom, slot_token, hard, spec, metadata)
        prepared.validate()
        validate_frame_context(prepared.features, prepared.coordinates)
        # Raw chemistry and the supplied pose are compact and always retained.
        # Repeated loop contexts can omit the large native feature tensor while
        # keeping exact maps, featurizer seeds and the complete context digest.
        torch.save(dict(slot_to_atom=slot_atom.to(torch.int16), slot_to_token=slot_token.to(torch.int16)),
                   output / "native-slot-map.pt")
        if persist_context:
            prepared.save(output / "context.pt")
        write_json(output / "context-metadata.json", metadata)
        report.update(status="prepared_features_not_scorer_qualified", passed=True, context_sha256=context_digest(prepared),
            prepared_sha256=sha256_file(output / "context.pt") if persist_context else None,
            full_native_feature_snapshot_retained=bool(persist_context),
            compact_supplied_input_sha256=sha256_file(output / "supplied-input.pt"),
            native_slot_map_sha256=sha256_file(output / "native-slot-map.pt"),
            metadata_sha256=sha256_file(output / "context-metadata.json"), native_pocket=list(pocket),
            original_msa_rows=original_msa_rows, ligand_identity=identity,
            receptor_frame_to_centered_translation=translation.tolist())
        write_json(output / "native-schema.json", schema)
        return SuppliedPoseContext(prepared, pocket, report)
    except Exception as error:
        report.update(status="failed_not_qualified", passed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(output / "context-receipt.json", report)
