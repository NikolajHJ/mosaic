"""Audit the actual pinned Boltz affinity input outside the gradient path.

Native ``StructureV2`` stores atom names/coordinates, not chemical labels. The
trusted, locally generated extra-molecule RDKit object supplies those labels;
the audit also checks their correspondence to the recreated native features.
Identity is stereo-free heavy-atom chemistry, including per-atom formal charge
and typed/aromatic bonds. This does not validate stereochemistry, isotopes,
explicit-hydrogen geometry, protonation ensembles or physical pose quality.

Only the CLI/native audit requires Boltz and an HPC allocation. Small helpers
are independently testable without importing the pretrained model. New workflow:
fixtures and native execution must be validated on HPC before results are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re

from latentmol.experiments import file_sha256, write_json


class ArtifactCorruption(RuntimeError):
    """A malformed or internally inconsistent native artifact is not chemistry."""


class IdentityMismatch(ValueError):
    """Readable native evidence does not reproduce the declared shared identity."""


def _require_hpc():
    if not os.environ.get('LSB_JOBID') or not os.environ.get('LATENTMOL_SOURCE_SHA256'):
        raise ValueError('Processed native audit requires HPC and immutable source identity')


def _heavy_molecule(molecule):
    from rdkit import Chem
    if molecule is None:
        raise IdentityMismatch('Unparsable molecular identity')
    value = Chem.RemoveHs(Chem.Mol(molecule))
    if any(a.GetAtomicNum() <= 1 or a.GetIsotope() or a.GetNumRadicalElectrons()
           for a in value.GetAtoms()):
        raise IdentityMismatch('Unsupported retained hydrogen, isotope or radical identity')
    Chem.RemoveStereochemistry(value)
    return value


def compare_molecules(expected_smiles, processed_molecule):
    """Atom reordering is harmless; redistribution of charge/bond types is not."""
    from rdkit import Chem
    expected = _heavy_molecule(Chem.MolFromSmiles(expected_smiles))
    actual = _heavy_molecule(processed_molecule)
    a = Chem.MolToSmiles(expected, isomericSmiles=False, canonical=True)
    b = Chem.MolToSmiles(actual, isomericSmiles=False, canonical=True)
    result = {'expected_canonical_smiles': a, 'processed_canonical_smiles': b,
              'heavy_atoms': actual.GetNumAtoms(),
              'processed_atomic_numbers': [x.GetAtomicNum() for x in actual.GetAtoms()],
              'processed_formal_charges': [x.GetFormalCharge() for x in actual.GetAtoms()],
              'processed_net_formal_charge': sum(x.GetFormalCharge() for x in actual.GetAtoms()),
              'same_molecular_identity': a == b}
    return result


def _protein_sequence(structure, chain_id, residue_to_letter):
    chains = [(i, c) for i, c in enumerate(structure.chains) if int(c['asym_id']) == chain_id]
    if len(chains) != 1:
        raise IdentityMismatch('Processed protein chain is missing or duplicated')
    index, chain = chains[0]
    if hasattr(structure, 'mask') and not bool(structure.mask[index]):
        raise IdentityMismatch('Processed protein chain is masked out')
    start, end = int(chain['res_idx']), int(chain['res_idx'] + chain['res_num'])
    residues = structure.residues[start:end]
    if len(residues) != int(chain['res_num']) or not all(bool(r['is_standard']) for r in residues):
        raise IdentityMismatch('Processed protein contains missing/nonstandard residues')
    try:
        return ''.join(residue_to_letter[str(r['name'])] for r in residues)
    except KeyError as exc:
        raise IdentityMismatch('Unsupported processed protein residue') from exc


def _msa_query(msa, token_names, residue_to_letter):
    if len(msa.sequences) == 0:
        raise IdentityMismatch('Processed MSA has no query')
    first = msa.sequences[0]
    query = msa.residues[int(first['res_start']):int(first['res_end'])]
    try:
        return ''.join(residue_to_letter[token_names[int(r['res_type'])]] for r in query)
    except (KeyError, IndexError) as exc:
        raise IdentityMismatch('Processed MSA query uses unsupported residue tokens') from exc


def check_target(expected_sequence, structure_sequence, pose_sequence, msa_query):
    """Verify full pre-crop sequences, rather than the smaller affinity crop."""
    observed = {'processed_protein_sequence_sha256': hashlib.sha256(structure_sequence.encode()).hexdigest(),
                'pose_protein_sequence_sha256': hashlib.sha256(pose_sequence.encode()).hexdigest(),
                'processed_msa_query_sha256': hashlib.sha256(msa_query.encode()).hexdigest()}
    if any(s != expected_sequence for s in (structure_sequence, pose_sequence, msa_query)):
        raise IdentityMismatch('Processed protein/pose/MSA query differs from the declared full target')
    return observed


def check_crop(expected_names, uncropped_names, cropped_names):
    """Require an injective name map covering every declared ligand heavy atom."""
    for names in (expected_names, uncropped_names, cropped_names):
        if len(set(names)) != len(names):
            raise ArtifactCorruption('Duplicate native ligand atom names')
    if set(expected_names) != set(uncropped_names) or set(expected_names) != set(cropped_names):
        raise IdentityMismatch('Affinity crop or native structure loses/replaces ligand atoms')
    return {'complete_ligand_coverage': True, 'cropped_ligand_heavy_atoms': len(cropped_names)}


def check_features(molecule, names, atom_rows, token_rows, features):
    """Compare native elements, local charges and the entire typed pair matrix.

    ``names`` is in native token order; RDKit atom order can differ. This checks
    charges on each mapped atom, not merely net charge. Native NONE is zero.
    """
    import numpy as np
    from latentmol.boltz.contracts import NATIVE_BOND_IDS
    heavy = _heavy_molecule(molecule)
    atoms = list(heavy.GetAtoms())
    if any(not a.HasProp('name') for a in atoms):
        raise ArtifactCorruption('Native RDKit molecule lacks atom-name correspondence')
    named = {a.GetProp('name'): a for a in atoms}
    check_crop(list(named), names, names)
    if len(named) != len(atoms):
        raise ArtifactCorruption('Native molecule has duplicate atom names')
    def array(key):
        value = features[key]
        return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
    element = array('ref_element')[0, atom_rows]
    charge = array('ref_charge')[0, atom_rows]
    expected_element = np.zeros_like(element)
    for row, name in enumerate(names):
        expected_element[row, named[name].GetAtomicNum()] = 1
    expected_charge = np.asarray([named[name].GetFormalCharge() for name in names])
    if not np.array_equal(element, expected_element) or not np.array_equal(charge, expected_charge):
        raise ArtifactCorruption('Native element/local formal-charge features differ from its RDKit molecule')
    native_pairs = array('type_bonds')[0][np.ix_(token_rows, token_rows)]
    expected_pairs = np.zeros((len(names), len(names)), dtype=np.int64)
    positions = {named[name].GetIdx(): row for row, name in enumerate(names)}
    for bond in heavy.GetBonds():
        kind = str(bond.GetBondType()).upper()
        if kind not in NATIVE_BOND_IDS:
            raise IdentityMismatch('Unsupported native molecular bond type: ' + kind)
        a, b = positions[bond.GetBeginAtomIdx()], positions[bond.GetEndAtomIdx()]
        expected_pairs[a, b] = expected_pairs[b, a] = NATIVE_BOND_IDS[kind]
    connectivity = array('token_bonds')[0][np.ix_(token_rows, token_rows)][..., 0]
    if (not np.array_equal(native_pairs, expected_pairs)
            or not np.array_equal(connectivity, expected_pairs != 0)):
        raise ArtifactCorruption('Native typed-bond/connectivity features differ from its RDKit molecule')
    return {'native_chemical_feature_parity': True,
            'mapped_atom_names': names, 'mapped_native_atom_rows': atom_rows,
            'mapped_native_token_rows': token_rows}


def _ligand_rows(tokenized):
    selected = [(i, t) for i, t in enumerate(tokenized.tokens) if bool(t['affinity_mask'])]
    if not selected or any(int(t['atom_num']) != 1 for _, t in selected):
        raise IdentityMismatch('Affinity ligand must have exactly one heavy atom per token')
    identities = {(int(t['asym_id']), int(t['res_idx']), str(t['res_name'])) for _, t in selected}
    if len(identities) != 1:
        raise IdentityMismatch('Native affinity tokens select more than one ligand residue')
    names = [str(tokenized.structure.atoms[int(t['atom_idx'])]['name']) for _, t in selected]
    return selected, names, next(iter(identities))


def audit_processed(prediction_dir: Path, cache: Path, expected: dict) -> dict:
    """Audit one completed fresh native prediction, never the differentiable path.

    ``prediction_dir`` is ``boltz_results_<record_id>``. Expected fields are
    smiles, target_sequence_sha256, msa_sha256, protein_sequence, record_id,
    input_yaml and msa_path. Mismatches return an ineligible observation;
    missing/corrupt infrastructure and internal feature inconsistencies raise.
    Generated native pickle/NPZ artifacts must belong to this trusted request.
    """
    _require_hpc()
    required = {'smiles', 'target_sequence_sha256', 'msa_sha256', 'protein_sequence',
                'record_id', 'input_yaml', 'msa_path'}
    if not isinstance(expected, dict) or not required <= set(expected):
        raise ValueError('Processed audit lacks its declared expected identity')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', expected['record_id']):
        raise ValueError('Unsafe native record identifier')
    if hashlib.sha256(expected['protein_sequence'].encode()).hexdigest() != expected['target_sequence_sha256']:
        raise ValueError('Declared target sequence/hash disagree')
    from latentmol.boltz.prepare import installed_boltz_provenance
    provenance = installed_boltz_provenance()
    processed = prediction_dir / 'processed'
    rid = expected['record_id']
    paths = {'input_yaml': Path(expected['input_yaml']), 'raw_msa': Path(expected['msa_path']),
             'manifest': processed / 'manifest.json', 'record': processed / 'records' / f'{rid}.json',
             'structure': processed / 'structures' / f'{rid}.npz',
             'molecules': processed / 'mols' / f'{rid}.pkl',
             'pose': prediction_dir / 'predictions' / rid / f'pre_affinity_{rid}.npz',
             'affinity': prediction_dir / 'predictions' / rid / f'affinity_{rid}.json'}
    hashes = {key: file_sha256(p) for key, p in paths.items()}
    if hashes['raw_msa'] != expected['msa_sha256']:
        raise ArtifactCorruption('Declared MSA asset hash changed')
    result = {'retention_eligible': False, 'failure': None, 'record_id': rid,
              'artifact_sha256': hashes, 'artifact_paths': {k: str(p.resolve()) for k, p in paths.items()},
              'native_provenance': provenance, 'processed_canonical_smiles': None,
              'heavy_atoms': None,
              'identity_scope': 'stereo-free heavy atoms, per-atom formal charge and typed/aromatic bonds',
              'limitations': ['Stereochemistry/isotopes/explicit-hydrogen geometry are not validated.',
                             'Features are reconstructed with pinned native code; no tensor hook into the earlier forward.',
                             'Affinity association is record/path/manifest/pose linkage, not an embedded score-input hash.']}
    try:
        _audit(paths, processed, prediction_dir, cache, expected, result)
    except IdentityMismatch as exc:
        result['failure'] = str(exc)
    for key, digest in result['artifact_sha256'].items():
        if file_sha256(Path(result['artifact_paths'][key])) != digest:
            raise ArtifactCorruption('Native artifact changed while its identity was audited')
    return result


def _audit(paths, processed, prediction_dir, cache, expected, result):
    """Native imports are delayed so pure identity fixtures remain portable."""
    import numpy as np
    import torch
    import yaml
    from rdkit import Chem
    from boltz.data import const
    from boltz.data.crop.affinity import AffinityCropper
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.module.inferencev2 import collate, load_input
    from boltz.data.mol import load_canonicals, load_molecules
    from boltz.data.parse.a3m import parse_a3m
    from boltz.data.parse.csv import parse_csv
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    from boltz.data.types import Manifest, Record, StructureV2
    template = yaml.safe_load(paths['input_yaml'].read_text())
    sequences = template.get('sequences', [])
    proteins = [x['protein'] for x in sequences if 'protein' in x]
    ligands = [x['ligand'] for x in sequences if 'ligand' in x]
    if (set(template) != {'version', 'sequences', 'properties'} or template['version'] != 1
            or len(sequences) != 2 or len(proteins) != 1 or len(ligands) != 1
            or set(proteins[0]) != {'id', 'sequence', 'msa'}
            or set(ligands[0]) != {'id', 'smiles'}
            or template.get('properties') != [{'affinity': {'binder': ligands[0]['id']}}]):
        raise IdentityMismatch('Unsupported modified/multiple-target or ambiguous native affinity input')
    if (paths['input_yaml'].stem != expected['record_id']
            or proteins[0]['sequence'] != expected['protein_sequence']):
        raise IdentityMismatch('Native input YAML target/record differs from its request')
    declared_msa = Path(proteins[0]['msa'])
    if not declared_msa.is_absolute():
        declared_msa = paths['input_yaml'].resolve().parent / declared_msa
    if file_sha256(declared_msa) != expected['msa_sha256']:
        raise ArtifactCorruption('Input YAML selects a different MSA asset')
    if not compare_molecules(expected['smiles'], Chem.MolFromSmiles(ligands[0]['smiles']))['same_molecular_identity']:
        raise IdentityMismatch('Native input YAML changes the requested molecule')
    manifest = Manifest.load(paths['manifest'])
    if len(manifest.records) != 1 or manifest.records[0].id != expected['record_id']:
        raise IdentityMismatch('Native manifest does not identify exactly the requested record')
    record = manifest.records[0]
    separate = Record.load(paths['record'])
    if record != separate:
        raise ArtifactCorruption('Native record file and manifest disagree')
    protein_chains = [c for c in record.chains if c.mol_type == const.chain_type_ids['PROTEIN']]
    ligand_chains = [c for c in record.chains if c.mol_type == const.chain_type_ids['NONPOLYMER']]
    if (len(record.chains) != 2 or len(protein_chains) != 1 or len(ligand_chains) != 1
            or record.affinity is None or record.affinity.chain_id != ligand_chains[0].chain_id
            or protein_chains[0].chain_name != proteins[0]['id']
            or ligand_chains[0].chain_name != ligands[0]['id']
            or any(not c.valid for c in record.chains) or record.templates):
        raise IdentityMismatch('Processed chain/affinity identities differ from the sole declared protein/ligand')
    if record.inference_options and (record.inference_options.pocket_constraints
                                     or record.inference_options.contact_constraints):
        raise IdentityMismatch('Unsupported extra native target constraints')
    optional = lambda name: processed / name if (processed / name).exists() else None
    data = load_input(record, prediction_dir / 'predictions', processed / 'msa',
                      constraints_dir=optional('constraints'), template_dir=optional('templates'),
                      extra_mols_dir=optional('mols'), affinity=True)
    original = StructureV2.load(paths['structure'])
    protein = protein_chains[0]
    msa = data.msa.get(protein.chain_id)
    if msa is None:
        raise IdentityMismatch('Native affinity input lacks the declared protein MSA')
    msa_file = processed / 'msa' / f'{protein.msa_id}.npz'
    result['artifact_sha256']['processed_msa'] = file_sha256(msa_file)
    result['artifact_paths']['processed_msa'] = str(msa_file.resolve())
    # The pinned CLI preprocesses at 8192 by default; the feature constant is
    # 16384 and must not be substituted for this distinct parsing limit.
    parse_limit = expected.get('max_msa_seqs', 8192)
    if type(parse_limit) is not int or parse_limit < 1:
        raise ValueError('Invalid declared native MSA parsing limit')
    parsed = (parse_a3m(paths['raw_msa'], taxonomy=None, max_seqs=parse_limit)
              if paths['raw_msa'].suffix == '.a3m'
              else parse_csv(paths['raw_msa'], max_seqs=parse_limit))
    if any(not np.array_equal(getattr(msa, k), getattr(parsed, k))
           for k in ('residues', 'deletions', 'sequences')):
        raise IdentityMismatch('Processed MSA differs from the pinned native parse of its declared file')
    result.update(check_target(expected['protein_sequence'],
                               _protein_sequence(original, protein.chain_id, const.prot_token_to_letter),
                               _protein_sequence(data.structure, protein.chain_id, const.prot_token_to_letter),
                               _msa_query(msa, const.tokens, const.prot_token_to_letter)))
    tokenizer = Boltz2Tokenizer()
    full = tokenizer.tokenize(data)
    before, full_names, full_identity = _ligand_rows(full)
    cropped = AffinityCropper().crop(full, max_tokens=256, max_atoms=2048)
    after, crop_names, crop_identity = _ligand_rows(cropped)
    if full_identity != crop_identity or full_identity[0] != ligand_chains[0].chain_id:
        raise IdentityMismatch('Affinity crop changes the selected ligand identity')
    receptor = [t for t in cropped.tokens if int(t['mol_type']) == const.chain_type_ids['PROTEIN']]
    if not receptor or any(int(t['asym_id']) != protein.chain_id for t in receptor):
        raise IdentityMismatch('Affinity crop loses/replaces the declared receptor')
    molecules = load_canonicals(cache / 'mols')
    molecules.update(data.extra_mols or {})
    molecules.update(load_molecules(cache / 'mols', set(cropped.tokens['res_name']) - set(molecules)))
    if full_identity[2] not in (data.extra_mols or {}):
        raise IdentityMismatch('SMILES affinity ligand is not the request-generated extra molecule')
    molecule = molecules[full_identity[2]]
    chemical = compare_molecules(expected['smiles'], molecule)
    result.update(chemical)
    if not chemical['same_molecular_identity']:
        raise IdentityMismatch('Actual processed ligand changes heavy-atom/charge/typed-bond identity')
    heavy = _heavy_molecule(molecule)
    if any(not a.HasProp('name') for a in heavy.GetAtoms()):
        raise ArtifactCorruption('Processed molecule lacks native heavy-atom names')
    expected_names = [a.GetProp('name') for a in heavy.GetAtoms()]
    result.update(check_crop(expected_names, full_names, crop_names))
    # Include the input and full predicted structure, not only the crop, in the
    # coverage assertion. Coordinates carry no independent element/charge labels.
    for structure in (original, data.structure):
        chain = [c for c in structure.chains if int(c['asym_id']) == ligand_chains[0].chain_id]
        if len(chain) != 1:
            raise IdentityMismatch('Full structure loses the affinity ligand chain')
        c = chain[0]
        structure_names = list(map(str, structure.atoms[int(c['atom_idx']):int(c['atom_idx'] + c['atom_num'])]['name']))
        check_crop(expected_names, structure_names, structure_names)
    # Match pinned PredictionDataset's featurizer, without its silent retry of
    # record zero. No scorer/model load or gradient is required for this audit.
    features = Boltz2Featurizer().process(
        cropped, molecules=molecules, random=np.random.default_rng(42), training=False,
        max_atoms=None, max_tokens=None, max_seqs=const.max_msa_seqs,
        pad_to_max_seqs=False, single_sequence_prop=0.0, compute_frames=True,
        compute_constraint_features=True, inference_pocket_constraints=None,
        inference_contact_constraints=None, override_method='other', compute_affinity=True)
    batch = collate([features])
    atom_rows, token_rows, row = [], [], 0
    for index, token in enumerate(cropped.tokens):
        if bool(token['affinity_mask']):
            atom_rows.append(row)
            token_rows.append(index)
        row += int(token['atom_num'])
    result.update(check_features(molecule, crop_names, atom_rows, token_rows, batch))
    if not torch.isfinite(batch['coords']).all():
        raise ArtifactCorruption('Native affinity pose has nonfinite coordinates')
    if batch['affinity_token_mask'][0].bool().nonzero().flatten().tolist() != token_rows:
        raise ArtifactCorruption('Featurizer affinity mask selects a different ligand')
    score = json.loads(paths['affinity'].read_text())
    if any(not math.isfinite(float(score[k])) for k in ('affinity_probability_binary', 'affinity_pred_value')):
        raise ArtifactCorruption('Native affinity output is missing/nonfinite')
    result.update(retention_eligible=True, native_affinity=score,
                  output_record_association=True, processed_msa_full_parse_equal=True,
                  native_msa_parse_limit=parse_limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-dir', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--expected', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_processed(args.prediction_dir, args.cache,
                             json.loads(args.expected.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({'retention_eligible': result['retention_eligible'],
                      'failure': result['failure'], 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
