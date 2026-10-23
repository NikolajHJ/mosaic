"""HPC-only preparation outside the differentiable path, using native Boltz code.

The example command derives a stereo-free tyrosine input from the pinned official
affinity example and supplies its bundled public MSA. The snapshot command reads
the official prediction output, follows its affinity crop/featurizer, and records
one exact graph-to-native-atom correspondence. No arbitrary pose is fabricated.
"""

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import urllib.request

import numpy as np
from rdkit import Chem
import torch
import yaml

from latentmol.chemistry import encode_smiles
from latentmol.experiments import file_sha256, source_manifest, write_json
from latentmol.representation import GraphSpec, HardGraphBatch
from .adapter import adapt_graph
from .contracts import BOLTZ_COMMIT, HF_REVISION, PreparedComplex


# Pinned official example bytes already used by source006/source029. Cached
# copies let compute-node preparation run without outbound internet access.
EXAMPLE_ASSET_SHA256 = {
    "official_affinity.yaml": "631bb444e0deea02db042cf2f766711d17a9372603d01694fbea804a12f7227e",
    "seq1.a3m": "d98f3b09f429e4fb087d67ca3b89f00c3165553444ef221ead6ed2cc01488bd4",
}


def installed_boltz_provenance() -> dict:
    """Require the exact Git installation, and report its package metadata."""
    dist = metadata.distribution("boltz")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != BOLTZ_COMMIT:
        raise ValueError("Install Boltz from the pinned Git commit; package version alone is insufficient")
    return {"boltz_commit": BOLTZ_COMMIT, "boltz_version": dist.version,
            "boltz_install": direct, "hf_revision": HF_REVISION}


def _make_example(output: Path, *, ligand_smiles: str | None, stem: str,
                  assets_dir: Path | None = None) -> Path:
    """Reuse the pinned protein/MSA while declaring one exact ligand input.

    The original example has a chiral tyrosine ligand; the derivative deliberately
    strips stereo to match the present topology-only codec. Replacing its ligand
    with a declared carbon chain creates a separate native prediction; the protein/MSA alone do not
    supply a ligand pose. Neither engineering smoke is a binding benchmark.
    """
    output.mkdir(parents=True, exist_ok=False)
    base = f"https://raw.githubusercontent.com/jwohlwend/boltz/{BOLTZ_COMMIT}/examples/"
    originals = {}
    for relative, local in (("affinity.yaml", "official_affinity.yaml"),
                            ("msa/seq1.a3m", "seq1.a3m")):
        path = output / local
        if assets_dir is None:
            urllib.request.urlretrieve(base + relative, path)
        else:
            cached = assets_dir / local
            if file_sha256(cached) != EXAMPLE_ASSET_SHA256[local]:
                raise ValueError(f"Cached official example asset hash mismatch: {local}")
            shutil.copyfile(cached, path)
        originals[local] = {"url": base + relative, "sha256": file_sha256(path)}
    value = yaml.safe_load((output / "official_affinity.yaml").read_text())
    sequence = value["sequences"][0]["protein"]["sequence"]
    msa_query = (output / "seq1.a3m").read_text().splitlines()[1]
    if sequence != msa_query:
        raise ValueError("Bundled MSA query differs from the official target sequence")
    value["sequences"][0]["protein"]["msa"] = str((output / "seq1.a3m").resolve())
    original_ligand = value["sequences"][1]["ligand"]["smiles"]
    seed = encode_smiles(original_ligand if ligand_smiles is None else ligand_smiles,
                         GraphSpec())
    value["sequences"][1]["ligand"]["smiles"] = seed["smiles"]
    path = output / f"{stem}.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    write_json(output / "input_manifest.json", {"assets": originals,
               "derived_yaml_sha256": file_sha256(path), "seed_smiles": seed["smiles"],
               "derived_stem": stem,
               "ligand_source": "official_example_stereo_stripped" if ligand_smiles is None
               else f"declared_linear_c{len(ligand_smiles)}_stereo_free",
               "target_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
               "msa_sha256": file_sha256(output / "seq1.a3m"),
               "original_example_ligand": original_ligand,
               "policy": "official sequence and public MSA; new native pose required; engineering smoke"})
    return path


def make_example(output: Path) -> Path:
    """Prepare the historical stereo-free tyrosine engineering smoke."""
    return _make_example(output, ligand_smiles=None, stem="tyrosine_smoke")


def make_chain_example(output: Path) -> Path:
    """Prepare a C13 ligand with the same public protein/MSA context."""
    return _make_example(output, ligand_smiles="CCCCCCCCCCCCC", stem="carbon_chain_smoke")


def make_count_example(output: Path, count: int, assets_dir: Path | None = None) -> Path:
    """Create one native carbon-chain input for a declared heavy-atom count."""
    if not 1 <= count <= GraphSpec().max_nodes:
        raise ValueError("Carbon-chain count is outside the graph capacity")
    return _make_example(output, ligand_smiles="C" * count,
                         stem=f"carbon_chain_{count:02d}", assets_dir=assets_dir)


def make_target_chain(output: Path, count: int, manifest: Path, target_id: str) -> Path:
    """Use one sealed target and its explicit MSA, without a network request.

    The panel currently uses a single query sequence as the A3M. This differs
    from the historical official-example MSA and is a declared engineering
    assay limitation. Native diffusion predicts a new context before scoring.
    """
    from latentmol.multimodel.study import verified_file
    if not 1 <= count <= GraphSpec().max_nodes:
        raise ValueError('Carbon-chain count outside graph capacity')
    panel = json.loads(manifest.read_text())
    if panel.get('version') != 1:
        raise ValueError('Unknown target manifest version')
    targets = [t for t in panel['targets'] if t['id'] == target_id]
    if len(targets) != 1:
        raise ValueError('Require exactly one declared target')
    target = targets[0]
    sequence = target['sequence']
    if hashlib.sha256(sequence.encode()).hexdigest() != target['sequence_sha256']:
        raise ValueError('Target sequence hash mismatch')
    raw = verified_file(manifest.parent / target['raw_record']['path'],
                        target['raw_record']['sha256'])
    record = json.loads(raw.read_text())
    if record['sequence']['value'] != sequence or record['primaryAccession'] != target['accession']:
        raise ValueError('Canonical sequence differs from its official record')
    msa = verified_file(manifest.parent / target['msa']['path'], target['msa']['sha256'])
    if msa.read_text().splitlines()[1] != sequence:
        raise ValueError('MSA query differs from target sequence')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(msa, output / 'seq1.a3m')
    stem = f'carbon_chain_{count:02d}'
    value = {'version': 1, 'sequences': [
        {'protein': {'id': 'A', 'sequence': sequence, 'msa': str((output / 'seq1.a3m').resolve())}},
        {'ligand': {'id': 'B', 'smiles': 'C' * count}}],
        'properties': [{'affinity': {'binder': 'B'}}]}
    path = output / f'{stem}.yaml'
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    write_json(output / 'input_manifest.json', {
        'target': target, 'target_manifest_sha256': file_sha256(manifest),
        'derived_yaml_sha256': file_sha256(path), 'seed_smiles': 'C' * count,
        'derived_stem': stem, 'target_sequence_sha256': target['sequence_sha256'],
        'msa_sha256': file_sha256(output / 'seq1.a3m'),
        'msa_policy': panel['msa_policy'], 'context_policy': panel['context_policy'],
        'policy': 'Native prediction required; sequence-only elemental-response assay'})
    return path


def snapshot(prediction_dir: Path, cache: Path, output: Path,
             ae_checkpoint: Path | None = None,
             input_manifest: Path | None = None) -> Path:
    """Snapshot a single official prediction with fail-fast native preprocessing.

    Native PredictionDataset silently retries record zero after some errors. We
    invoke the same loader/tokenizer/cropper/featurizer explicitly so every error
    fails this snapshot instead. Native loading reads a locally generated RDKit
    pickle from the trusted prediction run; that object is never saved here.
    """
    provenance = installed_boltz_provenance()
    from boltz.data import const
    from boltz.data.crop.affinity import AffinityCropper
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.module.inferencev2 import collate, load_input
    from boltz.data.mol import load_canonicals, load_molecules
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    from boltz.data.types import Manifest

    processed = prediction_dir / "processed"
    manifest_path = processed / "manifest.json"
    manifest = Manifest.load(manifest_path)
    if len(manifest.records) != 1 or manifest.records[0].affinity is None:
        raise ValueError("Require exactly one official affinity input record")
    record = manifest.records[0]
    optional_dir = lambda name: processed / name if (processed / name).exists() else None
    data = load_input(record, prediction_dir / "predictions", processed / "msa",
                      constraints_dir=optional_dir("constraints"), template_dir=optional_dir("templates"),
                      extra_mols_dir=optional_dir("mols"), affinity=True)
    tokenized = Boltz2Tokenizer().tokenize(data)
    tokenized = AffinityCropper().crop(tokenized, max_tokens=256, max_atoms=2048)
    molecules = load_canonicals(cache / "mols")
    molecules.update(data.extra_mols or {})
    molecules.update(load_molecules(cache / "mols", set(tokenized.tokens["res_name"]) - set(molecules)))
    options = record.inference_options
    torch.manual_seed(42)
    features = Boltz2Featurizer().process(
        tokenized, molecules=molecules, random=np.random.default_rng(42), training=False,
        max_atoms=None, max_tokens=None, max_seqs=const.max_msa_seqs, pad_to_max_seqs=False,
        single_sequence_prop=0.0, compute_frames=True, compute_constraint_features=True,
        inference_pocket_constraints=options.pocket_constraints if options else None,
        inference_contact_constraints=options.contact_constraints if options else None,
        override_method="other", compute_affinity=True,
    )
    batch = {key: value for key, value in collate([features]).items() if isinstance(value, torch.Tensor)}
    # Fix one small MSA subset for every reference/gradient/FD call. Native MSA
    # subsampling otherwise redraws randperm even in eval mode. Profiles remain
    # the native featurizer's profile of the complete available MSA.
    original_msa_rows = batch["msa"].shape[1]
    rng = torch.Generator().manual_seed(42)
    msa_rows = torch.cat((torch.zeros(1, dtype=torch.long),
                         torch.randperm(original_msa_rows - 1, generator=rng)[:127] + 1))
    for key in ("msa", "has_deletion", "deletion_value", "msa_paired", "msa_mask"):
        batch[key] = batch[key].index_select(1, msa_rows)
    ligand_tokens = tokenized.tokens[tokenized.tokens["affinity_mask"]]
    identities = {(int(row["asym_id"]), int(row["res_idx"]), str(row["res_name"])) for row in ligand_tokens}
    if len(identities) != 1 or not np.all(ligand_tokens["atom_num"] == 1):
        raise ValueError("Require one ligand residue with one native atom per token")
    res_name = next(iter(identities))[2]
    mol = Chem.RemoveHs(Chem.Mol(molecules[res_name]))
    Chem.RemoveStereochemistry(mol)
    spec = GraphSpec()
    if ae_checkpoint is not None:
        raw = torch.load(ae_checkpoint, map_location="cpu", weights_only=True)
        if raw.get("format_version") != 1:
            raise ValueError("Unknown AE checkpoint schema")
        spec = GraphSpec.from_dict(raw["graph_spec"])
    encoded = encode_smiles(Chem.MolToSmiles(mol, isomericSmiles=False), spec)
    input_identity = None
    if input_manifest is not None:
        input_identity = json.loads(input_manifest.read_text())
        if input_identity.get("seed_smiles") != encoded["smiles"]:
            raise ValueError("Native cropped ligand differs from the declared input ligand")
        if not all(input_identity.get(key) for key in
                   ("target_sequence_sha256", "msa_sha256", "derived_yaml_sha256", "derived_stem")):
            raise ValueError("Input manifest lacks target/MSA and native-input identity")
        declared_yaml = input_manifest.parent / f"{input_identity['derived_stem']}.yaml"
        declared_input = yaml.safe_load(declared_yaml.read_text())
        declared_proteins = [entry["protein"] for entry in declared_input["sequences"]
                             if "protein" in entry]
        if (file_sha256(declared_yaml) != input_identity["derived_yaml_sha256"]
                or record.id != input_identity["derived_stem"]
                or len(declared_proteins) != 1
                or hashlib.sha256(declared_proteins[0]["sequence"].encode()).hexdigest()
                   != input_identity["target_sequence_sha256"]
                or file_sha256(Path(declared_proteins[0]["msa"])) != input_identity["msa_sha256"]):
            raise ValueError("Declared protein/MSA/input YAML differs from the native record")
    query = Chem.MolFromSmiles(encoded["smiles"])
    match = mol.GetSubstructMatch(query, useChirality=False)
    if len(match) != mol.GetNumAtoms() or len(match) != len(ligand_tokens):
        raise ValueError("Seed molecule and cropped native ligand do not match completely")
    name_to_slot = {mol.GetAtomWithIdx(atom_index).GetProp("name"): slot for slot, atom_index in enumerate(match)}
    if len(name_to_slot) != len(match):
        raise ValueError("Native ligand atom names are not unique")
    slot_atom = torch.full((spec.max_nodes,), -1, dtype=torch.long)
    slot_token = torch.full((spec.max_nodes,), -1, dtype=torch.long)
    rows, atom_row = [], 0
    for token_row, token in enumerate(tokenized.tokens):
        if token["affinity_mask"]:
            name = str(tokenized.structure.atoms[int(token["atom_idx"])]["name"])
            slot = name_to_slot[name]
            if slot_atom[slot] != -1:
                raise ValueError("Duplicate atom name in native ligand structure")
            slot_atom[slot], slot_token[slot] = atom_row, token_row
            rows.append({"slot": slot, "atom_name": name, "native_atom": atom_row,
                         "native_token": token_row, "source_atom": int(token["atom_idx"])})
        atom_row += int(token["atom_num"])
    seed = HardGraphBatch(*(torch.from_numpy(encoded[key].astype(np.int64))[None]
                            for key in ("atom_types", "charges", "bond_types")))
    pose_path = prediction_dir / "predictions" / record.id / f"pre_affinity_{record.id}.npz"
    provenance.update(seed_smiles=encoded["smiles"], slot_map=sorted(rows, key=lambda row: row["slot"]),
                      native_manifest_sha256=file_sha256(manifest_path), reference_pose_sha256=file_sha256(pose_path),
                      prediction_dir=str(prediction_dir.resolve()), preparation_seed=42,
                      fixed_msa_row_indices=msa_rows.tolist(), original_msa_rows=original_msa_rows,
                      pose_policy="official pre_affinity predicted pose, centered by native featurizer; detached",
                      occupancy_policy="fixed complete ligand map; no atom birth/death",
                      fixed_features="pose, ref_pos, masks, atom names, token indices, crop, MSA, all other native fields")
    if input_identity is not None:
        provenance["input_manifest_sha256"] = file_sha256(input_manifest)
        provenance["target_sequence_sha256"] = input_identity["target_sequence_sha256"]
        provenance["msa_sha256"] = input_identity["msa_sha256"]
        provenance["derived_yaml_sha256"] = input_identity["derived_yaml_sha256"]
    prepared = PreparedComplex(batch, batch["coords"].detach().clone(), slot_atom, slot_token, seed, spec, provenance)
    prepared.validate()
    mapped, types = adapt_graph(seed.to_relaxed(spec), prepared)
    for name in ("ref_element", "ref_charge", "token_bonds"):
        torch.testing.assert_close(mapped[name], batch[name].to(mapped[name]), atol=0, rtol=0)
    torch.testing.assert_close(types, torch.nn.functional.one_hot(batch["type_bonds"].long(), 7).to(types), atol=0, rtol=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.save(output)
    write_json(output.with_suffix(".json"), {**provenance, "graph_spec": spec.to_dict(),
               "snapshot_sha256": file_sha256(output), "hard_feature_parity": True})
    return output


def count_catalog(root: Path, counts: list[int], output: Path) -> Path:
    """Index only finished, validated, distinct count-specific native snapshots.

    The catalog is generated after all predictions finish; it is not evidence
    that any active-set optimization has run or that poses are comparable.
    """
    if not counts or len(set(counts)) != len(counts) or output.exists():
        raise ValueError("Require unique counts and a new catalog path")
    return seal_count_catalog({count: root / f"count-{count:02d}" / "prepared.pt"
                               for count in counts}, output)


def seal_count_catalog(paths: dict[int, Path], output: Path, *, failed_counts=None) -> Path:
    """Seal reused and new immutable snapshots together, never mutate an input catalog.

    Preparation failures are explicit metadata, not entries pointing to incomplete
    tensors. A later unsupported selection must fail rather than choose a count.
    """
    if not paths or output.exists():
        raise ValueError("Require prepared contexts and a new catalog path")
    failures = failed_counts or {}
    if any(not str(key).isdigit() or not 1 <= int(key) <= GraphSpec().max_nodes
           for key in failures) or set(paths).intersection(map(int, failures)):
        raise ValueError("Prepared and failed count sets must be disjoint and within capacity")
    entries = {}
    common = None
    for count, path in sorted(paths.items()):
        if type(count) is not int or not 1 <= count <= GraphSpec().max_nodes:
            raise ValueError("Catalog count outside graph capacity")
        prepared = PreparedComplex.load(path)
        if int(prepared.active.sum()) != count:
            raise ValueError(f"Prepared count mismatch for {count}")
        identity = (prepared.metadata.get("target_sequence_sha256"),
                    prepared.metadata.get("msa_sha256"),
                    tuple(prepared.metadata.get("fixed_msa_row_indices", ())))
        if not all(identity[:2]) or not identity[2]:
            raise ValueError("Native snapshot lacks target/MSA identity")
        if common is not None and identity != common:
            raise ValueError("Count-specific contexts have different protein/MSA policies")
        common = identity
        entries[str(count)] = {"path": str(path.resolve().relative_to(output.resolve().parent))
                               if path.resolve().is_relative_to(output.resolve().parent)
                               else str(path.resolve()),
                               "sha256": file_sha256(path)}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"version": 1, "contexts": entries,
                        "failed_counts": failures,
                        "target_sequence_sha256": common[0], "msa_sha256": common[1],
                        "fixed_msa_row_indices": list(common[2]),
                        "source_manifest_sha256": source_manifest(Path.cwd())["sha256"],
                        "source_archive_sha256": os.environ.get("LATENTMOL_SOURCE_SHA256"),
                        "hpc_job_id": os.environ.get("LSB_JOBID"),
                        "status": "prepared_only_optimization_not_run"})
    return output


def main():
    """Prepare inputs or snapshot native features; run only within HPC jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    example = commands.add_parser("example")
    example.add_argument("--output", type=Path, required=True)
    chain_example = commands.add_parser("chain-example")
    chain_example.add_argument("--output", type=Path, required=True)
    count_example = commands.add_parser("chain-count-example")
    count_example.add_argument("--output", type=Path, required=True)
    count_example.add_argument("--count", type=int, required=True)
    count_example.add_argument("--assets-dir", type=Path,
                               help="Pinned official_affinity.yaml and seq1.a3m cache; no download")
    target_chain = commands.add_parser('target-chain')
    target_chain.add_argument('--output', type=Path, required=True)
    target_chain.add_argument('--count', type=int, required=True)
    target_chain.add_argument('--target-manifest', type=Path, required=True)
    target_chain.add_argument('--target', required=True)
    snap = commands.add_parser("snapshot")
    snap.add_argument("--prediction-dir", type=Path, required=True)
    snap.add_argument("--cache", type=Path, required=True)
    snap.add_argument("--output", type=Path, required=True)
    snap.add_argument("--ae-checkpoint", type=Path)
    snap.add_argument("--input-manifest", type=Path)
    catalog = commands.add_parser("count-catalog")
    catalog.add_argument("--root", type=Path, required=True)
    catalog.add_argument("--counts", type=int, nargs="+", required=True)
    catalog.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "example":
        print(make_example(args.output), flush=True)
    elif args.command == "chain-example":
        print(make_chain_example(args.output), flush=True)
    elif args.command == "chain-count-example":
        print(make_count_example(args.output, args.count, args.assets_dir), flush=True)
    elif args.command == 'target-chain':
        print(make_target_chain(args.output, args.count, args.target_manifest, args.target), flush=True)
    elif args.command == "count-catalog":
        print(count_catalog(args.root, args.counts, args.output), flush=True)
    else:
        print(snapshot(args.prediction_dir, args.cache, args.output,
                       args.ae_checkpoint, args.input_manifest), flush=True)


if __name__ == "__main__":
    main()
