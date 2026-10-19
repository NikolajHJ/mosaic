"""Prepare a fresh native six-atom BRD4/hexane context on allocated HPC.

Native prediction establishes the reference pose once. It is shared by both
fresh probability initializations; all subsequent affinity gradients hold it
fixed. No DBMol prepared contexts, decoders or latent representations are read.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def snapshot(prediction, cache, fixture, output):
    import numpy as np
    import torch
    from rdkit import Chem
    from boltz.data import const
    from boltz.data.crop.affinity import AffinityCropper
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.module.inferencev2 import collate, load_input
    from boltz.data.mol import load_canonicals, load_molecules
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    from boltz.data.types import Manifest
    from .native import BOLTZ_COMMIT, file_digest, validate_context

    processed = prediction / "processed"
    manifest_path = processed / "manifest.json"
    manifest = Manifest.load(manifest_path)
    if len(manifest.records) != 1 or manifest.records[0].affinity is None:
        raise ValueError("Require exactly one native BRD4/hexane affinity record")
    record = manifest.records[0]

    def optional(name):
        return processed / name if (processed / name).exists() else None

    data = load_input(
        record,
        prediction / "predictions",
        processed / "msa",
        constraints_dir=optional("constraints"),
        template_dir=optional("templates"),
        extra_mols_dir=optional("mols"),
        affinity=True,
    )
    tokenized = AffinityCropper().crop(
        Boltz2Tokenizer().tokenize(data), max_tokens=256, max_atoms=2048
    )
    molecules = load_canonicals(cache / "mols")
    molecules.update(data.extra_mols or {})
    molecules.update(
        load_molecules(
            cache / "mols", set(tokenized.tokens["res_name"]) - set(molecules)
        )
    )
    options = record.inference_options
    torch.manual_seed(42)
    features = Boltz2Featurizer().process(
        tokenized,
        molecules=molecules,
        random=np.random.default_rng(42),
        training=False,
        max_atoms=None,
        max_tokens=None,
        max_seqs=const.max_msa_seqs,
        pad_to_max_seqs=False,
        single_sequence_prop=0.0,
        compute_frames=True,
        compute_constraint_features=True,
        inference_pocket_constraints=options.pocket_constraints if options else None,
        inference_contact_constraints=options.contact_constraints if options else None,
        override_method="other",
        compute_affinity=True,
    )
    batch = {
        key: value
        for key, value in collate([features]).items()
        if isinstance(value, torch.Tensor)
    }
    ligand = tokenized.tokens[tokenized.tokens["affinity_mask"]]
    names = set(ligand["res_name"])
    if len(ligand) != 6 or len(names) != 1 or not np.all(ligand["atom_num"] == 1):
        raise ValueError("Native preprocessing changed six occupied ligand slots")
    mol = Chem.RemoveHs(Chem.Mol(molecules[next(iter(names))]))
    query = Chem.MolFromSmiles("CCCCCC")
    match = mol.GetSubstructMatch(query)
    if mol.GetNumAtoms() != 6 or len(match) != 6 or Chem.MolToSmiles(mol) != "CCCCCC":
        raise ValueError(
            "Native ligand identity is not the requested neutral carbon chain"
        )
    name_to_slot = {
        mol.GetAtomWithIdx(index).GetProp("name"): slot
        for slot, index in enumerate(match)
    }
    ai, ti = (
        torch.full((6,), -1, dtype=torch.long),
        torch.full((6,), -1, dtype=torch.long),
    )
    atom_row = 0
    for token_row, token in enumerate(tokenized.tokens):
        if token["affinity_mask"]:
            name = str(tokenized.structure.atoms[int(token["atom_idx"])]["name"])
            slot = name_to_slot[name]
            ai[slot], ti[slot] = atom_row, token_row
        atom_row += int(token["atom_num"])
    x = torch.nn.functional.one_hot(torch.zeros(6, dtype=torch.long), 6).float()
    i, j = torch.triu_indices(6, 6, 1)
    edges = torch.nn.functional.one_hot((j == i + 1).long(), 5).float()
    context = {
        "version": "mosaic-boltz-fixed-pose-v1",
        "features": batch,
        "coordinates": batch["coords"].detach().clone(),
        "atom_indices": ai,
        "token_indices": ti,
        "seed_X": x,
        "seed_E": edges,
    }
    validate_context(context)
    torch.save(context, output)
    return {
        "status": "prepared",
        "job_id": os.environ["LSB_JOBID"],
        "boltz_commit": BOLTZ_COMMIT,
        "context_sha256": file_digest(output),
        "fixture_sha256": file_digest(fixture),
        "native_manifest_sha256": file_digest(manifest_path),
        "input_sha256": file_digest(output.parent / "input.yaml"),
        "n_atoms": 6,
        "seed_smiles": "CCCCCC",
        "atom_indices": ai.tolist(),
        "token_indices": ti.tolist(),
        "protein_sequence": json.loads(fixture.read_text())["protein_sequence"],
        "msa_policy": "native single-sequence empty MSA, same deposited 3MXF sequence as prior Mosaic; no external server",
        "pose_policy": "new native predicted six-slot hexane pose; fixed during optimization; no crystal pose or ligand transplant",
        "frozen_fields": "coordinates, ref_pos, charge=0, atom names, crop, MSA, masks and all auxiliary fields",
        "prediction": str(prediction),
        "preparation_seed": 42,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Native preparation requires allocated DTU GPU")
    from .native import verify_installation

    verify_installation()
    fixture = json.loads(args.fixture.read_text())
    if (
        fixture["pdb_id"] != "3MXF"
        or fixture["n_heavy_atoms"] != 6
        or fixture["smiles"] != "CCCCCC"
    ):
        raise ValueError("Wrong Mosaic BRD4 six-slot fixture")
    args.output.mkdir(parents=True, exist_ok=False)
    specification = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": fixture["protein_sequence"],
                    "msa": "empty",
                }
            },
            {"ligand": {"id": "L", "smiles": "CCCCCC"}},
        ],
        "properties": [{"affinity": {"binder": "L"}}],
    }
    source = args.output / "input.yaml"
    source.write_text(json.dumps(specification, indent=2) + "\n")  # JSON is valid YAML.
    command = [
        sys.executable,
        "-m",
        "boltz.main",
        "predict",
        str(source),
        "--cache",
        str(args.cache),
        "--out_dir",
        str(args.output / "native"),
        "--model",
        "boltz2",
        "--recycling_steps",
        "0",
        "--sampling_steps",
        "25",
        "--diffusion_samples",
        "1",
        "--sampling_steps_affinity",
        "25",
        "--diffusion_samples_affinity",
        "1",
        "--seed",
        "17",
        "--no_kernels",
        "--output_format",
        "mmcif",
    ]
    (args.output / "command.json").write_text(json.dumps(command))
    with (args.output / "prediction.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    prediction = args.output / "native/boltz_results_input"
    report = snapshot(prediction, args.cache, args.fixture, args.output / "context.pt")
    (args.output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
