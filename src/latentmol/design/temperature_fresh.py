"""Re-featurize selected discrete temperature-study molecules with Protenix."""

import argparse
import json
import os
from pathlib import Path

from rdkit import Chem

from latentmol.multimodel.external import file_sha256


def group_candidates(rows):
    by_arm = {}
    for row in rows:
        by_arm.setdefault((row["start"], row["arm"]), []).append(row)
    return by_arm


def check_edit_limits(candidates, edit_limit):
    if type(edit_limit) is not int or edit_limit < 1:
        raise ValueError("Require a positive integer fresh edit budget")
    for rows in group_candidates(candidates).values():
        initials = sum(row["role"] == "initial" for row in rows)
        if (initials > 1 or len(rows) - initials > edit_limit
                or len({row["smiles"] for row in rows}) != len(rows)):
            raise ValueError("Fresh candidate arm exceeded its distinct edit budget")


def summarize_arms(rows):
    results = []
    for (start, arm), panel in group_candidates(rows).items():
        baseline = next((row for row in panel if row["role"] == "initial"), None)
        edits = [row for row in panel if row["role"] != "initial"]
        deltas = [row["loss_delta_from_valid_start"] for row in edits
                  if row["loss_delta_from_valid_start"] is not None]
        results.append(dict(
            start=start, arm=arm, requested_edits=len(edits),
            scored_edits=sum(row["loss"] is not None for row in edits),
            baseline_status="no_valid_initial" if baseline is None else baseline["status"],
            improving_edits=sum(delta < 0 for delta in deltas) if deltas else None,
            best_delta_from_valid_start=min(deltas, default=None),
        ))
    return results


def score_candidates(candidates, score):
    """Score each unique connected molecule once and retain its matched roles."""
    scored = {}
    rows = []
    for candidate in candidates:
        molecule = Chem.MolFromSmiles(candidate["smiles"])
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError("Fresh scoring requires selected valid connected graphs")
        key = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        if key not in scored:
            scored[key] = score(molecule, len(scored))
        rows.append(dict(candidate, **scored[key]))
    initial = {(row["start"], row["arm"]): row["loss"] for row in rows if row["role"] == "initial"}
    for row in rows:
        baseline = initial.get((row["start"], row["arm"]))
        row["loss_delta_from_valid_start"] = (row["loss"] - baseline
            if baseline is not None and row["loss"] is not None else None)
    return rows, len(scored)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plan")
    parser.add_argument("--plan-sha256")
    args = parser.parse_args(argv)
    job_id = os.environ.get("LSB_JOBID", "")
    if not job_id.isdigit():
        raise RuntimeError("Fresh Protenix scoring requires an allocated GPU job")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    candidates = json.loads((args.run / "fresh-candidates.json").read_text(encoding="utf-8"))["candidates"]
    if not isinstance(candidates, list) or len(candidates) > config["candidate_limit"]:
        raise ValueError("Unexpected fresh scoring candidate count")
    edit_limit = config.get("max_edit_candidates_per_arm")
    if edit_limit is not None:
        check_edit_limits(candidates, edit_limit)
    output = args.run / "fresh"
    output.mkdir(exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib")
    rows, unique = [], 0
    if candidates:
        from latentmol.protenix_occupancy.budget_native import score_topology
        from latentmol.protenix_occupancy.native import load_frozen_tiny
        from latentmol.protenix_occupancy.prepare import _bind_native

        assets = json.loads((args.config.parent / config["assets_config"]).read_text(encoding="utf-8"))["native"]
        pocket_path = Path(config["pocket"]["path"])
        if file_sha256(pocket_path) != config["pocket"]["sha256"]:
            raise ValueError("Fresh Protenix pocket definition changed")
        pocket = json.loads(pocket_path.read_text(encoding="utf-8"))
        residues = tuple(row["sequence_position"] for row in pocket["residues"])
        _bind_native(Path(assets["source_root"]), assets["source_sha256"], Path(config["ccd_root"]))
        model = load_frozen_tiny(**assets)
        snapshot = model.snapshot_state()

        def score(molecule, index):
            try:
                loss, details = score_topology(model, molecule, range(molecule.GetNumAtoms()),
                    pocket["protein_sequence"], residues, geometry_seed=config["geometry_seed"],
                    mol_path=output / f"candidate-{index:02d}.mol")
            except ValueError as exc:
                reason = str(exc)
                if reason.startswith("ETKDG failed for topology"):
                    return dict(status="embedding_failed", loss=None, reason=reason)
                if reason.startswith(("Ligand embedding lost", "Native input MOL did not preserve",
                                      "Native FILE import changed ligand", "Native FILE import needs one token")):
                    return dict(status="representation_failed", loss=None, reason=reason)
                raise
            return dict(status="scored", loss=loss, details=details)

        rows, unique = score_candidates(candidates, score)
        model.assert_unchanged(snapshot)
    extra = {} if edit_limit is None else dict(
        requested_candidates=len(candidates), max_edit_candidates_per_arm=edit_limit,
        arm_readouts=summarize_arms(rows),
    )
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(status="completed", job_id=job_id, unique_candidates=unique,
            geometry_seed=config["geometry_seed"], candidates=rows,
            objective="fresh native FILE negative mean log pocket-contact probability",
            interpretation="Fresh ligand conformers and native features, trunk distogram only. "
                           "Compare this separate contact readout within the matched panel.", **extra),
            stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
