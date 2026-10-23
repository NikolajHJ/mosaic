"""Predeclared fresh-pose native follow-up to paired embedding optimization.

All starts, terminal states and best-own-objective states remain in selection,
including invalid molecules. Valid canonical identities share one request per
pose seed. Native parsing regenerates chemistry-dependent fields and the
processed-input audit verifies identity. These are ordinary native affinity
predictions at newly generated poses, not the fixed-pose full DBMol objective.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time


VERSION = "epoch36-embedding-comparison-native-v1"
CLI_MAIN_SHA256 = "b5ded4dac1552af2e079c335d56e64ea580b4d5ca0202a9542385ce3e318c3b3"
ARMS = ("relaxed", "learned")
NATIVE_SETTINGS = {
    "structure_steps": 100, "affinity_steps": 100, "diffusion_samples": 1,
    "structure_recycles": 0, "affinity_recycles": 5, "precision": "bf16-mixed",
    "msa_rows": 128, "max_msa_seqs": 8192, "mw_correction": False,
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _canonical_identity(smiles):
    """Reject unsupported chemistry; canonical ordering never repairs a graph."""
    from rdkit import Chem
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or not molecule.GetNumAtoms() or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("Declared strict-valid candidate is unparsable or disconnected")
    allowed = {6, 7, 8, 9, 15, 16, 17, 35, 53}
    if any(a.GetAtomicNum() not in allowed or a.GetIsotope() or a.GetNumRadicalElectrons()
           or a.GetFormalCharge() not in (-2, -1, 0, 1, 2)
           or a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
           for a in molecule.GetAtoms()):
        raise ValueError("Candidate has unsupported element, isotope, radical, charge or stereo")
    if any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in molecule.GetBonds()):
        raise ValueError("Candidate contains unsupported bond stereochemistry")
    return Chem.MolToSmiles(molecule, isomericSmiles=False, canonical=True)


def freeze_selection(members: list[dict], *, pose_seeds=(17, 23),
                     native_settings=None, expected_case_count=3) -> dict:
    """Freeze all five declared roles per case before any native prediction.

    A member contains ``case_id``, ``arm`` (start/relaxed/learned), ``state``
    (baseline/terminal/best), ``smiles`` and ``failure``. Additional graph,
    latent or state hashes are preserved as caller evidence. This layer requires
    the caller's saved strict graph inspection; parsing here is an extra audit.
    """
    if type(expected_case_count) is not int or not 1 <= expected_case_count <= 3:
        raise ValueError("The bounded pilot supports one to three declared cases")
    if (not isinstance(pose_seeds, (tuple, list)) or len(pose_seeds) != 2
            or len(set(pose_seeds)) != 2
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in pose_seeds)):
        raise ValueError("Declare exactly two distinct native pose seeds")
    settings = deepcopy(NATIVE_SETTINGS if native_settings is None else native_settings)
    if (settings != NATIVE_SETTINGS
            or any(type(settings[key]) is not type(value) for key, value in NATIVE_SETTINGS.items())):
        raise ValueError("Native follow-up settings differ from the declared bounded pilot")
    if not isinstance(members, list) or len(members) != expected_case_count * 5:
        raise ValueError("Require one baseline and terminal/best of each arm for every case")
    frozen, cases, roles = [], [], set()
    required = {("start", "baseline"), *[(arm, state) for arm in ARMS for state in ("terminal", "best")]}
    for source in members:
        if not isinstance(source, dict) or not {"case_id", "arm", "state", "smiles", "failure"} <= set(source):
            raise ValueError("Native selection member lacks its identity and declared role")
        row = deepcopy(source)
        case = row["case_id"]
        if not isinstance(case, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", case):
            raise ValueError("Unsafe comparison case identifier")
        if case not in cases:
            cases.append(case)
        role = (row["arm"], row["state"])
        key = (case, *role)
        if role not in required or key in roles:
            raise ValueError("Duplicate or unsupported native candidate role")
        roles.add(key)
        smiles, failure = row["smiles"], row["failure"]
        if smiles is None:
            if not isinstance(failure, str) or not failure:
                raise ValueError("Invalid hard molecules must retain their failure reason")
            row["canonical_smiles"] = None
        else:
            if not isinstance(smiles, str) or not smiles or failure is not None:
                raise ValueError("Valid candidate SMILES and failure are inconsistent")
            row["canonical_smiles"] = _canonical_identity(smiles)
        frozen.append(row)
    if len(cases) != expected_case_count or roles != {(case, *role) for case in cases for role in required}:
        raise ValueError("Comparison cases have incomplete candidate roles")
    # Stable lexicographic identity order gives every shared identity identical
    # settings; it does not rank by either continuous or native outcome.
    identities = sorted({row["canonical_smiles"] for row in frozen if row["canonical_smiles"]})
    requests = []
    for smiles in identities:
        for seed in pose_seeds:
            identity = {"smiles": smiles, "seed": seed, "native_settings": settings}
            requests.append({"id": _digest(identity)[:24], "target_key": "shared", **identity})
    if len(requests) > 30:
        raise ValueError("Native follow-up exceeds the predeclared 30-request bound")
    value = {"version": VERSION, "case_ids": cases, "members": frozen,
             "pose_seeds": list(pose_seeds), "native_settings": settings,
             "requests": requests, "unique_valid_identities": len(identities),
             "invalid_members": sum(row["canonical_smiles"] is None for row in frozen),
             "selection_policy": "all baseline, terminal and best-own-objective states; no native-score selection"}
    return {**value, "selection_sha256": _digest(value)}


def load_target(template_path, expected_sequence_sha, expected_msa_sha, *, template_sha256=None):
    """Authenticate the same unmodified protein/MSA as prepared optimization."""
    from latentmol.experiments import file_sha256
    from .active_set_rescore import _target_identity
    path = Path(template_path)
    if template_sha256 is not None and file_sha256(path) != template_sha256:
        raise ValueError("Native template hash differs from the sealed plan")
    target = _target_identity(path, expected_sequence_sha, expected_msa_sha)
    protein = next(x["protein"] for x in target["template"]["sequences"] if "protein" in x)
    return {**target, "protein_sequence": protein["sequence"], "msa_path": protein["msa"],
            "template_path": str(path.resolve())}


def _verify_cli_defaults():
    """Bind the unexposed affinity recycles/precision/MW defaults to real code."""
    from importlib import metadata
    from latentmol.experiments import file_sha256
    from .prepare import installed_boltz_provenance
    distribution = metadata.distribution("boltz")
    path = Path(distribution.locate_file("boltz/main.py"))
    if file_sha256(path) != CLI_MAIN_SHA256:
        raise ValueError("Pinned native CLI default-setting source changed")
    return {"path": str(path.resolve()), "sha256": CLI_MAIN_SHA256,
            "installed_provenance": installed_boltz_provenance(),
            "unexposed_defaults": {"affinity_recycles": 5, "precision": "bf16-mixed", "mw_correction": False}}


def paired_report(selection, results):
    """Both native heads use complete eligible pairs at the same pose seed."""
    by_key = {(row["smiles"], row["seed"]): row for row in results}
    if len(by_key) != len(results):
        raise ValueError("Duplicate native results would overweight an identity")
    planned = {(row["smiles"], row["seed"]) for row in selection["requests"]}
    if not set(by_key) <= planned:
        raise ValueError("Native result is outside the frozen request denominator")
    rows = []
    for member in selection["members"]:
        if member["arm"] == "start":
            continue
        start = next(row for row in selection["members"]
                     if row["case_id"] == member["case_id"] and row["arm"] == "start")
        pairs = []
        for seed in selection["pose_seeds"]:
            initial = by_key.get((start["canonical_smiles"], seed))
            candidate = by_key.get((member["canonical_smiles"], seed))
            a = initial.get("affinity") if initial and initial.get("retention_eligible") is True and initial.get("status") == "completed" else None
            b = candidate.get("affinity") if candidate and candidate.get("retention_eligible") is True and candidate.get("status") == "completed" else None
            pair = {"seed": seed, "paired_identity_eligible": a is not None and b is not None,
                    "start_native_eligible": a is not None, "candidate_native_eligible": b is not None}
            if a is not None and b is not None:
                for key in ("affinity_pred_value", "affinity_probability_binary"):
                    av, bv = float(a[key]), float(b[key])
                    if not math.isfinite(av) or not math.isfinite(bv):
                        raise ValueError("Nonfinite native pair must not enter a gain report")
                    pair[key] = {"start": av, "candidate": bv, "candidate_minus_start": bv - av}
            pairs.append(pair)
        complete = all(pair["paired_identity_eligible"] for pair in pairs)
        rows.append({"case_id": member["case_id"], "arm": member["arm"], "state": member["state"],
                     "start_smiles": start["canonical_smiles"], "smiles": member["canonical_smiles"],
                     "failure": member["failure"], "identity_changed":
                     member["canonical_smiles"] != start["canonical_smiles"]
                     if member["canonical_smiles"] and start["canonical_smiles"] else None,
                     "paired_seeds": pairs, "both_seed_pairs_eligible": complete,
                     "mean_delta": {key: sum(pair[key]["candidate_minus_start"] for pair in pairs) / len(pairs)
                                    for key in ("affinity_pred_value", "affinity_probability_binary")}
                     if complete else None})
    return {"cases": rows, "case_count": len(selection["case_ids"]),
            "successful_native_requests": sum(row.get("retention_eligible") is True and row.get("status") == "completed" for row in results),
            "failed_native_requests": sum(row.get("status") == "native_failed" for row in results),
            "identity_ineligible_requests": sum(row.get("status") == "identity_ineligible" for row in results),
            "unattempted_requests": len(selection["requests"]) - len(results),
            "terminal_native_requests": len(results), "planned_native_requests": len(selection["requests"]),
            "all_requests_terminal": len(results) == len(selection["requests"]),
            "claim_limit": "Exploratory paired native affinities at fresh poses; not fixed-pose full DBMol, measured binding or statistical superiority."}


def run_followup(selection, target, cache, output, *, deadline,
                 native_timeout_seconds=1800, audit_timeout_seconds=600):
    """Run serial fresh requests after freezing selection, preserving failures.

    ``deadline`` is the allocation's monotonic shared internal deadline. No
    request launches without its full native+audit budget and cleanup reserve.
    Missing budget produces a partial record rather than fabricated failures.
    """
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise RuntimeError("Native comparison follow-up requires a sealed LSF allocation")
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise ValueError("Native follow-up requires a finite shared monotonic deadline")
    if any(type(value) is not int or value <= 0 for value in (native_timeout_seconds, audit_timeout_seconds)):
        raise ValueError("Native and audit child timeouts must be positive integer seconds")
    core = {key: value for key, value in selection.items() if key != "selection_sha256"}
    if selection.get("version") != VERSION or selection.get("selection_sha256") != _digest(core):
        raise ValueError("Frozen native selection identity changed")
    from latentmol.experiments import file_sha256
    from .fresh_trajectory import execute_request
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    _write(root / "selection.json", selection)
    _write(root / "target.json", target)
    cli = _verify_cli_defaults()
    _write(root / "native-cli-provenance.json", cli)
    target_digest = _digest(target)
    results = []
    protocol = {"native_timeout_seconds": native_timeout_seconds,
                "audit_timeout_seconds": audit_timeout_seconds}
    partial = False
    for request in selection["requests"]:
        if deadline - time.monotonic() <= native_timeout_seconds + audit_timeout_seconds + 120:
            partial = True
            break
        if (_digest(target) != target_digest
                or file_sha256(Path(target["msa_path"])) != target["msa_sha256"]
                or file_sha256(Path(target["template_path"])) != target["template_sha256"]):
            raise ValueError("Pinned native target/template/MSA changed")
        if file_sha256(Path(cli["path"])) != cli["sha256"]:
            raise ValueError("Pinned native CLI changed during follow-up")
        attempt = root / "requests" / request["id"]
        _write(root / "progress.json", {"status": "running_not_passed", "current_request": request,
                                      "completed_requests": len(results)})
        try:
            result = execute_request(deepcopy(request), deepcopy(target), Path(cache), attempt, protocol)
        except Exception as error:
            _write(root / "stopped.json", {"status": "failed_not_passed", "request": request,
                   "error_type": type(error).__name__, "error": str(error),
                   "completed_requests": len(results)})
            raise
        if result.get("id") != request["id"] or result.get("smiles") != request["smiles"] or result.get("seed") != request["seed"]:
            raise ValueError("Native result association differs from its frozen request")
        if (file_sha256(Path(target["msa_path"])) != target["msa_sha256"]
                or file_sha256(Path(target["template_path"])) != target["template_sha256"]):
            raise ValueError("Pinned native target asset changed during request")
        if file_sha256(Path(cli["path"])) != cli["sha256"]:
            raise ValueError("Pinned native CLI changed during request")
        results.append(result)
        _write(attempt / "result.json", result)
        _write(root / "native-results.json", results)
        _write(root / "summary.json", {**paired_report(selection, results), "status": "running_not_passed"})
    report = {**paired_report(selection, results), "status": "partial_runtime_not_passed" if partial else "completed",
              "selection_sha256": selection["selection_sha256"], "target_sha256": target_digest,
              "existing_vae_continuation": "untouched"}
    _write(root / "summary.json", report)
    return report
