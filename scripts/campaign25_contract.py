"""Standard-library contract for the new three-arm 25-slot campaign."""

import hashlib
import json
from pathlib import Path


ARMS = ("protenix", "boltz2", "boltz2_chemistry")
REFERENCE_SHA = "37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb"
MODEL_DIGESTS = {
    "protenix": "8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2",
    "boltz2": "395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16",
}

POCKET_POLICY = {
    "version": "brd4-jq1-heavy-neighborhood-v1", "pdb_id": "3MXF",
    "radius_angstrom": 6.0, "anchor_author_residues": [81, 140],
    "protenix_contact_cutoff_angstrom": 8.0, "boltz_anchor_max_distance_angstrom": 6.0,
    "minimum_ligand_fraction_within_pocket": 0.5, "diffusion_gradient": False,
}


def verify_frozen_response(response, model_digest, context_digest):
    """Require exact frozen digests and a passed worker response.

    The authenticated transport also includes id/elapsed_seconds. Those fields
    are not model/context state; retain them in the raw report, not in equality
    against the two-field expected identity. No digest or status check is waived.
    """
    if (
        response.get("status") != "passed"
        or response.get("model_digest") != model_digest
        or response.get("context_digest") != context_digest
    ):
        raise ValueError("Boltz frozen identity/status changed")
    return {"model_digest": response["model_digest"], "context_digest": response["context_digest"]}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def config25(config):
    version = config.get("version")
    pocket = version == "mosaic-brd4-jq1-25-three-arm-v1"
    if version not in ("mosaic-brd4-25-three-arm-v1", "mosaic-brd4-jq1-25-three-arm-v1"):
        raise ValueError("Unknown declared campaign version")
    if config.get("pocket") != (POCKET_POLICY if pocket else None):
        raise ValueError("Changed declared pocket protocol")
    expected = {
        "version": version, "n_atoms": 25,
        "steps": 10000, "seed": 17, "initial_step_size": 0.03,
        "max_backtracks": 10, "pilot_steps": 10,
        "allocation_seconds": 82800, "element_js_weight": 0.0,
        "initialization": {"kind": "chain_uniform", "mixing": 0.5},
    }
    if config.get("arm") not in ARMS or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Changed declared 25-slot three-arm campaign scope")
    active = config["arm"] == "boltz2_chemistry"
    if config.get("chemistry_enabled") is not active:
        raise ValueError("Arm/chemistry applicability mismatch")
    schedule = {
        "version": "delayed-geometric-v1", "initial": 0.001, "final": 50.0,
        "hold_updates": 5000, "total_updates": 10000,
    }
    if config.get("repair_schedule") != (schedule if active else None):
        raise ValueError("Changed declared repair schedule")
    return config


def verify_cpu_certificate(path, expected_sha, manifest):
    path = Path(path)
    if sha(path) != expected_sha:
        raise ValueError("CPU certificate bytes changed")
    record = json.loads(path.read_text())
    if record["status"] != "passed" or record["source_manifest_sha256"] != manifest:
        raise ValueError("CPU certificate does not cover this release")
    for proof in record["proofs"]:
        if sha(proof["path"]) != proof["sha256"]:
            raise ValueError("CPU final/test evidence changed")
    return record


def verify_readiness(path, expected_sha, config_path, manifest):
    path, config_path = Path(path), Path(config_path)
    if sha(path) != expected_sha:
        raise ValueError("Native readiness assessment bytes changed")
    assessment = json.loads(path.read_text())
    config = config25(json.loads(config_path.read_text()))
    if (
        assessment["status"] != "passed"
        or assessment["source_manifest_sha256"] != manifest
        or assessment["config_sha256"] != sha(config_path)
        or assessment["arm"] != config["arm"]
        or assessment["n_atoms"] != 25
        or assessment["model_digest"] != MODEL_DIGESTS["protenix" if config["arm"] == "protenix" else "boltz2"]
    ):
        raise ValueError("Native readiness does not certify this exact arm/configuration")
    if config.get("pocket") is not None and (
        len(assessment.get("pocket_definition_sha256", "")) != 64
        or assessment.get("pocket_validation_status") != "passed"
        or assessment.get("diffusion_gradient") is not False
    ):
        raise ValueError("Missing passed pocket/no-diffusion readiness binding")
    if config.get("pocket") is not None and not any(
        Path(p["path"]).name == "pocket_definition.json"
        and p["sha256"] == assessment["pocket_definition_sha256"]
        for p in assessment["proofs"]
    ):
        raise ValueError("Pocket definition bytes missing from native proofs")
    for proof in assessment["proofs"]:
        if sha(proof["path"]) != proof["sha256"]:
            raise ValueError("Native proof changed")
    return assessment
