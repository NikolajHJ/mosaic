"""Standard-library contract for the new three-arm 25-slot campaign."""

import hashlib
import json
import math
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

PROTENIX_LR03_VERSION = "mosaic-brd4-jq1-25-protenix-lr0p3-v1"
FIXED25_VERSION = "mosaic-brd4-jq1-25-fixed-step-v1"
FIXED25_LRBOOST_VERSION = "mosaic-brd4-jq1-25-fixed-step-lrboost-v1"
FIXED25_VERSIONS = (FIXED25_VERSION, FIXED25_LRBOOST_VERSION)
FIXED25_LRBOOST_RATES = {"protenix": 3.0, "boltz2": 0.05, "boltz2_chemistry": 0.05}


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
    lr03 = version == PROTENIX_LR03_VERSION
    fixed = version in FIXED25_VERSIONS
    boosted = version == FIXED25_LRBOOST_VERSION
    pocket = version in ("mosaic-brd4-jq1-25-three-arm-v1", PROTENIX_LR03_VERSION, *FIXED25_VERSIONS)
    if version not in ("mosaic-brd4-25-three-arm-v1", "mosaic-brd4-jq1-25-three-arm-v1", PROTENIX_LR03_VERSION, *FIXED25_VERSIONS):
        raise ValueError("Unknown declared campaign version")
    if lr03 and config.get("arm") != "protenix":
        raise ValueError("LR0.3 authorization covers Protenix only")
    if config.get("pocket") != (POCKET_POLICY if pocket else None):
        raise ValueError("Changed declared pocket protocol")
    if config.get('acceptance_policy') != ('fixed_step' if fixed else None):
        raise ValueError('Changed declared step acceptance policy')
    nominal = (FIXED25_LRBOOST_RATES.get(config.get('arm')) if boosted else
               ((0.3 if config.get('arm') == 'protenix' else 0.01) if fixed else (0.3 if lr03 else 0.03)))
    expected = {
        "version": version, "n_atoms": 25,
        "steps": 10000, "seed": 17, "initial_step_size": nominal,
        "max_backtracks": 1 if fixed else 10, "pilot_steps": 10,
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
    if boosted:
        preserved = {
            "loss": {"version": "repair-failure-guidance-v1", "extra_valid_weight": 1.0,
                     "partial_weight": 1.0, "local_bond_weight": 0.1,
                     "component_cut_weight": 0.1, "nonempty_weight": 0.1},
            "search": {"max_edits": 3, "checks_per_depth": 32, "beam_width": 4,
                       "probability_weight": 1.0, "probability_floor": 1e-12},
            "progress_search": {"version": "repair-progress-v1", "max_edits": 6,
                "checks_per_depth": 32, "beam_width": 4, "probability_weight": 1.0,
                "allow_atom_edits": False, "atom_edit_penalty": 3.0, "max_atom_edits": 1},
            "fixture": "tests/data/brd4_3mxf/fixture_25.json",
        }
        allowed = set(expected) | set(preserved) | {"arm", "chemistry_enabled", "repair_schedule", "pocket", "acceptance_policy", "label"}
        if set(config) != allowed or any(config.get(k) != v for k, v in preserved.items()):
            raise ValueError("LR-boost version changes only declared rates; objective/search/fixture fields must stay fixed")
        if any(type(config.get(k)) is not int for k in ("n_atoms", "steps", "seed", "max_backtracks", "pilot_steps", "allocation_seconds")):
            raise ValueError("LR-boost dimensions, key and budgets must be exact integers")
        if type(config.get("initial_step_size")) not in (int, float) or not math.isfinite(config["initial_step_size"]):
            raise ValueError("LR-boost learning rate must be finite")
    return config


def update_parameters(config):
    """Route the authenticated nominal rate to pilots and production alike."""
    config25(config)
    result = {"step_size": config["initial_step_size"], "max_backtracks": config["max_backtracks"]}
    if config['version'] in FIXED25_VERSIONS:
        result['acceptance_policy'] = 'fixed_step'
    return result


def verify_attempt_rates(result, config):
    """Check executed trial rates, including honest fallback/no-op updates."""
    policy = update_parameters(config)
    fixed = config['version'] in FIXED25_VERSIONS
    boosted = config['version'] == FIXED25_LRBOOST_VERSION
    if fixed and result.get('acceptance_policy') != 'fixed_step':
        raise ValueError('Actual optimization used an unexpected acceptance policy')
    for update in result["updates"]:
        attempts = update["attempts"]
        if not 1 <= len(attempts) <= policy["max_backtracks"] or any(
            a["trial"] != i or a["rate"] != policy["step_size"] * 0.5**i
            for i, a in enumerate(attempts)
        ):
            raise ValueError("Actual attempted rates differ from declared update policy")
        selected = update["selected_trial"]
        if not 0 <= selected < len(attempts) or update["accepted_step_size"] != attempts[selected]["rate"]:
            raise ValueError("Actual selected rate differs from attempted trial")
        if fixed and (update.get('fallback_used') is not False or selected != 0):
            raise ValueError('Fixed steps cannot be relabelled as a search/fallback')
        if boosted:
            candidate = attempts[0]
            numeric = [update.get(k) for k in ('fixed_before', 'fixed_after', 'accepted_step_size')]
            numeric += [candidate.get(k) for k in ('rate', 'autodiff_surrogate', 'independent_primal_surrogate', 'repair_weight')]
            components = candidate.get('components', {})
            if (update.get('selection_policy') != 'fixed_step' or type(selected) is not int
                    or type(candidate.get('trial')) is not int or candidate['trial'] != 0
                    or not components or not isinstance(components, dict)
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in numeric + list(components.values()))
                    or update.get('fixed_after') != candidate.get('independent_primal_surrogate')
                    or update.get('repair_weight') != candidate.get('repair_weight')
                    or type(candidate.get('strict')) is not bool or type(update.get('no_op')) is not bool
                    or type(update.get('fixed_increased')) is not bool
                    or update['fixed_increased'] is not (update['fixed_after'] > update['fixed_before'])):
                raise ValueError('LR-boost update must authenticate its one finite full-rate candidate')
    return policy


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
    if config["version"] in (PROTENIX_LR03_VERSION, *FIXED25_VERSIONS):
        summary = json.loads(path.with_name("summary.json").read_text())
        pilot = json.loads((path.parent / "early_pilot/report.json").read_text())
        if summary.get("optimization_parameters") != update_parameters(config) or pilot["status"] != "passed" or len(pilot["updates"]) != 10:
            raise ValueError("Missing executed declared-rate disposable pilot")
        if config['arm'] == 'protenix':
            initial_match = summary.get('baseline_initialization_match', {})
            if initial_match.get('status') != 'passed' or initial_match.get('job') != '29770179' or initial_match.get('sha256') != '351789aa62d3ff111a4d2050a062cf844a1f29a87a7e352af465ff4299fd1bbf':
                raise ValueError('Fresh declared-rate initialization differs from original run')
            if config['version'] == FIXED25_LRBOOST_VERSION:
                raw = summary.get('native_gradient_checks', {})
                updates = raw.get('updates', [])
                if (raw.get('status') != 'passed' or raw.get('requested_updates') != 1 or len(updates) != 1
                        or raw.get('primal_every_step') is not True or raw.get('acceptance_policy') != 'best_primal'
                        or len(updates[0].get('attempts', [])) != 1
                        or updates[0].get('selected_trial') != 0
                        or updates[0].get('accepted_step_size') != config['initial_step_size']
                        or updates[0]['attempts'][0].get('step_size') != config['initial_step_size']):
                    raise ValueError('LR-boost native Protenix one-step probe did not execute the declared single full-rate candidate')
        verify_attempt_rates(pilot, config)
        if config['arm'] == 'boltz2_chemistry':
            high = json.loads((path.parent / 'high_pilot/report.json').read_text())
            if high['status'] != 'passed' or len(high['updates']) != 10:
                raise ValueError('Missing fixed-step high-weight chemistry pilot')
            verify_attempt_rates(high, config)
    return assessment
