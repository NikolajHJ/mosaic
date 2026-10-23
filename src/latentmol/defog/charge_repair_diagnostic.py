"""Allocated-HPC exact-candidate test of an explicitly lossy charge interface.

Replay five captured candidates with identical native seeds/settings under the
legacy and opt-in protocols. Every prescribed attempt is retained. This does
not run Boltz, optimize z, change a failed historical gate or resample for
eligibility. A successful target-only FD check concerns the detached target
likelihood through the original frozen decoder, never DeFoG sampling gradients.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time

from latentmol.config_identity import verify_config_identity
from latentmol.experiments import ExperimentRun, file_sha256, source_manifest, write_json

SOURCE079_ARCHIVE = "744862c9a310f87ff078edb2ab3185fd2468eaa00339222d024c1ffe0ceebcce"
SOURCE079_MANIFEST = "21025b7bdb84ceeaf515ebe885ec0941491acf797a6ada66ffa02d73f1a5f29f"
CASE_IDS = ("prior-0-attempt-0", "prior-8-attempt-0", "prior-8-attempt-1",
            "prior-9-attempt-0", "prior-9-attempt-1")
CLAIM_LIMIT = ("Exact saved hard-candidate repair and frozen-target decoder-NLL checks only; "
               "charge-blind native conditioning, rule-assigned output charge, fixed occupancy; "
               "no Boltz evaluation, latent updates, native/draw/selection derivatives or production readiness.")


def read_asset(asset):
    """Authenticate original bytes before parsing; never rewrite parent records."""
    path = Path(asset["path"])
    if file_sha256(path) != asset["sha256"]:
        raise ValueError("Captured artifact changed: " + str(path))
    return json.loads(path.read_text())


def replay_request(original, source_sha256, *, projected):
    """Only protocol/provenance keys change; keep graph, RNG and native budget."""
    from latentmol.defog.codec import PROJECT_CHARGE_POLICY
    from latentmol.repair_guidance.provider import request_digest
    if original.get("version") != 1 or "charge_policy" in original:
        raise ValueError("Require an original strict v1 request")
    if request_digest(original) != original.get("request_sha256"):
        raise ValueError("Original request digest changed")
    result = deepcopy(original)
    result["adapter_source_sha256"] = source_sha256
    if projected:
        result.update(version=2, charge_policy=PROJECT_CHARGE_POLICY)
    result["request_sha256"] = request_digest(result)
    return result


def require_cpu_parity(root, source_root):
    """Require exactly one completed same-release native-builder parity record."""
    from latentmol.defog.codec import PROJECT_CHARGE_POLICY, NATIVE_CHARGE_RULE
    rows = list(Path(root).glob("*/run.json"))
    if len(rows) != 1:
        raise ValueError("Require one unambiguous native-builder parity run")
    record = json.loads(rows[0].read_text())
    verify_config_identity(record)
    summary_path = rows[0].with_name("summary.json")
    summary = json.loads(summary_path.read_text())
    current = source_manifest(source_root)["sha256"]
    scientific_fields = ("charge_policy", "rule", "native_source", "rdkit_version", "helper_source_sha256", "scope")
    cases = summary.get("cases")
    if (record["status"] != "completed" or record["source"]["sha256"] != current
            or record.get("source_archive_sha256") != os.environ.get("LATENTMOL_SOURCE_SHA256")
            or summary.get("passed") is not True or summary.get("native_builder_parity") is not True
            or summary.get("case_count") != 11 or summary.get("model_calls") != 0
            or summary.get("sampled_repairs") != 0
            or summary.get("charge_policy") != PROJECT_CHARGE_POLICY
            or summary.get("rule") != NATIVE_CHARGE_RULE
            or any(summary.get(key) != record["config"].get(key) for key in scientific_fields)
            or not isinstance(cases, list) or len(cases) != 11
            or any(row.get("passed") is not True for row in cases)
            or summary.get("helper_source_sha256") != file_sha256(source_root / "src/latentmol/defog/codec.py")):
        raise ValueError("Native-builder parity lacks completed same-release proof")
    return {"run": str(rows[0]), "run_sha256": file_sha256(rows[0]),
            "summary": str(summary_path), "summary_sha256": file_sha256(summary_path), "result": summary}


def execution_completeness(cases):
    """A successful target never conceals another prescribed attempt's failure."""
    strict = [row for case in cases if case["charged_input"]
              for row in case["attempts"] if row["mode"] == "strict_v1"]
    native = [row for case in cases for row in case["attempts"]
              if row["mode"] == "projected_v2" or not case["charged_input"]]
    counters = ("native_forward_calls", "native_transitions", "attempted_native_transitions")
    strict_passed = (len(strict) == 4 and all(row.get("status") == "rejected"
        and row.get("reason") == "input_representation_unsupported" and row.get("process_returncode") == 0
        and all(row.get("cost", {}).get(key) == 0 for key in counters) for row in strict))
    native_passed = (len(native) == 6 and all(row.get("status") in ("accepted", "rejected")
        and row.get("process_returncode") == 0
        and all(row.get("cost", {}).get(key) == 447 for key in counters) for row in native))
    return {"strict_charged_controls_passed": strict_passed,
            "supported_continuations_completed": native_passed,
            "strict_charged_control_count": len(strict), "supported_continuation_count": len(native)}


def run(plan_path, plan_sha256, run_root, parity_root):
    """One GPU worker at a time; parent decoder stays CPU and is loaded once."""
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Numerical repair diagnostics require an allocated DTU HPC job")
    plan = read_asset({"path": str(plan_path), "sha256": plan_sha256})
    if (plan.get("version") != 1 or tuple(row["id"] for row in plan["cases"]) != CASE_IDS
            or plan.get("source079_archive_sha256") != SOURCE079_ARCHIVE
            or plan.get("source079_manifest_sha256") != SOURCE079_MANIFEST):
        raise ValueError("Unexpected exact-candidate diagnostic plan")
    source_root = Path.cwd()
    parity = require_cpu_parity(parity_root, source_root)
    manifest = source_manifest(source_root)["sha256"]
    import torch
    from latentmol.boltz.active_set import load_initial_latent
    from latentmol.boltz.local_fd import validate_local_fd
    from latentmol.defog.codec import PROJECT_CHARGE_POLICY
    from latentmol.evaluate import load_model
    from latentmol.multimodel.study import hard_digest, module_digest, tensor_digest
    from latentmol.repair_guidance.config import RepairGuidanceConfig
    from latentmol.repair_guidance.loss import FrozenRepairTarget, target_score_at
    from latentmol.repair_guidance.provider import DeFoGRepairProvider, _target_from_record
    from latentmol.representation import GraphSpec
    started = time.monotonic()
    config = {"plan": str(plan_path), "plan_sha256": plan_sha256, "parity": parity,
              "scope": CLAIM_LIMIT, "worker_quota": 10, "native_continuation_quota": 6,
              "wall_budget_seconds": 4500, "fd_policy": "unchanged local_v1 target-only CPU"}
    with ExperimentRun(Path(run_root), "defog-charge-interface-replay", config, source_root) as experiment:
        summary = {"passed": False, "cases": [], "target_gradient_checks": [],
                   "boltz_calls": 0, "latent_updates": 0, "claim_limit": CLAIM_LIMIT}
        summary_path = experiment.path / "summary.json"
        persist = lambda: write_json(summary_path, summary)
        persist()
        selected = {}
        decoder = None
        decoder_before = None
        neutral_parity = False
        for item in plan["cases"]:
            if time.monotonic() - started > config["wall_budget_seconds"] - 300:
                raise TimeoutError("Declared replay budget exhausted; no hidden retries")
            original = read_asset(item["request"])
            parent = read_asset(item["parent_run"])
            parent_identity = verify_config_identity(parent, allow_legacy_active_set_contexts=True)
            pcfg = parent["config"]
            if (pcfg["source_archive_sha256"] != SOURCE079_ARCHIVE
                    or pcfg["source_sha256"] != SOURCE079_MANIFEST
                    or original["adapter_source_sha256"] != SOURCE079_MANIFEST):
                raise ValueError("Captured request/parent is not the authenticated source079 run")
            spec = GraphSpec.from_dict(original["spec"])
            candidate = _target_from_record(original["candidate"], spec)
            active = torch.zeros(spec.max_nodes, dtype=torch.bool)
            active[original["occupied_slots"]] = True
            if not torch.equal(candidate.atoms[0] != 0, active):
                raise ValueError("Captured candidate changed occupied slots")
            if original["device"] != "cuda" or original["steps"] != 1000 or original["t0"] != .8:
                raise ValueError("Captured native budget differs from the prescribed comparison")
            canonical_before = hard_digest(candidate)
            is_charged = bool((candidate.charges[0][active] != spec.formal_charges.index(0)).any())
            if is_charged != (item["prior"] in (8, 9)):
                raise ValueError("Charged/neutral captured cohort changed")
            record = {"id": item["id"], "parent_identity": parent_identity,
                      "original_candidate_sha256": canonical_before, "charged_input": is_charged,
                      "attempts": [], "source079_request_sha256": original["request_sha256"]}
            summary["cases"].append(record)
            pair = []
            for projected in (False, True):
                mode = "projected_v2" if projected else "strict_v1"
                folder = experiment.path / item["id"] / mode
                folder.mkdir(parents=True, exist_ok=False)
                raw_cfg = deepcopy(pcfg["repair_guidance"])
                if projected:
                    raw_cfg["provider"]["charge_policy"] = PROJECT_CHARGE_POLICY
                cfg = RepairGuidanceConfig.from_mapping(raw_cfg, source_root)
                # Parent settings must exactly describe the original worker.
                for key in ("source", "checkpoint", "t0", "steps", "initialization", "device", "sampling"):
                    if cfg.provider[key] != original[key]:
                        raise ValueError("Parent repair setting differs from captured request: " + key)
                provider = DeFoGRepairProvider(cfg, manifest)
                request = replay_request(original, manifest, projected=projected)
                row = {"mode": mode, "request_sha256": request["request_sha256"], "status": "started"}
                record["attempts"].append(row)
                persist()
                response, execution, response_path = provider._worker(request, folder)
                row.update(execution, response_path=str(response_path))
                target = None
                if response is None:
                    row.update(status="timeout", target_eligible=False)
                else:
                    target = provider._verify_response(response, request, spec, candidate, active,
                                                       execution["process_returncode"])
                    row.update(status=response["status"], reason=response.get("reason"), cost=response["cost"],
                               target_eligible=target is not None)
                    if not projected and is_charged:
                        if (response["status"] != "rejected" or response.get("reason") != "input_representation_unsupported"
                                or any(response["cost"][key] for key in ("native_forward_calls", "native_transitions", "attempted_native_transitions"))):
                            raise ValueError("Strict charged control did not reject before native inference")
                    if projected and response["status"] != "error":
                        if (response["cost"]["native_forward_calls"] != 447
                                or response["cost"]["native_transitions"] != 447):
                            raise ValueError("Projected candidate did not complete the unchanged native suffix")
                    if target is not None and projected and item["prior"] in (8, 9):
                        # Prescribed first eligible, retaining the second attempt too.
                        selected.setdefault(item["prior"], (target, active.clone(), response_path, pcfg, parent))
                if hard_digest(candidate) != canonical_before:
                    raise ValueError("Worker/validation modified the canonical candidate")
                pair.append(response)
                write_json(folder / "validation.json", row)
                persist()
            if item["prior"] == 0:
                neutral_parity = (all(value is not None and value["status"] == "accepted" for value in pair)
                                  and pair[0]["native_input_labels"] == pair[1]["native_input_labels"]
                                  and pair[0]["native_output_labels"] == pair[1]["native_output_labels"])
                record["neutral_native_label_parity"] = neutral_parity
                persist()
        # Repair-oracle completeness is distinct from the target-only gradient gate.
        summary["neutral_native_label_parity"] = neutral_parity
        summary["charged_priors_with_eligible_targets"] = sorted(selected)
        persist()
        for prior in (8, 9):
            if prior not in selected:
                continue
            hard, active, receipt, pcfg, parent = selected[prior]
            asset = pcfg["decoder"]
            if file_sha256(Path(asset["path"])) != asset["sha256"]:
                raise ValueError("Original decoder checkpoint changed")
            if decoder is None:
                decoder, _ = load_model(Path(asset["path"]), "cpu")
                decoder.requires_grad_(False)
                decoder_before = module_digest(decoder)
            if module_digest(decoder) != parent["decoder_state_sha256"]:
                raise ValueError("Frozen decoder differs from original parent state")
            target = FrozenRepairTarget.create(hard, active, refresh_step=0, receipt_path=receipt)
            point = load_initial_latent(pcfg["initial_latent"], decoder.config.latent, "cpu")
            folder = experiment.path / f"prior-{prior}-target-gradient"
            folder.mkdir()
            baseline, gradient = target_score_at(decoder, point, active, target,
                temperature=pcfg["temperature"], normalization=pcfg["repair_guidance"]["normalization"], gradients=True)
            row = {"prior": prior, "target_sha256": target.identity, "baseline": baseline,
                   "latent_sha256": tensor_digest({"z": point}), "gradient_sha256": tensor_digest({"gradient": gradient}),
                   "gradient_norm": float(gradient.double().norm()), "finite_differences": {}}
            summary["target_gradient_checks"].append(row)
            score = lambda endpoint: target_score_at(decoder, endpoint, active, target,
                temperature=pcfg["temperature"], normalization=pcfg["repair_guidance"]["normalization"])[0]
            validate_local_fd(point, gradient, score, baseline=baseline["loss"], seed=1729,
                rtol=.05, atol=1e-6, minimum_passes=2, report=row["finite_differences"], persist=persist)
            write_json(folder / "report.json", row)
            if module_digest(decoder) != decoder_before:
                raise ValueError("Frozen decoder state changed during target-gradient validation")
        summary.update(execution_completeness(summary["cases"]))
        summary.update(wall_seconds=time.monotonic() - started,
            decoder_state_unchanged=decoder is not None and module_digest(decoder) == decoder_before)
        summary["passed"] = (neutral_parity and sorted(selected) == [8, 9]
            and summary["strict_charged_controls_passed"] and summary["supported_continuations_completed"]
            and len(summary["target_gradient_checks"]) == 2
            and all(row["finite_differences"]["passed"] for row in summary["target_gradient_checks"]))
        persist()
        if not summary["passed"]:
            raise RuntimeError("Bounded charge-interface repair/gradient diagnostic did not pass; all attempts retained")
        experiment.annotate(charge_interface_repair_passed=True, production_readiness=False)
        return experiment.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--parity-root", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.plan, args.plan_sha256, args.run_root, args.parity_root))


if __name__ == "__main__":
    main()
