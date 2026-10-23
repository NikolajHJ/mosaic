"""Hard-candidate evaluation outside autograd, with explicit molecular identity.

Each native executable implements a tiny JSON request/response protocol. This
keeps native preprocessing, fresh poses and incompatible package environments
outside the gradient branch. No scorer, fake result or implicit repair is supplied
by this command boundary. Actual native evaluator executables require validation.
"""

import json
import math
from pathlib import Path
import subprocess
import time

from latentmol.experiments import file_sha256, write_json


def candidates(run_dir):
    """Recompute molecular identity; never trust a saved hash or SMILES alone."""
    import torch
    from latentmol.chemistry import inspect_graph
    from latentmol.representation import GraphSpec, HardGraphBatch
    from .study import hard_digest
    rows, seen = [], set()
    for path in sorted(Path(run_dir).glob("arms/*/candidates.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            spec = GraphSpec.from_dict(row["graph_spec"])
            labels = {key: torch.as_tensor(row["labels"][key]) for key in ("atoms", "charges", "bonds")}
            shapes = {"atoms": (spec.max_nodes,), "charges": (spec.max_nodes,), "bonds": (spec.max_nodes, spec.max_nodes)}
            if any(value.dtype != torch.int64 or tuple(value.shape) != shapes[key] for key, value in labels.items()):
                raise ValueError("Candidate categorical labels have changed type/shape")
            hard = HardGraphBatch(**{key: value[None] for key, value in labels.items()})
            if hard_digest(hard) != row["graph_sha256"]:
                raise ValueError("Candidate graph hash differs from saved labels")
            expected_id = f"{row['arm']}-s{row['step']:04d}-d{row['draw']:03d}"
            if row["candidate_id"] != expected_id or row["arm"] != path.parent.name or expected_id in seen:
                raise ValueError("Candidate id/arm/step/draw is inconsistent or duplicated")
            seen.add(expected_id)
            inspection = inspect_graph(labels["atoms"], labels["charges"], labels["bonds"], spec)
            if (inspection.smiles != row["smiles"] or inspection.failure_reason != row["failure"]
                    or int((labels["atoms"] != 0).sum()) != row["heavy_atoms"]):
                raise ValueError("Candidate SMILES/diagnostic differs from the raw graph")
            rows.append(row)
    return rows


def generation_provenance(run_dir, study):
    """Require a completed source experiment, while permitting new native scorers.

    Confirmation may evaluate a different set of models than optimization did.
    Both configurations are recorded, and the generating decoder/schema/seed and
    each complete arm stay tied to their original immutable run identities.
    """
    from .config import Study
    run_dir = Path(run_dir)
    record = json.loads((run_dir / "run.json").read_text())
    identity = json.loads((run_dir / "identity.json").read_text())
    generated = Study.from_dict(record["config"]["study"])
    if record.get("status") != "completed" or record["config"]["stage"] not in ("run", "pipeline"):
        raise ValueError("Native comparison requires a completed generating run/pipeline")
    if identity.get("decoder_sha256") != generated.decoder["sha256"] or identity.get("decoder_sha256") != study.decoder["sha256"]:
        raise ValueError("Confirmation decoder identity differs from the generating run")
    if identity.get("source_sha256") != record["source"]["sha256"]:
        raise ValueError("Generating source manifest and identity disagree")
    rows = candidates(run_dir)
    from latentmol.representation import GraphSpec
    for row in rows:
        if GraphSpec.from_dict(row["graph_spec"]).fingerprint() != identity.get("graph_spec_sha256"):
            raise ValueError("Candidate schema differs from generating run identity")
    for name in {row["arm"] for row in rows}:
        arm = json.loads((run_dir / "arms" / name / "arm.json").read_text())
        selected = [row for row in rows if row["arm"] == name]
        if arm.get("status") != "completed" or name not in record["config"]["arms"] or arm.get("attempted") != len(selected):
            raise ValueError("Source arm incomplete or candidate attempt count differs")
        if any(row["step"] < 0 or row["step"] > arm["steps"] for row in selected):
            raise ValueError("Candidate step outside its source arm")
    if not rows:
        raise ValueError("No completed hard candidates")
    return {"generation_config_sha256": generated.fingerprint(), "evaluation_config_sha256": study.fingerprint(),
            "run_manifest_sha256": file_sha256(run_dir / "run.json"),
            "identity_file_sha256": file_sha256(run_dir / "identity.json"),
            "candidate_file_sha256": {str(path.relative_to(run_dir)): file_sha256(path)
                                      for path in sorted(run_dir.glob("arms/*/candidates.jsonl"))}}


def endpoint_candidates(rows, budget):
    """Predeclare first draws at start/end; invalid draws consume selection slots."""
    selected = []
    for arm in sorted({row["arm"] for row in rows}):
        group = [row for row in rows if row["arm"] == arm]
        endpoints = {min(row["step"] for row in group), max(row["step"] for row in group)}
        for step in sorted(endpoints):
            selected += sorted((row for row in group if row["step"] == step), key=lambda r: r["draw"])[:budget]
    return selected


def validate_response(response, request):
    """Never call a score on substituted chemistry retention of the original."""
    if (response.get("candidate_id") != request["candidate_id"]
            or response.get("graph_sha256") != request["graph_sha256"]
            or response.get("input_smiles") != request["smiles"]):
        raise ValueError("Native response does not identify the requested molecule")
    if response.get("status") not in ("ok", "unsupported", "failed"):
        raise ValueError("Native response needs explicit ok/unsupported/failed status")
    if response["status"] != "ok":
        if not response.get("reason"):
            raise ValueError("Native failures require a reason")
        return response
    effective = response.get("effective_smiles")
    if not isinstance(effective, str) or not effective:
        raise ValueError("Native evaluator must report effective chemistry after its preprocessing")
    # Canonical comparison permits reordering, but never drops charge/stereo to
    # manufacture parity. Our shared schema's stereo loss remains documented.
    from rdkit import Chem
    incoming, outgoing = Chem.MolFromSmiles(request["smiles"]), Chem.MolFromSmiles(effective)
    if incoming is None or outgoing is None:
        raise ValueError("Native evaluator reported invalid effective SMILES")
    same = Chem.MolToSmiles(incoming, isomericSmiles=True) == Chem.MolToSmiles(outgoing, isomericSmiles=True)
    scores = response.get("scores")
    if not isinstance(scores, dict) or not scores or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in scores.values()):
        raise ValueError("Native response requires finite named scalar scores")
    if response.get("fresh_native_preprocessing") is not True:
        raise ValueError("Native confirmation must rebuild input features/geometry, not reuse fixed seed features")
    return {**response, "same_identity": same, "retention_eligible": same}


def confirm(study, run_dir, output_dir, event):
    """Run explicitly configured native executables without a shell or scorer fit."""
    provenance = generation_provenance(run_dir, study)
    rows = candidates(run_dir)
    if not rows:
        raise ValueError("No saved hard candidates")
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {"status": "completed", "source_run": str(Path(run_dir).resolve()), "provenance": provenance, "models": {}}
    for name, evaluator in study.native_evaluators.items():
        outcomes = []
        for candidate in endpoint_candidates(rows, evaluator["candidates_per_endpoint"]):
            job_dir = output_dir / name / candidate["candidate_id"]
            job_dir.mkdir(parents=True, exist_ok=False)
            request = {**candidate, "model": name, "native_evaluator_identity": evaluator["identity"]}
            request_path, response_path = job_dir / "request.json", job_dir / "response.json"
            write_json(request_path, request)
            result = {"candidate_id": candidate["candidate_id"], "arm": candidate["arm"],
                      "graph_sha256": candidate["graph_sha256"], "oracle_attempted": False}
            if not candidate["smiles"]:
                result.update(status="unsupported", reason="Invalid raw hard graph; no native call or repair")
            else:
                argv = [arg.replace("{input}", str(request_path.resolve())).replace("{output}", str(response_path.resolve()))
                        for arg in evaluator["command"]]
                write_json(job_dir / "command.json", {"argv": argv, "identity": evaluator["identity"]})
                start = time.perf_counter()
                result["oracle_attempted"] = True
                try:
                    with (job_dir / "command.log").open("w") as log:
                        process = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                                 timeout=evaluator["timeout_seconds"], check=False, shell=False)
                    if process.returncode:
                        raise RuntimeError(f"Native evaluator exited {process.returncode}")
                    response = validate_response(json.loads(response_path.read_text()), request)
                    result.update(response, response_sha256=file_sha256(response_path))
                except Exception as exc:
                    result.update(status="failed", reason=str(exc))
                result["elapsed_seconds"] = time.perf_counter() - start
            write_json(job_dir / "outcome.json", result)
            event({"native_model": name, **result})
            outcomes.append(result)
        summary["models"][name] = {"selected_attempts": len(outcomes),
                                  "oracle_attempts": sum(row["oracle_attempted"] for row in outcomes),
                                  "retention_eligible": sum(row.get("retention_eligible", False) for row in outcomes),
                                  "outcomes": outcomes}
    if not study.native_evaluators:
        summary["status"] = "not_configured"
        summary["reason"] = "Configure a native executable per model; no fresh-pose result is implied"
    write_json(output_dir / "summary.json", summary)
    return summary


def diagnose(run_dir, output_path, config):
    """Apply the existing graded repair-proximity diagnostic to saved raw graphs."""
    from latentmol.quality import GradedEvaluator, combine_quality
    from latentmol.representation import GraphSpec
    evaluators, records = {}, []
    for candidate in candidates(run_dir):
        spec = GraphSpec.from_dict(candidate["graph_spec"])
        evaluator = evaluators.setdefault(spec.fingerprint(), GradedEvaluator(spec, config=config))
        labels = candidate["labels"]
        diagnostic = evaluator.diagnose(labels["atoms"], labels["charges"], labels["bonds"])
        # diagnose_graph returns the bounded component keys consumed by the
        # established v2 combination. Missing/censored repair stays undefined.
        components = {key: diagnostic.get(key) for key in ("local_consistency", "connectedness", "repair_proximity")}
        components["validity"] = float(bool(diagnostic.get("valid", False)))
        score = combine_quality(components, config)
        records.append({"candidate_id": candidate["candidate_id"], "arm": candidate["arm"],
                        "graph_sha256": candidate["graph_sha256"], "diagnostic": diagnostic, "quality": score})
    value = {"status": "completed", "records": records,
             "compute": {key: evaluator.compute() for key, evaluator in evaluators.items()}}
    write_json(output_path, value)
    return value
