"""Fresh Nesso predictions for molecules selected by a completed Protenix search."""
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def _load(path):
    return json.loads(path.read_bytes())


def _contact(row, cycle):
    value = row["loss"]
    if row["status"] != "scored" or not math.isfinite(value):
        raise ValueError("Expected a successful finite discrete Protenix contact score")
    return dict(smiles=row["smiles"], cycle=cycle, branch=row["arm"], step=row["step"],
                contact_loss=value)


def _native(fresh, smiles):
    row = fresh(smiles)
    if row["status"] == "scored":
        if (not math.isfinite(row["affinity_pred_value"])
                or not 0 <= row["affinity_probability_binary"] <= 1
                or type(row["has_interface"]) is not bool):
            raise ValueError("Invalid fresh Nesso prediction")
    elif row["status"] != "feature_failed":
        raise ValueError(f"Unexpected fresh Nesso status {row['status']!r}")
    return row


def write_nesso_readout(root, fresh, *, seed, model_sha256, output_path=None):
    """Evaluate the founder and all new, distinct, successfully contact-scored molecules."""
    root = Path(root)
    output = Path(output_path) if output_path is not None else root / "nesso-readout.json"
    if output.exists():
        raise FileExistsError(output)
    run = _load(root / "run.json")
    result = _load(root / "result.json")
    if result["status"] not in {"completed", "frontier_exhausted", "duplicate_frontier_exhausted", "time_budget"}:
        raise ValueError("Nesso readout requires a final search receipt")
    parent, = run["parents"]
    baseline = _load(root / "founder-score" / "fresh" / "summary.json")
    raw_founder, = baseline["candidates"]
    founder = _contact(raw_founder, 0)
    if (baseline["status"] != "completed" or raw_founder["role"] != "initial"
            or founder["smiles"] != parent["smiles"] or founder["contact_loss"] != parent["fresh_loss"]):
        raise ValueError("Founder differs from the original Protenix scoring receipt")

    seen = {founder["smiles"]}
    molecules = []
    attempts = 0
    for milestone in result["cycles"]:
        cycle = milestone["cycle"]
        if not milestone["fresh_attempts"]:
            if milestone["fresh_scored"]:
                raise ValueError("Successful Protenix score without a scoring attempt")
            continue
        receipt = _load(root / f"cycle-{cycle:02d}" / "fresh" / "summary.json")
        rows = receipt["candidates"]
        if (receipt["status"] != "completed" or receipt["geometry_seed"] != baseline["geometry_seed"]
                or len(rows) != milestone["fresh_attempts"]
                or sum(r["status"] == "scored" for r in rows) != milestone["fresh_scored"]):
            raise ValueError("Incomplete or incompatible Protenix discovery receipt")
        for row in rows:
            if row["smiles"] in seen or row["role"] != "edit":
                raise ValueError("Expected a distinct non-founder discovery candidate")
            seen.add(row["smiles"])
            if row["status"] == "scored":
                molecules.append(_contact(row, cycle))
            elif row["loss"] is not None:
                raise ValueError("Unsuccessful Protenix attempt has a score")
        attempts += len(rows)
    if attempts != result["unique_fresh_attempts"]:
        raise ValueError("Discovery receipts and reported scoring budget disagree")

    baseline_native = founder["native_nesso"] = _native(fresh, founder["smiles"])
    for row in molecules:
        candidate = row["native_nesso"] = _native(fresh, row["smiles"])
        row.update(contact_change=row["contact_loss"] - founder["contact_loss"],
                   affinity_change=None, probability_change=None, improved_both=None, improved_all_three=None)
        if baseline_native["status"] == "scored" and candidate["status"] == "scored":
            row["affinity_change"] = candidate["affinity_pred_value"] - baseline_native["affinity_pred_value"]
            row["probability_change"] = candidate["affinity_probability_binary"] - baseline_native["affinity_probability_binary"]
            if baseline_native["has_interface"]:
                row["improved_both"] = candidate["has_interface"] and row["contact_change"] < 0 and row["affinity_change"] < 0
                row["improved_all_three"] = row["improved_both"] and row["probability_change"] > 0
    joint = [r for r in molecules if r["improved_both"]]
    comparable = baseline_native["status"] == "scored" and baseline_native["has_interface"]
    best = min(joint, key=lambda r: (r["contact_loss"], r["smiles"]), default=None)
    summary = dict(distinct_new=len(molecules), nesso_scored_new=sum(r["native_nesso"]["status"] == "scored" for r in molecules),
                   joint_hits=len(joint) if comparable else None,
                   triple_hits=sum(r["improved_all_three"] is True for r in joint) if comparable else None,
                   best_joint_by_contact=best["smiles"] if best else None)
    report = dict(status="completed", search_status=result["status"], search_job_id=run["job_id"], seed=seed, model_sha256=model_sha256,
                  checkpoint_sha256=run["checkpoint_sha256"], protenix_fresh_config_sha256=run["scoring_configs"]["fresh"],
                  geometry_seed=baseline["geometry_seed"], hard_recycles=5, completed_updates=result["completed_updates"],
                  planned=result["planned"], protenix_attempts_new=attempts, founder=founder, molecules=molecules, summary=summary)
    # Publish only the complete report, including on concurrent or retried invocations.
    with NamedTemporaryFile(mode="w", dir=output.parent, delete=False, encoding="utf-8") as tmp:
        pending = Path(tmp.name)
    try:
        pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.link(pending, output)
    finally:
        pending.unlink()
    return report
