"""Bounded, predeclared raw-versus-repair checkpoint outcomes for one long arm.

Uses the existing probability-aware repair audit and native Boltz CLI prediction
methods. No repair witness replaces the raw molecule. Invalid and budget-excluded
memberships remain in the denominator. Allocated HPC execution only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from run_boltz_defog_long_comparison import CHECKPOINTS, _read, _ref, _write, load_plan


def prescribed_memberships():
    return [{"step":s,"role":role,"seed":seed,"status":"not_attempted"}
            for s in CHECKPOINTS for role in ("raw","repaired") for seed in (17,23)]


def selected_molecules(candidates,repairs,*,terminal_step=500):
    """Prescribed argmax and weighted witness only; never rank on binding score."""
    by_step={r["step"]:r for r in candidates if r["draw"]==0}
    audited={r["step"]:r for r in repairs if r.get("arm")=="fixed_initial_mask"}
    if len(by_step)!=sum(r["draw"]==0 for r in candidates) or len(audited)!=len(repairs):
        raise ValueError("Duplicate or unexpected checkpoint candidate/repair rows")
    selected=[]
    for step in CHECKPOINTS:
        if step>terminal_step:
            if step in by_step:raise ValueError("Candidate exists beyond its authenticated terminal")
            selected.append({"step":step,**{role:{"smiles":None,"status":"not_reached","actual_terminal_step":terminal_step} for role in ("raw","repaired")}})
            continue
        raw=by_step.get(step);repair=audited.get(step)
        row={"step":step,"raw":{"smiles":None,"status":"missing_state"},
             "repaired":{"smiles":None,"status":"missing_state"}}
        if raw is not None:
            row["raw"]={"smiles":raw.get("smiles"),"status":"eligible" if raw.get("smiles") else "invalid_raw",
                "graph_sha256":raw.get("graph_sha256"),"failure":raw.get("failure"),"candidate_id":raw.get("candidate_id")}
        if repair is not None and repair.get("status")!="missing_state":
            result=repair.get("repair",{});witness=result.get("weighted_witness")
            if result.get("status")=="already_valid" and raw is not None and raw.get("smiles"):
                row["repaired"]={"smiles":raw["smiles"],"status":"eligible","edit_count":0,
                    "witness_kind":"already_valid_identity","weighted_cost":0.0}
            elif result.get("status")=="witness_found" and isinstance(witness,dict) and witness.get("smiles"):
                row["repaired"]={"smiles":witness["smiles"],"status":"eligible","edit_count":witness.get("edit_count"),
                    "weighted_cost":witness.get("weighted_cost"),"graph_id":witness.get("graph_id"),
                    "witness_kind":"bounded_rule_search_probability_weighted_repair"}
            else:
                row["repaired"]={"smiles":None,"status":"no_witness","search_status":result.get("status",repair.get("status"))}
        selected.append(row)
    return selected


def evaluate(plan_path,expected_sha,case_id,child,output):
    from latentmol.boltz.active_set_rescore import (_parent_target_identity,_target_identity,
        _read_candidates,_retention_result)
    from latentmol.boltz.chemical_audit import _catalog,describe
    from latentmol.boltz.native_rescore import _native_prediction
    from latentmol.boltz.prepare import installed_boltz_provenance
    from latentmol.boltz.stability_study import verify_asset
    from latentmol.boltz.trajectory_terminal import read_arm_terminal
    from latentmol.config_identity import verify_config_identity
    from latentmol.experiments import ExperimentRun,file_sha256
    from latentmol.probability_repair_audit import run_audit
    plan,configs,identity=load_plan(plan_path,expected_sha)
    if case_id not in configs:raise ValueError("Unknown comparison case")
    cfg=configs[case_id];child=Path(child).resolve();output=Path(output).resolve()
    if not child.is_relative_to(cfg.output_root.resolve()):raise ValueError("Checkpoint parent escaped the declared optimization root")
    parent=_read(child / "run.json");verify_config_identity(parent)
    cfg_ref=next(c["config"] for c in plan["cases"] if c["id"]==case_id)
    if (parent.get("status")!="completed" or parent["config"].get("config_file_sha256")!=cfg_ref["sha256"]
            or parent.get("source_archive_sha256")!=identity["source_archive_sha256"]
            or parent.get("source",{}).get("sha256")!=identity["source_manifest_sha256"]):
        raise ValueError("Fresh evaluation requires the completed authenticated long child")
    summary=_read(child / "summary.json")
    terminal=read_arm_terminal(child / "fixed_initial_mask",cfg.record(),summary["arms"]["fixed_initial_mask"])
    started=time.monotonic();deadline=started+plan["followup_timeout_seconds"]-30
    report={"status":"running","case_id":case_id,"child_run":str(child),"plan_sha256":expected_sha,
        "child_run_sha256":file_sha256(child / "run.json"),"memberships":prescribed_memberships(),
        "prediction_attempts":0,"maximum_prediction_attempts":plan["maximum_fresh_predictions_per_arm"],
        "pose_seeds":plan["pose_seeds"],"selection_policy":"prescribed checkpoints, raw argmax and bounded weighted witness; no native-score ranking",
        "prediction_priority":plan["prediction_priority"],
        "actual_terminal":terminal,
        "claim_limit":"Repair is a bounded observed witness, not nearest-valid distance. Native fresh-pose scores are predictions, not measured binding."}
    _write(output,report)
    root=Path(plan["output_root"]) / case_id
    try:
        protocol={"version":1,"name":"long-checkpoint-repair-"+case_id,"output_root":str(root / "repair-audit"),
            "parents":[{"id":case_id,"pilot":str(child),"run_sha256":report["child_run_sha256"],"steps":list(CHECKPOINTS)}],
            "search":plan["repair_search"]}
        protocol_path=output.with_name(case_id+"-checkpoint-repair-protocol.json");_write(protocol_path,protocol)
        repair_run=run_audit(protocol_path,expected_sha256=file_sha256(protocol_path))
        repairs=[json.loads(line) for line in (repair_run / "repairs.jsonl").read_text().splitlines() if line.strip()]
        candidate_path=child / "fixed_initial_mask/candidates.jsonl";candidate_sha=file_sha256(candidate_path)
        candidates=_read_candidates(candidate_path,parent["graph_spec_sha256"],500)
        expected={(s,d) for s in range(terminal["endpoint_step"]+1) for d in range(9)}
        if {(r["step"],r["draw"]) for r in candidates}!=expected:raise ValueError("Saved candidate membership differs from the actual terminal prefix")
        selected=selected_molecules(candidates,repairs,terminal_step=terminal["endpoint_step"])
        report["all_requested_checkpoints_observed"]=all(r["raw"]["status"] not in ("not_reached","missing_state") for r in selected)
        report["selection"]=selected;report["repair_audit"]={"run":str(repair_run),"summary":_ref(repair_run / "summary.json")}
        target,msa=_parent_target_identity(parent);native_identity=_target_identity(verify_asset(plan["native_template"]),target,msa)
        template=native_identity.pop("template");cache=Path(plan["cache"])
        if (file_sha256(cache / "boltz2_aff.ckpt")!=cfg.affinity_checkpoint["sha256"]
                or file_sha256(cache / "boltz2_conf.ckpt")!=cfg.dbmol.confidence_checkpoint["sha256"]):
            raise ValueError("Fresh prediction checkpoints differ from optimized frozen models")
        molecules=sorted({v[role]["smiles"] for v in selected for role in ("raw","repaired") if v[role]["smiles"]})
        audit_catalog=_catalog();chemical_identities={s:describe(s,audit_catalog) for s in molecules}
        evidence={"parent_run_id":parent["run_id"],"parent_run_sha256":report["child_run_sha256"],
            "plan_sha256":expected_sha,"case_id":case_id,"template_identity":native_identity,
            "selection":selected,"pose_seeds":plan["pose_seeds"],"prediction_priority":plan["prediction_priority"],"native_provenance":installed_boltz_provenance(),
            "budget":{"wall_seconds":plan["followup_timeout_seconds"],"maximum_predictions":24,"per_prediction_seconds":1800}}
        with ExperimentRun(root / "fresh-pose","long-checkpoint-fresh-pose",evidence,Path.cwd()) as experiment:
            _write(experiment.path / "identity_audit.json",chemical_identities)
            _write(experiment.path / "frozen_selection.json",selected)
            results={};native_rows=[];selection={v["step"]:v for v in selected}
            membership_index={(r["step"],r["role"],r["seed"]):r for r in report["memberships"]}
            for priority in plan["prediction_priority"]:
                row=membership_index[tuple(priority)]
                chosen=selection[row["step"]][row["role"]];row.update(deepcopy(chosen));key=(row["smiles"],row["seed"])
                if row["status"]!="eligible":continue
                if key in results:
                    row.update(status=results[key]["status"],prediction=deepcopy(results[key]),reused_exact_smiles_seed=True)
                    _write(output,report);continue
                remaining=deadline-time.monotonic()
                if (report["prediction_attempts"]>=plan["maximum_fresh_predictions_per_arm"]
                        or remaining<plan["fresh_prediction_timeout_seconds"]+10):
                    row["status"]="budget_excluded";row["remaining_seconds"]=remaining;_write(output,report);continue
                def attempted(value):
                    native_rows.append(value.copy());report["prediction_attempts"]+=1
                    _write(experiment.path / "native_results.json",native_rows);_write(output,report)
                prediction=_native_prediction(row["smiles"],len(results),row["seed"],template,experiment.path,cache,attempted)
                prediction=_retention_result(prediction,chemical_identities[row["smiles"]]);native_rows[-1]=prediction
                results[key]=prediction;row.update(status=prediction["status"],prediction=deepcopy(prediction),reused_exact_smiles_seed=False)
                _write(experiment.path / "native_results.json",native_rows);_write(output,report)
            if file_sha256(candidate_path)!=candidate_sha or file_sha256(child / "run.json")!=report["child_run_sha256"]:
                raise ValueError("Selected parent artifacts changed during checkpoint evaluation")
            unresolved={"not_attempted","eligible","budget_excluded","missing_state","failed"}
            report["status"]="completed" if not any(r["status"] in unresolved for r in report["memberships"]) else "incomplete"
            report["fresh_pose_run"]=str(experiment.path);report["seconds"]=time.monotonic()-started
            report["counts"]={status:sum(r["status"]==status for r in report["memberships"]) for status in sorted({r["status"] for r in report["memberships"]})}
            _write(experiment.path / "summary.json",report);_write(output,report)
            experiment.annotate(evaluation_complete=report["status"]=="completed")
    except Exception as error:
        report.update(status="failed",error=str(error),error_type=type(error).__name__,seconds=time.monotonic()-started)
        _write(output,report);raise
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--plan",type=Path,required=True);p.add_argument("--sha256",required=True);p.add_argument("--case",required=True);p.add_argument("--child",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args(argv)
    print(evaluate(a.plan,a.sha256,a.case,a.child,a.output),flush=True)


if __name__=="__main__":main()
