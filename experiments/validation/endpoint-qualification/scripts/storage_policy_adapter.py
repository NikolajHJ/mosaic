"""Explicit prospective adoption of shared live-quota policy; no scientific edits."""
import ast
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import subprocess
import time

shared=Path(__file__).with_name("storage_accounting.py")
if not shared.exists():shared=Path(__file__).resolve().parents[4]/"scripts/hpc/storage_accounting.py"
spec=importlib.util.spec_from_file_location("_shared_live_quota_policy",shared)
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)

def write(path,value):
    with Path(path).open("x",encoding="utf8") as f:json.dump(value,f,indent=2,allow_nan=False);f.write("\n")

def live_quota(parent):
    start=time.monotonic();commands=[];pools={}
    for index in range(1,7):
        argv=["beegfs-ctl","--getquota","--uid","306121","--storagepoolid="+str(index),"--csv",
              "--cfgFile=/etc/beegfs/beegfs-client4.conf"]
        r=subprocess.run(argv,capture_output=True,text=True,timeout=30)
        commands.append({"argv":argv,"returncode":r.returncode,"stdout":r.stdout,"stderr":r.stderr})
        policy.require(r.returncode==0,"Live quota command failed")
        pools[index]=policy.parse_pool(r.stdout,306121,"s234818")
    elapsed=time.monotonic()-start;policy.require(elapsed<=60,"Live quota observation exceeded60 seconds")
    result={"version":policy.VERSION,"observed_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":elapsed,
            "commands":commands,"pools":pools,"summary":policy.combine_pools(pools),"source":"storage-server exact-byte CSV"}
    path=Path(parent)/("live-quota-"+str(time.time_ns())+".json");write(path,result)
    return result,{"path":str(path),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

def embedding_budget(observation):
    main=next(i for i in observation["inventories"] if i["root"].endswith("/runs/source142-epoch36-guarded-embedding-long-attempt001"))
    policy.require(main["complete"] is True and main["nofollow"] is True,"Incomplete embedding inventory")
    root=Path(main["root"]);pilot=json.loads((root/"pilot/result.json").read_bytes())
    completed={};rates={};details=[]
    for case in ("case-001","case-002","case-003"):
        prefix="trajectories/"+case+"/";states=[];growing=0
        for row in main["rows"]:
            if not row["path"].startswith(prefix):continue
            rel=row["path"][len(prefix):]
            match=re.fullmatch(r"states/state-(\d{6})\.pt",rel)
            if match:states.append(int(match.group(1)))
            if rel.startswith(("states/","observations/","segments/","hard-cache/","best/")):
                growing+=row["bytes"]
        states.sort();policy.require(states and states==list(range(states[-1]+1)),"Embedding states not contiguous")
        completed[case]=states[-1];policy.require(completed[case]>=20,"Original embedding pilot missing")
        rates[case]=(growing+completed[case]-1)//completed[case]
        details.append({"case":case,"completed_updates":completed[case],"growing_output_bytes":growing,
                        "sampled_updates":completed[case],"measured_bytes_per_update_upper":rates[case]})
    result=policy.embedding_forecast(pilot["cases"],completed,rates)
    result.update(measured_current_output=details,original_remaining_bytes=observation["remaining_bytes"],
                  pilot_path=str(root/"pilot/result.json"),pilot_sha256=hashlib.sha256((root/"pilot/result.json").read_bytes()).hexdigest())
    return result

def replace(tree,before,after):
    old=ast.dump(ast.parse(before,mode="eval").body);new=ast.parse(after,mode="eval").body
    class Rewrite(ast.NodeTransformer):
        count=0
        def visit(self,node):
            if isinstance(node,ast.expr) and ast.dump(node)==old:
                self.count+=1;return ast.copy_location(deepcopy(new),node)
            return super().visit(node)
    transform=Rewrite();value=transform.visit(tree)
    policy.require(transform.count==1,"Storage integration shape changed: "+before)
    return ast.fix_missing_locations(value)

def install(rolling,plan,parent,source146=None):
    """Only newly sealed callers adopt this explicitly versioned resource policy."""
    policy.require(plan["storage_policy"]["version"]==policy.VERSION,"Explicit new resource-policy provenance required")
    unprofiled={};foreign={};original_parse=rolling.parse_scheduler
    def parse(raw):
        rows=original_parse(raw)
        if source146 is not None:
            import qualification_writer_capacity as observer
            foreign["observation"]=observer.observe(source146,rows)
            write(Path(parent)/("source146-live-policy-"+str(time.time_ns())+".json"),foreign["observation"])
        return rows
    rolling.parse_scheduler=parse
    classify=rolling.classify_jobs
    if source146 is not None:
        def classified(rows,known,own,source,accounted_external=()):
            return classify(rows,known,own,source,set(accounted_external)|set(foreign["observation"]["accounted_job_ids"]))
        rolling.classify_jobs=classified
    def unknown(row):
        policy.require(row["STAT"] in rolling.LIVE_STATES,"Unrecognized account state")
        if row["STAT"] in ("RUN","SSUSP","USUSP"):
            unprofiled[row["JOBID"]]=row
    tree=ast.Module([next(n for n in ast.parse(Path(rolling.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=="classify_jobs")],[])
    if "_runtime_namespace_owned" in rolling.__dict__:
        tree=replace(tree,"str(source) in command and jid in own_jobs","(str(source) in command and jid in own_jobs) or _runtime_namespace_owned(row, own_jobs, source)")
    tree=replace(tree,"c.require(row['STAT'] in ('PEND', 'PSUSP', 'WAIT'), 'Unrelated active account writer has no authenticated bounded declaration: ' + jid)","_policy_unknown(row)")
    rolling.__dict__["_policy_unknown"]=unknown
    exec(compile(tree,rolling.__file__,"exec"),rolling.__dict__)
    # classified() resolves the newly qualified function; preserve exact source146 ownership.
    new_classify=rolling.classify_jobs
    if source146 is not None:
        def classified_new(rows,known,own,source,accounted_external=()):
            return new_classify(rows,known,own,source,set(accounted_external)|set(foreign["observation"]["accounted_job_ids"]))
        rolling.classify_jobs=classified_new
    validate=rolling.c.validate_ledger
    def base(ledger,old_display_quota,disk,inodes):
        ledger=deepcopy(ledger)
        for goal in ledger["goals"]:
            if goal["id"]=="dbmol":
                goal["target_updates"]=30000;goal["later_updates_deferred"]=30000-goal["saved_updates"]-goal["next_updates"]
        rows,file_reserve=validate(ledger)
        measured,proof=live_quota(parent)
        scopes=[{"id":r["id"],"remaining_bytes":r["remaining_admitted_cap_bytes"],
                 "remaining_files":r["remaining_admitted_cap_files"]} for r in rows]
        external=[]
        if foreign:
            ob=foreign["observation"];external.append({"id":"source146","remaining_bytes":ob["remaining_bytes"],"remaining_files":ob["remaining_files"]})
        result=policy.assess(measured["summary"],disk,inodes,scopes,external,sorted(unprofiled))
        result.update(ledger=ledger,phase=ledger["phase"],source_manifest_sha256=ledger["source_manifest_sha256"],
            quota=measured,live_quota_proof=proof,scope_accounting=rows,resource_policy_changed_explicitly=True,
            historical_gate_results_changed=False,saved_results_changed=False,unprofiled_account_writers=list(unprofiled.values()),
            external_remaining_budgets=external,entire_tree_stale_charge_bytes=0,entire_tree_stale_charge_files=0,
            filesystem_inode_observation=inodes,filesystem_free_bytes=disk,
            future_admitted_bytes=sum(r["remaining_bytes"] for r in scopes),future_admitted_files=sum(r["remaining_files"] for r in scopes))
        for r in result["scope_accounting"]:
            r["existing_bytes_already_accounted_by_live_quota"]=r["actual_bytes"]
            r["entire_mutable_or_unproven_tree_stale_charge_bytes"]=0
            r["entire_mutable_or_unproven_tree_stale_charge_files"]=0
        return result
    rolling.c.assess=base
    def augment(result,observation,kind):
        result=dict(result);amount=observation["remaining_bytes"]
        if kind=="embedding":
            result["embedding_remaining_forecast"]=embedding_budget(observation)
            amount=result["embedding_remaining_forecast"]["remaining_bytes"]
        result["required_bytes"]+=amount;result["required_files"]+=observation["remaining_files"]
        result["external_remaining_budgets"].append({"id":kind,"remaining_bytes":amount,"remaining_files":observation["remaining_files"]})
        result["byte_margin"]=result["available_bytes"]-result["required_bytes"]
        result["file_margin"]=result["available_files"]-result["required_files"]
        result["status"]="passed" if result["byte_margin"]>=0 and result["file_margin"]>=0 else "failed"
        result["shared_"+kind+"_accounting"]=observation
        return result
    rolling.__dict__["_policy_augment"]=augment
    tree=ast.Module([next(n for n in ast.parse(Path(rolling.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=="_storage")],[])
    tree=replace(tree,"op.ROOT / 'storage-claims'","Path(op.runtime_root) / 'storage-claims'")
    for old,new in [("module.augment(result, shared)","_policy_augment(result, shared, 'vae')"),
                    ("embedding_module.augment(result, embedding)","_policy_augment(result, embedding, 'embedding')"),
                    ("probe_module.augment(result, probe)","_policy_augment(result, probe, 'runtime_probe')")]:tree=replace(tree,old,new)
    exec(compile(tree,rolling.__file__,"exec"),rolling.__dict__)
    return rolling
