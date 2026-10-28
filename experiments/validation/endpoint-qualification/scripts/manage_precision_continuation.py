"""Stdlib-only preparation of the new separate-reference continuation capsule.

Preparation may collect one read-only metadata packet; it never starts a model,
test, optimizer, scheduler submission or stage/upload. Existing attempt outputs
are refused. Staging and all numerical checks belong to allocated orchestration.
"""
import argparse
import ast
import base64
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile


REPO = Path(__file__).resolve().parents[4]
OWNER = REPO / "experiments/validation/endpoint-qualification"
TAG = "20261028-precision-continuation-21"
RUN = OWNER / "runs" / TAG
LOCAL = REPO / "results/hpc" / TAG
BASE = "/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01"
ROOT = BASE + "/nonchem-precision-continuation-21"
SOURCE = BASE + "/nonchem-precision-release-21/source"
PYTHON = "/work3/s234818/mosaic/envs/protenix-tiny/bin/python"
TORCH_PYTHON = "/work3/s234818/latentmol/env/boltz311/bin/python"
BASE_TAG = "20261027-continuation-measurement-09"
PRO_TAG = "20261028-protenix-endpoint-fix-09"
DB_TAG = "20261028-dbmol-endpoint-recovery-13"
PLAN_SENTINEL = "${OWN_PLAN_SHA256}"


def require(value, message):
    if not value: raise ValueError(message)


def raw(value): return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
def sha(value): return hashlib.sha256(value).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def local_capsule(tag, remote):
    directory = REPO / "results/hpc" / tag / "source01"
    manifest_bytes = (directory / "source_manifest.json").read_bytes(); manifest = json.loads(manifest_bytes)
    members = {name: (directory / name).read_bytes() for name in manifest["files"]}
    require(all(sha(members[name]) == digest for name, digest in manifest["files"].items()),
            "Previously authenticated local capsule bytes changed: " + tag)
    preparation = read(directory / "preparation.json")
    require(preparation["manifest_sha256"] == sha(manifest_bytes) and preparation["files"] == len(members),
            "Previous local preparation/source pin differs")
    return members, manifest, {"path": remote, "files": len(members), "manifest_sha256": sha(manifest_bytes)}


def bind_member(members, remote, name):
    data = members[name]
    return {"path": remote + "/" + name, "bytes": len(data), "sha256": sha(data)}


def deduplicate(rows):
    found = {}
    for row in rows:
        if row["path"] in found: require(found[row["path"]] == row, "Conflicting historical binding: " + row["path"])
        found[row["path"]] = deepcopy(row)
    return [found[key] for key in sorted(found)]


def qualified(tag, source, job, assessment):
    record = read(OWNER / "runs" / tag / "native-verification.json")
    require(record["job_id"] == job and record["original_results_changed"] is False
            and record["optimizer_jobs_submitted"] is False
            and record["source_manifest_sha256"] == source["manifest_sha256"], "Qualified native audit identity differs")
    rows = record["proofs"]
    def one(suffix):
        selected = [row for row in rows if row["path"].endswith(suffix)]
        require(len(selected) == 1, "Qualified actual proof absent/ambiguous: " + suffix)
        return selected[0]
    parent = BASE + ("/protenix-endpoint-fix-09/gpu" if tag == PRO_TAG else "/dbmol-endpoint-recovery-13/gpu")
    return {"source": source, "job_id": job, "assessment": one(assessment),
            "submission": one(parent + "/submission.json"), "stdout": one("/stdout." + job + ".log"),
            "proofs": deepcopy(rows)}


def metadata_program():
    return "BASE=" + repr(BASE) + "\nROOT=" + repr(ROOT) + "\nSOURCE=" + repr(SOURCE) + "\n" + r'''
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
base=Path(BASE);parent=base/'nonchem-endpoint-continuation-08/dbmol-segment-05000'
assert not Path(ROOT).exists() and not Path(SOURCE).parent.exists(),'Existing/unknown new attempt; inspect before recovery'
def binding(path):
 path=Path(path);assert path.is_absolute() and path.is_file() and all(not p.is_symlink() for p in (path,*path.parents))
 with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':digest}
def saved(path):
 b=Path(path).read_bytes();return {'binding':binding(path),'value':json.loads(b)}
invocation=saved(parent/'payload/endpoint-driver-invocation.json');argv=invocation['value']['argv']
assert argv[1]==str(base/'boltz-dbmol-pilot-release-02/source/scripts/dbmol_driver.py')
submission=saved(parent/'submission.json');assert submission['value']['job_id']=='29830999'
study=parent/'payload/study';summary=json.loads((study/'summary.json').read_bytes())
assert summary['status']=='failed' and summary['job_id']=='29830999'
lsf=(parent/'stdout.29830999.log').read_text();compact=''.join(lsf.split())
assert 'Subject:Job29830999:' in compact and 'Exitedwithexitcode1.' in compact and 'inqueue<gpul40s>' in compact
terminal=saved(study/'terminal_gradient_checks/report.json')
failed14=base/'nonchem-precision-continuation-14';source14=base/'nonchem-precision-release-14/source'
assert binding(source14/'source_manifest.json')['sha256']=='4c3f74166a57f35bd7c4eca8ffa58e53c0a12e7d232553acb70133d283d099ab'
jobs14=json.loads((failed14/'jobs.json').read_bytes());assert jobs14=={'cpu':'29838955','after-cpu':'29838956'}
assert not list(failed14.glob('*validate*')) and not list(failed14.glob('*segment*'))
lsf14=''.join((failed14/'cpu/stdout.29838955.log').read_text().split())
assert 'Subject:Job29838955:' in lsf14 and 'Exitedwithexitcode1.' in lsf14
failure14=saved(failed14/'cpu/payload/protenix-reference-startup-invocation.json')
assert failure14['value']['returncode']==1 and 'Closed Protenix reference specification differs' in failure14['value']['stderr']
proofs14=[binding(p) for p in (failed14/'cpu').rglob('*') if p.is_file()]
proofs14.append(binding(failed14/'jobs.json'))
for p in (failed14/'after-cpu').glob('*.log'):
 if p.stat().st_size:proofs14.append(binding(p))
proofs14.append(binding(failed14/'after-cpu/submission.json'))
failed={'job_id':'29830999','parent':str(parent),'endpoint':binding(study/'case-0/snapshots/step-2500.npz'),
 'summary':binding(study/'summary.json'),'stdout':binding(parent/'stdout.29830999.log'),'stderr':binding(parent/'stderr.29830999.log'),
 'states':binding(study/'case-0/states.jsonl'),'updates':binding(study/'case-0/updates.jsonl'),
 'submission':submission['binding'],'terminal':terminal['binding']}
inputs={'original_driver_argv':argv,'original_driver_invocation':invocation['binding'],
 'original_native_report':binding(study/'native_gradient_checks/report.json')}
for flag in ('--cpu-certificate','--reference','--pocket','--ready'):
 assert argv.count(flag)==1
 inputs[flag[2:].replace('-','_')]=binding(argv[argv.index(flag)+1])
for flag,key in (('--cpu-certificate-sha256','cpu_certificate'),('--pocket-sha256','pocket'),('--ready-sha256','ready')):
 assert argv[argv.index(flag)+1]==inputs[key]['sha256']
print(json.dumps({'observed_utc':datetime.now(timezone.utc).isoformat(),'failed_parent':failed,'inputs':inputs,
 'terminal':terminal,'invocation':invocation,'submission':submission,
 'engineering_predecessor':{'source':{'path':str(source14),'files':35,'manifest_sha256':binding(source14/'source_manifest.json')['sha256']},
 'cpu_job_id':'29838955','controller_job_id':'29838956','proofs':proofs14,'failure':failure14,
 'native_or_optimizer_submitted':False}},allow_nan=False))
'''


def collect_metadata():
    LOCAL.mkdir(parents=True, exist_ok=True)
    helper = module(OWNER / "scripts/manage_protenix_precision.py", "_precision14_metadata_transport")
    helper.LOCAL = LOCAL
    return helper.command(metadata_program(), "input-metadata01", 180)


def metadata_file(path):
    path = Path(path); value = read(path)
    receipt = read(path.parent / "receipt.json")
    require(receipt["returncode"] == 0 and receipt["stdout_sha256"] == sha(path.read_bytes()),
            "Read-only metadata packet is not authenticated by its completed transport")
    return value


def _scientific_argv(argv):
    replace = {"--mode", "--output", "--offset", "--segment-steps", "--resume-snapshot", "--resume-sha256", "--ready", "--ready-sha256"}
    result = []; index = 0
    while index < len(argv):
        if argv[index] in replace: index += 2
        else: result.append(argv[index]); index += 1
    return result


def make_plan(base_members, pro_members, pro_source, db_members, db_source, metadata):
    old = json.loads(base_members["plan.json"]); db_qualified = json.loads(db_members["plan.json"])
    pro = deepcopy(next(arm for arm in old["arms"] if arm["id"] == "protenix"))
    pro.update(segment_updates=30000, segment_steps=30000, offsets=[20000])
    db = deepcopy(db_qualified["original_arm"])
    require(metadata["failed_parent"]["endpoint"] == db_qualified["boundary"]
            and _scientific_argv(metadata["inputs"]["original_driver_argv"]) == _scientific_argv(db["original_argv"]),
            "Saved7500 boundary or original scientific command changed")
    db.update(start_global=7500, global_start=7500, target_global=10000, global_target=10000,
              segment_updates=2500, segment_steps=2500, offsets=[7500],
              parent_snapshot=deepcopy(db_qualified["boundary"]), failed_parent=metadata["failed_parent"])
    # The original scientific invocation has the established {'args': ...}
    # receipt consumed by the unchanged driver. The immediate parent wrapper's
    # {'argv': ...} receipt is separate ancestry, never a replacement for it.
    db["immediate_parent_ancestry"] = {
        "driver_invocation": deepcopy(metadata["inputs"]["original_driver_invocation"]),
        "driver_argv": deepcopy(metadata["inputs"]["original_driver_argv"]),
        "native_report": deepcopy(metadata["inputs"]["original_native_report"])}
    objectives = ("dbmol_affinity_loss", "dbmol_contact_loss", "dbmol_confidence_loss", "dbmol_total_loss")
    db["original_directions"] = {name: deepcopy(db_qualified["original_directions"]) if name == "dbmol_affinity_loss"
                                  else {"X": [], "E_upper": []} for name in objectives}
    db["expected"] = {"saved_parent_scientific_status": "failed", "parent_absolute_step": 7500,
        "original_terminal_report": metadata["terminal"]["value"],
        "historical_5000_expectations_preserved": db["expected"]}
    pro_spec = module(OWNER / "scripts/protenix_reference_worker.py", "_precision14_pro_spec").reference_spec()
    db_policy = module(REPO / "results/hpc" / DB_TAG / "source01/scripts/dbmol_endpoint_precision.py", "_precision14_db_policy_spec").finer_policy_spec()
    pro_policy = bind_member(pro_members, pro_source["path"], "scripts/protenix_precision_admission.py")
    db_policy_binding = bind_member(db_members, db_source["path"], "scripts/dbmol_endpoint_precision.py")
    environment = {"JAX_ENABLE_X64": "1", "JAX_PLATFORMS": "cuda,cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                   "XLA_FLAGS": "--xla_gpu_autotune_level=0", "PYTHONDONTWRITEBYTECODE": "1"}
    shared = {"max_session_bytes": 64 * 1024**2, "timeout_seconds": 5400,
              "co_residency_required": True, "runtime_fallback_allowed": False}
    reference = {"arms": {
        "protenix": {**shared, "policy_kind": "protenix-original-fp64", "historical_global": 20000,
            "historical_original_directions": deepcopy(pro["original_directions"]),
            "historical_failure": deepcopy(pro["failed_parent"]["terminal"]), "policy_module": pro_policy,
            "command_prefix": [PYTHON, "-u", SOURCE + "/scripts/protenix_reference_worker.py", "--reference-plan", SOURCE + "/plan.json",
                               "--reference-plan-sha256", PLAN_SENTINEL], "environment": environment,
            "expected_identity": {"identity": {"model_digest": pro_spec["original_model_digest"],
                "context_digest": pro_spec["original_context_digest"], "pocket_definition_sha256": pro_spec["pocket_definition_sha256"],
                "qualified_source_manifest_sha256": pro_source["manifest_sha256"],
                "scientific_source_manifest_sha256": pro["source"]["manifest_sha256"],
                "device": {"platform": "gpu", "id": 0, "kind": "NVIDIA L40S"}}}},
        "dbmol": {**shared, "policy_kind": "dbmol-finer-fp64-affinity", "historical_global": 7500,
            "historical_original_directions": deepcopy(db_qualified["original_directions"]),
            "historical_failure": deepcopy(db_qualified["original_admission"]), "policy_module": db_policy_binding,
            "precision_policy": db_policy,
            "command_prefix": [TORCH_PYTHON, "-u", SOURCE + "/scripts/dbmol_reference.py", *db_qualified["reference_argv"][3:]],
            "environment": {"PYTHONPATH": db["source"]["path"] + "/src:" + db["source"]["path"] + "/scripts",
                            "PYTHONDONTWRITEBYTECODE": "1"},
            "expected_identity": {"promotion": {"original_digest": "395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16"}}}}}
    plan = {"version": "precision-reference-continuation-plan-v1", "admission_policy": "precision-reference-continuation-policy-v1",
        "logit_policy": "prospective-fixed-reference-logit-admission-v1", "user_authorized_prospective_checks_before_children": True,
        "original_failures_preserved": True, "original_results_changed": False, "optimization_settings_changed": False,
        "monitor_remains_paused": True, "matrix_stride": 20, "loss_stride": 1,
        "root": ROOT, "control_root": ROOT, "source_root": SOURCE, "arms": [pro, db],
        "dispatch_arms": ["protenix", "dbmol"], "automatic_dispatch_arms": ["protenix", "dbmol"],
        "preserved_failed_admissions": deepcopy(old["blocked_arms"]),
        "precision_adapter_version": "precision-reference-continuation-adapter-v1",
        "precision_policy_by_arm": {"protenix": {"protenix_contact_loss": "protenix-separate-precision-admission-v1"},
            "dbmol": {"dbmol_affinity_loss": "dbmol-separate-precision-finer-admission-v1",
                      "dbmol_contact_loss": "prospective-endpoint-admission-v3", "dbmol_confidence_loss": "prospective-endpoint-admission-v3",
                      "dbmol_total_loss": "prospective-endpoint-admission-v3"}},
        "protenix_reference": pro_spec, "reference_configuration": reference,
        "base_driver": bind_member(base_members, BASE + "/nonchem-endpoint-release-09/source", "scripts/continuation_driver.py"),
        "qualified_admissions": {
            "protenix": qualified(PRO_TAG, pro_source, "29838596", "/gpu/payload/assessment.json"),
            "dbmol": qualified(DB_TAG, db_source, "29838712", "/gpu/payload/assessment.json")},
        "metadata_observed_utc": metadata["observed_utc"],
        "engineering_predecessor": metadata["engineering_predecessor"],
        "scope": "Fresh complete entry/nonraw/null/component/pilot/runtime/live-reference/startup/storage before child; original failures remain failed"}
    previous = REPO / "results/hpc/20261028-precision-continuation-16"
    current = read(previous / "current-metadata01/observation.json")
    details = read(previous / "current-metadata02/observation.json")
    for packet, field in ((current, "source15_proofs"), (current, "teacher_records"), (details, "records")):
        for record in packet[field]:
            data = record["text"].encode("utf8")
            require(len(data) == record["binding"]["bytes"] and sha(data) == record["binding"]["sha256"],
                    "Completed read-only metadata packet text differs from its byte binding")
    rows = json.loads(current["commands"][0]["stdout"])["RECORDS"]
    teacher = next(row for row in rows if row["JOBID"] == "29839884")
    report = next(row for row in rows if row["JOBID"] == "29796558")
    teacher_manifest = next(row["binding"] for row in current["teacher_records"]
                            if row["binding"]["sha256"] == "a15c8bb2a8ac7e9e12a65c54079e14c994e53b119b9a59e1a9788b5044d8686f")
    plan["launch_predecessor"] = {"source": {"path": BASE + "/nonchem-precision-release-15/source", "files": 35,
        "manifest_sha256": current["source15_manifest"]["sha256"]},
        "failed_jobs": {"protenix": "29839086", "dbmol": "29839088"},
        "proofs": [row["binding"] for row in current["source15_proofs"] if row["binding"]["bytes"] > 0],
        "original_results_changed": False, "failures_preserved": True}
    retired = read(REPO / "results/hpc/20261028-precision-continuation-17/retirement-proof01/observation.json")["assessment"]
    require(retired["value"]["status"] == "superseded_unstarted_native_jobs_and_dispatchers_intentionally_retired"
            and retired["value"]["optimizer_children"] == []
            and all(not p for p in retired["value"]["validation_study_presence"].values()),
            "Old small-chunk dispatch or native work is not safely retired")
    plan["superseded_allocation_plan"] = {"source": retired["value"]["source"],
        "retirement_assessment": retired["binding"], "original_results_changed": False,
        "CPU_pass_preserved": True, "reason": "User prefers full remaining work with measured reservations"}
    failed17 = read(REPO / "results/hpc/20261028-precision-continuation-18/failure17-01/observation.json")
    plan["engineering_fixture_predecessor"] = {"source": failed17["value"]["source"],
        "assessment": failed17["assessment"]["binding"], "original_results_changed": False}
    failed18 = read(LOCAL / "failure18-proof01/observation.json")
    plan["resource_failure_predecessor"] = {"source": failed18["value"]["source"],
        "assessment": failed18["assessment"]["binding"], "original_results_changed": False}
    plan["shared_vae"] = read(LOCAL / "shared-vae-profile.json")
    def deferred(name, saved, target):
        return {"id": name, "saved_updates": saved, "target_updates": target, "next_updates": 0,
                "next_scope_id": None, "later_updates_deferred": target - saved,
                "readmission_before_later_execution": True}
    plan["rolling_storage_policy"] = {"version": "mosaic-rolling-work3-adapter-v1",
        "control_cap_bytes": 2 * 1024**3, "control_cap_files": 30000,
        "release_cap_bytes": 64 * 1024**2, "release_cap_files": 3000,
        "known_mosaic_jobs": [
            {"job_id": "29839884", "command": teacher["COMMAND"],
             "root": "/work3/s234818/mosaic/runs/20261028-protenix-defog-30000-time-ramp-01",
             "cap_bytes": 4 * 1024**3, "cap_files": 30000, "role": "defog",
             "release_root": "/work3/s234818/mosaic/releases/20261028-protenix-defog-30000-time-ramp-01",
             "release_cap_bytes": 64 * 1024**2, "release_cap_files": 3000, "source_manifest": teacher_manifest},
            {"job_id": "29796558", "command": report["COMMAND"], "root": BASE + "/lrreset-analysis-03/combined-report",
             "cap_bytes": 2 * 1024**3, "cap_files": 20000, "role": "cache", "release_root": None,
             "release_cap_bytes": None, "release_cap_files": None, "source_manifest": details["records"][0]["binding"]}],
        "deferred_goals": [deferred("plain-boltz", 20000, 30000), deferred("protenix-chemistry", 10000, 50000),
                           deferred("boltz-chemistry", 10000, 30000)]}
    terminal = read(REPO / "results/hpc/20261028-precision-continuation-19/terminal-proof01/observation.json")
    plan["rolling_storage_policy"]["known_mosaic_jobs"][0]["terminal_proof"] = terminal["terminal_defog"]["binding"]
    plan["preflight_file_predecessor"] = {"source": terminal["failure19"]["value"]["source"],
        "assessment": terminal["failure19"]["binding"]}
    conflict = read(REPO / "results/hpc/20261028-precision-continuation-20/reader-conflict-proof01/observation.json")
    plan["home_report_reader_pin"] = "4db7ca7c58a8481cd4e17c4f8a2df4aafad3e94e190408a2a4c0db30da168165"
    plan["report_conflict_predecessor"] = {"source": conflict["value"]["source"], "assessment": conflict["assessment"]["binding"]}
    return plan


def startup_adoption(source, *, tests=False):
    """Change only declared metadata startup path literals; reverse the AST exactly."""
    before = ast.parse(source); changed = deepcopy(before); counts = {}
    old_root = BASE + "/dbmol-endpoint-recovery-13"
    replacements = {old_root + "/cpu/payload/reference-startup.json": ROOT + "/cpu/payload/dbmol-reference-startup.json"}
    if tests:
        replacements = {"cpu/payload/reference-startup.json": "cpu/payload/dbmol-reference-startup.json",
            "dbmol-endpoint-recovery-13": "nonchem-precision-continuation-21",
            old_root + "/gpu/payload/reference-startup.json": ROOT + "/gpu/payload/dbmol-reference-startup.json"}
    class Change(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in replacements:
                counts[node.value] = counts.get(node.value, 0) + 1
                return ast.copy_location(ast.Constant(replacements[node.value]), node)
            return node
    changed = Change().visit(changed); require(counts == {key: 1 for key in replacements}, "Unique startup metadata literals differ")
    inverse = {value: key for key, value in replacements.items()}
    class Restore(ast.NodeTransformer):
        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant(inverse[node.value]), node) if isinstance(node.value, str) and node.value in inverse else node
    restored = Restore().visit(deepcopy(changed))
    require(ast.dump(restored, include_attributes=False) == ast.dump(before, include_attributes=False),
            "Startup adoption changed more than metadata path literals")
    encoded = (ast.unparse(ast.fix_missing_locations(changed)) + "\n").encode()
    return encoded, {"status": "exact_startup_metadata_AST_adoption", "old_sha256": sha(source), "new_sha256": sha(encoded),
                     "changed_literals": replacements, "AST_reversal_exact": True, "numerical_AST_unchanged": True}


def prepare(counts_path, metadata_path=None):
    require((RUN / "README.md").is_file(), "Write the new run record before preparation")
    require(not (LOCAL / "source01").exists() and not (RUN / "plan.json").exists()
            and not (RUN / "source-manifest.json").exists() and not (RUN / "preparation.json").exists(),
            "Existing prepared/unknown attempt cannot be replaced")
    base_members, base_manifest, base_source = local_capsule(BASE_TAG, BASE + "/nonchem-endpoint-release-09/source")
    pro_members, pro_manifest, pro_source = local_capsule(PRO_TAG, BASE + "/protenix-endpoint-fix-release-09/source")
    db_members, db_manifest, db_source = local_capsule(DB_TAG, BASE + "/dbmol-endpoint-recovery-release-13/source")
    require(len(base_members) == 19 and pro_source["manifest_sha256"] == "f2043af74b8fa9ff485a44b80fcf96454bf8585868859389828c37f0073a3664"
            and db_source["manifest_sha256"] == "0f82d44bbb65ea957a7e5faa9e9837aba7fc1f2256f7344a96606b11ef3d664d", "Exact qualified predecessor scope differs")
    metadata = collect_metadata() if metadata_path is None else metadata_file(metadata_path)
    plan = make_plan(base_members, pro_members, pro_source, db_members, db_source, metadata)
    members = dict(base_members)
    new_files = ["scripts/precision_continuation_driver.py", "scripts/precision_continuation_hooks.py",
        "scripts/precision_continuation_ops.py", "scripts/precision_continuation_stage.lsf", "scripts/protenix_reference_worker.py",
        "scripts/recovery_capacity.py", "tests/test_precision_continuation_driver.py", "tests/test_precision_continuation_hooks.py",
        "tests/test_precision_continuation_ops.py", "tests/test_protenix_reference_worker.py", "tests/test_recovery_capacity.py",
        "scripts/cuda_ownership.py", "tests/test_cuda_ownership.py", "scripts/segment_capacity.py", "tests/test_segment_capacity.py",
        "scripts/rolling_capacity.py", "tests/test_rolling_capacity.py", "tests/test_runtime_reservations.py", "scripts/vae_shared_capacity.py", "tests/test_vae_shared_capacity.py", "scripts/home_report_reader.py", "tests/test_home_report_reader.py"]
    for name in new_files: members[name] = (OWNER / name).read_bytes().replace(b"\r\n", b"\n")
    for name in ("scripts/dbmol_endpoint_precision.py", "tests/test_dbmol_endpoint_precision.py"):
        members[name] = db_members[name]
    startup_audit = {}
    for name in ("scripts/dbmol_reference.py", "tests/test_dbmol_reference.py"):
        members[name], startup_audit[name] = startup_adoption(db_members[name], tests=name.startswith("tests/"))
    members["README.md"] = (RUN / "README.md").read_bytes().replace(b"\r\n", b"\n")
    members["plan.json"] = raw(plan)
    members["startup-adoption.json"] = raw(startup_audit)
    counts = read(counts_path)
    modules = {"tests." + Path(name).stem for name in members if name.startswith("tests/test_") and name.endswith(".py")}
    require(set(counts) == modules and all(type(count) is int and count > 0 for count in counts.values()), "Explicit full CPU fixture module counts differ")
    for name, data in members.items():
        if name.endswith(".py"): ast.parse(data, filename=name)
    proofs = list(base_manifest["preserved_proofs"]) + list(pro_manifest["preserved_proofs"])
    proofs += json.loads(db_members["plan.json"])["protected_inputs"]
    for arm in plan["arms"]:
        proofs.append(arm["config"])
        for group in (arm["inputs"], arm["failed_parent"], arm.get("immediate_parent_ancestry", {})):
            proofs += [row for row in group.values() if isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"}]
    proofs.append(plan["base_driver"])
    proofs += plan["engineering_predecessor"]["proofs"]
    proofs += plan["launch_predecessor"]["proofs"]
    proofs.append(plan["superseded_allocation_plan"]["retirement_assessment"])
    proofs.append(plan["engineering_fixture_predecessor"]["assessment"])
    proofs.append(plan["resource_failure_predecessor"]["assessment"])
    proofs.append(plan["preflight_file_predecessor"]["assessment"])
    proofs.append(plan["report_conflict_predecessor"]["assessment"])
    proofs += [row["terminal_proof"] for row in plan["rolling_storage_policy"]["known_mosaic_jobs"] if "terminal_proof" in row]
    proofs += [plan["shared_vae"][key] for key in ("plan", "archive", "manifest", "quota_before")]
    proofs += [row["source_manifest"] for row in plan["rolling_storage_policy"]["known_mosaic_jobs"]]
    for row in plan["qualified_admissions"].values(): proofs += row["proofs"]
    parents = deduplicate(list(base_manifest["parent_sources"]) + list(pro_manifest["parent_sources"])
        + list(db_manifest["parent_sources"]) + [base_source, pro_source, db_source, plan["engineering_predecessor"]["source"],
                                               plan["launch_predecessor"]["source"], plan["superseded_allocation_plan"]["source"],
                                               plan["engineering_fixture_predecessor"]["source"], plan["resource_failure_predecessor"]["source"], plan["preflight_file_predecessor"]["source"], plan["report_conflict_predecessor"]["source"]])
    manifest = {"version": "precision-reference-continuation-source-v1", "files": {name: sha(data) for name, data in sorted(members.items())},
        "cpu_counts": counts, "parent_sources": parents, "preserved_proofs": deduplicate(proofs),
        "preserved_failed_admissions": plan["preserved_failed_admissions"], "dispatch_arms": plan["dispatch_arms"],
        "original_results_changed": False, "original_failures_preserved": True, "optimization_settings_changed": False,
        "monitor_remains_paused": True, "base_compatibility": {"source": base_source,
            "all_original_script_and_test_bytes_unchanged": all(members[name] == data for name, data in base_members.items()
                if name.startswith(("scripts/", "tests/")))}, "startup_adoption": startup_audit,
        "scope": "Prospective complete child gates only; no original scientific recertification"}
    require(manifest["base_compatibility"]["all_original_script_and_test_bytes_unchanged"], "Inherited compatibility bytes changed")
    members["source_manifest.json"] = raw(manifest)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in sorted(members.items()):
            info = tarfile.TarInfo(name); info.mode = 0o444; info.size = len(data); info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    archive_bytes = buffer.getvalue()
    preparation = {"status": "prepared_not_validated_not_staged", "files": len(manifest["files"]),
        "manifest_sha256": sha(members["source_manifest.json"]), "plan_sha256": sha(members["plan.json"]),
        "archive_bytes": len(archive_bytes), "archive_sha256": sha(archive_bytes), "cpu_counts": counts,
        "jobs_submitted": False, "optimization_settings_changed": False, "original_results_changed": False}
    LOCAL.mkdir(parents=True, exist_ok=True); out = LOCAL / "source01"; out.mkdir(exist_ok=False)
    for name, data in members.items():
        target = out / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    (out / "source.tar.gz").write_bytes(archive_bytes); (out / "preparation.json").write_bytes(raw(preparation))
    for name, data in (("plan.json", members["plan.json"]), ("source-manifest.json", members["source_manifest.json"]),
                       ("preparation.json", raw(preparation)), ("startup-adoption.json", members["startup-adoption.json"])):
        with (RUN / name).open("xb") as stream: stream.write(data)
    print(json.dumps(preparation, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare",)); parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args(); prepare(args.counts, args.metadata)


if __name__ == "__main__": main()
