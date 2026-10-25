"""Prepare a fresh immutable saved-report capsule; never connect or submit.

The source03 archive is the immutable template. Only its bound plan and the
controller's root/source/job-prefix/archive-edition literals may change. The
generated start_prepared.py is a separate, explicit one-shot CPU-only action.
This helper does not poll HPC, update a viewer, or launch optimization jobs.
"""

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import tarfile


REPO = Path(__file__).resolve().parents[4]
LOCAL = REPO / "results/hpc/20261025-jq1-live-report-01"
TEMPLATE = LOCAL / "source03"
ARCHIVE_BYTES = 51389
ARCHIVE_SHA = "51459284779b90baf8f22d3effb7d33708d7c96d8ae5e424c73ccd13c3eeebd9"
MANIFEST_SHA = "ee69381d752bf32f3ae55cf76f69ae1330acf5b20c52ddc7c9ee3cd7f711218c"
PREFIX_SHA = "2094d1cfcf43eee3dc2210e8464375007c3a306835a42a2c8225a7cf94a8aeab"
KNOWN_ARMS = {"protenix", "boltz", "dbmol", "protenix_noisy"}
PLAN_VERSION = "jq1-live-saved-report-v1"
EXPECTED_COUNTS = {"tests.test_qualification": 19, "tests.test_report_data": 58,
                   "tests.test_report_ops": 13, "tests.test_viewer": 33}
BASE = "/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01"
PYTHON = "/work3/s234818/mosaic/envs/protenix-tiny/bin/python"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read_json(raw):
    def reject(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(raw, parse_constant=reject)


def archive_members(raw, expected_manifest):
    """Byte-only archive verification; never import or execute capsule code."""
    members = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as source:
        for member in source.getmembers():
            name = PurePosixPath(member.name)
            require(member.isfile() and not name.is_absolute() and ".." not in name.parts
                    and "\\" not in member.name and name.as_posix() == member.name,
                    "Unsafe/nonregular source archive member")
            require(member.name not in members, "Duplicate source archive member")
            members[member.name] = source.extractfile(member).read()
    require("source_manifest.json" in members and digest(members["source_manifest.json"]) == expected_manifest,
            "Exact source03 manifest changed")
    manifest = read_json(members["source_manifest.json"])
    require(set(members) == set(manifest["files"]) | {"source_manifest.json"},
            "Unlisted/missing source archive member")
    for name, expected in manifest["files"].items():
        require(digest(members[name]) == expected, "Source member changed: " + name)
        if name.endswith(".py"):
            ast.parse(members[name], filename=name)
    return members, manifest


def binding(row):
    require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"}, "Incomplete file binding")
    path = PurePosixPath(row["path"])
    require(path.is_absolute() and ".." not in path.parts and "\\" not in row["path"]
            and path.as_posix() == row["path"], "Noncanonical saved-file path")
    require(row["path"].startswith(("/work3/s234818/mosaic/", "/zhome/95/e/206121/mosaic-runs/")),
            "Refresh reads only the Mosaic saved-result trees")
    require(type(row["bytes"]) is int and row["bytes"] >= 0
            and re.fullmatch("[a-f0-9]{64}", row["sha256"]) is not None, "Invalid bytes/SHA binding")
    return row


def natural(value, name):
    require(type(value) is int and value >= 0, "Explicit nonnegative integer required: " + name)
    return value


def fresh_plan(raw, original, edition):
    plan = read_json(raw)
    require(plan.get("version") == PLAN_VERSION and plan.get("edition") == edition, "Wrong plan version/edition")
    require(plan.get("monitor_remains_paused") is True and plan.get("automatic_optimizer_submission") is False,
            "A refresh cannot resume monitoring or optimizer submission")
    captured = datetime.fromisoformat(plan["captured_at"].replace("Z", "+00:00"))
    require(captured.tzinfo is not None and 0 <= (datetime.now(timezone.utc) - captured).total_seconds() <= 4 * 3600,
            "Fresh timezone-aware plan capture within four hours is required")
    arms = plan.get("arms")
    require(isinstance(arms, list) and len(arms) == 4
            and {arm["id"] for arm in arms} == KNOWN_ARMS, "Exactly the four known saved-result arms are required")
    old = {arm["id"]: arm for arm in original["arms"]}
    proofs = {}
    for arm in arms:
        baseline = old[arm["id"]]
        for field in ("kind", "rate", "global_start", "global_target", "target_updates"):
            require(arm.get(field) == baseline[field], "Refresh changed scientific settings/range: " + field)
        require(isinstance(arm.get("label"), str) and isinstance(arm.get("status_detail"), str), "Explicit labels/status detail required")
        expected = arm.get("expected_counts")
        require(isinstance(expected, dict) and set(expected) == {"committed_updates", "scalar_states", "retained_frames", "latest_global_step"},
                "Each arm requires complete explicit expected_counts")
        for key, value in expected.items():
            natural(value, arm["id"] + "." + key)
        for alias, key in (("expected_committed_updates", "committed_updates"),
                           ("expected_retained_frames", "retained_frames")):
            require(alias not in arm or arm[alias] == expected[key], "Expected-count alias differs")
            arm[alias] = expected[key]
        require(expected["latest_global_step"] == arm["global_start"] + expected["committed_updates"]
                and expected["latest_global_step"] <= arm["global_target"], "Absolute expected progress/range differs")
        segments = arm.get("segments")
        require(isinstance(segments, list), "Explicit segment list is required")
        offset, updates, frames = arm["global_start"], 0, 0
        permitted = {(row["source_manifest_sha256"], json.dumps(row["config"], sort_keys=True))
                     for row in baseline["segments"]}
        for index, segment in enumerate(segments):
            require(isinstance(segment.get("id"), str) and re.fullmatch("[a-z][a-z0-9_-]*", segment["id"]), "Unsafe saved segment identity")
            require(segment.get("global_offset") == offset and type(segment.get("requested_updates")) is int
                    and segment["requested_updates"] > 0, "Segment budget/absolute offset differs")
            require(segment.get("scientific_status") in ("passed", "failed"),
                    "Source03 controller supports original terminal saved segments only; running data needs a separate validated controller")
            require(re.fullmatch("[1-9][0-9]*", str(segment.get("job_id", ""))) is not None
                    and isinstance(segment["job_id"], str), "Exact original job ID is required")
            require((segment.get("source_manifest_sha256"), json.dumps(segment.get("config"), sort_keys=True)) in permitted,
                    "Refresh changed the original scientific source/config")
            counts = segment.get("expected_counts")
            require(isinstance(counts, dict) and set(counts) == {"committed_updates", "scalar_states", "retained_frames"},
                    "Each segment requires complete explicit expected_counts")
            for key, value in counts.items():
                natural(value, segment["id"] + "." + key)
            require(counts["scalar_states"] == counts["committed_updates"] + 1
                    and 0 <= counts["committed_updates"] <= segment["requested_updates"]
                    and 1 <= counts["retained_frames"] <= counts["scalar_states"], "Expected segment prefix/counts differ")
            if segment["scientific_status"] == "passed":
                require(counts["committed_updates"] == segment["requested_updates"], "Passed original segment has an incomplete expected budget")
            for key in ("states", "updates", "summary", "config", "retained_index", "stdout", "stderr"):
                row = binding(segment[key])
                require(row["path"] not in proofs or proofs[row["path"]] == row, "Conflicting bindings for one original file")
                proofs[row["path"]] = row
            if arm["kind"] == "fixed":
                row = binding(segment["case_report"])
                require(row["path"] not in proofs or proofs[row["path"]] == row, "Conflicting original case binding")
                proofs[row["path"]] = row
            for key in ("case_directory", "anchor_directory"):
                path = PurePosixPath(segment[key])
                require(path.is_absolute() and ".." not in path.parts and "\\" not in segment[key]
                        and segment[key].startswith(("/work3/s234818/mosaic/", "/zhome/95/e/206121/mosaic-runs/")), "Invalid original anchor")
            offset += counts["committed_updates"]
            updates += counts["committed_updates"]
            frames += counts["retained_frames"] - int(index > 0)
        require(expected == {"committed_updates": updates, "scalar_states": updates + 1 if segments else 0,
                             "retained_frames": frames, "latest_global_step": offset}, "Arm/segment expected counts disagree")
        verdict = ("failed" if any(row["scientific_status"] == "failed" for row in segments)
                   else "passed" if segments and offset == arm["global_target"] else "partial" if segments else "not_started")
        require(arm.get("scientific_status") == verdict, "Refresh relabeled original scientific outcomes")
    return plan, list(proofs.values())


def launcher_prefix():
    path = Path(__file__).with_name("launch.py")
    tree = ast.parse(path.read_bytes(), filename=str(path))
    candidates = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "PREFIX" for target in node.targets)]
    require(len(candidates) == 1 and digest(candidates[0].encode()) == PREFIX_SHA, "Exact one-shot source03 launch prefix changed")
    return candidates[0]


COUNT_GUARD = r'''
# Metadata-only fresh-prefix checks. Matrix validation stays in allocated CPU.
plan=json.loads((source/'plan.json').read_bytes())
for arm in plan['arms']:
 total_updates=total_frames=0
 for number,segment in enumerate(arm['segments']):
  committed={}
  for field in ('states','updates'):
   raw=Path(segment[field]['path']).read_bytes();boundary=raw.rfind(b'\n')+1
   rows=[json.loads(line) for line in raw[:boundary].splitlines()]
   assert all(isinstance(row,dict) for row in rows)
   committed[field]=rows
  states,updates=committed['states'],committed['updates'];expected=segment['expected_counts']
  assert len(states)==expected['scalar_states']==len(updates)+1
  assert len(updates)==expected['committed_updates']
  assert all(type(row.get('step')) is int and row['step']==index for index,row in enumerate(states))
  assert all(row.get('global_step',segment['global_offset']+index)==segment['global_offset']+index for index,row in enumerate(states))
  assert all(type(row.get('snapshot_retained')) is bool for row in states)
  retained=sum(row['snapshot_retained'] for row in states)
  assert retained==expected['retained_frames']
  total_updates+=len(updates);total_frames+=retained-int(number>0)
 expected=arm['expected_counts']
 assert total_updates==expected['committed_updates']==arm['expected_committed_updates']
 assert total_frames==expected['retained_frames']==arm['expected_retained_frames']
 assert expected['scalar_states']==total_updates+int(bool(arm['segments']))
 assert expected['latest_global_step']==arm['global_start']+total_updates
'''


STARTER = r'''"""Explicit one-shot CPU report launch. Prepared capsules are never replayed."""
import argparse
from datetime import datetime,timezone
import hashlib,json,shlex,subprocess
from pathlib import Path

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
 if not args.execute:raise SystemExit('Preparation only. Inspect capsule, then supply --execute to start CPU/report jobs.')
 package=Path(__file__).resolve().parent;receipt=json.loads((package/'preparation.json').read_bytes())
 for name,length,pin in [('source.tar.gz',receipt['archive_bytes'],receipt['archive_sha256']),
                         ('source_manifest.json',receipt['manifest_bytes'],receipt['manifest_sha256']),
                         ('plan.json',receipt['plan_bytes'],receipt['plan_sha256']),
                         ('remote-program.py',receipt['program_bytes'],receipt['program_sha256'])]:
  raw=(package/name).read_bytes();assert len(raw)==length and hashlib.sha256(raw).hexdigest()==pin,'Prepared capsule bytes changed: '+name
 out=package.parent/('launch-'+receipt['generation']);out.mkdir(exist_ok=False)
 program=(package/'remote-program.py').read_bytes()
 command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
          'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
 status={'status':'unknown','program_sha256':receipt['program_sha256'],'manifest_sha256':receipt['manifest_sha256'],
         'generation':receipt['generation'],'edition':receipt['edition'],'argv':command,'optimizer_submission':False}
 (out/'receipt.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf8')
 (out/'program.py').write_bytes(program)
 try:
  with (out/'stdout.json').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
   run=subprocess.run(command,input=program,stdout=stdout,stderr=stderr,timeout=240)
  status.update(status='returned',returncode=run.returncode)
  assert run.returncode==0,'One-shot launch failed; preserve files/remote claim and inspect, never replay.'
 finally:
  status['captured_utc']=datetime.now(timezone.utc).isoformat()
  (out/'receipt.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf8')
 print((out/'stdout.json').read_text())

if __name__=='__main__':main()
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--generation", required=True, help="Fresh two-digit generation04..99")
    parser.add_argument("--edition", required=True, help="Fresh edition-0003 or later")
    args = parser.parse_args(argv)
    require(re.fullmatch("[0-9]{2}", args.generation) and 4 <= int(args.generation) <= 99, "Generation must be04..99")
    require(re.fullmatch("edition-[0-9]{4}", args.edition) and int(args.edition[-4:]) >= 3, "Edition must be0003 or later")
    require(re.fullmatch("[a-f0-9]{64}", args.plan_sha256) is not None, "Explicit fresh plan SHA is required")
    plan_raw = args.plan.read_bytes()
    require(digest(plan_raw) == args.plan_sha256, "Fresh bound plan changed")
    original_raw = (TEMPLATE / "source.tar.gz").read_bytes()
    require(len(original_raw) == ARCHIVE_BYTES and digest(original_raw) == ARCHIVE_SHA, "Exact sealed source03 archive changed")
    members, manifest = archive_members(original_raw, MANIFEST_SHA)
    require(len(manifest["files"]) == 17 and manifest["cpu_counts"] == EXPECTED_COUNTS,
            "Exact source03 fixture/source identities differ")
    plan, fresh_proofs = fresh_plan(plan_raw, read_json(members["plan.json"]), args.edition)
    source = BASE + "/nonchem-live-report-release-" + args.generation + "/source"
    root = BASE + "/nonchem-live-report-" + args.generation
    original_ops = members["scripts/report_ops.py"].decode()
    changed_ops = original_ops
    replacements = {"nonchem-live-report-release-03": "nonchem-live-report-release-" + args.generation,
                    "nonchem-live-report-03": "nonchem-live-report-" + args.generation,
                    "mosaic-live-report03-": "mosaic-live-report" + args.generation + "-",
                    "edition-0002": args.edition}
    for old, new in replacements.items():
        require(old in changed_ops, "Required controller metadata literal missing: " + old)
        changed_ops = changed_ops.replace(old, new)
    ast.parse(changed_ops, filename="scripts/report_ops.py")
    # Reversibility proves no nonmetadata controller edit occurred.
    restored = changed_ops
    for old, new in reversed(list(replacements.items())):
        restored = restored.replace(new, old)
    require(restored == original_ops, "Controller transformation changed nonmetadata code")
    members["scripts/report_ops.py"] = changed_ops.encode()
    members["plan.json"] = encoded(plan)
    new_manifest = deepcopy(manifest)
    proofs = {row["path"]: row for row in new_manifest["proofs"]}
    for row in fresh_proofs:
        require(row["path"] not in proofs or proofs[row["path"]] == row, "Fresh plan conflicts with an immutable historical proof")
        proofs[row["path"]] = row
    new_manifest["proofs"] = list(proofs.values())
    new_manifest["refresh_capsule"] = {"version": "jq1-live-report-refresh-v1", "generation": args.generation,
        "edition": args.edition, "template_manifest_sha256": MANIFEST_SHA,
        "template_archive_sha256": ARCHIVE_SHA, "input_plan_sha256": args.plan_sha256,
        "changes": ["bound_saved_plan", "controller_root_source_job_prefix_archive_edition_literals"],
        "scientific_renderer_qualification_template_tests_changed": False,
        "automatic_hpc_poll": False, "optimizer_submission": False}
    new_manifest["files"] = {name: digest(raw) for name, raw in sorted(members.items()) if name != "source_manifest.json"}
    members["source_manifest.json"] = encoded(new_manifest)
    for name, old_sha in manifest["files"].items():
        if name not in ("plan.json", "scripts/report_ops.py"):
            require(digest(members[name]) == old_sha, "Scientific/qualification/template/test bytes changed")
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as bundle:
            for name, raw in sorted(members.items()):
                info = tarfile.TarInfo(name); info.size = len(raw); info.mode = 0o444; info.mtime = 0
                bundle.addfile(info, io.BytesIO(raw))
    archive = output.getvalue()
    pin = digest(members["source_manifest.json"])
    checked, _ = archive_members(archive, pin)
    require(checked == members, "Prepared archive differs from exact member bytes")
    prefix = launcher_prefix()
    for old, new in replacements.items():
        prefix = prefix.replace(old, new)
    marker = "assert not root.exists() and not source.parent.exists(), 'Existing/unknown source or report root: never replay'"
    require(prefix.count(marker) == 1 and prefix.count("preflight=root/'preflight'") == 1, "Launch prefix metadata anchors changed")
    claim = "claim=base/'nonchem-live-report-" + args.generation + ".refresh-claim.json'\nwith claim.open('x') as stream:json.dump({'status':'unknown','manifest_sha256':PIN,'optimizer_submission':False},stream)\n"
    prefix = prefix.replace(marker, claim + marker).replace("preflight=root/'preflight'", COUNT_GUARD + "\npreflight=root/'preflight'")
    import base64
    program = "\n".join(["ARCHIVE=" + repr(base64.b64encode(archive).decode()), "LENGTH=" + str(len(archive)),
                         "DIGEST=" + repr(digest(archive)), "PIN=" + repr(pin), prefix])
    ast.parse(program, filename="remote-program.py")
    ast.parse(STARTER, filename="start_prepared.py")
    package = LOCAL / ("source" + args.generation)
    require(not (LOCAL / ("launch-" + args.generation)).exists(), "Existing/unknown local launch: never replay")
    package.mkdir(exist_ok=False)
    for name, raw in {"source.tar.gz": archive, "source_manifest.json": members["source_manifest.json"],
                      "plan.json": members["plan.json"], "remote-program.py": program.encode(),
                      "start_prepared.py": STARTER.encode()}.items():
        with (package / name).open("xb") as stream:
            stream.write(raw)
    preparation = {"status": "prepared_only_not_submitted", "generation": args.generation, "edition": args.edition,
        "template_archive_sha256": ARCHIVE_SHA, "manifest_sha256": pin, "manifest_bytes": len(members["source_manifest.json"]),
        "archive_bytes": len(archive), "archive_sha256": digest(archive), "files": len(members) - 1,
        "cpu_counts": EXPECTED_COUNTS, "plan_bytes": len(members["plan.json"]), "plan_sha256": digest(members["plan.json"]),
        "input_plan_sha256": args.plan_sha256, "program_bytes": len(program.encode()), "program_sha256": digest(program.encode()),
        "source": source, "root": root, "allocated_checks_passed": False, "automatic_hpc_poll": False,
        "upload_executed": False, "job_submitted": False, "optimizer_submission": False}
    with (package / "preparation.json").open("xb") as stream:
        stream.write(encoded(preparation))
    helper = REPO / "experiments/reporting/jq1-live-progress/scripts"
    commands = [
        "python " + shlex.quote(str(package / "start_prepared.py")) + " --execute",
        "python " + shlex.quote(str(helper / "observe.py")) + " --name observation-refresh" + args.generation +
        " --generation " + args.generation + " --edition " + args.edition,
        "python " + shlex.quote(str(helper / "publish.py")) + " --observation observation-refresh" + args.generation +
        " --generation " + args.generation + " --edition " + args.edition]
    print(json.dumps({"status": preparation["status"], "package": str(package), "manifest_sha256": pin,
                      "commands_not_executed": commands,
                      "publication_condition": "Actual CPU123pass and reportDONE/LSF/passedarchive evidence, independentbytechecks then HTTP/browserQA"}, indent=2))


if __name__ == "__main__":
    main()
