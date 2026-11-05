"""Submit a fresh seed-23 line-search repeat with one detached projection proposal."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import tarfile

from launch_score_transfer_audit import HELPER, REPO, dump, sha


BASELINE = "29890096"
ARM = "annealed-sparse-s23-projected"
PARENT_ARM = "annealed-sparse-s23-halving8"
UPSTREAM = "041c89ea5bdab024bf5671d32749a151f9558e48"
POLICY = "stage2-valence-proposal-v1"
WALL_MINUTES = 36 * 60


STAGE = r"""
from pathlib import Path, PurePosixPath
import hashlib, json, os, subprocess, sys, tarfile

root = Path(PACKET["remote"])
source, archive, run = root/"source", root/"source.tar.gz", root/"run"
def digest(path):
    with path.open("rb") as f:
        h = hashlib.sha256()
        for data in iter(lambda: f.read(1048576), b""):
            h.update(data)
        return h.hexdigest()

assert archive.stat().st_size == PACKET["archive_bytes"] and digest(archive) == PACKET["archive_sha256"]
source.mkdir(exist_ok=False)
with tarfile.open(archive) as f:
    members = f.getmembers()
    assert len(members) == PACKET["files"] + 1 and len({m.name for m in members}) == len(members)
    for member in members:
        name = PurePosixPath(member.name)
        assert member.isfile() and not name.is_absolute() and ".." not in name.parts and "\\" not in member.name
        target = source.joinpath(*name.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with f.extractfile(member) as inp, target.open("xb") as out:
            out.write(inp.read())
assert digest(source/"checksums.json") == PACKET["checksums_sha256"]
assert digest(source/"plan.json") == PACKET["plan_sha256"]
checks = json.loads((source/"checksums.json").read_text())
assert len(checks) == PACKET["files"]
for name, expected in checks.items():
    assert digest(source/name) == expected, name
for p in source.rglob("*"):
    if p.is_file():
        p.chmod(0o444)
for p in sorted((p for p in source.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
    p.chmod(0o555)
source.chmod(0o555)
run.mkdir(exist_ok=False)

sys.path.insert(0, str(source))
import chemistry_screen_ops as ops

plan_path = source/"plan.json"
plan = ops.read(plan_path)
old_path = Path(PACKET["baseline_source"])/"plan.json"
assert digest(old_path) == PACKET["baseline_plan_sha256"]
old = ops.read(old_path)
old_row = old["arms"][PARENT_ARM]
arm = plan["arms"][ARM]
for key, val in old_row.items():
    if key not in {"name", "budget_basis", "requested_wall_minutes", "internal_seconds"}:
        assert arm[key] == val, key
assert ops.read(Path(old["root"])/"jobs.json")[PARENT_ARM] == BASELINE

q = ops.command(["bqueues", "-l", "gpua100"])
assert q["returncode"] == 0 and "Open:Active" in q["stdout"] and "4320.0 min" in q["stdout"]
ops.write(root/"queue-proof.json", q, exclusive=True)
startup = ops.command([sys.executable, "-B", str(source/"protenix_chemistry_screen.py"),
                       "--plan", str(plan_path), "--arm", ARM, "--startup-only"])
ops.write(root/"metadata-startup.json", startup, exclusive=True)
assert startup["returncode"] == 0, startup
parsed = json.loads(startup["stdout"])
assert parsed["line_search_halvings"] == 8 and parsed["repair_proposal"] == arm["repair_proposal"]
assert parsed["configuration"] == old_row["screen"]
ops.write(root/"preflight-capacity.json", ops.capacity(plan, "projected-seed23-preflight"), exclusive=True)

previous = ops.command(["bjobs", "-a", "-noheader", "-o", "jobid stat user", BASELINE])
parts = previous["stdout"].strip().split()
if previous["returncode"] == 0 and len(parts) == 3 and parts[0] == BASELINE and parts[2] == "s234818":
    assert parts[1] in ("RUN", "PEND", "DONE", "EXIT", "USUSP", "SSUSP"), previous
    dependency = BASELINE if parts[1] not in ("DONE", "EXIT") else None
    history = None
else:
    history = ops.command(["bhist", "-l", BASELINE])
    assert history["returncode"] == 0 and "Job <"+BASELINE+">" in history["stdout"] and (
        "Done successfully" in history["stdout"] or "Exited with exit code" in history["stdout"]
    ), history
    dependency = None
ops.write(root/"baseline-scheduler.json", {"scheduler": previous, "history": history,
          "dependency": "ended("+BASELINE+")" if dependency else None}, exclusive=True)
job = ops.submit(plan_path, "run", ARM, arm["requested_wall_minutes"],
                 dependency=dependency, dependency_type="ended")
current = ops.command(["bjobs", "-a", "-w", job])
receipt = {"job_id":job, "arm":ARM, "scheduler":current, "dependency":dependency,
           "plan_sha256":digest(plan_path), "checksums_sha256":digest(source/"checksums.json")}
ops.write(root/"dispatch.json", receipt, exclusive=True)
print(json.dumps(receipt), flush=True)
"""


def prepare(tag, parent):
    recorded = json.loads((REPO/"experiments/brd4-jq1/protenix-line-search/campaign.json").read_bytes())
    base = parent/"source"
    assert recorded["job_id"] == BASELINE
    assert sha((base/"plan.json").read_bytes()) == recorded["plan_sha256"]
    assert sha((base/"checksums.json").read_bytes()) == recorded["checksums_sha256"]
    checks = json.loads((base/"checksums.json").read_bytes())
    local = REPO/"results/hpc"/tag
    source = local/"source"
    source.mkdir(parents=True, exist_ok=False)
    for name, expected in checks.items():
        data = (base/name).read_bytes()
        assert sha(data) == expected, name
        target = source/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    owner = Path(__file__).resolve().parent
    for src, dest in (
        (owner/"protenix_chemistry_screen.py", "protenix_chemistry_screen.py"),
        (owner/"projected_repair.py", "projected_repair.py"),
        (REPO/"src/mosaic/small_molecule/chemistry/preparation.py",
         "src/mosaic/small_molecule/chemistry/preparation.py"),
    ):
        (source/dest).write_bytes(src.read_bytes())
    operator = subprocess.check_output(["git", "show", UPSTREAM+":stage2/operators.py"], cwd=REPO)
    (source/"stage2_operators.py").write_bytes(operator)
    remote = "/work3/s234818/mosaic/runs/"+tag
    plan = json.loads((base/"plan.json").read_bytes())
    arm = plan["arms"][PARENT_ARM]
    arm.update(name=ARM, repair_proposal=POLICY, requested_wall_minutes=WALL_MINUTES,
               internal_seconds=WALL_MINUTES*60-600,
               budget_basis="1.8 s/update from matched A100 baseline plus 0.2 s/update extra CPU allowance, "
                            "25 percent margin and 900 seconds setup; 36 hours rounded")
    plan.update(root=remote+"/run", source_root=remote+"/source", code_source_root=remote+"/source",
                capacity_module=remote+"/source/chemistry_screen_ops.py", job_name_prefix="mosaic-"+tag,
                arms={ARM:arm}, probe_arms=[ARM], production_arms=[ARM],
                interpretation=["Fresh 50k same-seed A100 comparison against 29890096.",
                                "Original optimizer, schedules, and structural repair.",
                                "Extra detached Stage 2 repair proposal compared using the original probabilities."])
    dump(source/"plan.json", plan)
    paths = [p for p in sorted(source.rglob("*")) if p.is_file()]
    checks = {p.relative_to(source).as_posix():sha(p.read_bytes()) for p in paths}
    dump(source/"checksums.json", checks)
    archive = local/"source.tar.gz"
    with tarfile.open(archive, "w:gz") as f:
        for path in [*paths, source/"checksums.json"]:
            f.add(path, arcname=path.relative_to(source).as_posix())
    packet = dict(remote=remote, archive_bytes=archive.stat().st_size, archive_sha256=sha(archive.read_bytes()),
                  files=len(checks), plan_sha256=checks["plan.json"], baseline_job=BASELINE,
                  baseline_source=recorded["source"], baseline_plan_sha256=recorded["plan_sha256"],
                  checksums_sha256=sha((source/"checksums.json").read_bytes()), stage2_revision=UPSTREAM,
                  stage2_operator_sha256=sha(operator), wall_minutes=WALL_MINUTES)
    dump(local/"prepared.json", packet)
    return local, packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    assert re.fullmatch(r"[0-9a-z][0-9a-z-]+", args.tag), "Require a fresh safe experiment tag"
    local, packet = prepare(args.tag, args.parent)
    archive = local/"source.tar.gz"
    remote = packet["remote"]
    claim = subprocess.run([HELPER, "exec", "--timeout", "30", "mkdir "+remote],
                           capture_output=True, text=True, timeout=40)
    dump(local/"claim.json", dict(returncode=claim.returncode, stdout=claim.stdout, stderr=claim.stderr))
    assert claim.returncode == 0, "Output owner exists or access unavailable; reconcile"
    upload = subprocess.run([HELPER, "upload", "--timeout", "90", str(archive), remote+"/source.tar.gz"],
                            capture_output=True, text=True, timeout=105)
    dump(local/"upload.json", dict(returncode=upload.returncode, stdout=upload.stdout, stderr=upload.stderr))
    assert upload.returncode == 0, "Source transfer incomplete; inspect and preserve evidence"
    definitions = {"PACKET":packet, "BASELINE":BASELINE, "ARM":ARM, "PARENT_ARM":PARENT_ARM}
    program = "".join(f"{name}={value!r}\n" for name, value in definitions.items()) + STAGE
    (local/"stage-submit.py").write_text(program, encoding="utf-8", newline="\n")
    process = subprocess.run([HELPER, "exec", "--timeout", "220", "bash -lc 'python3 -B -'"],
                             input=program, text=True, capture_output=True, timeout=235)
    dump(local/"stage-submit-reply.json", dict(returncode=process.returncode,
                                               stdout=process.stdout, stderr=process.stderr))
    if process.returncode:
        raise RuntimeError("Inspect actual remote state before recovery: " + process.stderr[-2400:])
    dispatch = json.loads(process.stdout.splitlines()[-1])
    dump(local/"dispatch.json", dispatch)
    print(json.dumps(dispatch))


if __name__ == "__main__":
    main()
