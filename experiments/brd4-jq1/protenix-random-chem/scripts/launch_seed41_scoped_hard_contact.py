"""Stage a bounded L40S continuation from the verified valid seed-41 endpoint."""

import argparse
import json
from pathlib import Path
import re
import tarfile

from launch_masked_denoising import transport
from launch_score_transfer_audit import REMOTE, REPO, STAGE, dump, sha
from seed41_scoped_hard_contact import (
    ACCEPT_MARGIN, BATCH, CANDIDATE_CAP, PARENT_JOB, PARENT_STEP, ROUNDS, VERSION,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--parent-proof", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]{1,50}", args.tag):
        parser.error("Invalid immutable run tag")

    study = REPO / "experiments/brd4-jq1/protenix-random-chem"
    evidence = json.loads((study / "runs/20261105-seed41-atom-exchange-pair-01/terminal-verification.json").read_bytes())
    data = json.loads(args.parent_proof.read_bytes())
    if (evidence["job_id"] != PARENT_JOB or evidence["status"] != "verified_complete" or
            any(data["source_" + key] != evidence["source" if key == "manifest" else key]
                for key in ("result", "plan", "manifest", "lsf")) or
            data["frozen"] != evidence["frozen_identity"] or
            data["initial"]["source_step"] != PARENT_STEP or not data["initial"]["scoped_valid"]):
        raise ValueError("Expected verified valid seed-41 endpoint evidence")

    remote = "/work3/s234818/mosaic/runs/" + args.tag
    plan = {k: v for k, v in data.items() if k != "queue"}
    plan.update(version=VERSION, parent_job=PARENT_JOB, parent_step=PARENT_STEP,
                rounds=ROUNDS, batch=BATCH, candidate_cap=CANDIDATE_CAP, accept_margin=ACCEPT_MARGIN,
                output=remote + "/payload", entrypoint="seed41_scoped_hard_contact.py",
                queue="gpul40s", gpu_profile="l40s-48gb-v1",
                gpu_request="num=1:mode=exclusive_process:j_exclusive=yes:mps=no",
                cpu_slots=4, memory_mib_per_slot=8192, wall_minutes=30, internal_seconds=1500,
                complete_output_cap_mib=64, output_cap_bytes=56 * 1024**2, file_cap=120)
    local = REPO / "results/hpc" / args.tag
    source = local / "source"
    source.mkdir(parents=True, exist_ok=False)
    for name in ("seed41_scoped_hard_contact.py", "scoped_hard_neighborhood.py",
                 "atom_exchange_audit.py", "score_transfer_audit.py"):
        (source / name).write_bytes((study / "scripts" / name).read_bytes())
    (source / "storage_accounting.py").write_bytes((REPO / "scripts/hpc/storage_accounting.py").read_bytes())
    (source / "stage.sh").write_text(STAGE, encoding="utf-8", newline="\n")
    dump(source / "plan.json", plan)
    dump(source / "checksums.json", {"files": {
        p.name: {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
        for p in sorted(source.iterdir())
    }})
    archive = local / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(source.iterdir()):
            bundle.add(path, arcname=path.name)
    archive_sha = sha(archive.read_bytes())
    dump(local / "prepared.json", {
        "remote": remote, "archive_sha256": archive_sha, "archive_bytes": archive.stat().st_size,
        "source_manifest_sha256": sha((source / "checksums.json").read_bytes()),
        "plan_sha256": sha((source / "plan.json").read_bytes()),
        "parent_proof_sha256": sha(args.parent_proof.read_bytes()),
    })
    transport(local, "claim", ["exec", "--timeout", "30", "mkdir " + remote], 40)
    transport(local, "upload", ["upload", "--timeout", "90", str(archive), remote + "/source.tar.gz"], 105)
    program = f"ROOT={remote!r}\nARCHIVE_SHA={archive_sha!r}\nTAG={args.tag!r}\n" + REMOTE
    (local / "stage-submit.py").write_text(program, encoding="utf-8", newline="\n")
    stdout = transport(local, "transport", ["exec", "--timeout", "240", "bash -lc 'python3 -B -'"],
                       255, program=program)
    actual = json.loads(stdout)
    dump(local / "dispatch.json", actual)
    print(json.dumps(actual))


if __name__ == "__main__":
    main()
