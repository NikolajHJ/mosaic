"""Stage one held L40S job for the matched seed-41 repair comparison."""

import argparse
import json
from pathlib import Path
import re
import tarfile

from launch_masked_denoising import transport
from launch_score_transfer_audit import REMOTE, REPO, STAGE, dump, sha
from seed41_atom_exchange_pair import EXCHANGE, HARD_STRIDE, PARENT_JOB, PARENT_STEP, UPDATES, VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--parent-proof", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]{1,50}", args.tag):
        parser.error("Invalid immutable run tag")

    study = REPO / "experiments/brd4-jq1/protenix-random-chem"
    data = json.loads(args.parent_proof.read_bytes())
    saved = json.loads((study / "runs/20261104-masked-denoising-01/inputs.json").read_bytes())
    endpoint, = (p for p in saved["points"] if p["id"] == "sparse41-valid")
    if (data["endpoint"] != endpoint["snapshot"] or endpoint["producer_job_id"] != PARENT_JOB
            or data["summary"]["job_id"] != PARENT_JOB or not data["summary"]["full_budget_complete"]
            or "Done successfully" not in data["parent_history"]["stdout"]):
        raise ValueError("Expected the same verified seed-41 checkpoint")

    remote = "/work3/s234818/mosaic/runs/" + args.tag
    plan = {
        "version": VERSION, "parent_step": PARENT_STEP, "parent_job": PARENT_JOB,
        "parent_plan": data["parent_plan"], "parent_manifest": data["parent_manifest"],
        "parent_summary": data["summary_binding"], "parent_snapshot": data["endpoint"],
        "original_screen": data["original_screen"], "source_root": data["source_root"],
        "routing_source": data["routing_source"], "science_source": data["row"]["science_source"]["path"],
        "model_cache": saved["model_cache"], "output": remote + "/payload",
        "updates": UPDATES, "hard_stride": HARD_STRIDE, "atom_exchange": EXCHANGE,
        "entrypoint": "seed41_atom_exchange_pair.py",
        "queue": "gpul40s", "gpu_profile": "l40s-48gb-v1",
        "gpu_request": "num=1:mode=exclusive_process:j_exclusive=yes:mps=no",
        "cpu_slots": 8, "memory_mib_per_slot": 4096, "wall_minutes": 90,
        "internal_seconds": 5100, "complete_output_cap_mib": 512,
        "output_cap_bytes": 504 * 1024**2, "file_cap": 5000,
    }
    local = REPO / "results/hpc" / args.tag
    source = local / "source"
    source.mkdir(parents=True, exist_ok=False)
    for directory, names in (
        (study / "scripts", ("seed41_atom_exchange_pair.py", "atom_exchange_audit.py",
                             "atom_exchange_metrics.py", "score_transfer_audit.py")),
        (REPO / "src/mosaic/small_molecule/chemistry", ("atom_exchange.py", "structural.py",
                                                       "environment_policy.py", "preparation.py")),
        (REPO / "tests/small_molecule", ("test_chemistry_atom_exchange.py",
                                        "test_chemistry_atom_exchange_preparation.py",
                                        "test_chemistry_structural.py")),
        (REPO / "scripts/hpc", ("storage_accounting.py",)),
    ):
        for name in names:
            (source / name).write_bytes((directory / name).read_bytes())
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
