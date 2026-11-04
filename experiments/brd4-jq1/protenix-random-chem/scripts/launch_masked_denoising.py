"""Package and launch one immutable masked-denoising comparison on DTU."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import tarfile

from launch_score_transfer_audit import HELPER, HERE, REMOTE, REPO, STAGE, dump, sha
from masked_denoising_benchmark import ATTEMPTS, SCORE_CAP, VERSION


def transport(local, name, argv, timeout, *, program=None):
    result = subprocess.run([HELPER, *argv], input=program, text=True, capture_output=True, timeout=timeout)
    dump(local / (name + ".json"), {
        "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
    })
    if result.returncode:
        raise RuntimeError(f"{name} requires reconciliation in {local}: {result.stderr[-1800:]}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True, help="Existing frozen DeFoG loader")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]{1,50}", args.tag):
        parser.error("Invalid immutable run tag")

    plan = json.loads(args.inputs.read_text(encoding="utf-8"))
    probe = args.probe.read_bytes()
    if sha(probe) != plan.pop("defog_probe_sha256"):
        raise ValueError("Frozen DeFoG loader differs from the study inputs")
    remote = "/work3/s234818/mosaic/runs/" + args.tag
    plan.update(version=VERSION, seed=20261104, optimizer_updates=0,
                proposals_per_cell=ATTEMPTS, hard_score_cap_per_cell=SCORE_CAP,
                entrypoint="masked_denoising_benchmark.py", output=remote + "/payload",
                queue="gpuv100", cpu_slots=4, memory_mib_per_slot=8192, wall_minutes=60,
                internal_seconds=3000, complete_output_cap_mib=64,
                output_cap_bytes=32 * 1024**2, file_cap=120)
    plan["denoiser"].update(probe=remote + "/source/defog_probe.py",
                           log=remote + "/payload/defog-worker.log", steps=8, start_time=0.55)
    local = REPO / "results/hpc" / args.tag
    source = local / "source"
    source.mkdir(parents=True, exist_ok=False)
    for name in ("masked_denoising_benchmark.py", "masked_denoising_controls.py",
                 "masked_denoiser.py", "score_transfer_audit.py"):
        (source / name).write_bytes(HERE.with_name(name).read_bytes())
    (source / "storage_accounting.py").write_bytes((REPO / "scripts/hpc/storage_accounting.py").read_bytes())
    (source / "defog_probe.py").write_bytes(probe)
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
