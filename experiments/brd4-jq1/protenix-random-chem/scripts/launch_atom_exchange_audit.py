"""Submit one read-only atom exchange audit over the saved paired proposals."""

import argparse
import json
import re
import tarfile

from launch_masked_denoising import transport
from launch_score_transfer_audit import REMOTE, REPO, dump, sha


STAGE = r"""#!/bin/bash
set -euo pipefail
source_dir=$1
module load python3/3.12.11
unset PYTHONHOME
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
cd "$source_dir"
"$python" -B storage_accounting.py --output-root "$(dirname "$source_dir")" \
  --cap-mib 32 --file-cap 120 > "$(dirname "$source_dir")/capacity-entry.json"
exec "$python" -B -u atom_exchange_audit.py --plan "$source_dir/plan.json"
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]{1,50}", args.tag):
        parser.error("Invalid immutable run tag")

    study = REPO / "experiments/brd4-jq1/protenix-random-chem"
    completed = study / "runs/20261104-masked-denoising-01"
    inputs = json.loads((completed / "inputs.json").read_bytes())
    terminal = json.loads((completed / "terminal-verification.json").read_bytes())
    if terminal["status"] != "verified_complete" or terminal["job_id"] != "29890650":
        raise ValueError("Require the verified matched sampling run")
    remote = "/work3/s234818/mosaic/runs/" + args.tag
    plan = dict(version="mosaic-atom-exchange-audit-v1", source_root=inputs["source_root"],
                benchmark=terminal["result"], points=inputs["points"], output=remote + "/payload",
                optimizer_updates=0, atom_edit_penalty=1.0, max_substitutions=4,
                entrypoint="atom_exchange_audit.py", cpu_only=True, queue="hpc",
                cpu_slots=2, memory_mib_per_slot=4096, wall_minutes=15,
                internal_seconds=750, complete_output_cap_mib=32,
                output_cap_bytes=24 * 1024**2, file_cap=120)
    local = REPO / "results/hpc" / args.tag
    source = local / "source"
    source.mkdir(parents=True, exist_ok=False)
    for directory, names in (
        (study / "scripts", ("atom_exchange_audit.py", "score_transfer_audit.py")),
        (REPO / "src/mosaic/small_molecule/chemistry", ("atom_exchange.py", "structural.py", "environment_policy.py")),
        (REPO / "tests/small_molecule", ("test_chemistry_atom_exchange.py", "test_chemistry_structural.py")),
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
