"""Package named Mosaic run directories on allocated HPC, preserving evidence."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile


PYTEST_BASETEMPS = frozenset(
    ("pytest-unit-tmp", "pytest-unit-tmp-legacy", "pytest-unit-tmp-new")
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage", choices=("work3", "zhome"), default="work3")
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Evidence packaging requires allocated DTU HPC")
    root = (
        Path.home() / "mosaic-runs"
        if args.storage == "zhome"
        else Path("/work3/s234818/mosaic/runs")
    ).resolve()
    if not args.output.resolve().is_relative_to(root):
        raise ValueError("Archive must use the declared Mosaic storage root")
    files = {}
    pytest_aliases = {}
    for directory in args.run:
        directory = directory.resolve()
        if not directory.is_relative_to(root) or directory == root:
            raise ValueError(
                "Only explicitly named Mosaic run directories are supported"
            )
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                relative = path.relative_to(directory)
                # Only declared pytest basetemps may contain duplicate
                # ``current`` aliases. Never accept arbitrary prefix matches,
                # cross-basetemp links, or links out of the named run.
                if (
                    len(relative.parts) < 3
                    or relative.parts[0] != "payload"
                    or relative.parts[1] not in PYTEST_BASETEMPS
                    or not path.name.endswith("current")
                ):
                    raise ValueError(
                        "Non-pytest symlink evidence is unsupported: " + str(path)
                    )
                temporary = directory / "payload" / relative.parts[1]
                target = path.resolve(strict=True)
                if (
                    target == temporary
                    or not target.is_relative_to(temporary)
                    or not target.is_dir()
                ):
                    raise ValueError(
                        "Non-pytest symlink evidence is unsupported: " + str(path)
                    )
                # Pytest's current aliases duplicate directories already walked
                # under their real names. Record but never follow/archive links.
                pytest_aliases[path.relative_to(root).as_posix()] = str(
                    target.relative_to(root)
                )
                continue
            if path.is_file():
                files[path.relative_to(root).as_posix()] = path
    total = sum(path.stat().st_size for path in files.values())
    receipt_dir = args.output.parent / (args.output.name + ".preflight")
    quota_arguments = [
        str(Path("/work3/s234818/mosaic/envs/protenix-tiny/bin/python")),
        str(
            _layout_path((_repository_root(__file__) / 'scripts')
            / "hpc"
            / (
                "zhome_quota_preflight.py"
                if args.storage == "zhome"
                else "stage0_quota_preflight.py"
            ))
        ),
        "--run-dir",
        str(receipt_dir),
        "--minimum-free-gib",
        str(4 + (total * 1.01 + 1024**2) / 1024**3),
    ]
    if args.storage == "work3":
        quota_arguments.extend(["--storage-pool", "6"])
    subprocess.run(
        quota_arguments,
        check=True,
    )
    for path in receipt_dir.iterdir():
        if path.is_file():
            files["packaging-preflight/" + path.name] = path
    metadata = args.output.parent / (args.output.name + ".metadata")
    metadata.mkdir(exist_ok=False)
    job_ids = {os.environ["LSB_JOBID"]}
    for path in files.values():
        match = re.fullmatch(r"(?:stdout|stderr)\.(\d+)\.log", path.name)
        if match:
            job_ids.add(match.group(1))
    for job in sorted(job_ids):
        result = subprocess.run(
            ["bjobs", "-l", job],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        path = metadata / ("job-" + job + ".txt")
        path.write_text(result.stdout + result.stderr)
        files["scheduler/" + path.name] = path
    manifest = {
        "job_id": os.environ["LSB_JOBID"],
        "files": {},
        "omitted_pytest_current_aliases": pytest_aliases,
    }
    for name, path in files.items():
        with path.open("rb") as reader:
            digest = hashlib.file_digest(reader, "sha256").hexdigest()
        manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": digest}
    data = (json.dumps(manifest, indent=2) + "\n").encode()
    with args.output.open("xb") as output:
        with tarfile.open(fileobj=output, mode="w:gz") as handle:
            for name, path in files.items():
                handle.add(path, arcname=name, recursive=False)
            info = tarfile.TarInfo("evidence_manifest.json")
            info.size = len(data)
            handle.addfile(info, io.BytesIO(data))
    with args.output.open("rb") as reader:
        digest = hashlib.file_digest(reader, "sha256").hexdigest()
    receipt = {
        "path": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": digest,
        "files": len(files),
    }
    args.output.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
