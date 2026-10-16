"""Package named Mosaic run directories on allocated HPC, preserving evidence."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile


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
                if (
                    "pytest-unit-tmp" not in relative.parts
                    or not path.name.endswith("current")
                    or not path.resolve().is_relative_to(
                        directory / "payload/pytest-unit-tmp"
                    )
                    or not path.resolve().is_dir()
                ):
                    raise ValueError(
                        "Non-pytest symlink evidence is unsupported: " + str(path)
                    )
                # Pytest's current aliases duplicate directories already walked
                # under their real names. Record but never follow/archive links.
                pytest_aliases[path.relative_to(root).as_posix()] = str(
                    path.resolve().relative_to(root)
                )
                continue
            if path.is_file():
                files[path.relative_to(root).as_posix()] = path
    total = sum(path.stat().st_size for path in files.values())
    receipt_dir = args.output.parent / (args.output.name + ".preflight")
    quota_arguments = [
        str(Path("/work3/s234818/mosaic/envs/protenix-tiny/bin/python")),
        str(
            Path(__file__).parent
            / "hpc"
            / (
                "zhome_quota_preflight.py"
                if args.storage == "zhome"
                else "stage0_quota_preflight.py"
            )
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
