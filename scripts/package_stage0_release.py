"""Seal current local source bytes without importing Mosaic or running tests.

Packaging is permitted during the HPC outage. The manifest includes uncommitted
code and the plan/audit/ledger; its hash and archive hash are external receipts.
This command does not connect to HPC or transfer anything.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split("\0")
    files = {root / name for name in tracked if name and (root / name).is_file()}
    # Add all Stage 0 execution sources even before they are committed.
    for folder in ("src", "tests", "examples", "scripts"):
        files.update(
            path
            for path in (root / folder).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in (".pyc", ".pyo")
        )
    for name in (
        "docs/project_desc.pdf",
        "docs/stage_0_implementation_plan.md",
        "docs/stage_0_native_contract.md",
        "docs/stage_0_usage.md",
        "docs/experiment_ledger.md",
        "docs/HPC_HELP.tex",
        "docs/hpc_protenix_tiny.md",
        "docs/protenix_relaxed_gradient_tests.md",
        "docs/carbon_chain_100.md",
        "docs/carbon_chain_500.md",
        "docs/known_ligand_1000.md",
        "docs/brd4_5000.md",
        "docs/brd4_uniform_5000.md",
        "docs/brd4_noisy_uniform_5000.md",
        "docs/brd4_combined_report.md",
    ):
        path = root / name
        if path.is_file():
            files.add(path)
    manifest = {
        "git_revision_reference": revision,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "identity_policy": "current file bytes including uncommitted contents; commit is reference only",
        "files": {
            path.relative_to(root).as_posix(): file_sha256(path)
            for path in sorted(files)
        },
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (output / "source_manifest.json").write_bytes(manifest_bytes)
    archive = output / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(files):
            handle.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
        info = tarfile.TarInfo("source_manifest.json")
        info.size = len(manifest_bytes)
        info.mtime = 0
        info.mode = 0o444
        handle.addfile(info, io.BytesIO(manifest_bytes))
    receipt = {
        "source_archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": file_sha256(archive),
        "manifest_sha256": file_sha256(output / "source_manifest.json"),
        "files": len(files),
        "git_revision_reference": revision,
        "hpc_transfer": "not performed",
        "numerical_validation": "pending DTU HPC",
    }
    (output / "release_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
