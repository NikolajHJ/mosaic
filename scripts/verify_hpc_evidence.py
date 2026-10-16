"""Verify and safely extract an HPC evidence archive using only stdlib I/O.

This is a local byte-integrity check, not numerical model validation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("sha256")
    args = parser.parse_args()
    archive = args.archive.resolve()
    with archive.open("rb") as reader:
        actual = hashlib.file_digest(reader, "sha256").hexdigest()
    if actual != args.sha256.lower():
        raise ValueError("Downloaded archive checksum differs")
    output = archive.with_suffix("").with_suffix("")
    output.mkdir(exist_ok=False)
    with tarfile.open(archive) as handle:
        names = set()
        for member in handle.getmembers():
            target = (output / member.name).resolve()
            if (
                not target.is_relative_to(output)
                or not (member.isdir() or member.isfile())
                or member.name in names
            ):
                raise ValueError("Unsafe or duplicate archive member")
            names.add(member.name)
        handle.extractall(output, filter="data")
    manifest = json.loads((output / "evidence_manifest.json").read_text())
    for name, entry in manifest["files"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or path.stat().st_size != entry["bytes"]:
            raise ValueError("Evidence path/size differs: " + name)
        with path.open("rb") as reader:
            if hashlib.file_digest(reader, "sha256").hexdigest() != entry["sha256"]:
                raise ValueError("Evidence file checksum differs: " + name)
    receipt = {
        "archive": str(archive),
        "sha256": actual,
        "verified_files": len(manifest["files"]),
        "scope": "stdlib byte integrity only; no local numerical evaluation",
    }
    output.with_name(output.name + "-download.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
