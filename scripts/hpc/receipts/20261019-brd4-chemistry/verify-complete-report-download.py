"""Finish byte verification of this report's already safely extracted archive.

The original scripts/verify_hpc_evidence.py completed safe extraction before
its slow serial Windows per-file pass was stopped for this bounded I/O retry. This
operator reauthenticates the archive, its manifest and every extracted file;
it neither extracts again nor performs numerical or molecular evaluation.
Pass the complete-report-01 download directory as the sole argument.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
import tarfile
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    base = parser.parse_args().directory.resolve(strict=True)
    name = "20261019-brd4-chemistry-05-repackage"
    archive = base / f"{name}.tar.gz"
    receipt = json.loads((base / f"{name}.tar.receipt.json").read_text())
    remote = f"/zhome/95/e/206121/mosaic-runs/{name}.tar.gz"
    expected_sha = "c0309f186da134d48f69bc6fd8594535325934f4904def3aaad377b2452d0c89"
    require(receipt["path"] == remote, "Unexpected remote archive path")
    require(
        archive.stat().st_size == receipt["bytes"] == 249546618,
        "Archive byte count differs",
    )
    with archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    require(actual == receipt["sha256"] == expected_sha, "Archive hash differs")
    root = (base / name).resolve(strict=True)
    manifest_bytes = (root / "evidence_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    require(
        len(manifest["files"]) == receipt["files"] == 15995,
        "Manifest file count differs",
    )
    # The manifest itself must be the exact member of the authenticated archive.
    with tarfile.open(archive) as handle:
        names = set()
        found_manifest = False
        for member in handle:
            target = PurePosixPath(member.name)
            require(
                not target.is_absolute()
                and ".." not in target.parts
                and ":" not in member.name
                and "\\" not in member.name
                and (member.isdir() or member.isfile())
                and member.name not in names,
                "Unsafe or duplicate archive member: " + member.name,
            )
            names.add(member.name)
            if member.name == "evidence_manifest.json":
                require(
                    handle.extractfile(member).read() == manifest_bytes,
                    "Extracted manifest differs from archive",
                )
                found_manifest = True
        require(found_manifest, "Missing archive manifest")
    print("Archive, manifest and safe member names authenticated.", flush=True)

    def check(item):
        filename, entry = item
        path = (root / filename).resolve(strict=True)
        require(
            path.is_relative_to(root)
            and path.is_file()
            and path.stat().st_size == entry["bytes"],
            "Evidence path/size differs: " + filename,
        )
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        require(digest == entry["sha256"], "Evidence hash differs: " + filename)

    started = time.monotonic()
    # Bounded batches keep memory small while overlapping Windows file I/O.
    with ThreadPoolExecutor(max_workers=8) as pool:
        items = iter(manifest["files"].items())
        count = 0
        while batch := list(itertools.islice(items, 5000)):
            list(pool.map(check, batch))
            count += len(batch)
            print(
                f"{count}/{len(manifest['files'])} verified "
                f"({time.monotonic() - started:.0f}s)",
                flush=True,
            )
    result = {
        "archive": str(archive),
        "remote_path": remote,
        "bytes": receipt["bytes"],
        "sha256": actual,
        "verified_files": len(manifest["files"]),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "scope": "stdlib byte integrity only; no local numerical evaluation",
        "method": "authenticated post-extraction byte pass after interrupted loop",
    }
    (base / f"{name}-download.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
