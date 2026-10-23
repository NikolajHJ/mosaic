"""Authenticate an extracted source release without importing Mosaic."""

import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path(sys.argv[1]).resolve()
    manifest = root / "source_manifest.json"
    if sha256(manifest) != sys.argv[2]:
        raise ValueError("Source manifest does not match independently recorded SHA256")
    receipt = json.loads(manifest.read_text(encoding="utf-8"))
    for folder in ("src", "tests", "examples", "scripts"):
        for path in (root / folder).rglob("*"):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.relative_to(root).as_posix() not in receipt["files"]
            ):
                raise ValueError(
                    f"Unmanifested execution input: {path.relative_to(root)}"
                )
    for name, expected in receipt["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != expected:
            raise ValueError(f"Source content mismatch: {name}")
    print(f"Authenticated {len(receipt['files'])} immutable source files", flush=True)


if __name__ == "__main__":
    main()
