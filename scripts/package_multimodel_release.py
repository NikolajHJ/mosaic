"""Package current source, including uncommitted files, without touching HPC.

Local packaging is allowed; this performs no molecular/model numerical test.
Keep each output path unique. Source/config files must remain unchanged while
packaging; the before/after identity check rejects a concurrent edit.
"""

import argparse
import io
import json
from pathlib import Path
import tarfile

from latentmol.experiments import file_sha256, source_manifest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, default=Path.cwd())
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    before = source_manifest(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = set(before["files"])
    for directory in ("docs",):
        paths.update(str(path.relative_to(args.source_root)).replace("\\", "/")
                     for path in (args.source_root / directory).glob("*.md"))
    paths.update(name for name in ("AGENTS.md", "README.md", "requirements-boltz-hpc.lock.txt")
                 if (args.source_root / name).is_file())
    with tarfile.open(args.output, "x:gz") as archive:
        for relative in sorted(paths):
            archive.add(args.source_root / relative, arcname=relative, recursive=False)
        payload = json.dumps(before, indent=2).encode()
        info = tarfile.TarInfo("release_manifest.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    if source_manifest(args.source_root)["sha256"] != before["sha256"]:
        raise RuntimeError("Source changed while packaging; retain this failed attempt and create a new archive")
    receipt = {"archive": str(args.output.resolve()), "archive_sha256": file_sha256(args.output),
               "source_manifest_sha256": before["sha256"], "status": "packaged_not_submitted"}
    write_json(args.output.with_suffix(args.output.suffix + ".json"), receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
