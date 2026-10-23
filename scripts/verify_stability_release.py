"""Check the submitted release identity again inside LSF before running a stage.

Metadata-only guard; no models or numerical experiments run here. The identity
arguments belong to the recorded bsub intent, so an inherited environment or
changed archive cannot silently redefine which source was queued.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from latentmol.experiments import file_sha256, source_manifest


def verify(archive_sha, manifest_sha):
    if not os.environ.get("LSB_JOBID"):
        raise ValueError("DTU allocation required")
    archive = Path(os.environ["LATENTMOL_SOURCE_ARCHIVE"])
    if (file_sha256(archive) != archive_sha
            or os.environ.get("LATENTMOL_SOURCE_SHA256") != archive_sha):
        raise ValueError("Archive differs from the submitted immutable source")
    recorded = json.loads(Path("release_manifest.json").read_text())["sha256"]
    if recorded != manifest_sha or source_manifest(Path.cwd())["sha256"] != manifest_sha:
        raise ValueError("Executable source differs from the submitted manifest")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-sha256", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        p.error("Require a command after --")
    verify(a.archive_sha256, a.manifest_sha256)
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
