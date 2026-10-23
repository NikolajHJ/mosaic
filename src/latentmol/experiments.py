"""Local-to-the-cluster experiment records with immutable IDs and failure capture.

Each command and training run gets a new directory. Machine-readable metadata,
append-only metrics, stdout, and artifacts supplement the human experiment ledger.
No tracking server or internet connection is required on compute nodes.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import traceback
import uuid

from latentmol.config_identity import CONFIG_HASH_VERSION, config_sha256, normalize_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(root: Path) -> dict:
    """Hash executable source, tests and configs, including uncommitted content."""
    files = {}
    for folder in ("src", "tests", "configs", "scripts"):
        if (root / folder).exists():
            for path in sorted((root / folder).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    files[path.relative_to(root).as_posix()] = file_sha256(path)
    for name in ("pyproject.toml", "requirements-hpc.txt"):
        if (root / name).is_file():
            files[name] = file_sha256(root / name)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"root": str(root), "sha256": digest, "files": files}


def write_json(path: Path, value):
    """Replace a small metadata file atomically on its own filesystem."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class ExperimentRun:
    """Context manager that records both successful runs and ordinary exceptions."""

    def __init__(self, root: Path, name: str, config: dict, source_root: Path, *, report_kind: str | None = None):
        if report_kind not in (None, "molecule_optimization"):
            raise ValueError("Unsupported experiment presentation kind")
        self.report_kind = report_kind
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError("Run names may contain letters, digits, underscores and hyphens")
        # Normalize before hashing and saving. Integer count keys must survive
        # a JSON roundtrip without changing their ordering or losing collisions.
        config = normalize_json(config)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{stamp}_{name}_{uuid.uuid4().hex[:8]}"
        self.path = root / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.record = {
            "run_id": self.run_id, "name": name, "status": "created", "created_at": utc_now(),
            "config": config, "source": source_manifest(source_root),
            "source_archive_sha256": os.environ.get("LATENTMOL_SOURCE_SHA256"),
            "git_commit": os.environ.get("LATENTMOL_GIT_COMMIT"),
            "config_sha256": config_sha256(config),
            "config_hash_version": CONFIG_HASH_VERSION,
            "environment": {"host": socket.gethostname(), "python": sys.version,
                            "platform": platform.platform(), "job_id": os.environ.get("LSB_JOBID"),
                            "packages": {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"]}},
        }
        self._save()

    def _save(self):
        write_json(self.path / "run.json", self.record)

    def __enter__(self):
        self.record.update(status="running", started_at=utc_now())
        self._save()
        print(f"RUN_DIR={self.path}", flush=True)
        return self

    def log(self, values: dict):
        """Append one finite JSON metric event; never silently serialize NaN."""
        event = {"at": utc_now(), **values}
        with (self.path / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")
        print(json.dumps(event, allow_nan=False), flush=True)

    def annotate(self, **values):
        self.record.update(values)
        self._save()

    def __exit__(self, exception_type, exception, tb):
        self.record.update(status="failed" if exception else "completed", finished_at=utc_now())
        if exception:
            self.record["error"] = str(exception)
            (self.path / "traceback.txt").write_text("".join(traceback.format_exception(exception_type, exception, tb)))
        self._save()
        if self.report_kind == "molecule_optimization":
            # The terminal scientific record is authoritative and stays stable
            # while the derived report hashes it. Presentation has a separate
            # receipt; a renderer failure never replaces a scientific traceback.
            try:
                from latentmol.reporting.optimization import default_report
                default_report(self.path)
            except Exception as report_error:
                print(f"Interactive report presentation failed: {report_error}", file=sys.stderr)
        return False


def main():
    """Wrap preprocessing/tests/other commands so failures have durable records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command after -- is required")
    with ExperimentRun(args.run_root, args.name, {"command": command}, args.source_root) as run:
        with (run.path / "command.log").open("w") as logfile:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                logfile.write(line)
                logfile.flush()
                print(line, end="", flush=True)
            code = process.wait()
        run.annotate(returncode=code)
        if code:
            raise RuntimeError(f"Command exited with status {code}: {command}")


if __name__ == "__main__":
    main()
