"""Local source/environment receipts; no numerical work or network requests."""

import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess

from .io import file_sha256, identity_digest


def source_identity(root: str | Path) -> dict:
    """Hash execution and test contents, including all uncommitted files.

    Documentation and generated run artifacts are excluded from execution
    identity, but included in the separately produced immutable release bundle.
    """
    root = Path(root).resolve()
    paths = [root / "pyproject.toml"]
    for folder in ("src", "tests", "examples", "scripts"):
        paths.extend(
            path
            for path in (root / folder).rglob("*")
            if path.is_file()
            and path.suffix in (".py", ".json", ".lsf", ".sh", ".toml", ".txt")
            and "__pycache__" not in path.parts
        )
    files = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(set(paths))
    }
    archived_manifest = root / "source_manifest.json"
    revision = None
    if archived_manifest.exists():
        revision = json.loads(archived_manifest.read_text(encoding="utf-8"))[
            "git_revision_reference"
        ]
    else:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            revision = result.stdout.strip() if result.returncode == 0 else None
        except FileNotFoundError:
            pass
    return {
        "git_revision_reference": revision,
        "files": files,
        "execution_content_sha256": identity_digest(files),
    }


def environment_identity() -> dict:
    # Used by the allocated runtime entry point, never by static packaging.
    import jax

    packages = {}
    for dist in importlib.metadata.distributions():
        if dist.metadata["Name"]:
            record = {"version": dist.version}
            direct_url = dist.read_text("direct_url.json")
            if direct_url:
                record["direct_url"] = json.loads(direct_url)
            packages[dist.metadata["Name"].lower()] = record
    receipt = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": dict(sorted(packages.items())),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "devices": [
            {"platform": device.platform, "kind": device.device_kind, "id": device.id}
            for device in jax.devices()
        ],
    }
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        receipt["gpu_driver"] = (
            driver.stdout.strip() if driver.returncode == 0 else None
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        receipt["gpu_driver"] = None
    return receipt
