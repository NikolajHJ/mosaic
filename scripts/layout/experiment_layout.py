"""Working-source paths and legacy entry points, using only the standard library.

Historical immutable releases do not use this registry. A canonical candidate
is returned only if it actually exists alongside the referenced source tree.
"""

import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
_RECORDS = None


def records():
    global _RECORDS
    if _RECORDS is None:
        path = _ROOT / "experiments/migration-map.json"
        if not path.is_file():
            raise RuntimeError(
                "Incomplete migrated source release: include experiments/, "
                "scripts/layout/ and the migration map; do not package only legacy shims."
            )
        _RECORDS = json.loads(path.read_text(encoding="utf-8"))["files"]
        nested = _ROOT / "experiments/folder-layout.json"
        if nested.is_file():
            layout = json.loads(nested.read_text(encoding="utf-8"))
            _RECORDS.update(layout["shared_tests"])
            for old, adapter in layout["relocated_compatibility"].items():
                _RECORDS[old] = dict(_RECORDS[old], compatibility_path=adapter)
    return _RECORDS


def repository_root(_file=None):
    """The root containing this layout runtime, including minimal HPC releases."""
    return _ROOT


def activate(_file=None, *, include_tests=False):
    """Make unique canonical script modules importable by their established names.

    Names remain stable for native controller/test contracts. Modules are loaded
    normally into sys.modules, so monkeypatching and function globals still agree.
    Tests opt into their unique helper-module directories without importing them.
    """
    paths = sorted({
        str((_ROOT / item["path"]).parent)
        for old, item in records().items()
        if old.endswith(".py") and (include_tests or not old.startswith("tests/"))
    })
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)


def resolve(path):
    """Resolve a legacy working path; keep existing old sealed-source paths valid.

    This also supports synthetic source trees in allocated tests: absent canonical
    files fall back to the supplied legacy location rather than inventing inputs.
    No write, model import, numerical evaluation or scheduler access occurs here.
    """
    path = Path(path)
    parts = path.parts
    for index, part in enumerate(parts):
        if part not in {"scripts", "examples", "tests"}:
            continue
        old = Path(*parts[index:]).as_posix()
        item = records().get(old)
        if item is not None:
            base = Path(*parts[:index]) if index else Path(".")
            target = base / item["path"]
            if target.is_file():
                return target
    return path


def execute_legacy(namespace, target):
    """Execute a single canonical implementation in the legacy module namespace."""
    target = Path(target)
    if not target.is_file():
        raise RuntimeError(f"Incomplete migrated source release: missing {target}")
    activate(include_tests=True)
    namespace["__file__"] = str(target)
    exec(compile(target.read_bytes(), str(target), "exec"), namespace)


def release_aliases(root):
    """Read declared legacy aliases for a new archive, without changing the tree.

    Existing sealed releases are never modified. Alias bytes are ordinary source
    inputs authenticated in the new manifest, not post-authentication symlinks.
    """
    root = Path(root).resolve()
    layout = root / "experiments/folder-layout.json"
    if not layout.is_file():
        return {}
    result = {}
    for old, stored in json.loads(layout.read_text(encoding="utf-8"))["release_only_aliases"].items():
        old_path, source = root / old, (root / stored).resolve()
        if Path(old).parent.as_posix() not in {"scripts", "tests"} or not source.is_relative_to(root):
            raise ValueError(f"Invalid release compatibility alias: {old}")
        if not source.is_file():
            raise ValueError(f"Missing release compatibility source: {stored}")
        data = source.read_bytes()
        if old_path.exists() and old_path.read_bytes() != data:
            raise ValueError(f"Release compatibility alias conflicts with existing file: {old}")
        result[old] = data
    return result


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "resolve":
        print(resolve(Path(sys.argv[2])))
        return
    if len(sys.argv) == 3 and sys.argv[1] == "test-paths":
        from fnmatch import fnmatchcase
        paths = [resolve(_ROOT / old) for old in sorted(records())
                 if old.startswith("tests/") and fnmatchcase(old, sys.argv[2])]
        if not paths or any(not path.is_file() for path in paths):
            raise SystemExit("No complete canonical test selection for " + sys.argv[2])
        print("\n".join(str(path) for path in paths))
        return
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        import runpy
        target = resolve(_ROOT / sys.argv[2])
        if target.suffix != ".py" or not target.is_file():
            raise SystemExit(f"No Python implementation for {sys.argv[2]}")
        sys.argv = [str(target), *sys.argv[3:]]
        activate(include_tests=True)
        runpy.run_path(str(target), run_name="__main__")
        return
    raise SystemExit("usage: experiment_layout.py resolve PATH | test-paths OLD_GLOB | run OLD_PYTHON_PATH [ARGS ...]")


if __name__ == "__main__":
    main()
