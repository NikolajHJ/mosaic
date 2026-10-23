"""Side-effect-free source/config parsing; never import numerical project code.

This is a static check, not execution of the HPC fixtures or model validation.
The receipt records exact checked bytes and the Python environment.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    paths = sorted(set(root.glob("src/**/*.py")) | set(root.glob("scripts/**/*.py")) | set(root.glob("tests/**/*.py")))
    checked = {}
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        checked[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    configurations = [root / "configs/large_triangle_protocol.json", root / "configs/large_triangle_bindings.template.json"]
    for path in configurations:
        json.loads(path.read_text())
        checked[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"status": "passed_static_only", "python_files": len(paths), "json_files": len(configurations),
              "checked_files": checked, "command": [sys.executable, *sys.argv], "python": sys.version,
              "platform": platform.platform(), "scope": "AST and JSON parsing only; no numerical/project imports",
              "numerical_tests_executed": False, "hpc_access_attempted": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "python_files", "json_files", "scope")}))


if __name__ == "__main__":
    main()
