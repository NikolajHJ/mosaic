"""Extract detached DBMol repair code without importing or executing it.

This deliberately excludes Torch, decoders, training and the old expected-
valence objective. Original symbol bytes and source file hashes are retained
for review and parity checks on allocated HPC. No numerical work runs here.
"""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
import ast
import hashlib
import json
from pathlib import Path


SELECTIONS = {
    "representation": (
        ["GraphSpec"],
        "from dataclasses import asdict, dataclass\nimport hashlib\nimport json\n",
    ),
    "quality": (
        [
            "VERSION",
            "_array",
            "_labels",
            "graph_identity",
            "_components",
            "diagnose_graph",
        ],
        "import hashlib\nimport json\nimport numpy as np\nfrom rdkit import Chem, rdBase\nfrom .chemistry import inspect_graph\nfrom .representation import GraphSpec\n",
    ),
    "probability_repair": (
        [
            "ProbabilityRepairConfig",
            "_array",
            "validate_probabilities",
            "probability_identity",
            "measure_probability_edits",
        ],
        "from dataclasses import asdict, dataclass\nimport hashlib\nimport math\nimport numpy as np\nfrom .quality import graph_identity\n",
    ),
    "atom_preserving_chemistry": (
        [
            "VERSION",
            "repair_policy",
            "_key",
            "_bond_edits",
            "search_atom_preserving_repair",
        ],
        "import math\nimport numpy as np\nfrom .chemistry import encode_smiles\nfrom .probability_repair import ProbabilityRepairConfig, validate_probabilities, probability_identity, measure_probability_edits\nfrom .quality import diagnose_graph\nfrom .repair_search import round_robin_candidates\nfrom .representation import GraphSpec\n",
    ),
    "chemistry_regularization": (
        ["RULE_VERSION", "RULE_SOURCES", "ALLOWED_VALENCES"],
        "",
    ),
}
WHOLE = ("chemistry", "repair_search", "repair_progress")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh only an unmodified prior extraction",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_repository_root(__file__)
        / "src/mosaic/small_molecule/chemistry/_reference",
    )
    parser.add_argument(
        "--oracle-output",
        type=Path,
        default=_layout_path(_repository_root(__file__)
        / "tests/data/chemistry_oracle/src/latentmol"),
    )
    args = parser.parse_args()
    if args.refresh:
        previous = json.loads((args.output / "source_manifest.json").read_text())
        for name, entry in previous["files"].items():
            if digest((args.output / name).read_bytes()) != entry["extracted_sha256"]:
                raise ValueError("Existing reference extraction was modified: " + name)
        if args.oracle_output.exists():
            previous_raw = {
                name: entry["source_sha256"]
                for name, entry in previous["files"].items()
            }
            previous_raw.update(previous.get("oracle_source_sha256", {}))
            for name, expected in previous_raw.items():
                if digest((args.oracle_output / name).read_bytes()) != expected:
                    raise ValueError("Original oracle snapshot was modified: " + name)
    args.output.mkdir(parents=True, exist_ok=args.refresh)
    manifest = {
        "source": "DBMol_exploratory/src/latentmol",
        "method": "verbatim selected AST symbols or whole detached modules; explicit import headers",
        "files": {},
    }
    manifest["oracle_source_sha256"] = {
        name: digest((args.source / "src/latentmol" / name).read_bytes())
        for name in ("repair_failure_guidance.py", "sampling.py", "async_repair.py")
    }
    for name in (*SELECTIONS, *WHOLE):
        raw = (args.source / "src/latentmol" / f"{name}.py").read_bytes()
        source = raw.decode("utf-8-sig")
        if name in SELECTIONS:
            wanted, header = SELECTIONS[name]
            nodes = {}
            for node in ast.parse(source).body:
                if isinstance(
                    node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    nodes[node.name] = node
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                    for target in targets:
                        if isinstance(target, ast.Name):
                            nodes[target.id] = node
            lines = source.splitlines(keepends=True)
            pieces = []
            for symbol in wanted:
                node = nodes[symbol]
                first = min(
                    [node.lineno]
                    + [v.lineno for v in getattr(node, "decorator_list", [])]
                )
                pieces.append("".join(lines[first - 1 : node.end_lineno]))
            content = (
                '"""Detached reference extraction; see source_manifest.json. Do not edit manually."""\nfrom __future__ import annotations\n'
                + header
                + "\n\n"
                + "\n\n".join(pieces)
            )
        else:
            wanted, content = ["whole_module"], source
        data = content.encode("utf-8")
        (args.output / f"{name}.py").write_bytes(data)
        manifest["files"][f"{name}.py"] = {
            "source_sha256": digest(raw),
            "extracted_sha256": digest(data),
            "symbols": wanted,
        }
    # Keep the exact original Torch implementation isolated in test data.
    # Production only imports the smaller Torch-free extraction above.
    expected = {
        name: entry["source_sha256"] for name, entry in manifest["files"].items()
    }
    expected.update(manifest["oracle_source_sha256"])
    args.oracle_output.mkdir(parents=True, exist_ok=args.refresh)
    for name, sha in expected.items():
        raw = (args.source / "src/latentmol" / name).read_bytes()
        if digest(raw) != sha:
            raise ValueError("Source changed during extraction: " + name)
        (args.oracle_output / name).write_bytes(raw)
    (args.oracle_output / "__init__.py").write_text(
        '"""Unmodified DBMol numerical oracle modules, used only on allocated CPU."""\n',
        newline="\n",
    )
    (args.output / "__init__.py").write_text(
        '"""Pinned detached reference helpers; no learned model or training dependency."""\n',
        newline="\n",
    )
    (args.output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", newline="\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
