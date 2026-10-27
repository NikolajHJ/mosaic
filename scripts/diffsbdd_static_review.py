"""Record source/JSON/byte checks without importing any numerical module.

This is source inspection, not fixture execution or native scientific evidence.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect(output):
    paths = sorted((ROOT / "src/latentmol/diffsbdd").glob("*.py"))
    paths += sorted((ROOT / "tests").glob("test_diffsbdd_*.py"))
    paths += [ROOT / "scripts/diffsbdd_assets.py", ROOT / "scripts/diffsbdd_campaign.py",
              ROOT / "scripts/diffsbdd_static_review.py", ROOT / "scripts/hpc/diffsbdd_operator.py"]
    members, definitions = {}, 0
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 11))
        if path.parent.name == "tests":
            definitions += sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                               and node.name.startswith("test_") for node in ast.walk(tree))
        members[path.relative_to(ROOT).as_posix()] = sha(path)
    config = ROOT / "configs/diffsbdd_vae_alternation_v1.json"
    protocol = json.loads(config.read_text())
    pdb = ROOT / "experiments/diffsbdd-vae-alternation/inputs/3MXF.pdb"
    pdb_equal = sha(pdb) == protocol["pocket"]["receptor_sha256"]
    if not pdb_equal:
        raise ValueError("Immutable crystal bytes differ from the protocol")
    receipt = dict(version="diffsbdd-local-static-inspection-v1",
        observation_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope="AST parsing, JSON decoding and byte hashes only; no numerical imports or tests",
        parsed_python_files=len(paths), authored_test_function_definitions=definitions,
        ast_syntax="Python3.11", source_members=members, config_sha256=sha(config),
        crystal_sha256=sha(pdb), crystal_bytes_match_protocol=pdb_equal,
        numerical_tests_executed=0, models_loaded=0, training_updates=0,
        jobs_submitted=0, native_validation="not_run", mapping_validation="not_run")
    with Path(output).open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,indent=2,sort_keys=True)
        stream.write("\n")
    print(json.dumps({key:value for key,value in receipt.items() if key != "source_members"},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    inspect(parser.parse_args().output)


if __name__ == "__main__":
    main()
