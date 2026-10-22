"""Authenticate allocated CPU test outcomes before native GPU readiness."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    proofs, total = [], 0
    for name in ("hpc-contract", "legacy", "native-mapping", "chemistry"):
        path = args.run / (name + ".xml")
        cases = ET.parse(path).findall(".//testcase")
        if not cases or any(
            case.find(tag) is not None
            for case in cases
            for tag in ("failure", "error", "skipped")
        ):
            raise ValueError("Missing/failed/skipped CPU checks: " + name)
        if name == "hpc-contract" and len(cases) < 16:
            raise ValueError("HPC launch/temp-directory coverage unexpectedly reduced")
        total += len(cases)
        proofs.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "passed": len(cases),
            }
        )
    if total < 299:
        raise ValueError("CPU regression coverage unexpectedly reduced")
    result = {
        "status": "passed",
        "passed": total,
        "source_manifest_sha256": args.manifest,
        "proofs": proofs,
        "reference_sha256": hashlib.sha256(
            (args.run / "composition_reference.json").read_bytes()
        ).hexdigest(),
    }
    (args.run / "assessment.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
