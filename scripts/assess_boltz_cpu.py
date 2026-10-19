"""Authenticate allocated CPU test outcomes before native GPU readiness."""

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
    for name in ("legacy", "native-mapping", "chemistry"):
        path = args.run / (name + ".xml")
        cases = ET.parse(path).findall(".//testcase")
        if not cases or any(
            case.find(tag) is not None
            for case in cases
            for tag in ("failure", "error", "skipped")
        ):
            raise ValueError("Missing/failed/skipped CPU checks: " + name)
        total += len(cases)
        proofs.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "passed": len(cases),
            }
        )
    if total < 281:
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
