"""Read-only producer proof check, shared by the allocated native launcher."""

import hashlib
import json
from pathlib import Path
import sys


def verify(cpu, manifest):
    assessment = json.loads((cpu / "assessment.json").read_text())
    if (
        assessment["status"] != "passed"
        or assessment["passed"] < 281
        or assessment["source_manifest_sha256"] != manifest
    ):
        raise ValueError("CPU assessment does not certify this immutable release")
    for proof in assessment["proofs"]:
        if (
            hashlib.sha256(Path(proof["path"]).read_bytes()).hexdigest()
            != proof["sha256"]
        ):
            raise ValueError("CPU test evidence changed")
    if (
        hashlib.sha256((cpu / "composition_reference.json").read_bytes()).hexdigest()
        != "37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb"
    ):
        raise ValueError("Historical chemistry composition reference changed")
    return assessment


if __name__ == "__main__":
    verify(Path(sys.argv[1]), sys.argv[2])
