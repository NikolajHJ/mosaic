"""Read-only producer proof check, shared by the allocated native launcher."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import hashlib
import json
from pathlib import Path
import sys


def verify(cpu, manifest):
    assessment = json.loads((cpu / "assessment.json").read_text())
    if (
        assessment["status"] != "passed"
        or assessment["passed"] < 299
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
