"""Authenticate a ready resource assessment before the long BRD4 allocation."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--pilot-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.assessment.read_text())
    if (
        receipt["status"] != "ready"
        or receipt["pilot_summary_sha256"] != args.pilot_sha256
        or receipt["config_sha256"]
        != hashlib.sha256(args.config.read_bytes()).hexdigest()
    ):
        raise ValueError(
            "BRD4 launch is not bound to passed pilot/config/resource gates"
        )
    print(receipt["required_free_gib"])


if __name__ == "__main__":
    main()
