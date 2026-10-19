"""Authenticate actual allocated fixture/reference outputs before native launch."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_ramp_config(config, baseline):
    """Versioned campaign delta; retain every source/model/data/repair pin."""
    expected = {
        **baseline,
        "version": "brd4-contact-chemistry-ramp-v2",
        "steps": 10000,
        "repair_schedule": {
            "version": "delayed-geometric-v1",
            "initial": 0.001,
            "final": 50.0,
            "hold_updates": 5000,
            "total_updates": 10000,
        },
    }
    del expected["repair_weight"], expected["mixing"]
    recipes = (
        (
            {"kind": "chain_uniform", "mixing": 0.5},
            "50% chain/uniform · delayed chemistry",
        ),
        (
            {"kind": "dirichlet", "concentration": 1.0, "seed": 20261019},
            "Random Dirichlet · delayed chemistry",
        ),
    )
    if not any(
        config == {**expected, "initialization": recipe, "label": label}
        for recipe, label in recipes
    ):
        raise ValueError("Ramp configuration changed a declared setting or identity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--additional-config", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Readiness belongs to the allocated CPU job")
    config = json.loads(args.config.read_text())
    expected = {
        "version": "brd4-contact-chemistry-logits-v1",
        "n_atoms": 6,
        "mixing": 0.5,
        "seed": 17,
        "steps": 5000,
        "initial_step_size": 0.03,
        "max_backtracks": 10,
        "matmul_precision": "highest",
        "repair_weight": 0.5,
        "element_js_weight": 0.1,
    }
    if (
        any(config[k] != v for k, v in expected.items())
        or config["progress_search"]["allow_atom_edits"]
    ):
        raise ValueError("Declared chemistry run settings changed")
    if config["search"] != {
        "max_edits": 3,
        "checks_per_depth": 32,
        "beam_width": 4,
        "probability_weight": 1.0,
        "probability_floor": 1e-12,
    }:
        raise ValueError("Original repair budget changed")
    if any(
        config["progress_search"][k] != v
        for k, v in {
            "version": "repair-progress-v1",
            "max_edits": 6,
            "checks_per_depth": 32,
            "beam_width": 4,
            "probability_weight": 1.0,
            "allow_atom_edits": False,
            "atom_edit_penalty": 3.0,
            "max_atom_edits": 1,
        }.items()
    ):
        raise ValueError("Additional repair policy changed")
    if config["loss"] != {
        "version": "repair-failure-guidance-v1",
        "extra_valid_weight": 1.0,
        "partial_weight": 1.0,
        "local_bond_weight": 0.1,
        "component_cut_weight": 0.1,
        "nonempty_weight": 0.1,
    }:
        raise ValueError("Internal repair coefficients changed")
    for path in args.additional_config:
        validate_ramp_config(json.loads(path.read_text()), config)
    receipts = {}
    for name in ("legacy-unit.xml", "chemistry-unit.xml"):
        suites = list(ET.parse(args.run / name).getroot().iter("testsuite"))
        if not suites or any(
            int(v.get("failures", 0))
            + int(v.get("errors", 0))
            + int(v.get("skipped", 0))
            for v in suites
        ):
            raise ValueError(
                "Required allocated tests failed, skipped or are missing: " + name
            )
        receipts[name] = {
            "sha256": sha(args.run / name),
            "tests": sum(int(v.get("tests", 0)) for v in suites),
        }
        if receipts[name]["tests"] < 1:
            raise ValueError("Empty test proof: " + name)
    reference = json.loads((args.run / "composition_reference.json").read_text())
    oracles = json.loads((args.run / "chemistry_oracles.json").read_text())
    if (
        reference["status"] != "passed"
        or reference["data"] != config["composition_data"]
        or len(oracles["cases"]) != 12
    ):
        raise ValueError("Empirical/oracle evidence differs")
    result = {
        "status": "passed",
        "job_id": os.environ["LSB_JOBID"],
        "config_sha256": sha(args.config),
        "configs": {
            path.relative_to(Path(__file__).resolve().parents[1]).as_posix(): sha(path)
            for path in [
                args.config.resolve(),
                *(p.resolve() for p in args.additional_config),
            ]
        },
        "source_manifest_sha256": sha(
            Path(__file__).resolve().parents[1] / "source_manifest.json"
        ),
        "reference": str((args.run / "composition_reference.json").resolve()),
        "reference_sha256": sha(args.run / "composition_reference.json"),
        "oracles_sha256": sha(args.run / "chemistry_oracles.json"),
        "tests": receipts,
        "scope": "allocated CPU engineering/parity/reference evidence; native GPU pilot still required",
    }
    with (args.run / "assessment.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
