"""Seal actual native gates/pilots for both requested initializations."""

import argparse
import hashlib
import json
from pathlib import Path

from check_boltz_cpu_proof import verify


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    cpu = verify(args.cpu, args.manifest)
    identities, proofs, configs, runtimes = [], list(cpu["proofs"]), {}, {}
    for arm in ("mix50", "random"):
        root = args.run / arm
        summary = json.loads((root / "summary.json").read_text())
        if (
            summary["status"] != "passed"
            or summary["mode"] != "validate"
            or summary["native_hard_parity"]["status"] != "passed"
            or summary["native_gradient_checks"]["status"] != "passed"
            or summary["chemistry_assay"]["status"] != "passed"
            or summary["runtime_gate"]["status"] != "passed"
        ):
            raise ValueError("Native readiness gate failed: " + arm)
        for label in (
            "early_coupled_checks",
            "high_coupled_checks",
            "early_pilot",
            "high_pilot",
        ):
            report = json.loads((root / label / "report.json").read_text())
            if report["status"] != "passed":
                raise ValueError("Failed combined native proof: " + arm + "/" + label)
        identities.append(summary["native_identity"])
        configs[f"brd4_boltz_affinity_{arm}_10000.json"] = summary["config_sha256"]
        runtimes[arm] = summary["runtime_gate"]
        for path in root.rglob("*.json"):
            proofs.append({"path": str(path), "sha256": sha(path)})
    if identities[0] != identities[1]:
        raise ValueError("Arms used different native model/context identities")
    context = args.run / "prepared/context.pt"
    if sha(context) != identities[0]["context_file_sha256"]:
        raise ValueError("Prepared context changed after native gates")
    result = {
        "status": "passed",
        "source_manifest_sha256": args.manifest,
        "model_digest": identities[0]["model_digest"],
        "native_identity": identities[0],
        "context_sha256": sha(context),
        "reference_sha256": cpu["reference_sha256"],
        "configs": configs,
        "runtime_gates": runtimes,
        "proofs": proofs,
        "cpu_assessment_sha256": sha(args.cpu / "assessment.json"),
    }
    (args.run / "assessment.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
