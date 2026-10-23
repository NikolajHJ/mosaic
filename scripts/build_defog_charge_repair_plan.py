"""Metadata-only plan from the verified source079 capsule; no model imports.

Run locally to pin original remote request/run bytes. The plan neither submits
jobs nor regenerates candidates. Numerical replay runs only in an LSF job.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence
    archive = Path(str(root) + ".tar.gz")
    expected = "b24a477c6764458fe3cd827bd8d9c47fba1237fd607df8c93013af5c32916163"
    if sha(archive) != expected:
        raise ValueError("Require the final authenticated source079 capsule")
    origins = json.loads((root / "export_scope.json").read_text())["origins"]
    manifest = json.loads((root / "evidence_manifest.json").read_text())["files"]
    stems = {0: "20261008T232742Z_ramped-prior-0_58408977",
             8: "20261009T102534Z_ramped-prior-8_ca54aba9",
             9: "20261009T120026Z_ramped-prior-9_a607ab2d"}
    cases = []
    for prior, attempt in ((0, 0), (8, 0), (8, 1), (9, 0), (9, 1)):
        stem = f"campaign/prior-{prior}/optimization/{stems[prior]}"
        request = f"{stem}/fixed_initial_mask/repair-refresh-00000/attempt-{attempt:03d}/request.json"
        def asset(name):
            actual = sha(root / name)
            if actual != manifest[name]:
                raise ValueError("Extracted original changed: " + name)
            return {"path": origins[name], "sha256": actual}
        cases.append({"id": f"prior-{prior}-attempt-{attempt}", "prior": prior,
                      "attempt": attempt, "request": asset(request), "parent_run": asset(stem + "/run.json")})
    plan = {"version": 1, "source079_archive_sha256": "744862c9a310f87ff078edb2ab3185fd2468eaa00339222d024c1ffe0ceebcce",
            "source079_manifest_sha256": "21025b7bdb84ceeaf515ebe885ec0941491acf797a6ada66ffa02d73f1a5f29f",
            "evidence_archive_sha256": expected, "cases": cases,
            "selection": "all prescribed paired requests; first eligible projected target per charged prior",
            "scope": "exact candidate adapter comparison; no Boltz, latent updates or production release"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"plan": str(args.output), "sha256": sha(args.output), "cases": len(cases)}))


if __name__ == "__main__":
    main()
