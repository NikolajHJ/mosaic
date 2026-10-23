"""DTU disallows user bmod: replace a held, never-started pilot safely.

Science/source/body remain the sealed source071 experiment. Only job name,
log paths and scheduler dependency change. Submit once before retiring the
old PSUSP job; exclusive receipts protect against uncertain network outcomes.
"""
import argparse
import json
from pathlib import Path
import shlex

from latentmol.experiments import file_sha256, source_manifest
from submit_stability_study import authenticated, read, submit
from submit_decoder_budget import action, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=Path.cwd())
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--work-root", type=Path, default=Path("/work3/s234818/latentmol"))
    parser.add_argument("--recovery-receipts", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    work, release = args.work_root.resolve(strict=True), args.release.resolve(strict=True)
    if not release.is_relative_to(work / "releases") or not args.recovery_receipts.resolve().is_relative_to(work):
        raise ValueError("Require immutable release and declared DTU receipt root")
    authenticated(Path(str(release) + ".tar.gz"), args.archive_sha256)
    if source_manifest(release)["sha256"] != read(release / "release_manifest.json")["sha256"]:
        raise ValueError("Operator source changed")
    recovery = args.recovery_receipts
    last = read(recovery / "seed23-receipt.json")
    cpu = read(recovery / "cpu-receipt.json")
    if last["intent"]["identity"] != cpu["intent"]["identity"]:
        raise ValueError("Recovery receipts differ")
    rejected = read(recovery / "retarget-search-reply.json")
    if rejected["returncode"] == 0 or "Only administrators can run bmod" not in rejected["stderr"]:
        raise ValueError("Expected retained DTU bmod rejection")
    gate = read(recovery / "retarget-search-gates.json")
    authenticated(Path(gate["original_completed_wrapper"]), gate["original_completed_wrapper_sha256"])
    if read(Path(gate["original_completed_wrapper"]))["status"] != "completed" or gate["superset_cpu"] != cpu["job_id"]:
        raise ValueError("Pilot CPU proof differs")
    old_root = work / "submissions/source071-search-decoder-attempt001"
    old = read(old_root / "native-receipt.json")
    pilot = old["job_id"]
    replacement_root = recovery / "search-replacement"
    if not (replacement_root / "retire-held-reply.json").exists() and state(pilot) != "PSUSP":
        raise ValueError("Replace only a held pilot; never cancel a running experiment")
    old_identity = old["intent"]["identity"]
    authenticated(Path(old_identity["release"] + ".tar.gz"), old_identity["archive_sha256"])
    if source_manifest(Path(old_identity["release"]))["sha256"] != old_identity["manifest_sha256"]:
        raise ValueError("Original pilot source changed")
    if Path(old_identity["output_root"]).exists():
        raise ValueError("Pilot already created scientific outputs; do not overwrite")
    argv = list(old["intent"]["command"][:-1])
    name = "source071-search-budget-replacement-attempt001"
    for flag, value in {"-J": name, "-w": f"done({cpu['job_id']}) && ended({last['job_id']})",
                        "-oo": str(work / "logs" / (name + ".%J.out")),
                        "-eo": str(work / "logs" / (name + ".%J.err"))}.items():
        argv[argv.index(flag) + 1] = value
    body = shlex.split(old["intent"]["command"][-1])
    plan = [{"stage": "search", "cwd": old["intent"]["cwd"], "arguments": argv,
             "job_command": body, "dependency": None}]
    identity = {"original_pilot": pilot, "original_scientific_identity": old_identity,
                "operator_archive_sha256": args.archive_sha256, "recovery": last["intent"]["identity"],
                "last_recovery_job": last["job_id"], "superset_cpu": cpu["job_id"]}
    if not args.submit:
        print(json.dumps({"identity": identity, "plan": plan}, indent=2))
        return
    jobs = submit(plan, replacement_root, identity)
    print(json.dumps(jobs), flush=True)
    # Submission/source verification can take time. Recheck at cancellation
    # boundary so a concurrently resumed old experiment is never retired here.
    if not (replacement_root / "retire-held-reply.json").exists() and state(pilot) != "PSUSP":
        raise RuntimeError("Replacement submitted; old pilot is no longer held. Reconcile without cancelling a running job")
    action(replacement_root, "retire-held", ["bkill", pilot])
    print("Verified replacement submitted; old held pilot retired", flush=True)


if __name__ == "__main__":
    main()
