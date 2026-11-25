"""LSF entry point for one encoder candidate alongside the existing epoch-48 control."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def seal():
    """Build a local candidate release by replacing only the changed production sources."""
    from submit_epoch48_continuation import (CONTROL_ROOT, EARLY_TAG, ROOT, ROUNDTRIP_TAG,
                                             SUBMISSIONS, build_archive, sha)

    tag = "vae-e48-encoder3-nextleg-20261123a"
    old = SUBMISSIONS / ROUNDTRIP_TAG / "source.tar.gz"
    production = SUBMISSIONS / EARLY_TAG
    plan = json.loads((production / "plan.json").read_bytes())
    if sha(old) != plan["archive_sha256"]:
        raise ValueError("Production code snapshot differs from the planned release")
    changed = ["src/latentmol/models/encoder.py", "src/latentmol/models/autoencoder.py",
               "src/latentmol/training/projected_continuation.py", "src/latentmol/training/generated_roundtrip.py",
               "experiments/vae/run_checkpoint_comparison.sh"]
    added = ["src/latentmol/training/encoder_expansion.py", "src/latentmol/training/confidence_margins.py",
             "experiments/vae/encoder_upgrade_pilot.py", "experiments/vae/compare_encoder_upgrade.py",
             "experiments/vae/epoch48-encoder-upgrade.json"]
    sources = {p: (ROOT / p).read_bytes().replace(b"\r\n", b"\n") for p in changed + added}
    directory = SUBMISSIONS / tag
    archive = build_archive(directory, old, sources, changed, added)
    for name in ("hpc_launch.py", "vae_segment_chain.py"):
        shutil.copyfile(production / name, directory / name)
    shutil.copyfile(Path(__file__), directory / Path(__file__).name)
    release = dict(source_sha256=sha(archive), production_control_dir=CONTROL_ROOT + EARLY_TAG,
                   production_plan_sha256=sha(production / "plan.json"), callback=30264093)
    (directory / "release.json").write_text(json.dumps(release, indent=2), encoding="utf8")
    bundle = directory / "bundle.tar.gz"
    with tarfile.open(bundle, "x:gz") as tar:
        for name in ("release.json", "source.tar.gz", "hpc_launch.py", "vae_segment_chain.py", Path(__file__).name):
            tar.add(directory / name, arcname=name)
    print(json.dumps(dict(directory=str(directory), remote=CONTROL_ROOT + tag,
                          bundle_sha256=sha(bundle), **release), indent=2))


def launch(directory, plan, label, command, *, startup, duration):
    subprocess.run(["/usr/bin/python3", str(directory / "hpc_launch.py"), "--archive",
                    str(directory / "source.tar.gz"), "--archive-sha256", plan["source_sha256"],
                    "--release", str(directory / "release"), "--receipt", str(directory / f"{label}.launch.json"),
                    "--startup-seconds", str(startup), "--run-seconds", str(duration), "--", *command],
                   check=True)


def seal_early():
    """Fork the approved three-block experiment into two bounded allocations."""
    from submit_epoch48_continuation import CONTROL_ROOT, ROOT, SUBMISSIONS, build_archive, sha

    base = SUBMISSIONS / "vae-e48-encoder3-nextleg-20261123a"
    tag = "vae-e48-encoder3-early-8h-20261123a"
    dest = SUBMISSIONS / tag
    plan = json.loads((base / "release.json").read_bytes())
    if sha(base / "source.tar.gz") != plan["source_sha256"]:
        raise ValueError("Original candidate release changed")
    changed = ["experiments/vae/encoder_upgrade_pilot.py", "experiments/vae/run_checkpoint_comparison.sh"]
    added = ["experiments/vae/early_encoder_arms.py", "experiments/vae/encoder_interim.py"]
    changes = {p: (ROOT / p).read_bytes().replace(b"\r\n", b"\n") for p in changed + added}
    archive = build_archive(dest, base / "source.tar.gz", changes, changed, added)
    plan.update(source_sha256=sha(archive), superseded_release=CONTROL_ROOT + base.name,
                worker=30264092, superseded_candidate=30302772)
    (dest / "release.json").write_text(json.dumps(plan, indent=2), encoding="utf8")
    for name in ("hpc_launch.py", "vae_segment_chain.py"):
        shutil.copyfile(base / name, dest / name)
    for name in (Path(__file__).name, "stop_roundtrip_for_encoder_pair.py"):
        shutil.copyfile(Path(__file__).with_name(name), dest / name)
    with tarfile.open(dest / "bundle.tar.gz", "x:gz") as tar:
        for item in ("source.tar.gz", "hpc_launch.py", "vae_segment_chain.py", Path(__file__).name,
                     "stop_roundtrip_for_encoder_pair.py", "release.json"):
            tar.add(dest / item, arcname=item)
    print(json.dumps(dict(remote=CONTROL_ROOT + tag, bundle_sha256=sha(dest / "bundle.tar.gz"), **plan)))


def early(directory, plan, action):
    """A single control, a single candidate and two read-only reports."""
    prod = Path(plan["production_control_dir"])
    if hashlib.sha256((prod / "plan.json").read_bytes()).hexdigest() != plan["production_plan_sha256"]:
        raise ValueError("Unexpected production lineage")
    sys.path.insert(0, str(prod))
    from epoch48_segment_chain import submit_once
    from vae_segment_chain import read

    script = ["/usr/bin/python3", str(Path(__file__).resolve())]
    name = lambda role: f"e48-fork3-{role}"
    if action == "stage":
        ids = {}
        for role, wall in (("candidate", "24:00"), ("control", "08:00")):
            ids[role] = submit_once(directory, name(role), [
                "-H", "-q", "gpuv100", "-n", "8", "-R", "select[gpu32gb] span[hosts=1] rusage[mem=6144]",
                "-gpu", "num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=30G", "-W", wall,
                "-env", "all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
                *script, role])
        for role, cores, wall, dep in (("interim", 1, "00:10", f"ended({ids['control']})"),
                                        ("final", 4, "01:30", f"ended({ids['control']}) && ended({ids['candidate']})")):
            ids[role] = submit_once(directory, name(role), [
                "-q", "hpc", "-n", str(cores), "-R", "span[hosts=1] rusage[mem=2048]", "-W", wall,
                "-w", dep, *script, role])
        print(json.dumps(ids))
        return
    if int(os.environ["LSB_JOBID"]) != read(directory / (name(action) + ".receipt.json"))["id"]:
        raise ValueError("Allocation identity differs from approved job")
    if action in ("control", "candidate"):
        pinned = read(directory / "source/latest.json")
        if read(directory / "stop.json")["latest"] != pinned:
            raise ValueError("Missing final checkpoint handoff")
        base = read(prod / "plan.json")
        launch(directory, plan, action, ["/usr/bin/env", "LATENTMOL_CPU_THREADS=4",
               "LATENTMOL_EVALUATION_MODULE=experiments.vae.early_encoder_arms",
               "bash", "experiments/vae/run_checkpoint_comparison.sh", "--config", base["config"],
               "--parent-receipt", base["parent_receipt"], "--resume-receipt", str(directory / "source/latest.json"),
               "--source-sha256", pinned["sha256"], "--production-control", str(directory / "control"),
               "--output", str(directory / action), "--arm", action],
               startup=2400, duration=28500 if action == "control" else 85000)
        return
    module = "encoder_interim" if action == "interim" else "compare_encoder_upgrade"
    launch(directory, plan, action, ["bash", "-lc",
           'set -eu; module load python3/3.11.9; unset PYTHONHOME PYTHONPATH; '
           'source /work3/s234818/latentmol/env/boltz311/bin/activate; cd "$DBMOL_LOCAL_SOURCE"; '
           'export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; '
           f'exec python -m experiments.vae.{module} "$@"', action,
           "--control", str(directory / "control"), "--candidate", str(directory / "candidate"),
           "--output", str(directory / ("interim-report" if action == "interim" else "comparison"))],
           startup=420 if action == "interim" else 4800, duration=500 if action == "interim" else 5000)


def seal_next():
    """Overlay the adopted objective and candidate resume entry on the actual predecessor release."""
    from submit_epoch48_continuation import ROOT, SUBMISSIONS, build_archive, sha

    base = SUBMISSIONS / "vae-e48-encoder3-early-8h-20261123a"
    directory = SUBMISSIONS / "vae-e48-encoder-invented-bonds-20261124a"
    source = base / "source.tar.gz"
    previous = json.loads((base / "release.json").read_bytes())
    if sha(source) != previous["source_sha256"]:
        raise ValueError("Predecessor archive digest differs")
    changed = ["src/latentmol/training/projected_continuation.py", "src/latentmol/training/generated_roundtrip.py",
               "src/latentmol/training/occupancy_monitor.py", "experiments/vae/encoder_upgrade_pilot.py",
               "experiments/vae/run_checkpoint_comparison.sh"]
    added = ["src/latentmol/training/spurious_bonds.py", "experiments/vae/encoder_next_leg.py",
             "experiments/vae/epoch48-encoder-targets.json"]
    files = {name: (ROOT / name).read_bytes().replace(b"\r\n", b"\n") for name in changed + added}
    archive = build_archive(directory, source, files, changed, added)
    plan = dict(source_sha256=sha(archive), predecessor_directory=base.name,
                production_control_dir=previous["production_control_dir"], predecessor_job=30330582)
    (directory / "release.json").write_text(json.dumps(plan, indent=2), encoding="utf8")
    shutil.copyfile(base / "hpc_launch.py", directory / "hpc_launch.py")
    shutil.copyfile(Path(__file__), directory / Path(__file__).name)
    with tarfile.open(directory / "bundle.tar.gz", "x:gz") as bundle:
        for name in ("source.tar.gz", "release.json", "hpc_launch.py", Path(__file__).name):
            bundle.add(directory / name, arcname=name)
    print(json.dumps(plan))


def next_leg(directory, plan, action):
    predecessor = directory.parent / plan["predecessor_directory"]
    job = plan["predecessor_job"]
    label = "e48-encoder-invented-bonds"
    if json.loads((predecessor / "e48-fork3-candidate.receipt.json").read_bytes())["id"] != job:
        raise ValueError("Incorrect candidate predecessor")
    if action == "submit-next":
        sys.path.insert(0, plan["production_control_dir"])
        from epoch48_segment_chain import submit_once

        args = ["-q", "gpuv100", "-n", "8", "-R", "select[gpu32gb] span[hosts=1] rusage[mem=6144]",
                "-gpu", "num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=30G", "-W", "24:00",
                "-env", "all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
                "-w", f"done({job})", "/usr/bin/python3", str(Path(__file__).resolve()), "run-next"]
        print(json.dumps(dict(job=submit_once(directory, label, args), after=job)))
        return
    receipt = json.loads((directory / f"{label}.receipt.json").read_bytes())
    if int(os.environ["LSB_JOBID"]) != receipt["id"]:
        raise ValueError("Unexpected successor job identity")
    launch(directory, plan, label, ["/usr/bin/env", "LATENTMOL_CPU_THREADS=4",
           "LATENTMOL_EVALUATION_MODULE=experiments.vae.encoder_next_leg", "bash",
           "experiments/vae/run_checkpoint_comparison.sh", "--config", "experiments/vae/epoch48-seven-round-generated-roundtrip.json",
           "--source-dir", str(predecessor / "candidate"), "--policy", "experiments/vae/epoch48-encoder-targets.json",
           "--output", str(directory / "segment")], startup=2400, duration=85000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    early_actions = ("stage", "candidate", "control", "interim", "final")
    parser.add_argument("action", choices=("seal", "seal-early", "seal-next", "submit-next", "run-next",
                                           "submit", "train", "compare", *early_actions))
    args = parser.parse_args()
    if args.action == "seal":
        seal()
        return
    if args.action == "seal-early":
        seal_early()
        return
    if args.action == "seal-next":
        seal_next()
        return
    directory = Path(__file__).resolve().parent
    plan = json.loads((directory / "release.json").read_bytes())
    if args.action in ("submit-next", "run-next"):
        next_leg(directory, plan, args.action)
        return
    if args.action in early_actions:
        early(directory, plan, args.action)
        return
    # Reuse the production submitter's exclusive intent and LSF receipt. The
    # existing worker, callback and all their paths are read-only to this job.
    from vae_segment_chain import read, submit

    script = ["/usr/bin/python3", str(Path(__file__).resolve())]
    prod = Path(plan["production_control_dir"])
    base = read(prod / "plan.json")
    if (hashlib.sha256((prod / "plan.json").read_bytes()).hexdigest() != plan["production_plan_sha256"]
            or read(prod / "e48-r7-rtearly-after-s03.receipt.json")["id"] != plan["callback"]):
        raise ValueError("Production control plan or callback changed")
    candidate = directory / "candidate"
    control = Path(base["output_root"]) / "segment-04"
    job = "e48-encoder3-s04"
    report = "e48-encoder3-eval"
    if args.action == "submit":
        if candidate.exists():
            raise FileExistsError(candidate)
        job_id = submit(directory, job, [
            "-q", "gpuv100", "-n", "8", "-R", "select[gpu32gb] span[hosts=1] rusage[mem=6144]",
            "-gpu", "num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=30G", "-W", "24:00",
            "-env", "all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
            "-w", f"done({plan['callback']})", *script, "train"])
        print(json.dumps({"candidate": job_id, "after": plan["callback"]}), flush=True)
        return
    label = job if args.action == "train" else report
    assigned = read(directory / f"{label}.receipt.json")["id"]
    if int(os.environ["LSB_JOBID"]) != assigned:
        raise ValueError("Worker identity differs from its submission")
    if args.action == "train":
        # The preceding callback has now published the production control ID.
        # Queue evaluation once, so it can also report a failed/incomplete arm.
        other = read(prod / "e48-r7-rtearly-s04.receipt.json")["id"]
        eval_id = submit(directory, report, ["-q", "hpc", "-n", "4", "-R",
            "span[hosts=1] rusage[mem=2048]", "-W", "01:30", "-w",
            f"ended({assigned}) && ended({other})", *script, "compare"])
        print(json.dumps({"candidate": assigned, "control": other, "comparison": eval_id}), flush=True)
        launch(directory, plan, "train", ["/usr/bin/env", "LATENTMOL_CPU_THREADS=4",
               "LATENTMOL_EVALUATION_MODULE=experiments.vae.encoder_upgrade_pilot",
               "bash", "experiments/vae/run_checkpoint_comparison.sh", "--config", base["config"],
               "--parent-receipt", base["parent_receipt"], "--resume-receipt",
               str(control.parent / "segment-03" / "latest.json"), "--production-control",
               str(control), "--output", str(candidate)], startup=2400, duration=85000)
    else:
        for arm in (candidate, control):
            if not (arm / "summary.json").is_file():
                raise RuntimeError(f"No terminal summary for {arm}, so no comparison is available")
        launch(directory, plan, "compare", ["bash", "-lc",
               'set -eu; module load python3/3.11.9; unset PYTHONHOME PYTHONPATH; '
               'source /work3/s234818/latentmol/env/boltz311/bin/activate; cd "$DBMOL_LOCAL_SOURCE"; '
               'export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; '
               'exec python -m experiments.vae.compare_encoder_upgrade "$@"',
               "compare", "--control", str(control), "--candidate", str(candidate),
               "--output", str(directory / "comparison")], startup=4900, duration=5000)


if __name__ == "__main__":
    main()
