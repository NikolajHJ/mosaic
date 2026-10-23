"""Duplicate-safe density-study CPU, native-profile and serial full-case stages.

Model/data computations remain allocated. Login work authenticates bytes and
prepares argv only. Each stage requires the preceding engineering gate DONE;
each independent case repeats source checks, and no partial endpoint is ranked.
"""
import argparse
import json
from pathlib import Path
import re

from latentmol.decoder_density_protocol import read_protocol, cases
from latentmol.experiments import file_sha256, source_manifest
from submit_stability_study import authenticated, read, submit
from submit_ramped_random import require_completed_stage


def workflow(release, work, protocol, path, config_sha, archive_sha, manifest_sha,
             stage, cpu_job=None, gate_job=None, queue="gpul40s"):
    rows = []
    declarations = cases(protocol) if stage == "cases" else [{"label": stage}]
    for case in declarations:
        label = release.name + "-" + case["label"]
        cpu = stage == "cpu"
        arguments = ["bsub", "-J", label, "-q", "hpc" if cpu else queue,
            "-n", "2" if cpu else "4", "-R", "rusage[mem=4GB]" if cpu else "rusage[mem=16GB]",
            "-R", "span[hosts=1]", "-W", "0:45" if cpu else "2:00" if stage == "gate" else "23:00",
            "-oo", str(work / "logs" / (label + ".%J.out")),
            "-eo", str(work / "logs" / (label + ".%J.err"))]
        if not cpu:
            dependency = f"done({cpu_job})" + (f" && done({gate_job})" if stage == "cases" else "")
            arguments += ["-gpu", "num=1:mode=exclusive_process:j_exclusive=yes", "-w", dependency]
        campaign = work / "runs" / (release.name + "-density-study")
        tail = ["bash", "scripts/hpc/decoder_density_tests.sh"] if cpu else [
            "bash", "scripts/hpc/decoder_density_run.sh", str(path), config_sha, str(campaign),
            "case" if stage == "cases" else "gate"]
        if stage == "cases":
            tail += ["--case", str(case["index"])]
        command = ["env", f"LATENTMOL_SOURCE_ARCHIVE={release}.tar.gz", f"LATENTMOL_WORK_ROOT={work}",
            "bash", "scripts/hpc/boltz_run.sh", "python", "scripts/verify_stability_release.py",
            "--archive-sha256", archive_sha, "--manifest-sha256", manifest_sha,
            "--", "python", "-m", "latentmol.experiments", "--run-root", str(work / "runs" / label),
            "--name", "density-" + case["label"], "--", *tail]
        rows.append({"stage": case["label"], "cwd": str(release), "arguments": arguments,
                     "job_command": command, "dependency": None})
    return rows


def following_case(row, cpu_job, gate_job, preceding):
    """Resolve one serial dependency without duplicate -w or mutable argv."""
    if not all(isinstance(job, str) and re.fullmatch(r'[0-9]+', job) for job in (cpu_job, gate_job, preceding)):
        raise ValueError("Serial dependency needs verified numeric job IDs")
    arguments = list(row['arguments'])
    index = arguments.index('-w')
    arguments[index + 1] = f'done({cpu_job}) && done({gate_job}) && ended({preceding})'
    return {**row, 'arguments': arguments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu", "gate", "cases"), required=True)
    parser.add_argument("--release", type=Path, default=Path.cwd())
    parser.add_argument("--work-root", type=Path, default=Path('/work3/s234818/latentmol'))
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--queue", choices=("gpul40s", "gpuv100", "gpua100"), default="gpul40s")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    work, release = args.work_root.resolve(strict=True), args.release.resolve(strict=True)
    if (not str(work).startswith('/work3/') or not release.is_relative_to(work / 'releases')
            or not args.receipts.is_absolute() or not args.receipts.resolve().is_relative_to(work)):
        raise ValueError("Require immutable DTU release and contained receipt directory")
    authenticated(Path(str(release) + '.tar.gz'), args.archive_sha256)
    manifest = source_manifest(release)['sha256']
    if read(release / 'release_manifest.json')['sha256'] != manifest:
        raise ValueError("Density release changed after sealing")
    path = release / 'configs/decoder_density_study.json'
    config_sha = file_sha256(path)
    protocol = read_protocol(path, config_sha)
    authenticated(Path(protocol['initializer']), protocol['initializer_sha256'])
    authenticated(Path(protocol['data']) / 'dataset.npz', protocol['dataset_sha256'])
    if read(Path(protocol['data']) / 'manifest.json')['split_sha256'] != protocol['split_sha256']:
        raise ValueError("Protected density split changed")
    identity = {'release': str(release), 'archive_sha256': args.archive_sha256,
        'manifest_sha256': manifest, 'config_sha256': config_sha, 'queue': args.queue,
        'initializer_sha256': protocol['initializer_sha256'], 'dataset_sha256': protocol['dataset_sha256'],
        'split_sha256': protocol['split_sha256'], 'output_root': str(work / 'runs' / (release.name + '-density-study'))}
    cpu_job = require_completed_stage(args.receipts, 'cpu', identity) if args.stage != 'cpu' else None
    gate_job = require_completed_stage(args.receipts, 'gate', identity) if args.stage == 'cases' else None
    if gate_job:
        native = read(Path(identity['output_root']) / 'gate/passed.json')
        if native['status'] != 'passed' or native['protocol'] != protocol or native['source_archive_sha256'] != args.archive_sha256:
            raise ValueError("Native density profile belongs to another immutable experiment")
    rows = workflow(release, work, protocol, path, config_sha, args.archive_sha256, manifest,
                    args.stage, cpu_job, gate_job, args.queue)
    if not args.submit:
        print(json.dumps({'identity': identity, 'plan': rows}, indent=2))
        return
    (work / 'logs').mkdir(exist_ok=True)
    jobs = {}
    preceding = None
    for row in rows:
        if preceding:
            # Every case retains explicit CPU/native-gate protection. A failed
            # case ends its reservation but does not invalidate another seed.
            row = following_case(row, cpu_job, gate_job, preceding)
        jobs.update(submit([row], args.receipts, identity))
        preceding = jobs[row['stage']]
    print(json.dumps(jobs), flush=True)


if __name__ == '__main__':
    main()
