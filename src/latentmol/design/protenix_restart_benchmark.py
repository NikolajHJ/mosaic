"""Measure sibling Protenix throughput on one allocated GPU."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import os
import subprocess
from threading import Event, Thread
from time import perf_counter

import torch

from latentmol.chemistry import inspect_graph
from latentmol.design.latent_protenix import (
    LatentDesignConfig, _json, _model_identity, finish_point, hard_from_graph,
    latent_step, prepare_point, structural_target,
)
from latentmol.design.protenix_restart_parents import load_parent_panel
from latentmol.design.protenix_restart_proposals import propose_start
from latentmol.design.protenix_restart_study import parent_record
from latentmol.design.protenix_restarts import fresh_score_records, native_factory, restart_args, write_json
from latentmol.evaluation.evaluate import load_model


DEVICES = {
    "a100-80gb": ("A100", 78_000),
    "v100-32gb": ("V100", 31_000),
}
_ALLOCATION_FAILURES = ("out of memory", "out_of_memory", "resource exhausted", "resource_exhausted",
                        "cudaerrormemoryallocation", "cublas_status_alloc_failed")


def smi(query, *, gpu=None, section="gpu"):
    command = ["nvidia-smi", f"--query-{section}={query}", "--format=csv,noheader,nounits"]
    if gpu is not None:
        command += ["-i", gpu]
    return [tuple(item.strip() for item in line.split(",")) for line in
            subprocess.check_output(command, text=True, timeout=10).splitlines() if line.strip()]


def allocated_gpu(pid, device="a100-80gb"):
    model, minimum_mib = DEVICES[device]
    uuids = {uuid for uuid, process in smi("gpu_uuid,pid", section="compute-apps") if process == str(pid)}
    if len(uuids) != 1:
        raise ValueError(f"Cannot bind benchmark memory samples to native worker {pid}: {uuids}")
    uuid, = uuids
    (name, total), = smi("name,memory.total", gpu=uuid)
    if model not in name.upper() or int(total) < minimum_mib:
        raise ValueError(f"Pilot requires {device}, got {name} with {total} MiB")
    return dict(uuid=uuid, name=name, memory_total_mib=int(total))


def allocation_failure(error, stderr):
    """Identify device allocation errors, including worker EOF with an OOM log."""
    message = str(error)
    if any(marker in message.lower() for marker in _ALLOCATION_FAILURES):
        return message
    if not isinstance(error, (EOFError, BrokenPipeError)):
        return None
    try:
        with stderr.open("rb") as stream:
            end = stream.seek(0, os.SEEK_END)
            stream.seek(max(end - 65_536, 0))
            tail = stream.read().decode(errors="replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        if any(marker in line.lower() for marker in _ALLOCATION_FAILURES):
            return f"{message}: {line.strip()}"
    return None


class DeviceSamples:
    """Sample whole-device memory on the exclusive GPU, including cached allocations."""

    def __init__(self, gpu):
        self.gpu = gpu
        self.rows = []
        self.error = None
        self.stop = Event()
        self.thread = Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            try:
                (memory, utilization), = smi("memory.used,utilization.gpu", gpu=self.gpu)
                self.rows.append((perf_counter(), int(memory), int(utilization)))
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                self.error = str(error)
                return
            self.stop.wait(0.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()

    def between(self, start, end):
        rows = [row for row in self.rows if start <= row[0] <= end]
        return dict(samples=len(rows), sampled_peak_memory_mib=max((row[1] for row in rows), default=None),
                    mean_gpu_utilization_pct=sum(row[2] for row in rows) / len(rows) if rows else None,
                    sample_error=self.error)


def cpu_info():
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    models = {line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
              if line.startswith("model name")}
    cores = set()
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            cores.add(tuple((topology / field).read_text().strip()
                            for field in ("physical_package_id", "core_id")))
        except OSError:
            break
    return dict(models=sorted(models), affinity=allowed, physical_cores=len(cores),
                lsf_slots=os.environ.get("LSB_DJOB_NUMPROC"), omp_threads=os.environ.get("OMP_NUM_THREADS"))


def step_group(decoder, scorer, points, masks, generators, design, step, pool, overlap):
    apply = map if pool is None else pool.map
    started = perf_counter()
    prepared = list(apply(lambda pair: prepare_point(decoder, *pair, design, step), zip(points, masks, strict=True)))
    prepare_seconds = perf_counter() - started
    targets = ([pool.submit(structural_target, point.graph, decoder.spec, point.active, design, step)
                for point in prepared] if overlap else None)
    started = perf_counter()
    if len(points) == 1:
        scored = [scorer.score_at(prepared[0].graph, masks[0])]
    else:
        scored = scorer.score_batch([point.graph for point in prepared], masks)
    rpc_seconds = perf_counter() - started
    native_seconds = scorer.last_native_seconds
    started = perf_counter()
    bundles = [future.result() for future in targets] if targets is not None else [None] * len(points)
    target_wait_seconds = perf_counter() - started

    def finish(row):
        prepared_point, score, bundle, generator = row
        evaluation = finish_point(decoder, prepared_point, score, target_bundle=bundle)
        proposal, update = latent_step(evaluation["point"], evaluation["binding_gradient"],
                                       evaluation["gradient"], design, step, generator)
        hard = hard_from_graph(evaluation["graph"])
        checked = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], decoder.spec)
        smiles = checked.smiles
        return (proposal, evaluation["binding_gradient"], float(evaluation["binding"]),
                smiles if smiles and "." not in smiles else None, update["bound_factor"] < 1)

    started = perf_counter()
    rows = list(apply(finish, zip(prepared, scored, bundles, generators, strict=True)))
    return rows, dict(prepare_seconds=prepare_seconds, rpc_seconds=rpc_seconds, native_seconds=native_seconds,
                      target_wait_seconds=target_wait_seconds, finish_seconds=perf_counter() - started)


def agreement(reference, rows):
    differences, relative, cosine = [], [], []
    for expected, actual in zip(reference, rows, strict=True):
        grad, other = expected[1], actual[1]
        norm, other_norm = float(grad.norm()), float(other.norm())
        differences.append(abs(expected[2] - actual[2]))
        relative.append(float((grad - other).norm()) / max(norm, 1e-12))
        if norm and other_norm:
            cosine.append(float(torch.nn.functional.cosine_similarity(grad.flatten(), other.flatten(), dim=0)))
    return dict(max_loss_abs_delta=max(differences), max_binding_gradient_relative_l2=max(relative),
                min_binding_gradient_cosine=min(cosine, default=None))


def run_case(decoder, scorer, starts, design, case, *, updates):
    torch.set_num_threads(case["torch_threads"])
    points, masks, seeds = zip(*starts, strict=True)
    generators = [torch.Generator().manual_seed(seed) for seed in seeds]
    width, workers, overlap = (case[key] for key in ("batch_size", "cpu_workers", "overlap"))
    all_first, endpoints, candidates = [], [], {}
    counts = dict(valid=0, capped=0)
    warmup_native = []
    timing = dict.fromkeys(("prepare_seconds", "rpc_seconds", "native_seconds", "target_wait_seconds", "finish_seconds"), 0.)
    pool_context = ThreadPoolExecutor(max_workers=workers) if overlap or workers > 1 else nullcontext()
    start = perf_counter()
    with pool_context as pool:
        for offset in range(0, len(starts), width):
            group = slice(offset, offset + width)
            current = points[group]
            for step in range(updates):
                rows, durations = step_group(decoder, scorer, current, masks[group], generators[group],
                                             design, step, pool, overlap)
                if step == 0:
                    all_first.extend(rows)
                if case["name"] == "warmup":
                    warmup_native.append(durations["native_seconds"])
                current = tuple(row[0] for row in rows)
                for _, _, loss, smiles, capped in rows:
                    counts["capped"] += int(capped)
                    if smiles is not None:
                        counts["valid"] += 1
                        candidates[smiles] = min(loss, candidates.get(smiles, loss))
                for key in timing:
                    timing[key] += durations[key]
            endpoints.extend(current)
    end = perf_counter()
    total = len(starts) * updates
    report = dict(case, updates=total, elapsed_seconds=end - start, updates_per_second=total / (end - start),
                  valid_fraction=counts["valid"] / total, capped_updates=counts["capped"],
                  distinct_valid=len(candidates), **timing)
    if warmup_native:
        report["native_seconds_by_call"] = warmup_native
    return report, all_first, endpoints, candidates, (start, end)


def run_benchmark(args, config):
    design = LatentDesignConfig(**config["design"])
    cases = config["cases"]
    device = config.get("device", "a100-80gb")
    if not isinstance(device, str) or device not in DEVICES:
        raise ValueError(f"Unknown benchmark device: {device!r}")
    count = config["parents"] * config["children_per_parent"]
    if (config["study"] != "batch_throughput" or args.arm != "bench" or design.learning_rate != 60
            or not 1 <= design.updates <= 100 or not 1 <= config["repeats"] <= 3
            or not 1 <= config["warmup_steps"] <= 4 or not 0 <= config["fresh_attempts"] <= 16
            or not cases or count < 1 or cases[0]["name"] != "scalar-standard" or cases[0]["batch_size"] != 1
            or len({case["name"] for case in cases}) != len(cases)
            or any(case["batch_size"] not in (1, 2, 4, 8) or count % case["batch_size"]
                   or not 1 <= case["cpu_workers"] <= 4 or not 1 <= case["torch_threads"] <= 4
                   for case in cases)):
        raise ValueError("Incompatible bounded throughput pilot configuration")
    decoder, _ = load_model(args.checkpoint, "cpu")
    decoder.eval().requires_grad_(False)
    identity = _model_identity(decoder)
    parents = load_parent_panel(decoder, [Path(root) for root in config["parent_cohorts"]],
        count=config["parents"], seed=config["panel_seed"], temperature=design.temperature_initial,
        occupied_atoms=design.occupied_atoms, checkpoint_sha256=args.checkpoint_sha256)
    starts, proposals = [], []
    for index, parent in enumerate(parents):
        for _ in range(config["children_per_parent"]):
            seed = config["perturbation_seed"] + len(starts)
            point, offset, receipt = propose_start(decoder, parent.candidate, parent.active,
                policy="latent", seed=seed, temperature=design.temperature_initial)
            if offset is not None:
                raise ValueError("This pilot requires screened latent starts")
            starts.append((point, parent.active, design.noise_seed + seed))
            proposals.append(dict(parent_index=index, noise_seed=design.noise_seed + seed, **receipt))
    args.output.mkdir(parents=True)
    write_json(args.output / "run.json", dict(settings=asdict(design), cases=cases, cpu=cpu_info(),
        job_id=os.environ["LSB_JOBID"], decoder_sha256=identity, parents=[parent_record(p) for p in parents],
        starts=proposals, config=config))
    scorer = native_factory(args)(decoder.spec, starts[0][1], args.output)
    scorer.timeout = 2400.0
    trials, warmups, unavailable = [], [], []
    native_stderr = args.output / "native.stderr.log"
    reference = reference_endpoints = fresh_pool = None
    widths = sorted({case["batch_size"] for case in cases})
    warm_case = dict(name="warmup", batch_size=1, torch_threads=1, cpu_workers=1, overlap=False)
    try:
        scalar_warmup, _, _, _, _ = run_case(decoder, scorer, starts[:1], design, warm_case,
                                              updates=config["warmup_steps"])
        warmups.append(scalar_warmup)
        gpu = allocated_gpu(scorer.worker_pid, device)
        if ready := os.environ.get("DBMOL_LAUNCH_READY"):
            _json(ready, dict(status="models_and_inputs_ready", nonce=os.environ.get("DBMOL_LAUNCH_NONCE"),
                              job_id=os.environ["LSB_JOBID"]))
        write_json(args.output / "device.json", gpu)
        with DeviceSamples(gpu["uuid"]) as samples, (args.output / "timings.jsonl").open("x", buffering=1) as trace:
            for width in widths:
                phase = "warmup"
                try:
                    if width > 1:
                        warm, first, _, _, window = run_case(decoder, scorer, starts[:width], design,
                            dict(warm_case, batch_size=width), updates=config["warmup_steps"])
                        warmups.append(warm | samples.between(*window) | agreement(reference[:width], first))
                    print(json.dumps(dict(event="warmup", **warmups[-1]), allow_nan=False), file=trace, flush=True)
                    phase = "trial"
                    matching = [case for case in cases if case["batch_size"] == width]
                    for repeat in range(config["repeats"]):
                        for case in matching if repeat % 2 == 0 else reversed(matching):
                            row, first, endpoints, candidates, window = run_case(
                                decoder, scorer, starts, design, case, updates=design.updates)
                            if reference is None:
                                reference, reference_endpoints, fresh_pool = first, endpoints, candidates
                            row.update(samples.between(*window), repeat=repeat + 1, **agreement(reference, first),
                                max_endpoint_latent_l2_delta=max(float((a - b).norm()) for a, b in
                                                                 zip(reference_endpoints, endpoints, strict=True)))
                            trials.append(row)
                            print(json.dumps(dict(event="trial", **row), allow_nan=False), file=trace, flush=True)
                            print(f"{row['name']} repeat {row['repeat']} {row['updates_per_second']:.3f} updates/s", flush=True)
                except (RuntimeError, EOFError, BrokenPipeError) as error:
                    reason = allocation_failure(error, native_stderr)
                    if width == 1 or reason is None:
                        raise
                    failure = dict(batch_size=width, phase=phase, reason=reason)
                    unavailable.append(failure)
                    print(json.dumps(dict(event="unavailable", **failure), allow_nan=False), file=trace, flush=True)
                    break
    finally:
        try:
            scorer.close()
        except (EOFError, BrokenPipeError) as error:
            if not unavailable or allocation_failure(error, native_stderr) is None:
                raise
    if _model_identity(decoder) != identity or any(parameter.grad is not None for parameter in decoder.parameters()):
        raise ValueError("Frozen decoder parameters or gradients changed")
    ordered = sorted(fresh_pool, key=lambda smiles: (fresh_pool[smiles], smiles))[:config["fresh_attempts"]]
    result = dict(status="optimization_complete", gpu=gpu, warmups=warmups, trials=trials,
                  unavailable=unavailable, fresh_attempts=len(ordered),
                  native_shutdown=scorer.shutdown_status or ("allocation_failure" if unavailable else None))
    write_json(args.output / "optimization-result.json", result)
    if ordered:
        root = args.output / "fresh-probe"
        root.mkdir()
        started = perf_counter()
        scored = fresh_score_records(args, root, [dict(smiles=smiles, start="pilot", arm="scalar-standard",
                                                        step=0, role="edit") for smiles in ordered])
        result.update(fresh_elapsed_seconds=perf_counter() - started,
                      fresh_successful=sum(row["status"] == "scored" for row in scored))
    result["status"] = "completed"
    write_json(args.output / "result.json", result)


def main(argv=None):
    args = restart_args(__doc__, argv)
    run_benchmark(args, json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
