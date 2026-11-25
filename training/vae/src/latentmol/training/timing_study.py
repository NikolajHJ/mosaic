"""Matched, disposable timing runs for the combined and prior-only VAE recipes."""

from contextlib import contextmanager
import gc
import math
import os
from statistics import mean, median
from time import perf_counter

import torch

from latentmol.experiments import write_json
from latentmol.train import learning_rate_at_epoch, state_digest
from latentmol.training import combined_continuation as runner


class StepTimings:
    def __init__(self, device):
        self.device = device
        self.wall = {}
        self.events = {}

    @contextmanager
    def measure(self, name, *, cuda=False):
        events = None
        if cuda and self.device == "cuda":
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
        start = perf_counter()
        try:
            yield
        finally:
            self.wall[name] = self.wall.get(name, 0.) + (perf_counter() - start)
            if events is not None:
                events[1].record()
                self.events.setdefault(name, []).append(events)

    def read(self):
        """The caller has synchronized the CUDA stream at the end of the update."""
        return dict(wall_seconds=self.wall, cuda_stream_seconds={
            name: sum(start.elapsed_time(stop) for start, stop in pairs) / 1000
            for name, pairs in self.events.items()
        })


def _timed_arm(config, study, case, data, assets, order, resume, microbatch, output, device):
    parent, payload, occupancy, anchor, composition = assets
    start = resume["continuation"] if resume is not None else payload["continuation"]
    row = start["row"] if resume is not None else 0
    size = study["batch_size"]
    epoch = config.get("parent_epoch", 45) + 1
    if row + (study["warmup"] + study["updates"]) * size > len(data):
        raise ValueError("Timing window crosses the training epoch boundary")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, optimizer, training, generators = runner._fresh(parent, payload, anchor, size, device, resume,
        atom_refinement_dropout=config.get("atom_refinement_dropout"))
    for group in optimizer.param_groups:
        group["lr"] = learning_rate_at_epoch(training, epoch)
    records = []
    with runner._teacher(config, model.spec, anchor) as teacher:
        for index in range(study["warmup"] + study["updates"]):
            timings = StepTimings(device) if case["instrumented"] else None
            start_time = perf_counter()
            hard = runner._batch(data, order, row + index * size, size)
            loaded = perf_counter()
            metrics = runner._update(config, model, occupancy, optimizer, hard, training, composition, teacher,
                generators, start["updates"] + index, epoch, microbatch, timings=timings)
            if device == "cuda":
                torch.cuda.synchronize()
            seconds = perf_counter() - start_time
            if index >= study["warmup"]:
                record = dict(update=start["updates"] + index, seconds=seconds, batch_load_seconds=loaded-start_time)
                for key in ("real_points", "fresh_prior_points", "search_seconds", "search_checks",
                            "repair_targets", "repair_statuses", "prior_already_valid", "total",
                            "ordered_teacher_receipt_chain_sha256"):
                    if key in metrics:
                        record[key] = metrics[key]
                if timings is not None:
                    record.update(timings.read())
                records.append(record)
            if (index + 1) % 8 == 0:
                write_json(output / "progress.json", dict(case=case, microbatch=microbatch, updates=index+1))
    reserved = torch.cuda.max_memory_reserved() / torch.cuda.get_device_properties(0).total_memory if device == "cuda" else 0.
    return dict(case, records=records, reserved_fraction=reserved,
                weights_sha256=state_digest(model.state_dict()), microbatch_size=microbatch,
                mean_seconds_per_update=mean(x["seconds"] for x in records),
                real_points_per_second=sum(x["real_points"] for x in records) / sum(x["seconds"] for x in records))


def _summary(arm):
    records = arm["records"]
    fields = {key: [r[key] for r in records] for key in ("seconds", "batch_load_seconds", "search_seconds")}
    for field in ("wall_seconds", "cuda_stream_seconds"):
        for key in records[0].get(field, {}):
            fields[f"{field}.{key}"] = [r[field][key] for r in records]
    return {key: dict(mean=mean(values), median=median(values), p90=sorted(values)[math.ceil(.9*len(values))-1])
            for key, values in fields.items()}


def run_timing_study(config, identity, study, data, assets, output, device):
    prior_only = "prior_only" in config
    size = study["batch_size"]
    if size not in config["batch_sizes"] or any(type(study[k]) is not int or study[k] < 1 for k in ("updates", "warmup")):
        raise ValueError("Invalid VAE timing budget or batch shape")
    dataset = data.manifest["dataset_sha256"]
    resume = None
    if not prior_only:
        resume = torch.load(runner._authenticated(study["checkpoint"]), map_location="cpu", weights_only=True)
        if (resume["dataset_sha256"] != dataset or resume["config_sha256"] != identity
                or resume["combined_version"] != runner.VERSION or resume["batch_size"] != size
                or resume["microbatch_size"] != study["microbatch_size"] or resume["continuation"]["epoch"] != 45):
            raise ValueError("Worker comparison must resume the exact active training recipe")
    digest = None if assets[2] is None else state_digest(assets[2].state_dict())
    order = runner._order(len(data), config["shuffle_seed"], config.get("parent_epoch", 45) + 1)
    cases = ([dict(workers=n, instrumented=True) for n in study["worker_counts"]] if not prior_only else
             [dict(workers=config["execution"]["workers"], instrumented=enabled) for enabled in (False, True)])
    microbatches = runner._microbatch_sizes(config) if prior_only else [study["microbatch_size"]]
    attempts = []
    for microbatch in microbatches:
        arms = []
        for case in cases:
            trial = dict(config, execution=dict(config["execution"], workers=case["workers"]))
            try:
                arm = _timed_arm(trial, study, case, data, assets, order, resume, microbatch, output, device)
            except torch.OutOfMemoryError:
                arm = dict(case, microbatch_size=microbatch, error="CUDA out of memory")
            arms.append(arm)
            runner._check_frozen(assets[2], digest, assets[3])
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        attempts.extend(arms)
        if all("records" in arm and arm["reserved_fraction"] <= config["protocol"]["vram_fraction_limit"] for arm in arms):
            break
    else:
        write_json(output / "timing.json", dict(status="no_passing_pair", attempts=attempts))
        raise RuntimeError("No timing pair fit the configured memory limit")

    for arm in arms:
        arm["summary"] = _summary(arm)
    left, right = arms
    ordered = "ordered_teacher_receipt_chain_sha256"
    semantics = ("real_points", "fresh_prior_points", "repair_targets", "repair_statuses", ordered)
    matched = all(all(a.get(key) == b.get(key) for key in semantics)
                  for a, b in zip(left["records"], right["records"], strict=True))
    write_json(output / "timing.json", dict(status="completed", job_id=os.environ.get("LSB_JOBID"),
        execution=config["execution"],
        dataset_sha256=dataset, config_sha256=identity, study=study, training_molecules=len(data),
        device=torch.cuda.get_device_name(0) if device == "cuda" else device,
        same_repair_trace=matched, same_final_weights=left["weights_sha256"] == right["weights_sha256"],
        throughput_ratio=right["real_points_per_second"] / left["real_points_per_second"], attempts=attempts))
