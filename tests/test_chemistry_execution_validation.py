"""Exact execution-parity orchestration fixtures, allocated DTU HPC only.

The update is a tiny actual AdamW parameter loss with explicit RNG consumption.
No repair search or worker pool is spawned here; native full256+768 parity and
speed remain separate mandatory GPU checks against the real saved VAE.
"""
from dataclasses import replace
import json
import random

import numpy as np
import pytest
import torch

from latentmol import chemistry_continuation as continuation
from latentmol import chemistry_execution_validation as execution
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.train import capture_rng_state, restore_rng_state
from test_chemistry_continuation import _hard, _tiny_models


class Teacher:
    instances = []

    def __init__(self, spec, config, *, cache_capacity=128, backend="reference", workers=0):
        self.spec, self.config, self.cache_capacity = spec, config, cache_capacity
        self.backend, self.workers = backend, workers
        self._cache, self.closed = {"stale": "must-clear"}, False
        self._pool = object() if workers else None
        self.clears = 0
        self.instances.append(self)

    def clear_cache(self):
        self.clears += 1
        self._cache.clear()

    def drain(self):
        pass

    def close(self):
        self.closed = True
        # Cleanup must precede final RNG restoration, even if a future worker
        # shutdown implementation happens to consume a global random draw.
        random.random()


@pytest.fixture(autouse=True)
def restore_fixture_rng():
    initial = capture_rng_state({}, include_cuda=True)
    try:
        yield
    finally:
        restore_rng_state(initial, {})


def fixture(monkeypatch, tmp_path, defect=None):
    model, reference, config, protocol, composition = _tiny_models()
    config = replace(config, batch_size=256)
    protocol = {**protocol, "real_batch_size": 256, "microbatch_size": 128}
    hard = _hard(model.spec, ("CO", "CCO"))
    data = [{"atoms": hard.atoms[index % 2], "charges": hard.charges[index % 2],
             "bonds": hard.bonds[index % 2]} for index in range(256)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # Preserve genuine parent moments and mixed existing gradients, rather than
    # only the trivially empty optimizer/None-gradient phase-boundary case.
    sum(parameter.square().mean() for parameter in model.parameters()).backward()
    optimizer.step()
    next(model.parameters()).grad = None
    model.train(); model.decoder.eval()
    teacher = Teacher(model.spec, ProbabilityRepairConfig(), backend="optimized", workers=12)
    generators = {"train": torch.Generator().manual_seed(881), "validation": torch.Generator().manual_seed(882),
                  "fresh_prior": torch.Generator().manual_seed(883)}
    monkeypatch.setattr(execution, "AtomPreservingRepairTeacher", Teacher)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    calls = []

    def actual_update(model, reference, optimizer, hard, config, protocol, composition,
                      teacher, generators, update, epoch, phase_start_epoch, *, overlap=False, step_control=None):
        assert len(hard.atoms) == 256 and protocol["prior_ratio"] in (2, 3)
        assert not teacher._cache
        calls.append((teacher.backend, update, epoch, phase_start_epoch, hard.atoms.clone(), overlap))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        noise = random.random() + float(np.random.normal()) + float(torch.rand(()))
        noise += sum(float(torch.randn((), generator=generator)) for generator in generators.values())
        sum(parameter.square().mean() + .01 * noise * parameter.mean() for parameter in model.parameters()).backward()
        from latentmol.chemistry_optimizer_step import apply_optimizer_step
        step_receipt = apply_optimizer_step(optimizer, model.parameters(), step_control)
        teacher._cache["disposable"] = "must-not-be-benchmark-state"
        metrics = {"reconstruction": 1.25, "kl": .2, "repair_real": .3, "repair_prior": .7,
            "total": 2.25, "real_points": 256, "fresh_prior_points": 256 * protocol["prior_ratio"],
            "repair_targets": {"real": 1, "prior": 2}, "search_checks": 123,
            "repair_statuses": {"real": {"accepted": 1}, "prior": {"accepted": 2}},
            "ordered_teacher_receipt_chain_sha256": "c" * 64,
            "search_seconds": 10. if teacher.backend == "reference" else .25,
            "prepass_seconds": 11. if teacher.backend == "reference" else .5,
            "teacher_phase_seconds": {"classification_worker_seconds": 0. if teacher.backend == "reference" else .2}}
        if step_control is not None:
            metrics["optimizer_step_control"] = step_receipt
        if defect == "no_target":
            metrics["repair_targets"] = {"real": 0, "prior": 0}
        if teacher.backend == "optimized":
            if defect == "metric": metrics["search_checks"] += 1
            elif defect == "receipt": metrics["ordered_teacher_receipt_chain_sha256"] = "d" * 64
            elif defect == "gradient": next(model.parameters()).grad.add_(.125)
            elif defect == "rng": torch.rand(())
            elif defect == "exception": raise RuntimeError("Injected optimized update failure")
        return metrics

    monkeypatch.setattr(continuation, "chemistry_update", actual_update)
    arguments = {"model": model, "reference": reference, "optimizer": optimizer, "data": data,
        "cohorts": {"train": list(reversed(range(256)))}, "config": config, "protocol": protocol,
        "composition": composition, "optimized_teacher": teacher, "generators": generators,
        "epoch": 35, "updates": 230860, "output": tmp_path / "execution-parity.json"}
    initial = execution._snapshot(model, reference, optimizer, generators)
    return arguments, initial, calls


def test_explicit_quarter_step_is_identical_in_serial_and_optimized_parity_arms(monkeypatch, tmp_path):
    arguments, initial, _ = fixture(monkeypatch, tmp_path)
    policy = {"version": "adamw-displacement-scale-v1", "factor": .25}
    report = execution.native_execution_parity(**arguments, step_control=policy)
    assert report["passed"] and report["restoration"]["passed"]
    assert report["optimizer_step_control"] == policy
    for name in ("reference", "optimized"):
        receipt = report[name]["metrics"]["optimizer_step_control"]
        assert receipt["factor"] == .25 and receipt["optimizer_steps"] == 1
    restored = execution._snapshot(arguments["model"], arguments["reference"], arguments["optimizer"], arguments["generators"])
    assert all(continuation.nested_equal(value, restored[key]) for key, value in initial.items())


def assert_restored(arguments, initial):
    restored = execution._snapshot(arguments["model"], arguments["reference"], arguments["optimizer"], arguments["generators"])
    for key, expected in initial.items():
        assert continuation.nested_equal(expected, restored[key]), key
    assert not arguments["optimized_teacher"]._cache
    assert not arguments["optimized_teacher"].closed
    assert Teacher.instances[-1].backend == "reference" and Teacher.instances[-1].closed


def test_full_update_parity_excludes_only_timing_and_restores_every_parent_stream(monkeypatch, tmp_path):
    arguments, initial, calls = fixture(monkeypatch, tmp_path)
    pool = arguments["optimized_teacher"]._pool
    report = execution.native_execution_parity(**arguments)
    assert report["passed"] and report["status"] == "passed"
    assert all(report["checks"].values()) and report["restoration"]["passed"]
    assert report["excluded_metric_fields"] == ["search_seconds", "prepass_seconds", "teacher_phase_seconds"]
    assert report["reference"]["metrics"]["teacher_phase_seconds"] != report["optimized"]["metrics"]["teacher_phase_seconds"]
    assert report["reference"]["metrics"]["search_seconds"] != report["optimized"]["metrics"]["search_seconds"]
    assert report["reference"]["named_gradients"] == report["optimized"]["named_gradients"]
    assert report["execution"]["real_points"] == 256 and report["execution"]["fresh_prior_points"] == 768
    assert [row[:4] for row in calls] == [("reference", 230860, 36, 35), ("optimized", 230860, 36, 35)]
    assert torch.equal(calls[0][4], calls[1][4])
    assert report["speedup"] > 0 and report["training_updates_promoted"] == 0
    assert arguments["optimized_teacher"]._pool is pool
    assert json.loads(arguments["output"].read_text())["passed"] is True
    assert_restored(arguments, initial)


@pytest.mark.parametrize("defect,check", [("metric", "metrics_exact"), ("gradient", "gradients_exact"),
    ("receipt", "ordered_teacher_receipt_chain_exact"), ("rng", "rng_exact"),
    ("no_target", "repair_target_observed"), ("exception", None)])
def test_semantic_mismatch_or_execution_error_fails_and_preserves_a_failure_receipt(monkeypatch, tmp_path, defect, check):
    arguments, initial, _ = fixture(monkeypatch, tmp_path, defect)
    expected = RuntimeError if defect == "exception" else ValueError
    with pytest.raises(expected):
        execution.native_execution_parity(**arguments)
    report = json.loads(arguments["output"].read_text())
    assert report["status"] == "failed" and not report["passed"]
    assert report["restoration"]["passed"]
    if check is not None:
        assert report["checks"][check] is False
    assert_restored(arguments, initial)


def test_prior_result_is_immutable_and_never_reused_as_benchmark_state(monkeypatch, tmp_path):
    arguments, initial, calls = fixture(monkeypatch, tmp_path)
    arguments["output"].write_text('{"historical":true}')
    with pytest.raises(FileExistsError):
        execution.native_execution_parity(**arguments)
    assert arguments["output"].read_text() == '{"historical":true}'
    assert not calls
    restored = execution._snapshot(arguments["model"], arguments["reference"], arguments["optimizer"], arguments["generators"])
    assert all(continuation.nested_equal(value, restored[key]) for key, value in initial.items())


@pytest.mark.parametrize("ratio,microbatch,workers", [(3, 128, 12), (3, 256, 20), (3, 256, 28),
    (2, 128, 12), (2, 256, 20), (2, 256, 28)])
def test_same_variant_native_proof_routes_overlap_without_retiming_or_reweighting_reference(monkeypatch, tmp_path, ratio, microbatch, workers):
    arguments, initial, calls = fixture(monkeypatch, tmp_path)
    arguments["protocol"].update(prior_ratio=ratio, microbatch_size=microbatch)
    if ratio == 2:
        arguments["protocol"]["composition_prior_weight"] = 1.5
    arguments["optimized_teacher"].workers = workers
    report = execution.native_execution_parity(**arguments, overlap=True)
    assert report["passed"] and all(report["checks"].values())
    assert [row[5] for row in calls] == [False, True]
    assert report["execution"]["fresh_prior_points"] == 256 * ratio
    assert report["execution"]["composition_prior_weight"] * ratio == 3
    assert report["execution"]["microbatch_size"] == microbatch
    assert report["execution"]["optimized_workers"] == workers
    assert report["execution"]["optimized_overlap"] is True
    assert_restored(arguments, initial)


def test_ratio2_without_declared_composition_weight_never_runs_native_update(monkeypatch, tmp_path):
    arguments, initial, calls = fixture(monkeypatch, tmp_path)
    arguments["protocol"]["prior_ratio"] = 2
    with pytest.raises(ValueError, match="composition_prior_weight"):
        execution.native_execution_parity(**arguments, overlap=True)
    assert calls == []
    restored = execution._snapshot(arguments["model"], arguments["reference"], arguments["optimizer"], arguments["generators"])
    assert all(continuation.nested_equal(value, restored[key]) for key, value in initial.items())
