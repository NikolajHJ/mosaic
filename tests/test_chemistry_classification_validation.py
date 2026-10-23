"""Matched classification orchestration fixtures; allocated DTU HPC only.

Actual tiny AdamW updates exercise every restored state and RNG stream. Worker
lifecycle is substituted here; native classifier/process parity is covered by
test_chemistry_classification_pool and the mandatory full native comparison.
"""
from dataclasses import replace
import json
import random

import numpy as np
import pytest
import torch

from latentmol import chemistry_classification_validation as validation
from latentmol import chemistry_continuation as continuation
from latentmol.representation import HardGraphBatch
from latentmol.train import capture_rng_state, restore_rng_state
from test_chemistry_continuation import _hard, _tiny_models


@pytest.fixture(autouse=True)
def preserve_fixture_rng():
    state = capture_rng_state({}, include_cuda=True)
    try:
        yield
    finally:
        restore_rng_state(state, {})


def setup_fixture(monkeypatch, tmp_path, defect=None):
    model, reference, config, protocol, composition = _tiny_models()
    config = replace(config, batch_size=256)
    protocol = {**protocol, "real_batch_size": 256, "microbatch_size": 128,
        "prior_ratio": 2, "composition_prior_weight": 1.5,
        "failure_guidance": {"fixture": "native update substituted below"}}
    row = _hard(model.spec, ("CO",))
    hard = HardGraphBatch(*(value.expand((256,) + value.shape[1:]).clone()
        for value in (row.atoms, row.charges, row.bonds)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    sum(parameter.square().mean() for parameter in model.parameters()).backward()
    optimizer.step()
    next(model.parameters()).grad = None
    model.train(); model.decoder.eval()
    generators = {"train": torch.Generator().manual_seed(311),
        "validation": torch.Generator().manual_seed(312),
        "fresh_prior": torch.Generator().manual_seed(313)}
    teachers, events, calls = [], [], []

    class Pool:
        def map_classification(self, tasks):
            assert len(tasks) == 256
            assert all(spec == model.spec and len(raw) == 3 for raw, spec in tasks)
            events.append(("warm", len(tasks)))
            # Shutdown and startup may consume parent RNG; restoration must
            # follow warmup as well as closure before comparing either arm.
            random.random(); np.random.normal(); torch.rand(())
            if defect == "warm_exception":
                raise RuntimeError("Injected warmup failure")
            return [{}] * (255 if defect == "warm_count" else 256)

    class Teacher:
        def __init__(self, spec, received_protocol, execution, *, classification_execution=None):
            assert all(teacher.closed for teacher in teachers)
            assert execution == dict(backend="optimized", workers=12, torch_threads=4,
                                     gpu_cpus=16, overlap=arguments["overlap"])
            assert received_protocol is protocol and spec == model.spec
            self.policy = classification_execution
            self._pool, self.closed, self._cache = Pool(), False, {"stale": True}
            teachers.append(self)
            events.append(("open", self.policy))

        def clear_cache(self):
            self._cache.clear()

        def drain(self):
            events.append(("drain", self.policy))

        def close(self):
            self.closed = True
            events.append(("close", self.policy))
            random.random(); np.random.normal(); torch.rand(())
            if defect == "close_exception":
                raise RuntimeError("Injected close failure")

    def update(model, reference, optimizer, hard, config, protocol, composition,
               teacher, generators, updates, epoch, phase_start_epoch, *, overlap=False):
        assert not teacher._cache
        assert len(hard.atoms) == 256 and protocol["prior_ratio"] == 2
        calls.append((teacher.policy, updates, epoch, phase_start_epoch, overlap))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        noise = random.random() + float(np.random.normal()) + float(torch.rand(()))
        noise += sum(float(torch.randn((), generator=value)) for value in generators.values())
        sum(parameter.square().mean() + .01 * noise * parameter.mean()
            for parameter in model.parameters()).backward()
        optimizer.step()
        pooled = teacher.policy is not None
        metrics = dict(total=2.25, repair_real=.3, repair_prior=.7,
            real_points=256, fresh_prior_points=512, repair_targets={"real": 1, "prior": 2},
            chemical_classification={"chemical_failures": 3, "representation_failures": 1},
            ordered_teacher_receipt_chain_sha256="b" * 64,
            search_seconds=.25 if pooled else 1., prepass_seconds=.2 if pooled else .8,
            teacher_phase_seconds={"classification_worker_seconds": .5 if pooled else 0.})
        if pooled:
            if defect == "metric": metrics["chemical_classification"]["chemical_failures"] += 1
            elif defect == "receipt": metrics["ordered_teacher_receipt_chain_sha256"] = "c" * 64
            elif defect == "gradient": next(model.parameters()).grad.add_(.125)
            elif defect == "model":
                with torch.no_grad(): next(model.parameters()).add_(.125)
            elif defect == "reference":
                with torch.no_grad(): next(reference.parameters()).add_(.125)
            elif defect == "rng": torch.rand(())
            elif defect == "nonfinite": metrics["total"] = float("nan")
            elif defect == "exception": raise RuntimeError("Injected update failure")
        return metrics

    monkeypatch.setattr(validation, "build_repair_teacher", Teacher)
    monkeypatch.setattr(continuation, "chemistry_update", update)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    arguments = dict(model=model, reference=reference, optimizer=optimizer, hard=hard,
        config=config, protocol=protocol, composition=composition, generators=generators,
        epoch=36, updates=237456, phase_start_epoch=35, output=tmp_path / "paired.json", overlap=False)
    initial = validation._snapshot(model, reference, optimizer, generators)
    return arguments, initial, teachers, events, calls


def assert_restored(arguments, initial, teachers):
    final = validation._snapshot(arguments["model"], arguments["reference"],
        arguments["optimizer"], arguments["generators"])
    assert all(continuation.nested_equal(value, final[key]) for key, value in initial.items())
    assert all(teacher.closed for teacher in teachers)


@pytest.mark.parametrize("overlap", [False, True])
def test_matched_optimized12_updates_restore_state_and_close_pools_sequentially(monkeypatch, tmp_path, overlap):
    arguments, initial, teachers, events, calls = setup_fixture(monkeypatch, tmp_path)
    arguments["overlap"] = overlap
    report = validation.classification_execution_parity(**arguments)
    assert report["passed"] and report["restoration"]["passed"] and all(report["checks"].values())
    assert report["excluded_metric_fields"] == ["search_seconds", "prepass_seconds", "teacher_phase_seconds"]
    assert report["serial_classification"]["classification_execution"] is None
    assert report["pooled_classification"]["classification_execution"] == {
        "version": "repair-classification-pool-v1"}
    assert [row[1:] for row in calls] == [(237456, 37, 35, overlap)] * 2
    assert [row[0] for row in events] == ["open", "warm", "drain", "close"] * 2
    assert len(teachers) == 2 and report["training_updates_promoted"] == 0
    for name in ("serial_classification", "pooled_classification"):
        assert report[name]["warmup_hard_points"] == 256 and report[name]["warmup_seconds"] >= 0
        assert report[name]["seconds"] > 0
    assert report["serial_classification"]["named_gradients"] == report["pooled_classification"]["named_gradients"]
    assert json.loads(arguments["output"].read_text())["passed"] is True
    assert_restored(arguments, initial, teachers)


@pytest.mark.parametrize("defect,check", [("metric", "metrics_exact"),
    ("receipt", "ordered_teacher_receipt_chain_exact"), ("gradient", "gradients_exact"),
    ("model", "model_exact"), ("reference", "reference_unchanged"), ("rng", "rng_exact"),
    ("exception", None), ("nonfinite", None), ("warm_exception", None),
    ("warm_count", None), ("close_exception", None)])
def test_mismatch_or_failure_restores_every_stream_and_retains_failed_evidence(monkeypatch, tmp_path, defect, check):
    arguments, initial, teachers, _, _ = setup_fixture(monkeypatch, tmp_path, defect)
    with pytest.raises((ValueError, RuntimeError)):
        validation.classification_execution_parity(**arguments)
    report = json.loads(arguments["output"].read_text())
    assert not report["passed"] and report["status"] == "failed" and report["restoration"]["passed"]
    if check is not None: assert report["checks"][check] is False
    assert_restored(arguments, initial, teachers)


def test_existing_paired_receipt_is_not_reused_or_overwritten(monkeypatch, tmp_path):
    arguments, initial, teachers, events, calls = setup_fixture(monkeypatch, tmp_path)
    arguments["output"].write_text('{"historical":true}')
    with pytest.raises(FileExistsError): validation.classification_execution_parity(**arguments)
    assert arguments["output"].read_text() == '{"historical":true}' and not teachers and not events and not calls
    assert_restored(arguments, initial, teachers)


@pytest.mark.parametrize("field,value", [("real_batch_size", 255), ("microbatch_size", 64),
                                      ("failure_guidance", None)])
def test_invalid_native_scope_stops_before_workers(monkeypatch, tmp_path, field, value):
    arguments, initial, teachers, events, calls = setup_fixture(monkeypatch, tmp_path)
    arguments["protocol"][field] = value
    with pytest.raises(ValueError): validation.classification_execution_parity(**arguments)
    report = json.loads(arguments["output"].read_text())
    assert report["status"] == "failed" and report["restoration"]["passed"]
    assert not teachers and not events and not calls
    assert_restored(arguments, initial, teachers)
