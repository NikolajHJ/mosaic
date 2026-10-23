"""Explicit worker startup and cost-accounting fixtures; allocated DTU HPC only."""
from copy import deepcopy
import json
import random
from threading import BrokenBarrierError
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from latentmol import chemistry_worker_startup as startup
from latentmol import repair_worker_pool as pools
from latentmol.chemistry_classification import classify_repair_graph
from latentmol.chemistry_continuation import nested_equal
from latentmol.chemistry_execution_validation import _snapshot
from latentmol.quality import graph_identity
from latentmol.train import capture_rng_state, restore_rng_state
from test_chemistry_classification_validation import setup_fixture as parity_fixture
from test_chemistry_continuation import _hard
from latentmol.representation import GraphSpec


@pytest.fixture(autouse=True)
def preserve_rng():
    before = capture_rng_state({}, include_cuda=True)
    try:
        yield
    finally:
        restore_rng_state(before, {})


def fixture(monkeypatch, tmp_path, defect=None):
    args, _, _, _, _ = parity_fixture(monkeypatch, tmp_path)
    model, reference, optimizer = (args[key] for key in ("model", "reference", "optimizer"))
    hard = args["hard"]
    data = [dict(atoms=hard.atoms[i], charges=hard.charges[i], bonds=hard.bonds[i]) for i in range(256)]
    teacher = SimpleNamespace(spec=model.spec, backend="optimized", workers=12, _pool=None,
                              _cache={}, drain=lambda: None)
    classification = classify_repair_graph(hard.atoms[0],hard.charges[0],hard.bonds[0],model.spec)
    events = []
    class Pool:
        def __init__(self, workers):
            assert workers == 12
            self.workers, self.started, self.closed = workers, False, False
        def _map(self, tasks, validate, function):
            assert self.started and len(tasks) == 256 and function is startup._warm_task
            events.append("warm_map")
            random.random(); np.random.normal(); torch.rand(())
            if defect == "exception":raise RuntimeError("Injected native startup failure")
            results = []
            for raw, spec in tasks:
                results.append(dict(classification=deepcopy(classification), raw_sha256=graph_identity(*raw,spec),
                    schema_sha256=spec.fingerprint(), worker_seconds=.01,
                    startup=dict(version=startup.VERSION,pid=123,modules=list(startup.NATIVE_MODULES),
                                 initialized_workers=12,initializer_barrier_passed=True)))
            if defect == "missing":results.pop()
            elif defect == "raw":results[0]["raw_sha256"]="0"*64
            elif defect == "schema":results[0]["schema_sha256"]="0"*64
            elif defect == "workers":results[0]["startup"]["initialized_workers"]=11
            elif defect == "modules":results[0]["startup"]["modules"].pop()
            elif defect == "barrier":results[0]["startup"]["initializer_barrier_passed"]=False
            elif defect == "cache":teacher._cache["unexpected"] = True
            elif defect == "model":
                with torch.no_grad():next(model.parameters()).add_(.125)
            return results
        def close(self):
            events.append("close");self.closed=True
            random.random()
    def install(pool):
        events.append("start");pool.started=True
        random.random();np.random.normal();torch.rand(())
        if defect == "install":raise RuntimeError("Injected initializer creation failure")
        return SimpleNamespace(broken=False,n_waiting=0)
    monkeypatch.setattr(pools,"RepairWorkerPool",Pool)
    monkeypatch.setattr(startup,"_install_startup_executor",install)
    monkeypatch.setenv("LSB_JOBID","12345")
    plan=dict(worker_startup={"version":startup.VERSION},_plan_sha256="a"*64,
        archive_sha256="b"*64,manifest_sha256="c"*64,dataset_sha256="d"*64,
        starting_checkpoint={"sha256":"e"*64},
        execution=dict(backend="optimized",workers=12,torch_threads=4,gpu_cpus=16))
    arguments=dict(plan=plan,teacher=teacher,model=model,reference=reference,optimizer=optimizer,
        data=data,cohorts={"train":list(range(256))},generators=args["generators"],output=tmp_path/"startup.json")
    initial=_snapshot(model,reference,optimizer,args["generators"])
    return arguments,initial,events


def restored(arguments, initial):
    current=_snapshot(arguments["model"],arguments["reference"],arguments["optimizer"],arguments["generators"])
    assert all(nested_equal(value,current[key]) for key,value in initial.items())


def test_warmup_keeps_current_single_pool_cache_empty_and_restores_scientific_state(monkeypatch,tmp_path):
    arguments,initial,events=fixture(monkeypatch,tmp_path,"model")
    report=startup.warm_repair_workers(**arguments)
    assert report["passed"] and report["restoration"]["passed"] and report["warmup_real_points"]==256
    assert report["worker_initialization_barrier_passed"] and report["initialized_workers"]==12
    assert report["target_searches"]==report["optimizer_steps"]==report["random_samples"]==report["model_forwards"]==0
    assert report["elapsed_seconds"]>=report["worker_start_and_classification_seconds"]>0
    assert report["classification"]["attempts"]==256 and report["real_row_ids_sha256"]
    assert events==["start","warm_map"] and arguments["teacher"]._pool.started
    assert not arguments["teacher"]._pool.closed and not arguments["teacher"]._cache
    assert json.loads(arguments["output"].read_text())["passed"] is True
    restored(arguments,initial)


@pytest.mark.parametrize("defect",["exception","missing","raw","schema","workers","modules","barrier","cache","install"])
def test_failed_startup_is_not_retried_or_fabricated_and_restores_parent(monkeypatch,tmp_path,defect):
    arguments,initial,_=fixture(monkeypatch,tmp_path,defect)
    with pytest.raises((ValueError,RuntimeError)):startup.warm_repair_workers(**arguments)
    report=json.loads(arguments["output"].read_text())
    assert not report["passed"] and report["status"]=="failed" and report["restoration"]["passed"]
    assert arguments["teacher"]._pool.closed
    restored(arguments,initial)


@pytest.mark.parametrize("defect",["cache","warm","rows","policy","job","identity"])
def test_scope_guards_do_not_discard_existing_work_or_start_new_workers(monkeypatch,tmp_path,defect):
    arguments,initial,events=fixture(monkeypatch,tmp_path)
    if defect=="cache":arguments["teacher"]._cache["retained"]="scientific target"
    elif defect=="warm":arguments["teacher"]._pool=SimpleNamespace(started=True)
    elif defect=="rows":arguments["cohorts"]["train"]=[0]*256
    elif defect=="policy":arguments["plan"].pop("worker_startup")
    elif defect=="job":monkeypatch.delenv("LSB_JOBID")
    else:arguments["plan"]["archive_sha256"]="unknown"
    with pytest.raises(ValueError):startup.warm_repair_workers(**arguments)
    assert not events
    restored(arguments,initial)
    if defect=="cache":assert arguments["teacher"]._cache=={"retained":"scientific target"}


def test_startup_receipt_is_exclusive(monkeypatch,tmp_path):
    arguments,initial,events=fixture(monkeypatch,tmp_path)
    arguments["output"].write_text('{"old":true}')
    with pytest.raises(FileExistsError):startup.warm_repair_workers(**arguments)
    assert arguments["output"].read_text()=='{"old":true}' and not events
    restored(arguments,initial)


def test_initializer_imports_all_modules_after_native_thread_initialization_then_waits(monkeypatch):
    events=[]
    monkeypatch.setattr(pools,"_initialize_worker",lambda:events.append("native_threads"))
    monkeypatch.setattr(startup.importlib,"import_module",lambda name:events.append(name))
    monkeypatch.setattr(startup,"_STARTUP_PROOF",None)
    barrier=SimpleNamespace(parties=12,wait=lambda **kwargs:events.append(kwargs))
    pools._initialize_worker_after(startup._initialize_warm_worker,(barrier,))
    assert events==["native_threads",*startup.NATIVE_MODULES,{"timeout":180.}]
    assert startup._STARTUP_PROOF["initializer_barrier_passed"] and startup._STARTUP_PROOF["initialized_workers"]==12


def test_broken_initializer_barrier_never_returns_a_ready_proof(monkeypatch):
    monkeypatch.setattr(startup.importlib,"import_module",lambda name:None)
    monkeypatch.setattr(startup,"_STARTUP_PROOF",None)
    def broken(**kwargs):raise BrokenBarrierError
    with pytest.raises(BrokenBarrierError):startup._initialize_warm_worker(SimpleNamespace(parties=12,wait=broken))
    assert startup._STARTUP_PROOF is None


def test_actual_two_worker_initializer_barrier_and_native_tasks_share_one_executor():
    spec=GraphSpec(max_nodes=4);hard=_hard(spec,("CO","CCO"))
    tasks=[(tuple(v[i].numpy().copy() for v in (hard.atoms,hard.charges,hard.bonds)),spec) for i in range(2)]
    with pools.RepairWorkerPool(2) as pool:
        barrier=startup._install_startup_executor(pool)
        results=pool._map(tasks,pools._validate_classification_task,startup._warm_task)
        executor=pool._executor
        assert not barrier.broken and barrier.n_waiting==0
        assert all(row["startup"]["initialized_workers"]==2 for row in results)
        assert all(row["startup"]["modules"]==list(startup.NATIVE_MODULES) for row in results)
        assert [row["classification"] for row in results]==[classify_repair_graph(*raw,spec) for raw,_ in tasks]
        assert len(pool.map_classification(tasks))==2 and pool._executor is executor
        with pytest.raises(ValueError,match="pristine cold"):pool.start()


def test_startup_is_charged_once_without_trimming_any_actual_recurring_slow_update():
    result=startup.project_warmed_epoch([8.,9.,8.],1688539,256,panel_seconds=2010.,startup_seconds=30.)
    assert result["epoch_updates"]==6596
    assert result["conservative_epoch_seconds"]==9.*6596+2010.+600.+30.
    assert result["projected_mean_epoch_seconds"]==sum([8.,9.,8.])/3*6596+2010.+600.+30.
    # A one-time 30-second initialization must not be projected 6,596 times.
    old=max([38.,8.,8.])*6596+2010.+600.
    corrected=startup.project_warmed_epoch([8.,8.,8.],1688539,256,panel_seconds=2010.,startup_seconds=30.)
    assert old-corrected["conservative_epoch_seconds"]==30.*(6596-1)
    # The helper never silently drops a cold/slow wall if the caller includes it.
    retained=startup.project_warmed_epoch([38.,8.,8.],1688539,256,panel_seconds=2010.)
    assert retained["slowest_update_seconds"]==38.


@pytest.mark.parametrize("value",[float("nan"),float("inf"),True,-1.])
def test_invalid_startup_or_panel_overhead_is_rejected(value):
    with pytest.raises(ValueError):startup.project_warmed_epoch([8.],256,256,startup_seconds=value)
    with pytest.raises(ValueError):startup.project_warmed_epoch([8.],256,256,panel_seconds=value)


@pytest.mark.parametrize("value",[True,{}, {"version":"other"},{"version":startup.VERSION,"skip":True}])
def test_policy_is_explicit_strict_and_preserves_absent_default(value):
    assert startup.validate_worker_startup(None) is None
    with pytest.raises(ValueError):startup.validate_worker_startup(value)
