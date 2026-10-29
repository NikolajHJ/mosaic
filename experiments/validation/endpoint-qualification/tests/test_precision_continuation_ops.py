"""Allocated metadata-only controls for the two-arm precision child controller."""
from copy import deepcopy
import ast
import errno
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/precision_continuation_ops.py"
SPEC = importlib.util.spec_from_file_location("precision_continuation_ops_fixture", PATH)
o = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = o
SPEC.loader.exec_module(o)
PIN = "a" * 64
REAL_SEGMENT_BUDGET = o.segment_budget


def measured_budget(arm, offset):
    assert arm in o.DISPATCH_ARMS and offset in o.ARMS[arm]["offsets"]
    walltime, seconds = ("07:15", 26100) if arm == "protenix" else ("05:15", 18900)
    return {"version": "precision-bounded-continuation-segments-v1", "status": "passed", "arm": arm,
            "segment_updates": o.ARMS[arm]["segment_steps"], "walltime": walltime,
            "walltime_seconds": seconds, "limit_seconds": seconds - 600,
            "required_total_seconds": seconds - 850, "reference_seconds": 10800,
            "validation_summary": {"path": str(o.parent_path('validate', arm, offset) / 'payload/study/summary.json'),
                                   "bytes": 100, "sha256": PIN}}


@pytest.fixture(autouse=True)
def allocated_only(monkeypatch):
    assert os.environ.get("LSB_JOBID"), "Controller fixtures require allocated DTU HPC"
    # These are resource/provenance controls, not measured native pilot results.
    # Native qualification separately supplies the real immutable measurement.
    monkeypatch.setattr(o, 'segment_budget', measured_budget)


@pytest.fixture
def owned(tmp_path, monkeypatch):
    source = tmp_path / "release/source"
    root = tmp_path / "run"
    source.mkdir(parents=True)
    root.mkdir()
    monkeypatch.setattr(o, "SOURCE", source)
    monkeypatch.setattr(o, "ROOT", root)
    return source, root


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf8")
    return o.binding(path)


def plan():
    policies = {"protenix": {"protenix_contact_loss": "protenix-separate-precision-admission-v1"},
                "dbmol": {"dbmol_affinity_loss": "dbmol-separate-precision-finer-admission-v1",
                          "dbmol_contact_loss": "prospective-endpoint-admission-v3",
                          "dbmol_confidence_loss": "prospective-endpoint-admission-v3",
                          "dbmol_total_loss": "prospective-endpoint-admission-v3"}}
    refs = {name: {"command_prefix": ["/reference/python", "/reference/worker.py"], "environment": {},
                  "expected_identity": {"status": "ready_full_double_reference"},
                  "historical_global": o.ARMS[name]["offsets"][0],
                  "historical_original_directions": {"X": [], "E_upper": []},
                  "max_session_bytes": 64 * 1024**2,
                  "policy_kind": "protenix-original-fp64" if name == "protenix" else "dbmol-finer-fp64-affinity"}
            for name in o.DISPATCH_ARMS}
    arms = []
    for name in o.DISPATCH_ARMS:
        values, old = o.ARMS[name], o.ORIGINALS[name]
        source = o.BASE / old["release"]
        arms.append({"id": name, "offsets": list(values["offsets"]), "segment_steps": values["segment_steps"],
                     "segment_updates": values["segment_steps"], "start_global": values["offsets"][0],
                     "target_global": values["global_target"], "global_target": values["global_target"], "rate": values["rate"],
                     "source": {"path": str(source), "manifest_sha256": old["manifest"], "files": old["files"]},
                     "config": {"path": str(source / old["config"]), "bytes": old["config_bytes"], "sha256": old["config_sha256"]},
                     "failed_parent": {"job_id": old["job_id"], "endpoint": {"path": "/saved/endpoint.npz", "bytes": 1, "sha256": PIN}},
                     "inputs": {"original_driver_argv": [o.PYTHON, "/original/driver.py", "--mode", "run"]}})
    return {"version": o.PLAN_VERSION, "admission_policy": "precision-reference-continuation-policy-v1",
            "logit_policy": "prospective-fixed-reference-logit-admission-v1",
            "user_authorized_prospective_checks_before_children": True, "original_failures_preserved": True,
            "optimization_settings_changed": False, "monitor_remains_paused": True,
            "matrix_stride": 20, "loss_stride": 1, "dispatch_arms": list(o.DISPATCH_ARMS),
            "automatic_dispatch_arms": list(o.AUTOMATIC_ARMS),
            "preserved_failed_admissions": deepcopy(o.PRESERVED_FAILED_ADMISSIONS),
            "root": str(o.ROOT), "control_root": str(o.ROOT), "source_root": str(o.SOURCE),
            "precision_adapter_version": "precision-reference-continuation-adapter-v1",
            "precision_policy_by_arm": policies, "reference_configuration": {"arms": refs}, "arms": arms}


@pytest.mark.parametrize("arm,offsets,rate,target,steps", [
    ("protenix", [20000], .3, 50000, 30000),
    ("dbmol", [7500], .0025, 10000, 2500)])
def test_exact_two_arm_range_rate_and_immediate_saved_boundary(arm, offsets, rate, target, steps):
    current = plan()
    row = o.arm_plan(current, arm)
    assert row["offsets"] == offsets and row["rate"] == rate
    assert row["global_target"] == target and row["segment_steps"] == steps
    assert {r["id"] for r in current["arms"]} == {"protenix", "dbmol"}
    assert o.ORIGINALS["dbmol"]["job_id"] == "29830999"
    for offset in offsets:
        assert o.label("validate", arm, offset) == arm + "-validate-" + str(offset).zfill(5)


@pytest.mark.parametrize("failure", ["enable_boltz", "missing_arm", "duplicate_arm", "rate", "dbmol_5000",
                                    "target", "steps", "source", "config", "predecessor", "missing_argv",
                                    "policy", "preserved_failure", "stride", "loss_stride", "automatic", "root_alias"])
def test_changed_scope_source_history_or_science_rejected(failure):
    current = plan()
    if failure == "enable_boltz": current["dispatch_arms"].append("boltz")
    elif failure == "missing_arm": current["arms"].pop()
    elif failure == "duplicate_arm": current["arms"][1] = deepcopy(current["arms"][0])
    elif failure == "rate": current["arms"][0]["rate"] = .03
    elif failure == "dbmol_5000": current["arms"][1]["offsets"] = [5000, 7500]
    elif failure == "target": current["arms"][0]["global_target"] = 60000
    elif failure == "steps": current["arms"][1]["segment_steps"] = 1000
    elif failure == "source": current["arms"][1]["source"]["manifest_sha256"] = "b" * 64
    elif failure == "config": current["arms"][0]["config"]["sha256"] = "b" * 64
    elif failure == "predecessor": current["arms"][1]["failed_parent"]["job_id"] = "29810971"
    elif failure == "missing_argv": current["arms"][1]["inputs"].pop("original_driver_argv")
    elif failure == "policy": current["admission_policy"] = "prospective-endpoint-admission-v3"
    elif failure == "preserved_failure": current["preserved_failed_admissions"]["protenix"]["status"] = "passed"
    elif failure == "stride": current["matrix_stride"] = 1
    elif failure == "loss_stride": current["loss_stride"] = 20
    elif failure == "automatic": current["automatic_dispatch_arms"] = ["boltz"]
    else: current["root"] += "-old"
    with pytest.raises(ValueError): o.validate_plan(current)


@pytest.mark.parametrize("phase", ["cpu", "after-cpu", "validate", "after-validate", "segment", "after-segment"])
def test_actual_resources_wrapper_and_dependency_are_declared(phase):
    callback = phase.startswith("after-")
    arm, offset = ("all", 0) if phase in ("cpu", "after-cpu") else ("dbmol", 7500)
    argv = o.argv(phase, PIN, arm, offset, "ended(111)" if callback else None)
    assert argv[-5:] == [str(o.SOURCE), PIN, phase, arm, str(offset)]
    assert str(o.SOURCE / "scripts/precision_continuation_stage.lsf") in argv
    assert argv[argv.index("-n") + 1] == ("4" if phase in ("cpu", "validate", "segment") else "1")
    assert argv[argv.index("-q") + 1] == ("gpul40s" if phase in ("validate", "segment") else "hpc")
    assert ("num=1:mode=exclusive_process:mps=yes" in argv) is (phase in ("validate", "segment"))
    assert ("-w" in argv) is callback
    assert argv[argv.index("-W") + 1] == (measured_budget(arm, offset)["walltime"] if phase == "segment" else "02:00" if phase == "validate"
                                          else "01:00" if phase == "cpu" else "00:20")


def test_segment_budget_delegates_to_exact_own_complete_qualification(owned, monkeypatch):
    calls = []
    expected = measured_budget('dbmol', 7500)
    def budget(path, arm):
        calls.append((path, arm))
        return expected
    monkeypatch.setattr(o, '_module', lambda path, name: SimpleNamespace(runtime_budget=budget))
    assert REAL_SEGMENT_BUDGET('dbmol', 7500) is expected
    assert calls == [(o.parent_path('validate', 'dbmol', 7500) / 'payload/study/summary.json', 'dbmol')]


@pytest.mark.parametrize('arm,offset', [('boltz', 10000), ('dbmol', 5000), ('dbmol', 8750), ('dbmol', 10000),
                                      ('protenix', 21000), ('protenix', 22500), ('protenix', 30000), ('protenix', 50000)])
def test_segment_budget_cannot_measure_old_completed_or_undeclared_range(owned, monkeypatch, arm, offset):
    def forbidden(*args): raise AssertionError('Undeclared source must not import')
    monkeypatch.setattr(o, '_module', forbidden)
    with pytest.raises(ValueError): REAL_SEGMENT_BUDGET(arm, offset)


def test_actual_segment_submission_keeps_exact_measured_receipt_and_complete_remaining_budget(owned, monkeypatch):
    calls = []
    def submitted(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='Job <111> is submitted to queue <gpul40s>.', stderr='')
    monkeypatch.setattr(o.subprocess, 'run', submitted)
    result = o.submit('segment', PIN, 'dbmol', 7500)
    assert result['runtime_budget'] == measured_budget('dbmol', 7500)
    assert result['argv'][result['argv'].index('-W') + 1] == '05:15'
    assert result['argv'][result['argv'].index('-gpu') + 1] == 'num=1:mode=exclusive_process:mps=yes'
    assert result['argv'] == calls[0]
    parent = o.parent_path('segment', 'dbmol', 7500)
    assert o.read(parent / 'submission.json') == result
    assert o.read(parent / 'submission-started.json')['runtime_budget'] == result['runtime_budget']


@pytest.mark.parametrize('failure', ['old_walltime', 'missing_mps', 'other_job'])
def test_own_startup_rejects_unmeasured_walltime_or_missing_gpu_sharing(owned, monkeypatch, failure):
    monkeypatch.setenv('LSB_QUEUE', 'gpul40s')
    monkeypatch.setenv('LSB_DJOB_NUMPROC', '4')
    command = o.argv('segment', PIN, 'dbmol', 7500)
    sub = {'status': 'submitted', 'job_id': os.environ['LSB_JOBID'], 'source_manifest_sha256': PIN,
           'argv': command, 'dependency': None, 'runtime_budget': measured_budget('dbmol', 7500)}
    if failure == 'old_walltime': command[command.index('-W') + 1] = '23:00'
    elif failure == 'missing_mps': command[command.index('-gpu') + 1] = 'num=1:mode=exclusive_process'
    else: sub['job_id'] = '99999999'
    save(o.parent_path('segment', 'dbmol', 7500) / 'submission.json', sub)
    with pytest.raises(ValueError): o.own('segment', PIN, 'dbmol', 7500)


def execution_fixture(owned, monkeypatch, phase):
    source, root = owned
    current = plan()
    current['qualified_admissions'] = {name: {'assessment': save(root / (name + '-qualified.json'), {'status': 'passed'})}
                                       for name in o.DISPATCH_ARMS}
    save(source / 'plan.json', current)
    manifest = {'version': o.VERSION, 'files': {'plan.json': o.sha(source / 'plan.json')}}
    save(source / 'source_manifest.json', manifest)
    parent = o.parent_path(phase, 'dbmol', 7500)
    save(parent / 'submission.json', {'status': 'submitted', 'job_id': os.environ['LSB_JOBID']})
    events = []
    monkeypatch.setattr(o, 'verify', lambda *args: manifest)
    monkeypatch.setattr(o, 'own', lambda *args: {'runtime_budget': measured_budget('dbmol', 7500)})
    monkeypatch.setattr(o, 'external', lambda *args: events.append('external') or {})
    monkeypatch.setattr(o, 'storage', lambda *args: events.append('storage') or {'status': 'passed'})
    monkeypatch.setattr(o, 'certificate', lambda *args: events.append('cpu_certificate'))
    monkeypatch.setattr(o, 'verify_ready', lambda *args: events.append('ready'))
    monkeypatch.setattr(o, 'verify_segment_startup', lambda *args: events.append('actual_startup'))
    def probe(output, **kwargs):
        events.append('cuda_probe')
        assert output == parent / 'cuda-ownership'
        assert kwargs['submission'] == parent / 'submission.json'
        assert kwargs['expected_job_id'] == os.environ['LSB_JOBID']
        assert set(kwargs['input_bindings']) == {'source_manifest', 'plan', 'context_identity'}
        return {'status': 'passed'}
    monkeypatch.setattr(o, '_module', lambda *args: SimpleNamespace(run_probe=probe))
    monkeypatch.setattr(o, 'driver_argv', lambda *args: events.append('build_driver') or ['bound_driver'])
    def child(command, timeout, *args):
        events.append('driver')
        assert command == ['bound_driver']
        assert 0 < timeout <= (measured_budget('dbmol', 7500)['limit_seconds'] if phase == 'segment' else 6900)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(o, 'run_child', child)
    monkeypatch.setattr(o, 'verify_result', lambda *args: events.append('verify_result'))
    return events, parent


@pytest.mark.parametrize('phase', ['validate', 'segment'])
def test_real_two_client_cuda_probe_precedes_driver_with_shared_measured_deadline(owned, monkeypatch, phase):
    events, parent = execution_fixture(owned, monkeypatch, phase)
    o.execute(PIN, phase, 'dbmol', 7500)
    assert events.index('storage') < events.index('cuda_probe') < events.index('build_driver') < events.index('driver')
    assert events[-1] == 'verify_result'
    assert o.read(parent / 'driver-invocation.json')['argv'] == ['bound_driver']


@pytest.mark.parametrize('phase', ['validate', 'segment'])
def test_cuda_ownership_failure_blocks_driver_and_preserves_original_failure(owned, monkeypatch, phase):
    events, parent = execution_fixture(owned, monkeypatch, phase)
    def failed(*args, **kwargs):
        raise ValueError('Actual two-client CUDA ownership failed')
    monkeypatch.setattr(o, '_module', lambda *args: SimpleNamespace(run_probe=failed))
    with pytest.raises(ValueError, match='two-client'): o.execute(PIN, phase, 'dbmol', 7500)
    assert 'build_driver' not in events and 'driver' not in events
    assert not (parent / 'driver-invocation.json').exists()


@pytest.mark.parametrize('changed', ['missing', 'different_limit', 'changed_measurement'])
def test_entry_rejects_changed_or_missing_measured_runtime_receipt_before_any_execution(owned, monkeypatch, changed):
    events, parent = execution_fixture(owned, monkeypatch, 'segment')
    receipt = measured_budget('dbmol', 7500)
    if changed == 'different_limit': receipt['limit_seconds'] += 900
    elif changed == 'changed_measurement': receipt['validation_summary']['sha256'] = 'b' * 64
    monkeypatch.setattr(o, 'own', lambda *args: {} if changed == 'missing' else {'runtime_budget': receipt})
    with pytest.raises(ValueError, match='runtime receipt'): o.execute(PIN, 'segment', 'dbmol', 7500)
    assert events == [] and not (parent / 'driver-invocation.json').exists()


def supervised_child_fixture(owned, monkeypatch, *, finish=650, returncode=0, ignore_term=False):
    parent = owned[1] / 'supervised-child'
    parent.mkdir()
    clock = {'now': 0.0}
    events, launches, signals = [], [], []
    class Child:
        pid = 43210
        def __init__(self):
            self.returncode = None
            self.waits = []
        def poll(self):
            if self.returncode is None and clock['now'] >= finish:
                self.returncode = returncode
            return self.returncode
        def wait(self, timeout):
            self.waits.append(timeout)
            if self.returncode is None:
                raise o.subprocess.TimeoutExpired(['owned-driver'], timeout)
            return self.returncode
    child = Child()
    def popen(command, **kwargs):
        launches.append((command, kwargs))
        return child
    def signal_group(group, value):
        signals.append((group, value))
        assert group == child.pid
        if value == o.signal.SIGKILL or not ignore_term:
            child.returncode = -value
    def gate(path, phase):
        events.append(('storage', clock['now'], Path(path), phase))
        return {'status': 'passed'}
    monkeypatch.setattr(o.time, 'monotonic', lambda: clock['now'])
    monkeypatch.setattr(o.time, 'sleep', lambda seconds: clock.__setitem__('now', clock['now'] + seconds))
    monkeypatch.setattr(o.subprocess, 'Popen', popen)
    monkeypatch.setattr(o.os, 'killpg', signal_group)
    monkeypatch.setattr(o, 'storage', gate)
    monkeypatch.setattr(o, 'verify', lambda *args: events.append(('verify', clock['now'])))
    monkeypatch.setattr(o, 'external', lambda *args: events.append(('external', clock['now'])))
    return parent, child, clock, events, launches, signals


def test_live_child_uses_owned_session_and_periodic_capacity_without_restarting_driver(owned, monkeypatch):
    parent, child, clock, events, launches, signals = supervised_child_fixture(owned, monkeypatch)
    result = o.run_child(['owned-driver'], 1000, parent, {}, {}, PIN)
    assert result.args == ['owned-driver'] and result.returncode == 0
    assert launches == [(['owned-driver'], {'start_new_session': True})]
    checks = [row for row in events if row[0] == 'storage']
    assert [(row[1], row[2], row[3]) for row in checks] == [(300, parent, 'periodic'), (600, parent, 'periodic')]
    assert signals == [] and child.waits == []
    receipt = o.read(parent / 'post-execution-preservation.json')
    assert receipt['execution_status'] == 'returned' and receipt['returncode'] == 0
    assert receipt['preservation_status'] == 'passed' and receipt['original_results_changed'] is False
    assert receipt['owned_process_group'] == child.pid


@pytest.mark.parametrize('failure', ['capacity_status', 'capacity_exception', 'deadline'])
def test_capacity_or_runtime_failure_terminates_only_owned_group_and_preserves_evidence(owned, monkeypatch, failure):
    parent, child, clock, events, launches, signals = supervised_child_fixture(owned, monkeypatch, finish=float('inf'))
    if failure != 'deadline':
        def rejected(*args):
            if failure == 'capacity_exception': raise ValueError('Actual current quota unavailable')
            return {'status': 'failed'}
        monkeypatch.setattr(o, 'storage', rejected)
    with pytest.raises(ValueError):
        o.run_child(['owned-driver'], 12 if failure == 'deadline' else 1000, parent, {}, {}, PIN)
    assert signals == [(child.pid, o.signal.SIGTERM)] and child.waits == [15]
    receipt = o.read(parent / 'post-execution-preservation.json')
    assert receipt['owned_process_group_reaped'] is True and receipt['execution_status'] == 'unknown'
    assert 'execution_error' in receipt and receipt['preservation_status'] == 'passed'
    assert receipt['original_results_changed'] is False
    assert [row[0] for row in events][-2:] == ['verify', 'external']
    if failure == 'deadline': assert clock['now'] == 12


def test_unresponsive_owned_group_is_killed_after_bounded_grace_and_reaped(owned, monkeypatch):
    parent, child, clock, events, launches, signals = supervised_child_fixture(owned, monkeypatch,
                                                                          finish=float('inf'), ignore_term=True)
    with pytest.raises(ValueError, match='deadline'):
        o.run_child(['owned-driver'], 12, parent, {}, {}, PIN)
    assert signals == [(child.pid, o.signal.SIGTERM), (child.pid, o.signal.SIGKILL)]
    assert child.waits == [15, 15]
    assert o.read(parent / 'post-execution-preservation.json')['owned_process_group_reaped'] is True


@pytest.mark.parametrize('timeout', [0, -1, True, float('inf'), float('nan')])
def test_invalid_runtime_budget_cannot_spawn_a_driver(owned, monkeypatch, timeout):
    parent, child, clock, events, launches, signals = supervised_child_fixture(owned, monkeypatch)
    with pytest.raises(ValueError, match='Finite positive'):
        o.run_child(['owned-driver'], timeout, parent, {}, {}, PIN)
    assert launches == signals == []
    receipt = o.read(parent / 'post-execution-preservation.json')
    assert 'execution_error' in receipt and receipt['preservation_status'] == 'passed'


def test_periodic_metadata_cannot_consume_internal_budget_and_admit_a_late_child(owned, monkeypatch):
    parent, child, clock, events, launches, signals = supervised_child_fixture(owned, monkeypatch, finish=320)
    def slow_scan(*args):
        clock['now'] += 40
        return {'status': 'passed'}
    monkeypatch.setattr(o, 'storage', slow_scan)
    with pytest.raises(ValueError, match='Periodic metadata exhausted'):
        o.run_child(['owned-driver'], 330, parent, {}, {}, PIN)
    assert clock['now'] == 340 and signals == []
    receipt = o.read(parent / 'post-execution-preservation.json')
    assert receipt['execution_status'] == 'unknown' and 'execution_error' in receipt


def test_child_finishing_exactly_at_internal_cap_is_rejected_without_unowned_cleanup(owned, monkeypatch):
    parent, child, clock, events, launches, signals = supervised_child_fixture(owned, monkeypatch, finish=12)
    with pytest.raises(ValueError, match='outside its internal deadline'):
        o.run_child(['owned-driver'], 12, parent, {}, {}, PIN)
    assert clock['now'] == 12 and signals == [] and child.returncode == 0
    receipt = o.read(parent / 'post-execution-preservation.json')
    assert receipt['execution_status'] == 'unknown' and receipt['preservation_status'] == 'passed'


def test_nonzero_child_result_is_preserved_without_masquerading_as_success(owned, monkeypatch):
    parent, child, clock, events, launches, signals = supervised_child_fixture(owned, monkeypatch, finish=5, returncode=7)
    result = o.run_child(['owned-driver'], 1000, parent, {}, {}, PIN)
    assert result.returncode == 7 and signals == []
    receipt = o.read(parent / 'post-execution-preservation.json')
    assert receipt['returncode'] == 7 and receipt['execution_status'] == 'returned'
    assert receipt['preservation_status'] == 'passed' and receipt['original_results_changed'] is False


def test_popen_failure_has_no_unowned_cleanup_and_still_authenticates_saved_inputs(owned, monkeypatch):
    parent, child, clock, events, launches, signals = supervised_child_fixture(owned, monkeypatch)
    def failed(*args, **kwargs): raise OSError('Process creation failed')
    monkeypatch.setattr(o.subprocess, 'Popen', failed)
    with pytest.raises(OSError, match='creation failed'):
        o.run_child(['owned-driver'], 1000, parent, {}, {}, PIN)
    assert signals == [] and [row[0] for row in events] == ['verify', 'external']
    receipt = o.read(parent / 'post-execution-preservation.json')
    assert 'execution_error' in receipt and receipt['preservation_status'] == 'passed'


def test_postexecution_input_change_is_failed_and_retained_even_after_child_returns(owned, monkeypatch):
    parent, child, clock, events, launches, signals = supervised_child_fixture(owned, monkeypatch, finish=5)
    def changed(*args): raise ValueError('Protected source bytes changed')
    monkeypatch.setattr(o, 'verify', changed)
    with pytest.raises(ValueError, match='source bytes changed'):
        o.run_child(['owned-driver'], 1000, parent, {}, {}, PIN)
    receipt = o.read(parent / 'post-execution-preservation.json')
    assert receipt['execution_status'] == 'returned' and receipt['preservation_status'] == 'failed'
    assert receipt['original_results_changed'] is None and 'preservation_error' in receipt


@pytest.mark.parametrize("phase,dependency", [("validate", "ended(111)"), ("segment", "done(111)"),
                                              ("after-validate", None), ("after-segment", "done(111)")])
def test_interarm_or_wrong_dependency_cannot_be_introduced(phase, dependency):
    with pytest.raises(ValueError): o.argv(phase, PIN, "dbmol", 7500, dependency)


def test_unknown_submission_preserves_exclusive_marker_and_prevents_duplicate(owned, monkeypatch):
    def unknown(*args, **kwargs): raise TimeoutError("unknown actual bsub outcome")
    monkeypatch.setattr(o.subprocess, "run", unknown)
    with pytest.raises(ValueError, match="Unknown"): o.submit("cpu", PIN)
    root = owned[1]
    assert o.read(root / "cpu/submission.json")["status"] == "unknown"
    assert (root / "cpu/submission-started.json").is_file()
    with pytest.raises(FileExistsError): o.submit("cpu", PIN)


def test_actual_submission_keeps_only_own_ended_pair(owned, monkeypatch):
    calls = []
    def submit(phase, pin, arm, offset, dependency=None):
        calls.append((phase, arm, offset, dependency))
        return {"job_id": str(111 + len(calls))}
    monkeypatch.setattr(o, "submit", submit)
    assert o._pair("validate", PIN, "dbmol", 7500) == {"producer": "112", "callback": "113"}
    assert calls == [("validate", "dbmol", 7500, None), ("after-validate", "dbmol", 7500, "ended(112)")]


def test_initial_checkpoint_is_immediate7500_failed_parent(owned):
    current = plan()
    exact = save(owned[1] / "saved7500.json", {"original": "unchanged7500"})
    current["arms"][1]["failed_parent"]["endpoint"] = exact
    assert o.input_snapshot(current, "dbmol", 7500) == exact
    with pytest.raises(ValueError): o.input_snapshot(current, "dbmol", 5000)


@pytest.mark.parametrize('offset', [22500, 30000, 40000, 50000])
def test_no_intermediate_Pro_child_can_be_invented_even_with_passed_partial_certificate(owned, offset):
    source, root = owned
    current = plan()
    save(source / "plan.json", current)
    save(source / "source_manifest.json", {"files": {"plan.json": o.sha(source / "plan.json")}})
    cert_path = root / (o.label("segment", "protenix", 20000) + "-certificate.json")
    save(cert_path, {"status": "failed", "arm": "protenix", "global_end": offset})
    with pytest.raises(ValueError, match="Undeclared input boundary"): o.input_snapshot(current, "protenix", offset)
    proof = save(root / "parent-proof.json", {"status": "passed_new_child"})
    terminal = save(root / ("terminal" + str(offset) + ".json"), {"exact": "partial_or_completed_endpoint"})
    save(cert_path, {"status": "passed", "arm": "protenix", "global_end": offset,
                     "source_manifest_sha256": o.sha(source / "source_manifest.json"),
                     "plan_sha256": o.sha(source / "plan.json"), "proofs": [proof], "terminal_snapshot": terminal})
    with pytest.raises(ValueError, match="Undeclared input boundary"): o.input_snapshot(current, "protenix", offset)


@pytest.mark.parametrize("failure", ["version", "status", "certify", "changed", "unbound"])
def test_original_pro_failure_is_preserved_and_never_recertified(owned, monkeypatch, failure):
    root = owned[1]
    observation = {"version": "prospective-endpoint-admission-v1", "status": "failed", "can_admit_new_child": False,
                   "original_run_failed_preserved": True, "can_certify_original_study": False,
                   "frozen_unchanged": True, "field_outcomes": {"E_upper": {"passed": False}}}
    original = save(root / "original-failure.json", observation)
    declared = {"protenix": {"status": "blocked_failed_endpoint_admission", "source01_admission": original}}
    monkeypatch.setattr(o, "PRESERVED_FAILED_ADMISSIONS", declared)
    current = plan()
    manifest = {"preserved_failed_admissions": deepcopy(declared), "preserved_proofs": [original]}
    assert o.preserved_failed_admission(manifest, current) == original
    if failure == "version": observation["version"] = "new"
    elif failure == "status": observation["status"] = "passed"
    elif failure == "certify": observation["can_certify_original_study"] = True
    elif failure == "changed": observation["field_outcomes"]["E_upper"]["passed"] = True
    else: manifest["preserved_proofs"] = []
    if failure != "unbound":
        rebound = save(root / "original-failure.json", observation)
        declared["protenix"]["source01_admission"] = rebound
        current["preserved_failed_admissions"] = deepcopy(declared)
        manifest = {"preserved_failed_admissions": deepcopy(declared), "preserved_proofs": [rebound]}
    with pytest.raises(ValueError): o.preserved_failed_admission(manifest, current)


def qualified_fixture(owned, monkeypatch):
    root = owned[1]
    expected = deepcopy(o.QUALIFIED_NATIVE)
    declarations = {}
    for name, spec in expected.items():
        native_root = root / ("qualified-" + name)
        source = root / ("qualified-source-" + name)
        source.mkdir()
        spec["root"] = str(native_root)
        spec["source"]["path"] = str(source)
        proofs = []
        def artifact(relative, value):
            row = save(native_root / relative, value)
            proofs.append(row)
            return row
        plan_row = save(source / "plan.json", {"scope": "metadata toy qualification"})
        proofs.append(plan_row)
        preserved_proof = artifact("cpu/raw-proof.json", {"fixture": "passed"})
        artifact("cpu-certificate.json", {"status": "passed", "source_manifest_sha256": spec["source"]["manifest_sha256"],
            "cases": 1, "proofs": [preserved_proof]})
        policy = "protenix-separate-precision-admission-v1" if name == "protenix" else "dbmol-separate-precision-finer-admission-v1"
        admission = {"version": policy, "status": "passed", "can_admit_new_child": True,
                     "can_certify_original_study": False, "original_failed_outcome_preserved": True,
                     "frozen_unchanged": True, "baseline_fp32_unchanged": True, "optimization_updates": 0,
                     "failures": [], "field_outcomes": {"X": {"passed": True}, "E_upper": {"passed": True}},
                     "directions": [{"field": "X", "passed": True}, {"field": "E_upper", "passed": True}]}
        prefix = "gpu/payload/reference/" if name == "protenix" else "gpu/payload/native/"
        admission_row = artifact(prefix + ("admission.json" if name == "protenix" else "native-admission.json"), admission)
        for negative in ("zero", "reversed", "scaled"):
            artifact(prefix + negative + ("-control.json" if name == "protenix" else "-admission.json"),
                     {"status": "failed", "can_admit_new_child": False, "can_certify_original_study": False})
        if name == "protenix":
            reference = artifact(prefix + "assessment.json", {"status": "passed_new_prospective_precision_measurement",
                "updates": 0, "admission": admission_row, "original_results_changed": False})
            production = artifact("gpu/payload/production/assessment.json", {"status": "passed_fp32_measurements"})
            workers = {}
            for worker in ("fp32", "fp64"):
                preservation = artifact("gpu/payload/" + worker + "-final-preservation.json", {
                    "status": "passed", "updates": 0, "protected_inputs_unchanged": True,
                    "frozen_unchanged": True, "preservation_errors": []})
                workers[worker] = {"assessment": production if worker == "fp32" else reference, "preservation": preservation}
            artifact("gpu/payload/protected-before.json", {"saved": "unchanged"})
            artifact("gpu/payload/protected-after.json", {"saved": "unchanged"})
            artifact(prefix + "native-dtype-audit.json", {"status": "passed", "floating_operations_float64": True,
                                                        "narrow_float_avals": []})
            assessment = {"status": "prospective_precision_gate_passed_original_failures_preserved", "job_id": spec["job_id"],
                "updates": 0, "original_results_changed": False, "optimization_settings_changed": False,
                "can_certify_or_restart_original_study": False, "production_optimizer_executed": False,
                "frozen_unchanged": True, "preservation_errors": [], "plan": plan_row,
                "reference_assessment": reference, "worker_proofs": workers}
        else:
            inner = artifact(prefix + "assessment.json", {"status": "passed_affinity_reference_only", "error": None,
                "optimizer_updates": 0, "original_results_changed": False, "can_submit_continuation": False,
                "remaining_admission": "all_components_raw_null_parity_pilot_runtime_startup_full_storage"})
            artifact(prefix + "preservation.json", {"status": "passed", "before": {"saved": "unchanged"},
                                                    "after": {"saved": "unchanged"}})
            artifact(prefix + "reference-operation-audit.json", {"status": "closed", "audit": {
                "status": "passed_full_operation_FP64_audit", "narrow_floating_operations": 0,
                "production_execution_changed": False}})
            artifact(prefix + "production-frozen-after.json", {"frozen": True})
            assessment = {"status": "passed_affinity_reference_only", "optimizer_jobs_submitted": False, "native_assessment": inner}
        assessment_row = artifact("gpu/payload/assessment.json", assessment)
        command = ["bsub", "-q", "gpul40s", "-n", "4", "-R", "span[hosts=1]", "-R", "rusage[mem=8GB]",
                   "-gpu", "num=1:mode=exclusive_process", "-W", "02:00", "-J", spec["name"],
                   "-oo", str(native_root / "gpu/stdout.%J.log"), "-eo", str(native_root / "gpu/stderr.%J.log"),
                   "bash", "-l", str(source / "scripts" / spec["wrapper"]), str(source),
                   spec["source"]["manifest_sha256"], "gpu"]
        submitted = artifact("gpu/submission.json", {"status": "submitted", "job_id": spec["job_id"], "phase": "gpu",
            "dependency": None, "source_manifest_sha256": spec["source"]["manifest_sha256"], "argv": command, "returncode": 0})
        stdout = native_root / "gpu" / ("stdout." + spec["job_id"] + ".log")
        stdout.write_text("Subject: Job " + spec["job_id"] + ": <" + spec["name"] + "> in cluster <dcc> Done\n"
            "Job was executed in queue <gpul40s>.\nSuccessfully completed.\n", encoding="utf8")
        stdout_row = o.binding(stdout); proofs.append(stdout_row)
        declarations[name] = {"source": deepcopy(spec["source"]), "job_id": spec["job_id"], "assessment": assessment_row,
                              "submission": submitted, "stdout": stdout_row, "proofs": proofs}
    monkeypatch.setattr(o, "QUALIFIED_NATIVE", expected)
    def verified(source, pin):
        spec = next(spec for spec in expected.values() if spec["source"]["path"] == str(source))
        assert spec["source"]["manifest_sha256"] == pin
        return {"files": {str(i): PIN for i in range(spec["source"]["files"])}, "cpu_counts": {"tests.test_toy": 1}}
    monkeypatch.setattr(o, "verify", verified)
    return {"qualified_admissions": declarations}


def test_both_exact_qualified_native_passes_are_component_only_adoption(owned, monkeypatch):
    current = qualified_fixture(owned, monkeypatch)
    result = o.qualified_admissions(current)
    assert result["status"] == "passed_readonly_native_qualification_adoption"
    assert result["full_continuation_admission_established"] is result["original_results_changed"] is False
    assert set(result["arms"]) == {"protenix", "dbmol"}
    assert all(row["full_continuation_admission_established"] is False for row in result["arms"].values())


@pytest.mark.parametrize("failure", ["missing_arm", "wrong_job", "source", "proof_changed", "lsf_failed",
                                    "overall_native_failed", "negative_admitted", "original_recertified"])
def test_qualified_native_missing_wrong_or_failed_evidence_cannot_authorize_children(owned, monkeypatch, failure):
    current = qualified_fixture(owned, monkeypatch)
    declarations = current["qualified_admissions"]
    row = declarations["dbmol"]
    if failure == "missing_arm": del declarations["protenix"]
    elif failure == "wrong_job": row["job_id"] = "999"
    elif failure == "source": row["source"]["manifest_sha256"] = "f" * 64
    else:
        if failure == "proof_changed":
            Path(row["assessment"]["path"]).write_text("changed", encoding="utf8")
        elif failure == "lsf_failed":
            Path(row["stdout"]["path"]).write_text("Exited with exit code 1.\n", encoding="utf8")
        else:
            proof = (row["assessment"] if failure == "overall_native_failed" else next(
                proof for proof in row["proofs"] if proof["path"].endswith("/zero-admission.json")))
            value = o.read(proof["path"])
            if failure == "overall_native_failed": value["status"] = "failed"
            elif failure == "negative_admitted": value.update(status="passed", can_admit_new_child=True)
            else: value["can_certify_original_study"] = True
            replacement = save(Path(proof["path"]), value)
            row["proofs"] = [replacement if item["path"] == replacement["path"] else item for item in row["proofs"]]
            if failure == "overall_native_failed": row["assessment"] = replacement
    with pytest.raises(ValueError): o.qualified_admissions(current)


def capacity_fixture(owned, monkeypatch, available=80 * 1024**3):
    source, root = owned
    parent = root / "storage-test"
    parent.mkdir()
    prior = {"status": "passed", "required_bytes": 50 * o.GIB, "available_bytes": available,
             "reserve_bytes": 4 * o.GIB, "file_reserve": 100000, "original_gate_changed": False,
             "original_full_forecast": {"version": "selected-nonchem-full-storage-v1", "selected": {
                 "protenix50k40k": {"remaining": 3 * o.GIB}, "dbmol10k": {"remaining": o.GIB}}}}
    calls = []
    def gate(where, phase):
        calls.append((Path(where), phase))
        return deepcopy(prior)
    monkeypatch.setattr(o, "_module", lambda *args: SimpleNamespace(gate=gate))
    monkeypatch.setattr(o, "metadata_bytes", lambda: 0)
    monkeypatch.setattr(o, "owned_bytes", lambda path: 100 if path == root else 200)
    return parent, prior, calls, gate


@pytest.mark.parametrize('status', ['passed', 'failed'])
def test_production_storage_delegates_only_to_new_rolling_gate(owned, monkeypatch, status):
    parent = owned[1] / 'new-entry'
    parent.mkdir()
    calls = []
    receipt = {'status': status, 'reserve_bytes': 4 * o.GIB, 'file_reserve': 100000}
    def gate(controller, current_parent, phase):
        calls.append((controller, current_parent, phase))
        return receipt
    modules = []
    def module(path, name):
        modules.append(path)
        return SimpleNamespace(storage=gate)
    monkeypatch.setattr(o, '_module', module)
    if status == 'passed': assert o.storage(parent, 'entry') is receipt
    else:
        with pytest.raises(ValueError, match='Rolling'): o.storage(parent, 'entry')
    assert calls == [(o, parent, 'entry')]
    assert modules == [o.SOURCE / 'scripts/rolling_capacity.py']


def test_storage_adds_all_twelve_reference_caps_control_and_entire_new_tree_charge(owned, monkeypatch):
    parent, prior, calls, gate = capacity_fixture(owned, monkeypatch)
    result = o.legacy_storage(parent, "dispatch")
    assert result["status"] == "passed" and len(calls) == 1
    assert result["required_bytes"] == prior["required_bytes"] + 1280 * 1024**2
    assert result["available_bytes"] == prior["available_bytes"] - 700
    assert len(result["entire_new_tree_stale_charge"]) == 4  # Failed14 and current15 remain fully charged.
    assert result["additional_complete_reference_and_control_bytes"] == 1280 * 1024**2
    assert result["reference_sessions"] == 12 and result["reference_session_cap"] == 64 * 1024**2
    assert result["forecasts_removed"] is result["original_gate_changed"] is False
    assert result["reserve_bytes"] == 4 * o.GIB and result["file_reserve"] == 100000
    assert o.read(parent / "continuation-storage-dispatch.json") == result


def test_actual_storage_funds_fresh_pilot_positive_excess_without_borrowing_old_future(owned, monkeypatch):
    parent, prior, calls, gate = capacity_fixture(owned, monkeypatch)
    save(owned[1] / "projections.json", {"dbmol": {"unit_bytes": 20000000, "remaining_offsets": [7500]}})
    excess, details = o.projection_excess_from_gate(prior)
    assert excess > 0
    result = o.legacy_storage(parent, "dispatch")
    assert result["required_bytes"] == prior["required_bytes"] + 1280 * 1024**2 + excess
    assert result["additional_positive_pilot_projection_bytes"] == excess
    assert result["new_pilot_projection_details"] == details
    assert prior["original_full_forecast"]["selected"]["dbmol10k"]["remaining"] == o.GIB


def test_projection_uses_smallest_authenticated_inherited_remaining_budget(owned):
    save(owned[1] / "projections.json", {"dbmol": {"unit_bytes": 20000000, "remaining_offsets": [7500]}})
    gates = {"one": {"version": "selected-nonchem-full-storage-v1", "selected": {
        "protenix50k40k": {"remaining": 9 * o.GIB}, "dbmol10k": {"remaining": 5 * o.GIB}}},
        "two": [{"version": "selected-nonchem-full-storage-v1", "selected": {
            "protenix50k40k": {"remaining": 3 * o.GIB}, "dbmol10k": {"remaining": o.GIB}}}]}
    amount, details = o.projection_excess_from_gate(gates)
    assert amount > 0 and details["dbmol"]["original_unfunded_projection"] == o.GIB


@pytest.mark.parametrize("failure", ["missing_profiles", "negative_remaining", "boolean_remaining"])
def test_fresh_projection_cannot_use_missing_or_nonfinite_inherited_full_budget(owned, failure):
    prior = {"version": "selected-nonchem-full-storage-v1", "selected": {
        "protenix50k40k": {"remaining": o.GIB}, "dbmol10k": {"remaining": o.GIB}}}
    if failure == "missing_profiles": del prior["selected"]["dbmol10k"]
    elif failure == "negative_remaining": prior["selected"]["dbmol10k"]["remaining"] = -1
    else: prior["selected"]["dbmol10k"]["remaining"] = True
    with pytest.raises(ValueError): o.projection_excess_from_gate(prior)


@pytest.mark.parametrize("field,value", [("status", "failed"), ("original_gate_changed", True),
                                        ("reserve_bytes", 3 * 1024**3), ("file_reserve", 99999)])
def test_inherited_storage_or_reserve_failure_cannot_be_waived(owned, monkeypatch, field, value):
    parent, prior, calls, gate = capacity_fixture(owned, monkeypatch)
    prior[field] = value
    with pytest.raises(ValueError): o.legacy_storage(parent, "entry")
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["new_capacity", "control_cap"])
def test_new_full_storage_failure_is_preserved_without_retry(owned, monkeypatch, failure):
    parent, prior, calls, gate = capacity_fixture(owned, monkeypatch, available=50 * 1024**3 + 1000)
    if failure == "control_cap":
        prior["available_bytes"] = 80 * o.GIB
        monkeypatch.setattr(o, "metadata_bytes", lambda: o.CONTROL_CAP + 1)
    with pytest.raises(ValueError): o.legacy_storage(parent, "entry")
    assert len(calls) == 1
    failure_path = parent / "whole-precision-entry-1/full-precision-storage.json"
    assert failure_path.is_file() and o.read(failure_path)["status"] == "failed"
    assert not (parent / "whole-precision-entry-2").exists()


def test_exact_owned_atomic_race_retries_whole_inventory_and_keeps_failed_attempt(owned, monkeypatch):
    parent, prior, calls, normal = capacity_fixture(owned, monkeypatch)
    atomic = o.BASE / "nonchem-endpoint-continuation-09/boltz-segment-10000/payload/study/case-0/references/latest-reference.json.tmp"
    attempts = []
    def racing(where, phase):
        attempts.append(Path(where))
        if len(attempts) == 1: raise FileNotFoundError(errno.ENOENT, "owned atomic reference replaced", str(atomic))
        return deepcopy(prior)
    monkeypatch.setattr(o, "_module", lambda *args: SimpleNamespace(gate=racing))
    assert o.legacy_storage(parent, "dispatch")["status"] == "passed"
    assert len(attempts) == 2
    incomplete = o.read(attempts[0] / "incomplete-owned-atomic-inventory.json")
    assert incomplete["partial_totals_used"] is incomplete["files_skipped"] is False


@pytest.mark.parametrize("kind", ["unrelated_missing", "exhausted_atomic", "capacity"])
def test_inventory_retry_never_skips_unrelated_files_or_retries_other_failures(owned, monkeypatch, kind):
    parent, prior, calls, normal = capacity_fixture(owned, monkeypatch)
    atomic = o.BASE / "nonchem-endpoint-continuation-09/boltz-segment-10000/payload/study/case-0/references/latest-reference.json.tmp"
    attempts = []
    def failing(where, phase):
        attempts.append(Path(where))
        if kind == "capacity": raise ValueError("no capacity")
        raise FileNotFoundError(errno.ENOENT, "missing", str(atomic if kind == "exhausted_atomic" else parent / "unrelated-proof"))
    monkeypatch.setattr(o, "_module", lambda *args: SimpleNamespace(gate=failing))
    with pytest.raises((ValueError, FileNotFoundError)): o.legacy_storage(parent, "entry")
    assert len(attempts) == (3 if kind == "exhausted_atomic" else 1)


def test_projection_only_adds_positive_fresh_pilot_excess_and_never_reduces_old_future():
    original = {"selected": {"protenix50k40k": {"remaining": 3 * o.GIB}, "dbmol10k": {"remaining": o.GIB}}}
    forecast = {"dbmol": {"unit_bytes": 20, "remaining_offsets": [7500]}}
    extra, details = o.projection_excess(original, forecast)
    assert extra >= 0 and original["selected"]["dbmol10k"]["remaining"] == o.GIB
    forecast["dbmol"]["unit_bytes"] = 100000000
    larger, details = o.projection_excess(original, forecast)
    assert larger > extra and larger == details["dbmol"]["positive_excess_bytes"]
    assert details["dbmol"]["remaining_offsets"] == [7500]


@pytest.mark.parametrize("offsets", [[5000, 7500], [7500, 5000], [], [9999], [7500, 8750], [8750]])
def test_dbmol_projection_suffix_cannot_restore_completed_or_invented_updates(offsets):
    with pytest.raises(ValueError):
        o.projection_excess({"selected": {"dbmol10k": {"remaining": 1}}},
                            {"dbmol": {"unit_bytes": 20, "remaining_offsets": offsets}})


def result_fixture(owned, monkeypatch):
    source, root = owned
    current = plan()
    snapshot = save(root / "saved7500.json", {"original": "preserved7500"})
    row = next(item for item in current["arms"] if item["id"] == "dbmol")
    row["failed_parent"]["endpoint"] = snapshot
    save(source / "plan.json", current)
    proof = save(root / "new-native-proof.json", {"passed": True})
    result = {"version": o.RESULT_VERSION, "status": "passed", "job_id": "111", "arm": "dbmol",
              "phase": "validate", "offset": 7500, "global_end": 7500, "source_manifest_sha256": PIN,
              "plan_sha256": o.sha(source / "plan.json"), "config_sha256": row["config"]["sha256"],
              "original_results_changed": False, "frozen_unchanged": True, "original_failed_parent_preserved": True,
              "scientific_source_manifest_sha256": row["source"]["manifest_sha256"],
              "endpoint_policy": "precision-reference-continuation-policy-v1", "logit_records": [],
              "checks": {key: True for key in o.CHECKS}, "proofs": [proof], "input_snapshot": snapshot,
              "projections": {"maximum_pilot_update_bytes": 2000},
              "applicability": {"pilot_high": "not_applicable_chemistry_disabled",
                                "terminal_admission": "not_applicable_validation_phase",
                                "case_jsonl": "not_applicable_validation_phase", "fp32_boundary": "required",
                                "null_controls": "original_null_controls"}}
    parent = o.parent_path("validate", "dbmol", 7500)
    save(parent / "submission.json", {"status": "submitted", "job_id": "111"})
    save(parent / "post-execution-preservation.json", {"preservation_status": "passed",
          "source_manifest_sha256": PIN, "original_results_changed": False})
    path = parent / "payload/assessment.json"
    save(path, result)
    calls = []
    def verify_precision(*args):
        calls.append(args)
        return args[0]
    monkeypatch.setattr(o, "_module", lambda *args: SimpleNamespace(verify_precision_result=verify_precision))
    return current, result, path, calls


def test_result_requires_additional_reference_adapter_verification(owned, monkeypatch):
    current, result, path, calls = result_fixture(owned, monkeypatch)
    assert o.verify_result(current, PIN, "validate", "dbmol", 7500) == result
    assert len(calls) == 1 and calls[0][0] == result and calls[0][2:6] == (PIN, "validate", "dbmol", 7500)
    def reject(*args): raise ValueError("Reference co-residency failed")
    monkeypatch.setattr(o, "_module", lambda *args: SimpleNamespace(verify_precision_result=reject))
    with pytest.raises(ValueError, match="co-residency"): o.verify_result(current, PIN, "validate", "dbmol", 7500)


@pytest.mark.parametrize("failure", ["result_failed", "old_policy", "wrong_source", "wrong_config", "wrong_job",
                                    "frozen", "original_changed", "missing_check", "false_check", "snapshot", "global_end"])
def test_failed_identity_nonraw_gate_or_policy_blocks_before_reference_verifier(owned, monkeypatch, failure):
    current, result, path, calls = result_fixture(owned, monkeypatch)
    if failure == "result_failed": result["status"] = "failed"
    elif failure == "old_policy": result["endpoint_policy"] = "prospective-endpoint-admission-v3"
    elif failure == "wrong_source": result["source_manifest_sha256"] = "b" * 64
    elif failure == "wrong_config": result["config_sha256"] = "b" * 64
    elif failure == "wrong_job": result["job_id"] = "112"
    elif failure == "frozen": result["frozen_unchanged"] = False
    elif failure == "original_changed": result["original_results_changed"] = True
    elif failure == "missing_check": del result["checks"]["null_controls"]
    elif failure == "false_check": result["checks"]["hard_parity"] = False
    elif failure == "snapshot": result["input_snapshot"] = {**result["input_snapshot"], "sha256": "b" * 64}
    else: result["global_end"] = 10000
    save(path, result)
    with pytest.raises(ValueError): o.verify_result(current, PIN, "validate", "dbmol", 7500)
    if failure != "global_end": assert calls == []


def reference_startup_fixture(owned, monkeypatch):
    source, root = owned
    current = plan()
    for arm, config in current["reference_configuration"]["arms"].items():
        config["command_prefix"] = [o.PYTHON, str(source / "scripts" /
            ("protenix_reference_worker.py" if arm == "protenix" else "dbmol_reference.py"))]
        if arm == "protenix": config["command_prefix"] += ["--reference-plan-sha256", "${OWN_PLAN_SHA256}"]
    save(source / "plan.json", current)
    save(root / "cpu/submission.json", {"status": "submitted", "job_id": os.environ["LSB_JOBID"]})
    payload = root / "cpu/payload"
    payload.mkdir()
    calls = []
    def expanded(config, plan_sha):
        return [plan_sha if item == "${OWN_PLAN_SHA256}" else item for item in config["command_prefix"]]
    monkeypatch.setattr(o, "_module", lambda *args: SimpleNamespace(expanded_command=expanded))
    def run(command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[command.index("--startup-output") + 1])
        receipt = {"status": "passed_metadata_only", "numerical_setup_entered": False,
                   "optimizer_updates": 0, "job_id": os.environ["LSB_JOBID"]}
        if output.name.startswith("protenix"):
            receipt.update(own_source_manifest_sha256=PIN, plan_sha256=o.sha(source / "plan.json"))
        save(output, receipt)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(o.subprocess, "run", run)
    return current, payload, calls


def test_both_actual_reference_startup_commands_are_metadata_only_and_file_bound(owned, monkeypatch):
    current, payload, calls = reference_startup_fixture(owned, monkeypatch)
    reports = o.cpu_reference_startups(current, PIN, payload, o.time.monotonic() + 1000)
    assert len(reports) == len(calls) == 2
    for command, kwargs in calls:
        assert command[-3:] == ["--startup-only", "--startup-output", command[-1]]
        assert kwargs["env"]["JAX_PLATFORMS"] == "cpu" and kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert "${OWN_PLAN_SHA256}" not in command
        assert Path(command[-1]).is_relative_to(payload)
    assert len(o.reference_startup_proofs({"reference_checks": reports}, current, PIN)) == 4


@pytest.mark.parametrize("failure", ["numerical", "updates", "wrong_owner", "source_pin", "argv", "missing_reference"])
def test_reference_startup_missing_changed_or_cross_job_receipts_block_native_dispatch(owned, monkeypatch, failure):
    current, payload, calls = reference_startup_fixture(owned, monkeypatch)
    reports = o.cpu_reference_startups(current, PIN, payload, o.time.monotonic() + 1000)
    if failure == "missing_reference": reports.pop()
    else:
        row = reports[0]
        key = "invocation" if failure == "argv" else "receipt"
        value = o.read(row[key]["path"])
        if failure == "numerical": value["numerical_setup_entered"] = True
        elif failure == "updates": value["optimizer_updates"] = 1
        elif failure == "wrong_owner": value["job_id"] = "99999999"
        elif failure == "source_pin": value["own_source_manifest_sha256"] = "b" * 64
        else: value["argv"][0] = "/unbound/python"
        row[key] = save(Path(row[key]["path"]), value)
    with pytest.raises(ValueError): o.reference_startup_proofs({"reference_checks": reports}, current, PIN)


def test_no_scientific_import_or_optimizer_entry_exists_in_controller():
    tree = ast.parse(PATH.read_bytes())
    top = {a.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for a in node.names}
    assert not top & {"numpy", "jax", "torch", "equinox", "mosaic"}
    assert o.DISPATCH_ARMS == ("protenix", "dbmol") and o.AUTOMATIC_ARMS == o.DISPATCH_ARMS
    assert o.SOURCE != o.BASE / "nonchem-endpoint-release-09/source"


def test_gpu_argv_removes_inherited_callback_visibility():
    from cuda_ownership import SUBMISSION_ENV
    command = o.argv("validate", PIN, "protenix", 20000)
    assert command.count("-env") == 1 and command[command.index("-env")+1] == SUBMISSION_ENV
