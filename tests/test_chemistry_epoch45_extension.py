"""Allocated metadata/tamper contracts; fake evidence is not a native result."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from latentmol import chemistry_epoch45_extension as extension
from latentmol.chemistry_valid_reference_training_policy import POLICY as ORIGINAL


def plan():
    return dict(epoch45_extension=deepcopy(extension.POLICY), valid_reference_continuation=deepcopy(ORIGINAL))


def test_original_scope_is_absent_without_new_marker_and_original_markers_are_not_mutated():
    p = dict(valid_reference_continuation=deepcopy(ORIGINAL))
    before = deepcopy(p)
    assert extension.validate_extension(p) is None and p == before
    p = plan(); before = deepcopy(p)
    assert extension.validate_extension(p) == dict(version=extension.VERSION, start_epoch=40,
        end_epoch=45, purpose="validated_reference_training")
    assert p == before and p["valid_reference_continuation"]["end_epoch"] == 40


@pytest.mark.parametrize("field,value", [("start_epoch", 39), ("end_epoch", 46), ("end_epoch", True),
    ("checkpoint_sha256", "a"*64), ("checkpoint", "/tmp/epoch40.pt"), ("updates", 263839),
    ("source_plan_sha256", "b"*64), ("completion_sha256", "c"*64), ("branch_sha256", "d"*64),
    ("save_every_epoch", False), ("version", "unknown")])
def test_extension_is_exact_and_rejects_changed_scope_or_source(field, value):
    p = plan(); p["epoch45_extension"][field] = value
    with pytest.raises(ValueError): extension.validate_extension(p)


@pytest.mark.parametrize("change", [lambda p:p["epoch45_extension"].update(extra=1),
    lambda p:p["epoch45_extension"]["generation"].update(seeds=[904017]),
    lambda p:p["epoch45_extension"]["generation"].update(prior_count=0),
    lambda p:p["valid_reference_continuation"].update(end_epoch=45),
    lambda p:p.pop("valid_reference_continuation")])
def test_monitoring_and_original_authorization_remain_closed(change):
    p = plan(); change(p)
    with pytest.raises(ValueError): extension.validate_extension(p)


def test_operational_initial_checkpoint_is40_without_replacing_original36_reference():
    p = plan(); p["starting_checkpoint"] = dict(epoch=36, sha256="original36")
    assert extension.initial_checkpoint(p) == dict(epoch=40, updates=263840,
        path=extension.CHECKPOINT, sha256=extension.CHECKPOINT_SHA)
    assert p["starting_checkpoint"]["epoch"] == 36
    assert extension.initial_checkpoint({"starting_checkpoint": {"epoch":36}}) == {"epoch":36}


def test_parent_algorithm_is_authenticated_old_package_not_new_metadata_package(tmp_path, monkeypatch):
    package = tmp_path / "src/latentmol"; package.mkdir(parents=True)
    (package / "model.py").write_text("MODEL = 1\n")
    old = dict(release=str(tmp_path), _plan_sha256=extension.SOURCE_PLAN_SHA)
    monkeypatch.setattr(extension, "authenticate_source", lambda p:(old, {}, {}, []))
    current = {"sha256":"new metadata identity"}
    identity, algorithm = extension.resume_identity(plan(), extension.CHECKPOINT, extension.CHECKPOINT_SHA, current)
    assert identity == old and algorithm != current
    expected = {"model.py": hashlib.sha256((package / "model.py").read_bytes()).hexdigest()}
    assert algorithm == dict(files=expected, sha256=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest())
    with pytest.raises(ValueError): extension.resume_identity(plan(), extension.CHECKPOINT, "a"*64, current)
    p=plan()
    assert extension.resume_identity(p, "/new/epoch41.pt", "b"*64, current) == (p,current)


def _source_pair(tmp_path, before, after, name="src/latentmol/chemistry_continuation.py"):
    old, new = tmp_path / "old", tmp_path / "new"
    for root, source in ((old,before),(new,after)):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
    return old,new,{name:"a"},{name:"b"}


def test_source_delta_permits_named_metadata_boundary_only(tmp_path):
    old="CONSTANT=1\ndef setup():\n return 1\ndef chemistry_update():\n return 2\n"
    assert extension.compare_sources(*_source_pair(tmp_path,old,old.replace("return 1","return 3")))


@pytest.mark.parametrize("change", [lambda s:s.replace("CONSTANT=1","CONSTANT=2"),
    lambda s:s.replace("return 2","return 4"), lambda s:s+"\ndef hidden_reward():\n return 1\n"])
def test_source_delta_rejects_changed_loss_or_numerical_constant(tmp_path,change):
    old="CONSTANT=1\ndef setup():\n return 1\ndef chemistry_update():\n return 2\n"
    with pytest.raises(ValueError): extension.compare_sources(*_source_pair(tmp_path,old,change(old)))


def test_source_delta_rejects_deleted_files_configuration_and_unknown_new_code(tmp_path):
    for old,new in (({"model.py":"a"},{}),({"configs/loss.yaml":"a"},{"configs/loss.yaml":"b"}),
                    ({},{"src/latentmol/secret_reward.py":"a"})):
        with pytest.raises(ValueError): extension.compare_sources(tmp_path,tmp_path,old,new)


def test_worker_preserves_generated_argv_and_uses_single_wrapped_gpu_allocation(monkeypatch):
    import sys
    scripts=Path(__file__).resolve().parents[1]/"scripts/hpc"
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location("epoch45_controller_fixture",scripts/"chemistry_epoch45_campaign.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    c=object.__new__(module.Epoch45Campaign);c.path=Path("/plan.json");c.sha="a"*64
    checked=[]
    monkeypatch.setattr(c,"check_launch",lambda command:checked.append(list(command)))
    monkeypatch.setattr(module.Campaign,"worker",lambda self,command,label:command)
    command=["python","-m","latentmol.chemistry_continuation","train","--segment","1",
             "--resume",extension.CHECKPOINT,"--resume-sha256",extension.CHECKPOINT_SHA]
    wrapped=c.worker(command,"first")
    assert checked == [command] and wrapped[0:2] == ["python","scripts/hpc/chemistry_epoch45_worker.py"]
    assert wrapped[wrapped.index("--")+1:] == command


def test_short_policy_admits45_only_with_exact_extension():
    from test_chemistry_valid_reference_training_policy import training_plan
    from latentmol.chemistry_short_continuation import continuation_end_epoch, expected_stop_reason
    p=training_plan()
    assert continuation_end_epoch(p) == 40
    p["epoch45_extension"]=deepcopy(extension.POLICY)
    assert continuation_end_epoch(p) == 45 and expected_stop_reason(p,45) == "bounded_horizon"
    assert expected_stop_reason(p,44) == "wall_budget_epoch_boundary"
    with pytest.raises(ValueError): expected_stop_reason(p,46)


@pytest.mark.parametrize("failed_phase", ["none", "baseline", "trainer", "observe"])
def test_fresh_coordinator_stays_tensor_free_and_stops_before_later_children(tmp_path,failed_phase):
    """Metadata control-flow test in a fresh CPU process, even in CUDA pytest.

    Phase process launches are replaced by deterministic file-producing stubs;
    the real observer/phase functions have separate allocated tiny-model tests.
    The real allocation additionally launches two genuine CUDA ownership probes
    only AFTER the CUDA pytest parent has exited, avoiding exclusive-GPU
    contention introduced by the test process itself.
    """
    import subprocess
    import sys
    root=Path(__file__).resolve().parents[1]
    script=r'''
import hashlib, importlib.abc, importlib.util, json, os, sys
from pathlib import Path
from types import SimpleNamespace
root, output, failed = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path[:0] = [str(root/'src'), str(root/'scripts/hpc')]
class NoTensorImports(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('torch','numpy','rdkit'):
   raise AssertionError('Coordinator imported a scientific module: '+fullname)
sys.meta_path.insert(0, NoTensorImports())
spec=importlib.util.spec_from_file_location('fresh_epoch45_worker',root/'scripts/hpc/chemistry_epoch45_worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
from latentmol.experiments import file_sha256
plan=dict(campaign=str(output),protocol=dict(segment_wall_seconds=23*3600),
 runtime_policy_amendment=dict(version='chemistry-epoch-fit-amendment-v1',factor=1.25,reserve_seconds=1800))
trace=[];deadlines={}
class Campaign:
 def cuda_check(self):trace.append('cuda-check')
worker._metadata=lambda args:(Campaign(),plan)
os.environ['LSB_JOBID']='12345'
output.mkdir(parents=True,exist_ok=True)
(output/'stability_gate.json').write_text(json.dumps(dict(conservative_epoch_seconds=2000.)))
args=SimpleNamespace(plan=output/'plan.json',plan_sha256='a'*64,segment=1,
 command=['--','python','-m','latentmol.chemistry_continuation','train','--segment','1'])
def fake_child(command,root,segment,label,deadline):
 phase=command[command.index('--phase')+1] if '--phase' in command else 'trainer'
 trace.append(label);deadlines[label]=deadline
 if phase==failed:raise RuntimeError('injected '+failed+' phase failure')
 if phase=='ownership-probe':
  index=int(command[command.index('--probe-index')+1]);pid=200000+index
  path=output/'cuda-ownership/segment-001'/f'probe-{index:03d}.json';path.parent.mkdir(parents=True,exist_ok=True)
  path.write_text(json.dumps(dict(status='passed',process_policy=worker.PROCESS_POLICY,
   plan_sha256=args.plan_sha256,job_id=os.environ['LSB_JOBID'],child_pid=pid,
   coordinator_pid=os.getpid(),probe_index=index,cuda_initialized=True)))
  return dict(child_pid=pid)
 if phase=='baseline':
  (output/'epoch40-handoff.json').write_text(json.dumps(dict(status='passed',plan_sha256=args.plan_sha256)))
 elif phase=='trainer':
  assert command==args.command[1:]
  run=output/'run';run.mkdir()
  (run/'metrics.jsonl').write_text(json.dumps(dict(epoch=41,updates=270436,
   checkpoint=str(run/'epoch_0041.pt'),checkpoint_sha256='b'*64))+'\n')
  (output/'segment-001-completion.json').write_text(json.dumps(dict(completed_epoch=41,run_dir=str(run))))
 elif phase=='observe':
  path=output/'generation/epoch0041/metrics.json';path.parent.mkdir(parents=True,exist_ok=True)
  path.write_text(json.dumps(dict(epoch=41)))
  receipt=output/'observation-receipts/epoch0041.json';receipt.parent.mkdir(parents=True,exist_ok=True)
  receipt.write_text(json.dumps(dict(path=str(path),sha256=file_sha256(path))))
 return dict(child_pid=200100)
worker._run_child=fake_child
try:
 worker.run(args)
 assert failed=='none'
except RuntimeError as error:
 assert failed!='none' and str(error)=='injected '+failed+' phase failure'
assert not {'torch','numpy','rdkit'} & set(sys.modules)
base=['cuda-check','ownership-probe-001','ownership-probe-002','baseline','trainer','observe-epoch0041']
expected=base if failed=='none' else base[:base.index('observe-epoch0041' if failed=='observe' else failed)+1]
assert trace==expected,(trace,expected)
if failed=='none':
 assert deadlines['trainer']==deadlines['baseline']-1800
 assert deadlines['observe-epoch0041']==deadlines['baseline']
 saved=json.loads((output/'generation-segment-001.json').read_text())
 assert saved['status']=='passed_observation_integrity' and saved['observed_epochs']==[41]
else:
 assert not (output/'generation-segment-001.json').exists()
print(json.dumps(dict(trace=trace,scientific_imports=[])))
'''
    result=subprocess.run([sys.executable,"-c",script,str(root),str(tmp_path/"campaign"),failed_phase],
                          capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    record=json.loads(result.stdout.splitlines()[-1])
    assert record["scientific_imports"]==[]
