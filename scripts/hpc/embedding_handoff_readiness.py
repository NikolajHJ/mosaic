"""Allocated CPU readiness for a control-only source142 handoff repair.

All checks are standard-library byte/receipt work. The real inherited pilot
entrypoint parses its exact generated argv in a separate read-only process;
execution is intercepted at native_smoke before CUDA/model setup. No original
source, campaign directory, fitted tensor or calibration is written here.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


VERSION = "embedding-handoff-readiness-v2"
CONFIG_VERSION = "embedding-handoff-recovery-v1"
PARENT_CONFIG = "configs/epoch36_embedding_long.json"
CONTROLLER = "scripts/hpc/embedding_long_campaign.py"
ORIGINAL_CPU_TESTS = 450
PROFILE = "mkl_compatible_four_threads_v1"
ENVIRONMENT = dict(MKL_CBWR="COMPATIBLE", MKL_NUM_THREADS="4", OMP_NUM_THREADS="4")
SHA = re.compile(r"^[0-9a-f]{64}$")
CHECKPOINT_SHA = "c8d86551849c7f7e1de5218b7121cbecacb7f9dbc9ee11a5aea5972187a81551"
CALIBRATION_SHA = "1c436618bc6dc26dfb049908b11477afc5f2e7b291132db4577a99631f0c69f8"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    path = Path(path)
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("Unbounded readiness metadata")
    return json.loads(path.read_text(), parse_constant=lambda value:
                      (_ for _ in ()).throw(ValueError("Nonfinite metadata: " + value)))


def verify(asset):
    if (not isinstance(asset, dict) or set(asset) != {"path", "sha256"}
            or not isinstance(asset["path"], str) or not isinstance(asset["sha256"], str)
            or SHA.fullmatch(asset["sha256"]) is None or digest(asset["path"]) != asset["sha256"]):
        raise ValueError("Readiness asset authentication failed")
    return Path(asset["path"])


def asset(path):
    return dict(path=str(Path(path).resolve()), sha256=digest(path))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def junit(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    if not cases or any(list(row.iter(tag)) for row in cases for tag in ("failure", "error", "skipped")):
        raise ValueError("Readiness requires nonempty zero-failure/error/skip JUnit")
    if any(int(suite.get(key, "0")) != 0 for suite in root.iter("testsuite")
           for key in ("failures", "errors", "skipped")):
        raise ValueError("JUnit aggregate is not all passed")
    return dict(tests=len(cases), failures=0, errors=0, skipped=0)


def _identity(parent):
    value = {key: parent[key] for key in ("source_sha256", "manifest_sha256", "plan_sha256")}
    if any(not isinstance(v, str) or SHA.fullmatch(v) is None for v in value.values()):
        raise ValueError("Malformed original source/manifest/plan identity")
    return value


def _execution(execution, result, result_asset, identity, node):
    if (execution.get("identity") != identity or execution.get("node") != node
            or execution.get("status") != "completed" or execution.get("returncode") != 0
            or execution.get("error") is not None
            or execution.get("job_id") != result.get("job_id")
            or not re.fullmatch(r"[0-9]+", str(execution.get("job_id", "")))
            or execution.get("result_sha256") != result_asset["sha256"]):
        raise ValueError("Original allocated execution/result receipt differs: " + node)


def validate_parent(config):
    """Reauthenticate complete parent source and passed CPU/fit artifact bytes."""
    if config.get("version") != CONFIG_VERSION:
        raise ValueError("Unknown handoff recovery configuration")
    parent, artifacts = config["parent"], config["parent"]["artifacts"]
    identity = _identity(parent)
    release, campaign = Path(parent["release"]).resolve(), Path(parent["campaign"]).resolve()
    controller = release / CONTROLLER
    if digest(controller) != parent["controller_sha256"]:
        raise ValueError("Original controller changed")
    operator = load_module(Path(__file__).with_name("embedding_operator.py"), "_handoff_readiness_operator")
    operator.authenticate(release, identity["source_sha256"], identity["plan_sha256"], PARENT_CONFIG)
    if digest(release / "embedding-release-manifest.json") != identity["manifest_sha256"]:
        raise ValueError("Original manifest byte identity changed")
    original_config = read(release / PARENT_CONFIG)
    for value in artifacts.values():
        verify(value)
    required = {"cpu_result", "cpu_execution", "fit_result", "fit_execution", "checkpoint", "calibration", "fit_terminal"}
    if not required <= set(artifacts):
        raise ValueError("Missing original CPU/fit/terminal artifact bindings")
    cpu, fit = read(artifacts["cpu_result"]["path"]), read(artifacts["fit_result"]["path"])
    for name, result in (("cpu", cpu), ("fit", fit)):
        if (result.get("status") != "passed" or result.get("stage") != name
                or any(result.get(k) != v for k, v in identity.items())
                or result.get("existing_vae_continuation") != "untouched"):
            raise ValueError("Original stage is not exact-source passed: " + name)
        for value in result["artifacts"].values():
            path = verify(value).resolve()
            if not path.is_relative_to(campaign / name):
                raise ValueError("Original artifact escaped its exact stage directory")
        _execution(read(artifacts[name + "_execution"]["path"]), result,
                   artifacts[name + "_result"], identity, name)
    count = junit(verify(cpu["artifacts"]["junit"]))
    if (count["tests"] != ORIGINAL_CPU_TESTS or cpu.get("tests_passed") != ORIGINAL_CPU_TESTS
            or cpu.get("cpu_execution", {}).get("profile") != PROFILE
            or cpu["cpu_execution"].get("environment") != ENVIRONMENT
            or cpu["starts_readiness"].get("all_three_parent_starts_exact") is not True
            or cpu["starts_readiness"].get("frozen_vae_unchanged") is not True
            or cpu["artifacts"]["starts"]["sha256"] != original_config["precision_correction"]["parent_starts"]["sha256"]
            or cpu["frozen_identities"] != fit["frozen_identities"]):
        raise ValueError("Original 450-case CPU/profile/shared-start readiness changed")
    model = dict(checkpoint_sha256=CHECKPOINT_SHA, calibration_sha256=CALIBRATION_SHA)
    if (fit.get("model_identity") != model or fit.get("fit_completed") is not True
            or fit.get("calibration_passed") is not True
            or fit["artifacts"]["checkpoint"] != artifacts["checkpoint"]
            or fit["artifacts"]["calibration"] != artifacts["calibration"]):
        raise ValueError("Recovery would substitute or retrain the completed mapping")
    fitted = fit["fit"]
    bindings = {k: identity[k] for k in ("source_sha256", "plan_sha256")}
    if (fitted.get("status") != "completed_long_bridge_fit_not_native_validated"
            or fitted.get("bindings") != bindings or fitted.get("completed_steps") != 5000
            or fitted.get("requested_steps") != 5000 or fitted.get("dev_improved") is not True
            or fitted.get("frozen_vae_and_bank_unchanged") is not True
            or fitted.get("warm_start_unchanged") is not True or fitted.get("audit_consumed") is not True
            or fitted.get("checkpoint_sha256") != CHECKPOINT_SHA
            or fitted.get("calibration_sha256") != CALIBRATION_SHA
            or not math.isfinite(fitted["best_dev_loss"]) or not math.isfinite(fitted["initial_dev_loss"])
            or not fitted["best_dev_loss"] < fitted["initial_dev_loss"]):
        raise ValueError("Completed supervised-fit frozen/dev selection proof changed")
    calibration = read(artifacts["calibration"]["path"])
    training = read(verify(fit["artifacts"]["training_plan"]))
    data = read(verify(fit["artifacts"]["paired_data_manifest"]))
    if (calibration.get("version") != "epoch36-bridge-dev-q99-calibration-v1"
            or calibration.get("bindings") != bindings or training.get("bindings") != bindings
            or calibration.get("bridge_sha256") != CHECKPOINT_SHA
            or calibration.get("split") != "dev" or calibration.get("quantile") != .99
            or calibration.get("quantile_method") != "nearest_rank" or calibration.get("cap_floor") != 1e-8
            or calibration.get("audit_used_for_calibration") is not False
            or calibration.get("desired_score_labels") is not False
            or calibration.get("warm_start") != original_config["bridge"]
            or training.get("warm_start") != original_config["bridge"]
            or training.get("desired_score_labels") is not False
            or calibration.get("training_plan_sha256") != fit["artifacts"]["training_plan"]["sha256"]
            or fitted.get("training_plan_sha256") != fit["artifacts"]["training_plan"]["sha256"]
            or calibration.get("dataset_manifest_sha256") != fit["artifacts"]["paired_data_manifest"]["sha256"]
            or fitted.get("dataset_manifest_sha256") != calibration["dataset_manifest_sha256"]
            or calibration.get("vae_checkpoint") != fitted["vae_checkpoint"]
            or calibration.get("vae_state_sha256") != fitted["vae_state_sha256"]
            or calibration.get("bank_state_sha256") != fitted["bank_state_sha256"]
            or data["long_data"].get("benchmark_starts") != original_config["precision_correction"]["parent_starts"]):
        raise ValueError("Completed development calibration/model/data binding changed")
    names = sorted(row["key"] + "/mse" for row in calibration["bank_metadata"]["projections"])
    if calibration.get("metric_names") != names or set(calibration["caps"]) != set(names) or set(calibration["dev_errors"]) != set(names):
        raise ValueError("Calibration projection coverage is incomplete")
    for key in names:
        values = calibration["dev_errors"][key]
        if (not values or len(values) != calibration["dev_examples"]
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)
                or calibration["caps"][key] != max(1e-8, sorted(values)[math.ceil(.99 * len(values)) - 1])):
            raise ValueError("Development q99 cap changed; never recalibrate during recovery")
    terminal = Path(artifacts["fit_terminal"]["path"]).read_text()
    job = fit["job_id"]
    if (f"Subject: Job {job}: <{campaign.name}-fit> in cluster <dcc> Done" not in terminal
            or f"JOB={job} " not in terminal or f"SOURCE={release}" not in terminal
            or "Successfully completed." not in terminal):
        raise ValueError("Original successful exact-name LSF terminal proof is missing")
    emitted = []
    for line in terminal.splitlines():
        if line.startswith("{"):
            try:
                emitted.append(json.loads(line))
            except ValueError:
                pass
    execution = read(artifacts["fit_execution"]["path"])
    if execution not in emitted:
        raise ValueError("LSF output does not contain the exact passed allocated execution receipt")
    return dict(status="passed", original_identity=identity,
        original_controller_sha256=parent["controller_sha256"], original_cpu_junit=count,
        original_artifacts=artifacts, original_stage_artifacts={"cpu": cpu["artifacts"], "fit": fit["artifacts"]},
        model_identity=model, tensor_semantics="Exact tensor-file SHA authenticated; inherited CPU450/fit proof adopted, no unpickling",
        new_calibration=False, refit=False, optimization="not_run")


PROBE = r'''
import builtins,hashlib,importlib,json,os,sys
from pathlib import Path
from unittest.mock import patch
spec=json.loads(sys.argv[1]); command=spec['command']; root=Path(spec['release'])
sys.dont_write_bytecode=True
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
    raise RuntimeError('CPU dispatch probe requires explicitly hidden CUDA devices')
original_import=builtins.__import__
def checked_import(name,*args,**kwargs):
    if name.split('.')[0] in {'rdkit','boltz'}:
        raise RuntimeError('Forbidden native/chemistry import before dispatch stop: '+name)
    return original_import(name,*args,**kwargs)
builtins.__import__=checked_import
# Preserve the real package __init__: adapter/contracts legitimately import
# CPU torch. Hiding devices precedes every framework import in this subprocess.
module=importlib.import_module('latentmol.boltz.embedding_long_experiment')
expected=root/'src/latentmol/boltz/embedding_long_experiment.py'
if Path(module.__file__).resolve()!=expected.resolve() or hashlib.sha256(expected.read_bytes()).hexdigest()!=spec['module_sha256']:
    raise ValueError('Actual inherited worker module binding differs')
torch=sys.modules.get('torch')
if torch is None or torch.cuda.is_initialized():
    raise RuntimeError('Authentic eager CPU torch import must leave CUDA uninitialized')
cuda_initialized_before=torch.cuda.is_initialized()
class StopBeforeNative(RuntimeError): pass
calls=[]; writes=[]; made=[]
counts=dict(cuda_initializations=0,native_model_loads=0)
output=Path(spec['output'])
def stop(config,campaign,directory,deadline):
    if Path(campaign)!=Path(spec['campaign']) or Path(directory)!=output:
        raise ValueError('Actual parsed pilot dispatch paths differ')
    calls.append('native_smoke'); raise StopBeforeNative('intentional_before_cuda_or_models')
def mkdir(path,*args,**kwargs):
    if path!=output: raise RuntimeError('Unexpected readiness directory mutation: '+str(path))
    made.append(str(path))
def no_write(path,value): writes.append(dict(path=str(path),status=value.get('status')))
def no_cuda(*args,**kwargs):
    counts['cuda_initializations']+=1
    raise RuntimeError('Forbidden CUDA initialization/device setup in CPU dispatch probe')
def no_model(*args,**kwargs):
    counts['native_model_loads']+=1
    raise RuntimeError('Forbidden model/checkpoint setup in CPU dispatch probe')
def audit(event,args):
    if event=='open':
        mode=args[1];flags=args[2]
        if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):
            raise RuntimeError('Forbidden inherited probe file write')
    if event in {'subprocess.Popen','os.system','os.remove','os.rename','os.mkdir','os.rmdir'}:
        raise RuntimeError('Forbidden inherited probe process/filesystem mutation: '+event)
sys.addaudithook(audit)
from contextlib import ExitStack
with ExitStack() as guards:
    for name in ('_lazy_init','init','set_device','device','_exchange_device','_maybe_exchange_device',
                 'is_available','device_count','current_device','get_device_properties','get_device_name'):
        if hasattr(torch.cuda,name): guards.enter_context(patch.object(torch.cuda,name,no_cuda))
    if hasattr(torch._C,'_cuda_init'): guards.enter_context(patch.object(torch._C,'_cuda_init',no_cuda))
    guards.enter_context(patch.object(torch,'load',no_model))
    guards.enter_context(patch.object(torch.nn.Module,'__init__',no_model))
    for name in ('fit','trajectory','load_long_bridge','native_target','execute_native'):
        guards.enter_context(patch.object(module,name,no_model))
    guards.enter_context(patch.object(module,'native_smoke',stop))
    guards.enter_context(patch.object(module,'write',no_write))
    guards.enter_context(patch.object(Path,'mkdir',mkdir))
    guards.enter_context(patch.object(sys,'argv',[command[2],*command[3:]]))
    try: module.main()
    except StopBeforeNative: pass
    else: raise RuntimeError('Actual worker did not reach expected pilot dispatch')
if calls!=['native_smoke'] or made!=[str(output)] or len(writes)!=1 or writes[0]['path']!=str(output/'result.json'):
    raise RuntimeError('Unexpected original pilot parse/dispatch evidence')
cuda_initialized_after=torch.cuda.is_initialized()
cpu_modules=sorted({name.split('.')[0] for name in sys.modules if name.split('.')[0] in {'torch','numpy'}})
if cuda_initialized_after or any(counts.values()) or any(name.split('.')[0] in {'rdkit','boltz'} for name in sys.modules):
    raise RuntimeError('CPU dispatch probe reached forbidden CUDA/native work')
print('HANDOFF_CLI_PROBE_JSON='+json.dumps(dict(status='passed',argv=command,actual_worker_sha256=spec['module_sha256'],reached='native_smoke',stop='before_cuda_or_model_setup',parent_writes=0,job_submissions=0,updates=0,**counts,cpu_module_imports=cpu_modules,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_initialized_before=cuda_initialized_before,cuda_initialized_after=cuda_initialized_after)))
'''


def pilot_cli_probe(config, *, timeout_seconds=120):
    recovery = load_module(Path(__file__).with_name("embedding_handoff_recovery.py"), "_handoff_readiness_recovery")
    campaign = recovery.make_parent_campaign(config)
    command = campaign.worker_command("pilot", {})
    parent = config["parent"]
    if (command[:3] != [sys.executable, "-m", "latentmol.boltz.embedding_long_experiment"]
            or command[command.index("--stage") + 1] != "pilot"
            or command[command.index("--source-sha256") + 1] != parent["source_sha256"]
            or command[command.index("--plan-sha256") + 1] != parent["plan_sha256"]
            or command[command.index("--bridge-sha256") + 1] != CHECKPOINT_SHA
            or command[command.index("--calibration-sha256") + 1] != CALIBRATION_SHA):
        raise ValueError("Generated pilot CLI changed inherited model/source/plan")
    release = Path(parent["release"])
    manifest = read(release / "embedding-release-manifest.json")
    spec = dict(command=command, release=str(release), campaign=parent["campaign"],
        output=str(Path(parent["campaign"]) / "pilot"),
        module_sha256=manifest["members"]["src/latentmol/boltz/embedding_long_experiment.py"])
    environment = dict(os.environ, LATENTMOL_SOURCE_SHA256=parent["source_sha256"],
                       PYTHONPATH=str(release / "src"), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    process = subprocess.run([sys.executable, "-c", PROBE, json.dumps(spec)], cwd=release,
        env=environment, capture_output=True, text=True, timeout=timeout_seconds)
    lines = [line for line in process.stdout.splitlines() if line.startswith("HANDOFF_CLI_PROBE_JSON=")]
    if process.returncode or len(lines) != 1:
        raise RuntimeError("Actual inherited pilot entrypoint failed pre-model dispatch: " + process.stderr[-8000:])
    receipt = json.loads(lines[0].split("=", 1)[1])
    if (receipt.get("status") != "passed" or receipt.get("argv") != command
            or receipt.get("actual_worker_sha256") != spec["module_sha256"]
            or receipt.get("reached") != "native_smoke" or receipt.get("stop") != "before_cuda_or_model_setup"
            or any(type(receipt.get(key)) is not int or receipt[key] != 0 for key in
                   ("parent_writes", "job_submissions", "cuda_initializations", "native_model_loads", "updates"))
            or receipt.get("cuda_visible_devices") != ""
            or receipt.get("cuda_initialized_before") is not False or receipt.get("cuda_initialized_after") is not False
            or not isinstance(receipt.get("cpu_module_imports"), list)
            or any(not isinstance(name, str) for name in receipt["cpu_module_imports"])
            or receipt["cpu_module_imports"] != sorted(set(receipt["cpu_module_imports"]))
            or not {"torch"} <= set(receipt["cpu_module_imports"]) <= {"torch", "numpy"}):
        raise ValueError("Actual inherited pilot argv receipt differs")
    return receipt


def run(config, output, *, repair_source_sha256, repair_manifest_sha256, repair_plan_sha256, deadline=None):
    if not os.environ.get("LSB_JOBID") or os.environ.get("LATENTMOL_SOURCE_SHA256") != repair_source_sha256:
        raise RuntimeError("Repair readiness requires the exact allocated repair-source CPU job")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    adopted = validate_parent(config)
    tests = config["readiness"]["cpu_tests"]
    if (not isinstance(tests, list) or not tests or any(not isinstance(v, str)
            or not v.startswith("tests/test_embedding_handoff_") or not v.endswith(".py") or ".." in Path(v).parts for v in tests)):
        raise ValueError("Readiness test scope must be explicit handoff-only fixtures")
    remaining = 2400 if deadline is None else int(deadline - time.monotonic())
    if remaining < 150:
        raise TimeoutError("Readiness lacks bounded CPU test/dispatch headroom")
    command = [sys.executable, "-m", "pytest", "-q", *tests, "--junitxml", str(output / "cpu-tests.xml")]
    with (output / "cpu-tests.log").open("xb") as stream:
        process = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), stdout=stream, stderr=subprocess.STDOUT,
            timeout=remaining - 120)
    if process.returncode:
        raise RuntimeError("Allocated control-only readiness tests failed")
    counts = junit(output / "cpu-tests.xml")
    cli = pilot_cli_probe(config, timeout_seconds=120)
    write(output / "pilot-cli.json", cli)
    if validate_parent(config) != adopted:
        raise ValueError("Original parent inputs changed during readiness")
    receipt = dict(version=VERSION, status="passed", source_sha256=repair_source_sha256,
        manifest_sha256=repair_manifest_sha256, plan_sha256=repair_plan_sha256,
        original_identity=adopted["original_identity"],
        original_controller_sha256=adopted["original_controller_sha256"],
        job_id=os.environ["LSB_JOBID"], junit={**asset(output / "cpu-tests.xml"), **counts},
        parent=adopted, actual_pilot_cli=cli, actual_pilot_cli_asset=asset(output / "pilot-cli.json"), calibration_changed=False,
        model_training="not_run", optimization="not_run", existing_vae_continuation="untouched")
    write(output / "readiness.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=2400)
    args = parser.parse_args()
    if digest(args.config) != args.plan_sha256 or args.seconds < 1:
        raise ValueError("Repair readiness plan binding differs")
    release = Path(__file__).resolve().parents[2]
    if args.config.resolve() != release / "configs/embedding_handoff_recovery.json":
        raise ValueError("Readiness must use this repair release's exact configuration")
    operator = load_module(Path(__file__).with_name("embedding_operator.py"), "_handoff_repair_operator")
    operator.authenticate(release, args.source_sha256, args.plan_sha256,
                          "configs/embedding_handoff_recovery.json")
    if digest(release / "embedding-release-manifest.json") != args.manifest_sha256:
        raise ValueError("Repair release manifest identity differs")
    result = run(read(args.config), args.output, repair_source_sha256=args.source_sha256,
        repair_manifest_sha256=args.manifest_sha256, repair_plan_sha256=args.plan_sha256,
        deadline=time.monotonic() + args.seconds)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
