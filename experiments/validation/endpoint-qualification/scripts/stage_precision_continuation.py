"""One-shot metadata staging of the authenticated two-arm continuation capsule.

This command submits allocated CPU admission plus its conditional controller.
It performs no numerical work and cannot bypass a failed/unknown prerequisite.
"""
import base64
import importlib.util
import json
from pathlib import Path


OWNER = Path(__file__).resolve().parents[1]
REPO = OWNER.parents[2]
TAG = "20261028-precision-continuation-21"
RUN = OWNER / "runs" / TAG
LOCAL = REPO / "results/hpc" / TAG


def main():
    spec = importlib.util.spec_from_file_location("_precision14_transport", OWNER / "scripts/manage_protenix_precision.py")
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    helper.LOCAL = LOCAL
    prepared = json.loads((LOCAL / "source01/preparation.json").read_bytes())
    archive = (LOCAL / "source01/source.tar.gz").read_bytes()
    assert len(archive) == prepared["archive_bytes"] and helper.sha(archive) == prepared["archive_sha256"]
    program = "PREP=" + repr(prepared) + "\nARCHIVE=" + repr(base64.b64encode(archive).decode()) + "\n" + r'''
import base64,hashlib,io,json,os,sys,tarfile,types,subprocess
from pathlib import Path,PurePosixPath
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=base/'nonchem-precision-continuation-21';source=base/'nonchem-precision-release-21/source'
assert not root.exists() and not source.parent.exists(),'Existing/unknown attempt: inspect actual receipts, never repeat staging'
data=base64.b64decode(ARCHIVE,validate=True)
assert len(data)==PREP['archive_bytes'] and hashlib.sha256(data).hexdigest()==PREP['archive_sha256']
members={}
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 for item in archive.getmembers():
  path=PurePosixPath(item.name)
  assert item.isfile() and not path.is_absolute() and '..' not in path.parts and '\\' not in item.name and item.name not in members
  members[item.name]=archive.extractfile(item).read();assert len(members[item.name])==item.size
manifest=json.loads(members['source_manifest.json']);plan=json.loads(members['plan.json'])
pin=hashlib.sha256(members['source_manifest.json']).hexdigest()
assert pin==PREP['manifest_sha256'] and hashlib.sha256(members['plan.json']).hexdigest()==PREP['plan_sha256']
assert set(members)==set(manifest['files'])|{'source_manifest.json'} and len(manifest['files'])==PREP['files']
assert all(hashlib.sha256(members[name]).hexdigest()==digest for name,digest in manifest['files'].items())
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
# Unsealed candidate metadata is checked against actual immutable inputs BEFORE
# any new remote path is created. This executes only the stdlib orchestrator.
op=types.ModuleType('_precision14_candidate_metadata');sys.modules[op.__name__]=op
op.__file__=str(source/'scripts/precision_continuation_ops.py')
exec(compile(members['scripts/precision_continuation_ops.py'],op.__file__,'exec'),op.__dict__)
assert op.ROOT==root and op.SOURCE==source
adoption=op.external(manifest,plan)
adapter=types.ModuleType('_precision14_candidate_adapter');sys.modules[adapter.__name__]=adapter
exec(compile(members['scripts/precision_continuation_driver.py'],str(source/'scripts/precision_continuation_driver.py'),'exec'),adapter.__dict__)
adapter.validate_adapter_plan(plan)
root.mkdir();source.mkdir(parents=True)
for name,content in members.items():
 target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as stream:stream.write(content)
 target.chmod(0o444)
with (source.parent/'source.tar.gz').open('xb') as stream:stream.write(data)
(source.parent/'source.tar.gz').chmod(0o444)
for directory in sorted((p for p in source.rglob('*') if p.is_dir()),reverse=True):directory.chmod(0o555)
source.chmod(0o555)
syntax=subprocess.run(['bash','-n',str(source/'scripts/precision_continuation_stage.lsf')],capture_output=True,text=True)
assert syntax.returncode==0,syntax.stderr
sys.path.insert(0,str(source/'scripts'))
assert op.verify(source,pin)==manifest
preflight=root/'preflight';preflight.mkdir()
gate=op.storage(preflight,'preflight')
cpu=op.submit('cpu',pin);callback=op.submit('after-cpu',pin,dependency='ended('+cpu['job_id']+')')
result={'status':'allocated_cpu_submitted_native_and_optimizer_not_established','preparation':PREP,
 'cpu':cpu,'after_cpu':callback,'storage':gate,'qualified_adoption':adoption,
 'optimization_settings_changed':False,'original_results_changed':False,'monitor_remains_paused':True}
op.write(root/'bootstrap-complete.json',result)
print(json.dumps(result,allow_nan=False))
'''
    result = helper.command(program, "stage01", 480)
    with (RUN / "submission.json").open("xb") as stream: stream.write(helper.raw(result))
    print(json.dumps({"status": result["status"], "cpu": result["cpu"]["job_id"],
                      "after_cpu": result["after_cpu"]["job_id"],
                      "storage_required": result["storage"]["required_bytes"],
                      "storage_available": result["storage"]["available_bytes"]}, indent=2))


if __name__ == "__main__": main()
