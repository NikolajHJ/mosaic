"""Stdlib-only preparation/staging/observation for DBMol precision qualification."""
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile

OWNER = Path(__file__).resolve().parents[1]
REPO = OWNER.parents[2]
TAG = '20261027-dbmol-endpoint-recovery-10'
RUN = OWNER / 'runs' / TAG
LOCAL = REPO / 'results/hpc' / TAG
BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
ROOT = BASE + '/dbmol-endpoint-recovery-10'
SOURCE = BASE + '/dbmol-endpoint-recovery-release-10/source'
COUNTS = {'tests.test_dbmol_reference': 12, 'tests.test_dbmol_endpoint_precision': 10,
          'tests.test_recovery_capacity': 10}


def raw(value): return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
def sha(value): return hashlib.sha256(value).hexdigest()


def connection(program, name, timeout=180):
    # Reuse only the existing stdlib SSH/receipt function, never its staging or
    # submission functions. Its local output is explicitly this independent run.
    import manage_protenix_precision as helper
    LOCAL.mkdir(parents=True, exist_ok=True); helper.LOCAL = LOCAL
    return helper.command(program, name, timeout)


def prepare():
    program = r'''
import hashlib,json
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
assert not (base/'dbmol-endpoint-recovery-10').exists() and not (base/'dbmol-endpoint-recovery-release-10').exists(),'Existing/unknown new outcome; inspect before staging'
def binding(p):
 p=Path(p);assert p.is_file() and not p.is_symlink()
 with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha}
old=base/'nonchem-endpoint-release-08/source';m=json.loads((old/'source_manifest.json').read_bytes())
assert binding(old/'source_manifest.json')['sha256']=='3ba7a6b30e6671076fee73910a06a622757734a43d30a589b9a89dd2b92c6877'
for name,digest in m['files'].items():assert binding(old/name)['sha256']==digest
oldplan=json.loads((old/'plan.json').read_bytes());arm=next(a for a in oldplan['arms'] if a['id']=='dbmol')
saved=base/'nonchem-endpoint-continuation-08/dbmol-segment-05000';study=saved/'payload/study'
assert json.loads((saved/'payload/assessment.json').read_bytes())['status']=='failed'
lsf=''.join((saved/'stdout.29830999.log').read_text().split());assert lsf.count('Subject:Job29830999:')==1 and 'Exitedwithexitcode1.' in lsf
protected={b['path']:b for b in m['preserved_proofs']}
for tree in (saved,base/'nonchem-endpoint-continuation-08/dbmol-after-segment-05000'):
 for p in tree.rglob('*'):
  assert not p.is_symlink() and (p.is_dir() or p.is_file())
  if p.is_file():protected[str(p)]=binding(p)
boundary=binding(study/'case-0/snapshots/step-2500.npz')
admission=binding(study/'terminal_gradient_checks-prospective/dbmol_affinity_loss/admission.json')
original=json.loads(Path(admission['path']).read_bytes());assert original['status']=='failed' and original['can_admit_new_child'] is False
row=next(d for d in original['directions'] if d['field']=='X' and d['row']==1)
assert row['source']==0 and row['target']==4 and row['pair_passed'] is True and row['finer_resolved_disagreements']==[.0001]
science=Path(arm['source']['path']);python='/work3/s234818/latentmol/env/boltz311/bin/python'
context='/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-01/prepare/payload/prepared/context.pt'
context_pin='a18a331269170aac3281b0a2ecad313aab81756c9b77e048d7e2cbaa4d8f1093'
assert binding(context)['sha256']==context_pin;protected[context]=binding(context)
checkpoint='/work3/s234818/latentmol/cache/boltz2/boltz2_aff.ckpt';structure='/work3/s234818/latentmol/cache/boltz2/boltz2_conf.ckpt'
for p in (checkpoint,structure):protected[p]=binding(p)
production=[python,'-u',str(science/'scripts/dbmol_worker.py'),'--context',context,'--context-sha256',context_pin,'--checkpoint',checkpoint,'--structure-checkpoint',structure]
reference=[python,'-u',str(base/'dbmol-endpoint-recovery-release-10/source/scripts/dbmol_reference.py'),'--context',context,'--context-sha256',context_pin,'--checkpoint',checkpoint,'--model-digest','395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16']
parents=m['parent_sources']+[{'path':str(old),'files':19,'manifest_sha256':binding(old/'source_manifest.json')['sha256']},
 {'path':str(base/'protenix-endpoint-fix-release-07/source'),'files':18,'manifest_sha256':'63db0a002c8aca2a282d50f9e7227d57b69ac6820b8ae0e06287ec95770dd8a3'},
 {'path':str(base/'protenix-endpoint-fix-release-08/source'),'files':20,'manifest_sha256':'7082a76a2da1849cfdf377735ebd7ccbc77e298395c7f2f154f18904a1db6856'},
 {'path':str(base/'chemistry-native-qualification-release-04/source'),'files':12,'manifest_sha256':'5d8f4ef0f8bb820e76d2e92442492f604393e79b58725fd01f9bb042b3114863'}]
parents=list({p['path']:p for p in parents}.values())
plan={'version':'dbmol-separate-affinity-reference-v1','root':str(base/'dbmol-endpoint-recovery-10'),
 'boundary':boundary,'original_admission':admission,'original_arm':arm,'science_source':arm['source'],
 'protected_inputs':list(protected.values()),'production_argv':production,'reference_argv':reference,
 'original_directions':{'X':[{'row':1,'source':0,'target':4}],'E_upper':[]},'global_step':7500,
 'target_global':10000,'rate':.0025,'original_results_changed':False,'optimizer_updates':0,
 'can_submit_continuation':False,'monitor_remains_paused':True}
print(json.dumps({'plan':plan,'parent_sources':parents}))
'''
    inputs = connection(program, 'input-metadata01', 180)
    members = {}
    names = ['README.md', 'scripts/README.md', 'tests/README.md',
             'scripts/dbmol_reference.py', 'scripts/dbmol_endpoint_precision.py',
             'scripts/dbmol_recovery_ops.py', 'scripts/dbmol_recovery_stage.lsf', 'scripts/recovery_capacity.py',
             'tests/test_dbmol_reference.py', 'tests/test_dbmol_endpoint_precision.py', 'tests/test_recovery_capacity.py']
    for name in names:
        content = (OWNER / name).read_bytes().replace(b'\r\n', b'\n')
        if name.endswith('.py'): ast.parse(content, filename=name)
        members[name] = content
    members['plan.json'] = raw(inputs['plan']); members['protocol.md'] = (RUN / 'README.md').read_bytes().replace(b'\r\n', b'\n')
    m = {'version': 'dbmol-separate-affinity-reference-source-v1', 'files': {n: sha(b) for n, b in members.items()},
         'parent_sources': inputs['parent_sources'], 'cpu_counts': COUNTS,
         'original_results_changed': False, 'optimization_settings_changed': False, 'optimizer_updates': 0}
    members['source_manifest.json'] = raw(m); folder = LOCAL / 'source01'; folder.mkdir(exist_ok=False)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name, content in members.items():
            target = folder / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content)
            info = tarfile.TarInfo(name); info.mode = 0o444; info.mtime = 0; info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    archive = buffer.getvalue(); (folder / 'source.tar.gz').write_bytes(archive)
    result = {'status': 'prepared_not_validated', 'files': len(m['files']), 'manifest_sha256': sha(members['source_manifest.json']),
              'archive_bytes': len(archive), 'archive_sha256': sha(archive), 'plan_sha256': sha(members['plan.json']), 'cpu_counts': COUNTS}
    for name, content in (('plan.json', members['plan.json']), ('source-manifest.json', members['source_manifest.json']),
                          ('preparation.json', raw(result))): (RUN / name).write_bytes(content)
    (folder / 'preparation.json').write_bytes(raw(result)); print(json.dumps(result, indent=2))


def stage():
    folder = LOCAL / 'source01'; prep = json.loads((folder / 'preparation.json').read_bytes()); archive = (folder / 'source.tar.gz').read_bytes()
    assert len(archive) == prep['archive_bytes'] and sha(archive) == prep['archive_sha256']
    program = 'PREP=' + repr(prep) + '\nARCHIVE=' + repr(base64.b64encode(archive).decode()) + '\n' + r'''
import base64,hashlib,io,json,sys,tarfile,subprocess
from pathlib import Path,PurePosixPath
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01');root=base/'dbmol-endpoint-recovery-10';source=base/'dbmol-endpoint-recovery-release-10/source'
assert not root.exists() and not source.parent.exists(),'Existing/unknown staging; inspect rather than repeat'
data=base64.b64decode(ARCHIVE);assert hashlib.sha256(data).hexdigest()==PREP['archive_sha256']
members={}
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 for item in archive:
  p=PurePosixPath(item.name);assert item.isfile() and not p.is_absolute() and '..' not in p.parts and '\\' not in item.name and item.name not in members
  members[item.name]=archive.extractfile(item).read();assert len(members[item.name])==item.size
manifest=json.loads(members['source_manifest.json']);assert hashlib.sha256(members['source_manifest.json']).hexdigest()==PREP['manifest_sha256']
assert set(members)==set(manifest['files'])|{'source_manifest.json'} and len(manifest['files'])==PREP['files']
for name,digest in manifest['files'].items():assert hashlib.sha256(members[name]).hexdigest()==digest
root.mkdir();source.mkdir(parents=True)
for name,content in members.items():
 p=source/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as stream:stream.write(content)
 p.chmod(0o444)
with (source.parent/'source.tar.gz').open('xb') as stream:stream.write(data)
(source.parent/'source.tar.gz').chmod(0o444)
for p in sorted((p for p in source.rglob('*') if p.is_dir()),reverse=True):p.chmod(0o555)
source.chmod(0o555);subprocess.run(['bash','-n',str(source/'scripts/dbmol_recovery_stage.lsf')],check=True)
sys.dont_write_bytecode=True;sys.path.insert(0,str(source/'scripts'));import dbmol_recovery_ops as op
m,plan=op.external(op.helpers(),PREP['manifest_sha256'])
preflight=root/'preflight';preflight.mkdir();import recovery_capacity;gate=recovery_capacity.gate(preflight,'preflight')
cpu=op.submit('cpu',PREP['manifest_sha256']);callback=op.submit('after-cpu',PREP['manifest_sha256'],'ended('+cpu['job_id']+')')
result={'status':'CPU_submitted_native_unvalidated','preparation':PREP,'cpu':cpu,'after_cpu':callback,'storage':gate}
op.helpers().write(root/'bootstrap-complete.json',result);print(json.dumps(result))
'''
    result = connection(program, 'stage01', 420); (RUN / 'submission.json').write_bytes(raw(result))
    print(json.dumps({'status': result['status'], 'cpu': result['cpu']['job_id'], 'callback': result['after_cpu']['job_id'],
                      'required_bytes': result['storage']['required_bytes'], 'available_bytes': result['storage']['available_bytes']}, indent=2))


def observe(name):
    program = 'ROOT=' + repr(ROOT) + '\n' + r'''
import json,base64,hashlib,subprocess,datetime
from pathlib import Path
root=Path(ROOT);files=[];jobs=json.loads((root/'jobs.json').read_bytes()) if (root/'jobs.json').is_file() else {}
for p in root.rglob('*'):
 if not p.is_file() or p.is_symlink() or p.stat().st_size>3000000 or p.suffix not in ('.json','.xml','.log'):continue
 b=p.read_bytes();files.append({'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()})
q=subprocess.run(['bjobs','-l',*jobs.values()],capture_output=True,text=True,timeout=40) if jobs else None
print(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':jobs,'files':files,
 'scheduler':None if q is None else {'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}}))
'''
    result = connection(program, name, 120)
    print(json.dumps({'observed_utc': result['observed_utc'], 'jobs': result['jobs'], 'assessments':
        [{'path': r['path'], 'status': json.loads(base64.b64decode(r['base64'])).get('status')}
         for r in result['files'] if r['path'].endswith('/payload/assessment.json')]}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('prepare', 'stage', 'observe'))
    p.add_argument('--name'); args = p.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'stage': stage()
    else:
        assert args.name and args.name.replace('-', '').isalnum(); observe(args.name)


if __name__ == '__main__': main()
