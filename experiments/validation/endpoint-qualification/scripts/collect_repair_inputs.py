"""One read-only connection for blocked-run proofs and immutable runtime source.

No model/optimizer execution, original writes, deletion or submission is possible.
Use a fresh label; a failed/partial attempt is preserved and never overwritten.
"""
import argparse
import ast
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261026-blocked-runs-repair-01'

PROGRAM=r'''

from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,re,stat,subprocess
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
record={'captured_utc':datetime.now(timezone.utc).isoformat(),'read_only':True,'files':[],'sources':{},'missing':[]};seen=set();total=0;ids=set()
def collect(p):
 global total
 if str(p) in seen:return
 seen.add(str(p))
 if not p.exists():record['missing'].append(str(p));return
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink() and s.st_size<=10*1024**2
 raw=p.read_bytes();assert len(raw)==s.st_size;total+=len(raw);assert total<48*1024**2
 record['files'].append({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()})
for rootname,srcname,pin in [('nonchem-endpoint-continuation-06','nonchem-endpoint-release-06/source','1519d5761069f73fb6659bc1740cb04c155a04bc3218f143a3f28a73e65a1f69'),('protenix-endpoint-fix-03','protenix-endpoint-fix-release-03/source','a701b12bd1e69f5aa7169ad5339f2456c3d728b855257b4270437d37f9da68f3'),('chemistry-production-03','chemistry-production-release-03/source','c73796b041d8432ef1634e8c56920006170a094d3bdc74867c08f3b5415126d7')]:
 src=base/srcname;root=base/rootname;raw=(src/'source_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==pin
 manifest=json.loads(raw)
 for rel,digest in manifest['files'].items():
  p=src/rel;assert not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==digest
 record['sources'][str(src)]={'manifest_sha256':pin,'files_rehashed':len(manifest['files'])};collect(src/'source_manifest.json');collect(src/'plan.json');collect(root/'jobs.json')
 jobs=json.loads((root/'jobs.json').read_bytes());ids.update(jobs.values())
 for name,job in jobs.items():
  for p in (root/name).glob('*.json'):collect(p)
  for p in (root/name).glob('*.log'):collect(p)
for arm,label in [('boltz','boltz-validate-10000'),('dbmol','dbmol-validate-05000')]:
 study=base/'nonchem-endpoint-continuation-06'/label/'payload/study'
 for p in (study/'native_gradient_checks-prospective').rglob('admission.json'):collect(p)
 for p in (study/'early_pilot/combined_initial_checks').glob('*.json'):collect(p)
 for p in (study/'native_gradient_checks').glob('report.json'):collect(p)
collect(base/'protenix-endpoint-fix-03/gpu/payload/fp64-final-preservation.json')
path=Path('/work3/s234818/mosaic/envs/protenix-tiny/lib/python3.12/site-packages/protenix/protenij.py')
assert hashlib.sha256(path.read_bytes()).hexdigest()=='37f337839034afd2a814c1ed6b76867b67918f73dd56cdd365e67b8fed882a6d';collect(path)
for name in ('features.py','protenix.py'):
 collect(base/'protenix-extended-release-03/source/src/mosaic/small_molecule'/name)
for rel in ('scripts/full_model_pilot.py','scripts/gpu_local_bridge.py'):
 collect(base/'chemistry-full-model-release-02/source'/rel)
r=subprocess.run(['bjobs','-l',*sorted(ids,key=int)],capture_output=True,text=True,timeout=45)
record['scheduler']={'job_ids':sorted(ids,key=int),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};record['compact_bytes']=total
print(json.dumps(record))

'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',required=True);args=parser.parse_args()
    if REPO.name!='mosaic' or re.fullmatch('[a-z0-9][a-z0-9-]{0,60}',args.label) is None:
        raise ValueError('Authorized repository and fresh safe label required')
    ast.parse(PROGRAM)
    out=LOCAL/args.label;out.mkdir(parents=True,exist_ok=False)
    (out/'program.py').write_text(PROGRAM,encoding='utf8',newline='\n')
    command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1',
             'dtu-hpc','bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    receipt={'status':'unknown_readonly_outcome','read_only':True,
             'started_utc':datetime.now(timezone.utc).isoformat(),
             'program_sha256':hashlib.sha256(PROGRAM.encode()).hexdigest()}
    try:
        with (out/'raw.json').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
            process=subprocess.run(command,input=PROGRAM.encode(),stdout=stdout,stderr=stderr,timeout=100)
        receipt['returncode']=process.returncode
        if process.returncode:raise RuntimeError('Single read-only SSH connection failed; no retry')
        packet=json.loads((out/'raw.json').read_bytes());proofs=[]
        for row in packet['files']:
            raw=base64.b64decode(row['base64'],validate=True)
            if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
                raise ValueError('Delivered input bytes changed')
            proofs.append({key:row[key] for key in ('path','bytes','sha256')})
        (out/'verification.json').write_text(json.dumps({'status':'downloaded_metadata_bytes_verified',
            'files':proofs,'numerical_evaluations':0},indent=2)+'\n',encoding='utf8',newline='\n')
        receipt.update(status='read_only_input_observation_completed',captured_utc=packet['captured_utc'],
                       files=len(proofs))
    except BaseException as error:
        receipt.update(status='failed_or_interrupted_readonly_observation',error=repr(error));raise
    finally:
        receipt['ended_utc']=datetime.now(timezone.utc).isoformat()
        if (out/'raw.json').exists():receipt['raw_sha256']=hashlib.sha256((out/'raw.json').read_bytes()).hexdigest()
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
