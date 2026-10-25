"""Read-only remote byte/JSON metadata for the new immutable pilot plans."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / 'results/hpc/20261025-chemistry-full-model-01/inputs-01'
PROGRAM = r'''
import hashlib,json,os
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
def bound(path):
 p=Path(path);assert p.is_file() and not p.is_symlink()
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
profiles={};proofs=[]
for key,rel,source,pin,count in [
 ('protenix','protenix-extended-03/random-native-0000','protenix-extended-release-03/source','da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562',643),
 ('boltz','boltz-lr0025-03/boltz2_chemistry-native-0000','boltz-lr0025-release-03/source','c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6',635)]:
 root=base/rel;source=base/source
 m=bound(source/'source_manifest.json');assert m['sha256']==pin
 assert len(json.loads((source/'source_manifest.json').read_bytes())['files'])==count
 invocation=root/'payload/driver-invocation.json';a=json.loads(invocation.read_bytes())['args']
 row={'source':{'path':str(source),'manifest_sha256':pin,'files':count},'driver':bound(a[1]),'inputs':{}}
 for name,flag in [('config','--config'),('cpu_certificate','--cpu-certificate'),('reference','--reference'),('pocket','--pocket')]:
  row['inputs'][name]=bound(a[a.index(flag)+1])
 if key=='boltz':
  row['inputs']['prepared_context']=bound(Path(a[a.index('--prepared')+1])/'context.pt')
  row['worker_python']=a[a.index('--worker-python')+1]
 row['config']=json.loads(Path(row['inputs']['config']['path']).read_bytes())
 row['original_driver_invocation']=bound(invocation)
 row['original_args']=a
 profiles[key]=row;proofs.append(bound(invocation))
parent=base/'chemistry-search-speed-02'
for rel in ['cpu/payload/assessment.json','cpu/payload/tests.xml','cpu/payload/benchmark/assessment.json','cpu/stdout.29809477.log',
            'gpu/payload/assessment.json','gpu/payload/tests.xml','gpu/payload/benchmark/assessment.json','gpu/stdout.29809554.log',
            'plan.json','initial-submissions.json']:
 proofs.append(bound(parent/rel))
ready=base/'restart-lrreset-04/validate-protenix/payload/study/assessment.json'
proofs.append(bound(ready))
print(json.dumps({'profiles':profiles,'preserved_proofs':proofs,'protenix_context_assessment':json.loads(ready.read_bytes())}))
'''


def main():
    OUT.mkdir(exist_ok=False)
    code = PROGRAM.encode()
    (OUT / 'program.py').write_bytes(code)
    args = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
            'dtu-hpc', 'bash -l -c ' + shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    with (OUT / 'raw.json').open('xb') as output, (OUT / 'stderr.txt').open('xb') as error:
        result = subprocess.run(args, input=code, stdout=output, stderr=error, timeout=120)
    record = {'returncode': result.returncode, 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'program_sha256': hashlib.sha256(code).hexdigest(),
              'stdout_sha256': hashlib.sha256((OUT / 'raw.json').read_bytes()).hexdigest()}
    (OUT / 'receipt.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))
    if result.returncode:
        print((OUT / 'stderr.txt').read_text(errors='replace')[-4000:])
    else:
        data = json.loads((OUT / 'raw.json').read_bytes())
        print(json.dumps({'profile_inputs': {k: v['inputs'] for k, v in data['profiles'].items()},
                          'context_assessment_keys': list(data['protenix_context_assessment'])}, indent=2))


if __name__ == '__main__':
    main()
