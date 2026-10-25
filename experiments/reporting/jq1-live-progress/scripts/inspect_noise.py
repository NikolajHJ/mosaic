"""One additional read-only original-noise source/config/proof inspection."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REPO = Path(__file__).resolve().parents[4]
PROGRAM = r'''
import hashlib,json,sys
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
sys.path.insert(0,str(base/'nonchem-stride20-startup-release-03/source'))
import selected_contract as c
def b(path):
 path=Path(path);assert path.is_file() and not path.is_symlink()
 h=hashlib.sha256(path.read_bytes()).hexdigest()
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':h}
source=c.NOISE_SOURCE
config=source/c.NOISE_STUDY/'runs/20261023-protenix-noisy-01/config.json'
man=json.loads((source/'source_manifest.json').read_bytes())
proofs=[]
for phase in ['noise-gpu']:
 parent=base/'nonchem-stride20-02'/phase
 if parent.exists():
  for path in parent.iterdir():
   if path.is_file() and (path.name=='submission.json' or path.name.startswith(('stdout.','stderr.'))):
    proofs.append({'binding':b(path),'text':path.read_text() if path.stat().st_size<200000 else None})
for parent in [base/'nonchem-stride20-startup-release-03/source',base/'nonchem-stride20-release-02/source']:
 proofs.append({'binding':b(parent/'source_manifest.json')})
print(json.dumps({'source':{'path':str(source),'manifest_sha256':b(source/'source_manifest.json')['sha256'],'files':len(man['files'])},'config':b(config),'proofs':proofs,'mutated':False}))
'''


def main():
    root = REPO / 'results/hpc/20261025-jq1-live-report-01/inputs-02'
    root.mkdir(parents=True, exist_ok=False)
    (root/'program.py').write_text(PROGRAM, encoding='utf8', newline='\n')
    command = ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
               'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    receipt = {'status':'unknown','program_sha256':hashlib.sha256(PROGRAM.encode()).hexdigest()}
    try:
        with (root/'raw.json').open('xb') as out, (root/'stderr.txt').open('xb') as err:
            run=subprocess.run(command,input=PROGRAM.encode(),stdout=out,stderr=err,timeout=90)
        receipt.update(status='returned',returncode=run.returncode)
        assert run.returncode==0
    finally:
        receipt.update(captured_utc=datetime.now(timezone.utc).isoformat(),raw_sha256=hashlib.sha256((root/'raw.json').read_bytes()).hexdigest())
        (root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    value=json.loads((root/'raw.json').read_bytes())
    print(json.dumps({k:value[k] for k in ('source','config')},indent=2))


if __name__=='__main__': main()
