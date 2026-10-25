"""One compact read-only query and byte-preserving compact evidence transfer."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess

REPO = Path(__file__).resolve().parents[4]
LOCAL = REPO / 'results/hpc/20261025-chemistry-full-model-01'
PROGRAM = r'''
from datetime import datetime,timezone
import base64,hashlib,json,os,stat,subprocess
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=base/'chemistry-full-model-01';source=base/'chemistry-full-model-release-01/source'
assert root.is_dir() and source.is_dir() and not root.is_symlink() and not source.is_symlink()
jobs=json.loads((root/'jobs.json').read_bytes()); assert set(jobs)<=set(['cpu','after-cpu','protenix','boltz'])
q=subprocess.run(['bjobs','-l',*list(jobs.values())],capture_output=True,text=True,timeout=30)
files=[];total=0
for directory,dirs,names in os.walk(root,followlinks=False):
 dirs[:]=[name for name in dirs if name not in ['tmp','cache','cuda-cache','pytest-cache']]
 for name in names:
  path=Path(directory)/name
  if path.suffix not in ['.json','.xml','.log','.txt']:continue
  if not stat.S_ISREG(path.lstat().st_mode):continue
  raw=path.read_bytes();total+=len(raw);assert total<64*1024**2,'Compact evidence cap'
  files.append({'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                'base64':base64.b64encode(raw).decode()})
for relative in ['source_manifest.json','plans/protenix.json','plans/boltz.json']:
 raw=(source/relative).read_bytes();files.append({'path':'_source/'+relative,'bytes':len(raw),
 'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()})
print(json.dumps({'captured_utc':datetime.now(timezone.utc).isoformat(),'root':str(root),'source':str(source),
 'jobs':jobs,'scheduler':{'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr},'files':files,
 'scope':'Compact JSON/XML/LSF/log bytes only; original native arrays are authenticated remote references, not downloaded'}))
'''


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('name')
    args = parser.parse_args()
    assert args.name.isalnum() or (args.name.replace('-', '').isalnum() and '/' not in args.name)
    out = LOCAL / ('observation-' + args.name); out.mkdir(exist_ok=False)
    program = PROGRAM.encode(); (out/'program.py').write_bytes(program)
    command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
               'dtu-hpc', 'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    with (out/'raw.json').open('xb') as output, (out/'stderr.txt').open('xb') as error:
        result = subprocess.run(command, input=program, stdout=output, stderr=error, timeout=120)
    receipt = {'returncode': result.returncode, 'completed_utc': datetime.now(timezone.utc).isoformat(),
               'program_sha256': hashlib.sha256(program).hexdigest(),
               'raw_sha256': hashlib.sha256((out/'raw.json').read_bytes()).hexdigest()}
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))
    if result.returncode:
        print((out/'stderr.txt').read_text(errors='replace')[-4000:]); raise SystemExit(result.returncode)
    data = json.loads((out/'raw.json').read_bytes())
    seen = set(); verified = []
    for row in data['files']:
        relative = PurePosixPath(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts and '\\' not in row['path']
        assert row['path'] not in seen; seen.add(row['path'])
        raw = base64.b64decode(row['base64'], validate=True)
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        path = out/'files'/Path(*relative.parts); path.parent.mkdir(exist_ok=True, parents=True)
        with path.open('xb') as stream: stream.write(raw)
        verified.append({key: row[key] for key in ['path','bytes','sha256']})
    (out/'byte-verification.json').write_text(json.dumps({'status':'passed_compact_byte_transfer_only',
                   'files': verified, 'raw_sha256': receipt['raw_sha256'], 'scientific_pass_claim':False}, indent=2)+'\n')
    print(json.dumps({'jobs':data['jobs'],'files':len(verified),'captured_utc':data['captured_utc']}))
    print(data['scheduler']['stdout'][-10000:]); print(data['scheduler']['stderr'])


if __name__ == '__main__':
    main()
