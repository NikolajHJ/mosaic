"""One read-only metadata query for the exact prior generated-command inputs."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REPO=Path(__file__).resolve().parents[4]
OUT=REPO/'results/hpc/20261025-continuation-admission-01/arguments'
EVIDENCE=REPO/'results/hpc/20261025-nonchem-stride20-startup-03/evidence-06/files'


def main():
    paths=set()
    for number in ('0117','0040','0099'):
        argv=json.loads((EVIDENCE/number/'driver-invocation.json').read_bytes())['args']
        for flag in ('--cpu-certificate','--reference','--pocket','--ready'):
            paths.add(argv[argv.index(flag)+1])
    program='PATHS='+repr(sorted(paths))+'\n'+'''from pathlib import Path
import hashlib,json
result=[]
for name in PATHS:
 p=Path(name);assert p.is_file() and not p.is_symlink()
 with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
 result.append({'path':name,'bytes':p.stat().st_size,'sha256':digest})
print(json.dumps(result))
'''
    OUT.mkdir(exist_ok=False)
    (OUT/'program.py').write_text(program,encoding='utf8',newline='\n')
    argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
          'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    with (OUT/'bindings.json').open('xb') as output,(OUT/'stderr.txt').open('xb') as error:
        result=subprocess.run(argv,input=program.encode(),stdout=output,stderr=error,timeout=75)
    (OUT/'receipt.json').write_text(json.dumps({'returncode':result.returncode,'argv':argv,
        'program_sha256':hashlib.sha256(program.encode()).hexdigest(),
        'raw_sha256':hashlib.sha256((OUT/'bindings.json').read_bytes()).hexdigest(),
        'mutated':False},indent=2)+'\n',encoding='utf8',newline='\n')
    assert result.returncode==0,(OUT/'stderr.txt').read_text()
    print(json.dumps(json.loads((OUT/'bindings.json').read_bytes()),indent=2))


if __name__=='__main__':main()
