"""Print read-only metadata for one saved-point audit; no scientific imports."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path('/work3/s234818/mosaic/runs')
ROWS = [
    ('annealed-sparse-s17', BASE/'20261102-sparse-annealed-50k-01/source01', [0, 20000, 30000, 50000]),
    ('structural-s17', BASE/'20261102-exclusive-gpu-recovery-01/screen01', [0, 10000, 20000, 30000]),
    ('structural10-chain-s17', BASE/'20261102-structural-strong-noise-01/source02', [0, 10000, 20000, 30000]),
    ('structural10-noise-chain-s17', BASE/'20261102-structural-strong-noise-01/source02', [0, 10000, 20000, 30000]),
]

def binding(path):
    path = Path(path)
    if not path.is_file():
        return {'path':str(path), 'exists':False}
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            h.update(block)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}

def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=25)
    return {'argv':argv,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}

def main():
    result = {'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'arms':[]}
    for name, home, steps in ROWS:
        plan_path = home/'source/plan.json'
        plan = json.loads(plan_path.read_text())
        row = plan['arms'][name]
        study = home/'run'/name/'payload/study'
        entry = {'arm':name,'plan':binding(plan_path),'row':row,'code_source_root':plan['code_source_root'],
                 'routing_source':plan['routing_source'],'study':str(study), 'files':[]}
        for step in steps:
            entry['files'].append({'step':step,
                 'snapshot':binding(study/'snapshots'/('step-%05d.npz'%step)),
                 'reference':binding(study/'references'/('step-%05d.json.gz'%step))})
        entry['snapshot_names'] = [p.name for p in sorted((study/'snapshots').glob('*'))[:3]]
        entry['reference_names'] = [p.name for p in sorted((study/'references').glob('*'))[:3]]
        for key in ('summary.json','report.json'):
            path = study.parent/key if key=='summary.json' else study/key
            if path.is_file(): entry[key]={'binding':binding(path),'data':json.loads(path.read_text())}
        result['arms'].append(entry)
    result['account'] = command(['bjobs','-u','s234818','-w'])
    result['queues'] = command(['bqueues','-w','gpuv100','gpua10','gpua40','gpul40s','gpua100'])
    result['gpu_hosts'] = command(['bhosts','-gpu'])
    print(json.dumps(result, allow_nan=False))

if __name__=='__main__': main()
