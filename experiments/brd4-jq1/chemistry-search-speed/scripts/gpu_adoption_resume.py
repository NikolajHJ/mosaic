"""Single fresh admission of authenticated, staged but never submitted capsule03.

The initial preflight failed closed on a concurrently replaced native .partial
file. Preserve it, inspect all unknown markers and perform one complete NEW
capacity observation. No seal, numerical policy or storage formula changes.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
LOCAL = REPO / 'results/hpc/20261025-chemistry-speed-gpu-adoption-03'
RUN = HERE / 'runs/20261025-chemistry-speed-gpu-adoption-03'


def main():
    from gpu_adoption_launch import REMOTE
    prep = json.loads((RUN / 'campaign.json').read_bytes())
    assert prep['status'] == 'AST_byte_packaged_not_allocated_pass'
    first = LOCAL / 'launch-01'
    assert json.loads((first / 'result.json').read_bytes())['returncode'] == 1
    failure = (first / 'stderr.txt').read_bytes()
    assert b'FileNotFoundError' in failure and b'trials/latest-trial.partial' in failure
    prefix = r'''
import hashlib,json,os,sys
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=base/'chemistry-search-speed-gpu-adoption-03'
source=base/'chemistry-search-speed-gpu-adoption-release-03/source'
assert root.is_dir() and source.is_dir()
assert hashlib.sha256((root/'source.tar.gz').read_bytes()).hexdigest()==SHA
assert (root/'source.tar.gz').stat().st_size==LENGTH
assert not any(os.path.lexists(root/p) for p in ['jobs.json','cpu','after-cpu','dispatch.lock','admission-recheck-started.json'])
assert not os.path.lexists(base/'chemistry-search-speed-02/gpu'),'Unknown GPU outcome; never duplicate'
sys.path.insert(0,str(source/'scripts'));import gpu_adoption as w
w.manifest(PIN);parent=w.original_proofs()
w.write(root/'admission-recheck-started.json',{'reason':'Prior complete preflight failed on replaced active trial partial; no job submitted',
                                           'first_local_failure_sha256':FAILURE,'source_manifest_sha256':PIN})
pre=root/'preflight-recheck-01';pre.mkdir(exist_ok=False)
gate=parent.storage(pre,'gpu-adoption03-fresh-preflight')
'''
    remainder = REMOTE[REMOTE.index('jobs={};submissions={}'):]
    bindings = {'PIN': prep['manifest_sha256'], 'SHA': prep['archive_sha256'],
                'LENGTH': prep['archive_bytes'], 'FAILURE': hashlib.sha256(failure).hexdigest()}
    code = '\n'.join(k + '=' + repr(v) for k, v in bindings.items()) + '\n' + prefix + remainder
    # The reused submission body imports these in its initial script; retain here.
    code = 'import subprocess\n' + code
    ast.parse(code)
    out = LOCAL / 'admission-recheck-01'; out.mkdir(exist_ok=False)
    (out / 'remote-program.py').write_text(code, encoding='utf-8', newline='\n')
    (out / 'started.json').write_text(json.dumps({'started_utc': datetime.now(timezone.utc).isoformat(),
                                                 'remote_program_sha256': hashlib.sha256(code.encode()).hexdigest()}, indent=2))
    argv = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
            'dtu-hpc', 'bash -l -c ' + shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    with (out / 'stdout.json').open('xb') as f, (out / 'stderr.txt').open('xb') as e:
        r = subprocess.run(argv, input=code.encode(), stdout=f, stderr=e, timeout=240)
    status = {'returncode': r.returncode, 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'stdout_sha256': hashlib.sha256((out / 'stdout.json').read_bytes()).hexdigest()}
    (out / 'result.json').write_text(json.dumps(status, indent=2) + '\n'); print(json.dumps(status))
    if r.returncode:
        print((out / 'stderr.txt').read_text(errors='replace')[-6000:])
        raise SystemExit(r.returncode)
    actual = json.loads((out / 'stdout.json').read_bytes())
    (RUN / 'campaign.json').write_text(json.dumps({**prep, **actual,
        'initial_preflight': 'failed_closed_on_concurrent_trial_partial_preserved',
        'first_failure_sha256': hashlib.sha256(failure).hexdigest()}, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'jobs': actual['jobs'], 'metadata_cases': actual['metadata_cases'],
                      'source_manifest_sha256': actual['manifest_sha256']}, indent=2))


if __name__ == '__main__':
    main()
