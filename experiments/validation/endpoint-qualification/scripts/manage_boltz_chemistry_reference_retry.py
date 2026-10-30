"""Retry only the actually failed Boltz chemistry transport in a new release."""
import argparse
import ast
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

OWNER=Path(__file__).resolve().parents[1];REPO=OWNER.parents[2]
TAG='20261030-boltz-objective-precision-06';LOCAL=REPO/'results/hpc'/TAG;RUN=OWNER/'runs'/TAG
BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01';SCOPE=BASE+'/nonchem-precision-continuation-26'
ROOT=SCOPE+'/runtime-production-05/boltz-precision-06';SOURCE=SCOPE+'/boltz-precision-release-06/source'
OLDROOT=SCOPE+'/runtime-production-05/boltz-precision-05';OLDSOURCE=SCOPE+'/boltz-precision-release-05/source'
OLDPIN='9f562f02918f02bdc2a5e732ab4a3f74b4752ade0935c7894aba6530fb0fba1c'


def raw(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()


def prepare():
    out=LOCAL/'source01';assert not out.exists(),'Immutable prepared retry exists'
    old=REPO/'results/hpc/20261030-boltz-objective-precision-05/source01'
    b=(old/'source_manifest.json').read_bytes();assert sha(b)==OLDPIN;m=json.loads(b)
    members={n:(old/n).read_bytes() for n in m['files']}
    assert all(sha(members[n])==h for n,h in m['files'].items())
    plan=json.loads(members['plan.json']);plan.update(root=ROOT,source_root=SOURCE)
    plan['arms']={'boltz_chemistry':plan['arms']['boltz_chemistry']}
    packet=json.loads((REPO/'results/hpc/20261030-boltz-objective-precision-05/status05/stdout.json').read_bytes())
    sub=packet['submissions']['gpu-boltz_chemistry'];assert sub['job_id']=='29853437'
    proofs=[]
    for suffix in ('stdout.29853437.log','stderr.29853437.log','payload/execution.log','payload/production-chemistry-GPU/stderr.log'):
        r=packet['files'][OLDROOT+'/gpu-boltz_chemistry/'+suffix];data=base64.b64decode(r['base64'],validate=True)
        assert sha(data)==r['sha256'] and len(data)==r['bytes']
        if suffix=='stdout.29853437.log':assert b'Exited with exit code 1.' in data and b'queue <gpul40s>' in data
        if suffix=='payload/production-chemistry-GPU/stderr.log':assert b"ModuleNotFoundError: No module named 'mosaic'" in data
        proofs.append({k:r[k] for k in ('path','bytes','sha256')})
    assert OLDROOT+'/gpu-boltz_chemistry/payload/assessment.json' not in packet['files']
    plan['inputs']+=proofs;plan['actual_failed_chemistry_predecessor']={'job_id':'29853437','root':OLDROOT,
        'error':'matrix_worker_missing_Mosaic_import_path_before_combined_gradient_calls','proofs':proofs}
    m['parent_sources'].append({'path':OLDSOURCE,'manifest_sha256':OLDPIN,'files':17})
    for name in ('scripts/boltz_precision_driver.py','scripts/boltz_precision_ops.py','tests/test_boltz_precision_portable.py'):
        members[name]=(OWNER/name).read_bytes().replace(b'\r\n',b'\n')
    members['plan.json']=raw(plan);members['README.md']=(RUN/'README.md').read_bytes().replace(b'\r\n',b'\n')
    m['jax_cpu_counts']['tests.test_boltz_precision_portable']=13
    m['all_cpu_counts']['tests.test_boltz_precision_portable']=13
    assert sum(m['all_cpu_counts'].values())==84
    m['files']={n:sha(b) for n,b in members.items()}
    members['source_manifest.json']=raw(m);out.mkdir(parents=True);buffer=io.BytesIO()
    for name,b in members.items():
        if name.endswith('.py'):ast.parse(b,filename=name)
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name,b in members.items():
            path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
            row=tarfile.TarInfo(name);row.size=len(b);row.mode=0o444;archive.addfile(row,io.BytesIO(b))
    data=buffer.getvalue();(out/'source.tar.gz').write_bytes(data)
    value={'status':'prepared_only_failed_chemistry_retry_not_qualified','tag':TAG,'root':ROOT,'source':SOURCE,'files':len(m['files']),
        'manifest_sha256':sha(members['source_manifest.json']),'plan_sha256':sha(members['plan.json']),
        'archive_bytes':len(data),'archive_sha256':sha(data),'CPU_cases':84,'optimizer_jobs':[]}
    for name,b in [('preparation.json',raw(value)),('campaign.json',raw(value)),('source-manifest.json',members['source_manifest.json'])]:
        (RUN/name).write_bytes(b)
    (out/'preparation.json').write_bytes(raw(value));print(json.dumps(value))


def stage():
    p=OWNER/'scripts/manage_boltz_precision.py';spec=importlib.util.spec_from_file_location('_prior_only_transport',p)
    prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    prior.TAG=TAG;prior.LOCAL=LOCAL;prior.RUN=RUN;prior.ROOT=ROOT;prior.SOURCE=SOURCE
    prior.STAGE=prior.STAGE.replace("runtime-production-05/boltz-precision-05'","runtime-production-05/boltz-precision-06'").replace("boltz-precision-release-05/source'","boltz-precision-release-06/source'")
    prior.stage()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','stage'));a=p.parse_args()
    prepare() if a.action=='prepare' else stage()
