"""Standard-library immutable capsule packaging; no tests or numerical imports."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parents[1]
REPO=HERE.parents[2]
RUN=HERE/'runs/20261025-continuation-admission-01'
LOCAL=REPO/'results/hpc/20261025-continuation-admission-01'


def encoded(value):
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--counts',type=Path,required=True)
    parser.add_argument('--attempt',choices=('01','02'),default='01')
    args=parser.parse_args()
    global RUN,LOCAL
    if args.attempt=='02':
        RUN=HERE/'runs/20261025-continuation-worker-fix-02'
        LOCAL=REPO/'results/hpc/20261025-continuation-worker-fix-02'
        LOCAL.mkdir(exist_ok=True)
    counts=json.loads(args.counts.read_bytes())
    assert set(counts)=={'tests.test_qualification','tests.test_admission',
                         'tests.test_continuation_driver','tests.test_continuation_ops'}
    assert all(type(n) is int and n>0 for n in counts.values())
    plan=json.loads((RUN/'plan.json').read_bytes())
    report=json.loads((REPO/'results/hpc/20261025-jq1-live-report-01/source04/source_manifest.json').read_bytes())
    parents=[r for r in report['parent_sources'] if r['files']!=642]
    if args.attempt=='02':
        parents.append({'path':plan['control_root'].rsplit('/',1)[0]+'/nonchem-endpoint-release-01/source',
                        'files':14,'manifest_sha256':'08e53ba66312b06e5f1c2836a3863479ceb85e11e9364c23026079301d2cd713'})
    proofs={}
    for arm in plan['arms']:
        for group in (arm['inputs'],arm['failed_parent']):
            for row in group.values():
                if isinstance(row,dict) and set(row)=={'path','bytes','sha256'}:
                    if row['path'] in proofs:assert proofs[row['path']]==row
                    proofs[row['path']]=row
        proofs[arm['config']['path']]=arm['config']
    if args.attempt=='02':
        for row in json.loads((RUN/'failure-adoption.json').read_bytes())['proofs']:
            proofs[row['path']]=row
    members={}
    names=['README.md','scripts/README.md','tests/README.md',
           'scripts/qualification.py','scripts/admission.py','scripts/continuation_driver.py',
           'scripts/continuation_ops.py','scripts/continuation_stage.lsf',
           'tests/test_qualification.py','tests/test_admission.py',
           'tests/test_continuation_driver.py','tests/test_continuation_ops.py']
    for name in names:
        content=(HERE/name).read_bytes().replace(b'\r\n',b'\n')
        if name.endswith('.py'):ast.parse(content,filename=name)
        members[name]=content
    members['plan.json']=(RUN/'plan.json').read_bytes()
    policy=HERE/'runs/20261025-continuation-admission-01/prospective-policy.md'
    members['prospective-policy.md']=policy.read_bytes().replace(b'\r\n',b'\n')
    manifest={'version':'endpoint-aware-continuation-source-v1',
              'files':{name:hashlib.sha256(data).hexdigest() for name,data in sorted(members.items())},
              'cpu_counts':counts,'parent_sources':parents,
              'preserved_proofs':list(proofs.values()),
              'original_failures_preserved':True,'original_results_changed':False,
              'policy_scope':'Explicit prospective child admission; original failed studies remain failed',
              'monitor_remains_paused':True,'optimization_settings_changed':False}
    if args.attempt=='02':
        manifest['dispatch_arms']=plan['dispatch_arms']
        manifest['blocked_arms']=plan['blocked_arms']
    members['source_manifest.json']=encoded(manifest)
    pin=hashlib.sha256(members['source_manifest.json']).hexdigest()
    out=LOCAL/'source01';out.mkdir(exist_ok=False)
    for name,content in members.items():
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as archive:
        for name,content in sorted(members.items()):
            info=tarfile.TarInfo(name);info.size=len(content);info.mode=0o444;info.mtime=0
            archive.addfile(info,io.BytesIO(content))
    raw=buf.getvalue();(out/'source.tar.gz').write_bytes(raw)
    receipt={'version':manifest['version'],'status':'prepared_not_validated',
             'manifest_sha256':pin,'files':len(manifest['files']),
             'archive_bytes':len(raw),'archive_sha256':hashlib.sha256(raw).hexdigest(),
             'plan_sha256':hashlib.sha256(members['plan.json']).hexdigest(),
             'cpu_counts':counts,'scientific_source_changed':False,'jobs_submitted':False}
    (out/'preparation.json').write_bytes(encoded(receipt))
    (RUN/'source-manifest.json').write_bytes(members['source_manifest.json'])
    (RUN/'preparation.json').write_bytes(encoded(receipt))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
