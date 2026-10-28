"""Prepare the explicit six-arm saved-data update; local stdlib bytes only."""
import argparse
import ast
import base64
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import tarfile

import refresh_capsule as previous

OWNER = Path(__file__).resolve().parents[1]
REPO = OWNER.parents[2]
CAPTURE = REPO / 'results/hpc/20261028-jq1-current-report-05'
COUNTS = {'tests.test_qualification':19, 'tests.test_report_data':73,
          'tests.test_report_ops':16, 'tests.test_viewer':41}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generation',default='08');parser.add_argument('--edition',default='edition-0004')
    args=parser.parse_args()
    assert args.generation.isdigit() and len(args.generation)==2 and int(args.generation)>=5
    assert args.edition.startswith('edition-') and len(args.edition)==12 and args.edition[-4:].isdigit()
    package=previous.LOCAL/('source'+args.generation)
    assert not package.exists() and not (previous.LOCAL/('launch-'+args.generation)).exists()
    capture=json.loads((CAPTURE/'capture01/stdout.json').read_bytes())
    transport=json.loads((CAPTURE/'capture01/receipt.json').read_bytes())
    assert transport['returncode']==0 and transport['stdout_sha256']==previous.digest((CAPTURE/'capture01/stdout.json').read_bytes())
    original=(previous.TEMPLATE/'source.tar.gz').read_bytes()
    assert len(original)==previous.ARCHIVE_BYTES and previous.digest(original)==previous.ARCHIVE_SHA
    members,old=previous.archive_members(original,previous.MANIFEST_SHA)
    manifest=deepcopy(old);plan=deepcopy(capture['plan']);plan['edition']=args.edition
    assert {a['id'] for a in plan['arms']}=={'protenix','boltz','dbmol','protenix_noisy','protenix_chemistry','boltz_chemistry'}
    for arm in plan['arms']:
        arm['continuation_status']='validation_queued' if arm['id'] in ('protenix','dbmol') else 'completed' if arm['id']=='protenix_noisy' else 'blocked'
        arm['status_detail']=arm['status_detail'].replace('queued','submitted').replace('to30,000','to 30,000').replace('toward100','toward 100').replace('after5,000','after 5,000').replace('at50,000','at 50,000').replace('at30,000','at 30,000').replace('First10,000','First 10,000').replace('Absolute20,000','Absolute 20,000')
    changed=['scripts/report_data.py','scripts/report_ops.py','templates/report.html',
             'tests/test_report_data.py','tests/test_report_ops.py','tests/test_viewer.py']
    for name in changed:
        data=(OWNER/name).read_bytes().replace(b'\r\n',b'\n')
        if name=='scripts/report_ops.py':
            data=data.replace(b'nonchem-live-report-release-08',('nonchem-live-report-release-'+args.generation).encode()).replace(b'nonchem-live-report-08',('nonchem-live-report-'+args.generation).encode()).replace(b'mosaic-live-report08-',('mosaic-live-report'+args.generation+'-').encode()).replace(b'edition-0004',args.edition.encode())
        if name.endswith('.py'):ast.parse(data,filename=name)
        members[name]=data
    run=OWNER/'runs'/('20261028-jq1-current-report-'+args.generation)
    assert (run/'README.md').is_file()
    members['README.md']=(run/'README.md').read_bytes().replace(b'\r\n',b'\n')
    members['plan.json']=previous.encoded(plan)
    sources={r['path']:r for r in capture['parent_sources']}
    sources[previous.BASE+'/nonchem-live-report-release-04/source']={'path':previous.BASE+'/nonchem-live-report-release-04/source','files':17,'manifest_sha256':previous.MANIFEST_SHA}
    sources[previous.BASE+'/nonchem-precision-release-15/source']={'path':previous.BASE+'/nonchem-precision-release-15/source','files':35,'manifest_sha256':'5448503965365f4db726d558df4ac91a41a7f711366b6ae47ae936993fd0c8bb'}
    proofs=list(capture['proofs'])
    failed=previous.LOCAL/'observation-current05-01/raw.json'
    if int(args.generation)>=6:
        failure=json.loads(failed.read_bytes());assert failure['source_manifest_sha256']=='3933f22fe049ee7a1af854bb94d7d1caca5824969e978c53dd44717a1e327423'
        sources[previous.BASE+'/nonchem-live-report-release-05/source']={'path':previous.BASE+'/nonchem-live-report-release-05/source','files':17,'manifest_sha256':failure['source_manifest_sha256']}
        for row in failure['files']:
            if row['relative'].startswith('cpu/'):
                proofs.append({'path':previous.BASE+'/nonchem-live-report-05/'+row['relative'],'bytes':row['bytes'],'sha256':row['sha256']})
    if int(args.generation)>=8:
        prior=json.loads((previous.LOCAL/'observation-current07-04/raw.json').read_bytes())
        assert prior['publication_ready'] is True and prior['source_manifest_sha256']=='7e19c46a1b1a062710f44d994f33cb57ced0a484c709ae320503503ebd24008d'
        for phase,job in prior['jobs'].items():
            query=prior['queries'][phase];compact=''.join(query['stdout'].split())
            assert query['returncode']==0 and 'Job<'+job+'>' in compact and 'Status<DONE>' in compact
        sources[previous.BASE+'/nonchem-live-report-release-07/source']={'path':previous.BASE+'/nonchem-live-report-release-07/source','files':17,'manifest_sha256':prior['source_manifest_sha256']}
        proofs.extend({'path':previous.BASE+'/nonchem-live-report-07/'+r['relative'],'bytes':r['bytes'],'sha256':r['sha256']} for r in prior['files'])
        proofs.append(prior['archive_receipt']['archive'])
    manifest.update(parent_sources=list(sources.values()),proofs=proofs,cpu_counts=COUNTS,
        refresh_capsule={'version':'jq1-six-arm-current-update-v1','generation':args.generation,'edition':args.edition,
            'original_results_changed':False,'optimizer_submission':False,'monitor_remains_paused':True,
            'changes':'Latest saved segments; two chemistry arms; continuation labels; full current independent storage; retained refined layout'})
    manifest['files']={name:previous.digest(data) for name,data in sorted(members.items()) if name!='source_manifest.json'}
    members['source_manifest.json']=previous.encoded(manifest)
    buffer=io.BytesIO()
    with gzip.GzipFile(fileobj=buffer,mode='wb',filename='',mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as archive:
            for name,data in sorted(members.items()):
                item=tarfile.TarInfo(name);item.mode=0o444;item.mtime=0;item.size=len(data);archive.addfile(item,io.BytesIO(data))
    archive=buffer.getvalue();pin=previous.digest(members['source_manifest.json'])
    checked,_=previous.archive_members(archive,pin);assert checked==members
    prefix=previous.launcher_prefix()
    for old,new in {'nonchem-live-report-release-03':'nonchem-live-report-release-'+args.generation,
                    'nonchem-live-report-03':'nonchem-live-report-'+args.generation}.items():prefix=prefix.replace(old,new)
    anchor="assert not root.exists() and not source.parent.exists(), 'Existing/unknown source or report root: never replay'"
    assert prefix.count(anchor)==1
    prefix=prefix.replace(anchor,"claim=base/'nonchem-live-report-"+args.generation+".current-claim.json'\nwith claim.open('x') as stream:json.dump({'status':'unknown','manifest_sha256':PIN,'optimizer_submission':False},stream)\n"+anchor)
    prefix=prefix.replace("preflight=root/'preflight'",previous.COUNT_GUARD+"\npreflight=root/'preflight'")
    program='\n'.join(['ARCHIVE='+repr(base64.b64encode(archive).decode()),'LENGTH='+str(len(archive)),
                       'DIGEST='+repr(previous.digest(archive)),'PIN='+repr(pin),prefix])
    ast.parse(program)
    package.mkdir()
    artifacts={'source.tar.gz':archive,'source_manifest.json':members['source_manifest.json'],'plan.json':members['plan.json'],
               'remote-program.py':program.encode(),'start_prepared.py':previous.STARTER.encode()}
    for name,data in artifacts.items():(package/name).write_bytes(data)
    prep={'status':'prepared_not_submitted','generation':args.generation,'edition':args.edition,'files':len(members)-1,
        'manifest_sha256':pin,'manifest_bytes':len(members['source_manifest.json']),
        'archive_bytes':len(archive),'archive_sha256':previous.digest(archive),'cpu_counts':COUNTS,
        'plan_bytes':len(members['plan.json']),'plan_sha256':previous.digest(members['plan.json']),
        'program_bytes':len(program.encode()),'program_sha256':previous.digest(program.encode()),
        'optimizer_submission':False,'monitor_remains_paused':True}
    (package/'preparation.json').write_bytes(previous.encoded(prep))
    for name,data in [('plan.json',members['plan.json']),('source-manifest.json',members['source_manifest.json']),('preparation.json',previous.encoded(prep))]:
        with (run/name).open('xb') as stream:stream.write(data)
    print(json.dumps(prep,indent=2))

if __name__=='__main__':main()
