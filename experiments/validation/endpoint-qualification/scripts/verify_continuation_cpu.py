"""Independent stdlib raw-byte/XML/schema audit of allocated CPU observations."""
from collections import Counter
import argparse
import base64
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import xml.etree.ElementTree as ET

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261025-continuation-admission-01'
RUN=REPO/'experiments/validation/endpoint-qualification/runs/20261025-continuation-admission-01'


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--observation',type=Path,required=True);args=parser.parse_args()
    observed=json.loads(args.observation.read_bytes())
    manifest=json.loads((LOCAL/'source01/source_manifest.json').read_bytes())
    preparation=json.loads((LOCAL/'source01/preparation.json').read_bytes())
    submission=json.loads((RUN/'submission.json').read_bytes())
    assert digest((LOCAL/'source01/source_manifest.json').read_bytes())==preparation['manifest_sha256']
    assert submission['manifest_sha256']==preparation['manifest_sha256']
    decoded={};files=[];destination=args.observation.parent/'files';destination.mkdir(exist_ok=False)
    for parent,record in observed['parents'].items():
        assert re.fullmatch(r'[a-z0-9-]+',parent)
        for rel,row in record['files'].items():
            if 'base64' not in row:continue
            path=PurePosixPath(rel);assert not path.is_absolute() and '..' not in path.parts and '\\' not in rel
            data=base64.b64decode(row['base64'],validate=True)
            assert len(data)==row['binding']['bytes'] and digest(data)==row['binding']['sha256']
            target=destination/parent/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            decoded[(parent,rel)]=data;files.append(row['binding'])
    def raw(parent,rel):return decoded[(parent,rel)]
    def document(parent,rel):return json.loads(raw(parent,rel))
    def terminal(parent,job):
        actual=observed['scheduler'][job];assert actual['returncode']==0
        text=''.join(actual['stdout'].split())
        assert re.findall(r'Job<([^>]+)>',text)==[job]
        assert re.findall(r'Status<([^>]+)>',text)==['DONE']
        assert re.findall(r'Queue<([^>]+)>',text)==['hpc']
        log=''.join(raw(parent,'stdout.'+job+'.log').decode().split())
        assert log.count('Subject:Job')==1 and 'Subject:Job'+job+':<mosaic-endpoint01-'+parent+'>incluster<dcc>Done' in log
        assert re.findall(r'inqueue<([^>]+)>',log)==['hpc'] and log.count('Successfullycompleted.')==1
        assert 'Exitedwithexit' not in log
    cpu=submission['cpu']['job_id'];callback=submission['after_cpu']['job_id']
    terminal('cpu',cpu);terminal('after-cpu',callback)
    for parent,expected in [('cpu',submission['cpu']),('after-cpu',submission['after_cpu'])]:
        assert observed['jobs'][parent]==expected['job_id']
        assert observed['parents'][parent]['submission']==expected
    xml=ET.fromstring(raw('cpu','payload/tests.xml'))
    cases=list(xml.iter('testcase'));assert cases and all(not list(row) for row in cases)
    assert len({(row.get('classname'),row.get('name')) for row in cases})==len(cases)
    counts=dict(Counter(row.get('classname') for row in cases));assert counts==manifest['cpu_counts']
    for suite in xml.iter('testsuite'):
        assert all(int(suite.get(key,'0'))==0 for key in ('errors','failures','skipped'))
    assessment=document('cpu','payload/assessment.json')
    assert assessment['status']=='passed' and assessment['cases']==len(cases)
    assert assessment['source_manifest_sha256']==preparation['manifest_sha256']
    assert assessment['plan_sha256']==preparation['plan_sha256']
    starts=document('cpu','payload/actual-startup-checks.json')
    assert starts['status']=='passed_metadata_only' and starts['native_readiness_established'] is False
    assert starts['fresh_prospective_readiness_deferred_until_native_validation'] is True
    assert len(starts['checks'])==6 and assessment['actual_startup_checks']==starts['checks']
    assert {(r['arm'],r['phase']) for r in starts['checks']}=={(a,p) for a in ('protenix','boltz','dbmol') for p in ('validate','segment')}
    for row in starts['checks']:
        rel='payload/startup-'+row['arm']+'-'+row['phase']+'/endpoint-startup.json'
        value=document('cpu',rel)
        assert value['status']=='passed_metadata_only' and value['numerical_setup_entered'] is False and value['updates']==0
        assert value['readiness_deferred'] is (row['phase']=='segment')
        assert value['job_id']==cpu
        assert digest(raw('cpu',rel))==row['receipt']['sha256']
    for parent in ('cpu','after-cpu'):
        gate=document(parent,'continuation-storage-entry.json')
        assert gate['status']=='passed' and gate['original_gate_changed'] is False
        assert gate['required_bytes']<=gate['available_bytes'] and gate['reserve_bytes']==4*1024**3 and gate['file_reserve']==100000
    certificate=observed['certificates']['cpu-certificate.json']
    assert certificate['data']['status']=='passed' and certificate['data']['cases']==len(cases)
    assert certificate['data']['source_manifest_sha256']==preparation['manifest_sha256']
    assert certificate['data']['plan_sha256']==preparation['plan_sha256']
    assert certificate['data']['original_scientific_cpu_replaced'] is False
    lookup={r['path']:r for r in files}
    assert all(lookup[r['path']]==r for r in certificate['data']['proofs'])
    report={'status':'passed','scope':'independent local stdlib bytes/XML/schema, not native numerical validation',
            'captured_utc':observed['captured_utc'],'source_manifest_sha256':preparation['manifest_sha256'],
            'cpu_job':cpu,'callback_job':callback,'raw_cases':len(cases),'modules':counts,
            'actual_startup_commands':6,'native_readiness_established':False,
            'certificate':certificate['binding'],'downloaded_files':len(files),
            'observation_sha256':digest(args.observation.read_bytes()),'original_results_changed':False}
    target=args.observation.parent/'cpu-verification.json'
    with target.open('x',encoding='utf8',newline='\n') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    (RUN/'cpu-verification.json').write_bytes(target.read_bytes())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
