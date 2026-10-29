"""Independent stdlib byte/XML/LSF/vector audit; never evaluates a model."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import xml.etree.ElementTree as ET

OWNER=Path(__file__).resolve().parents[1]
REPO=OWNER.parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--label',required=True);a=p.parse_args()
    assert a.tag.replace('-','').isalnum() and a.label.replace('-','').isalnum()
    local=REPO/'results/hpc'/a.tag;source=local/'source01';run=OWNER/'runs'/a.tag
    def read(p):return json.loads(Path(p).read_bytes())
    def sha(b):return hashlib.sha256(b).hexdigest()
    manifest=read(source/'source_manifest.json');pin=sha((source/'source_manifest.json').read_bytes());plan=read(source/'plan.json')
    for name,digest in manifest['files'].items():assert sha((source/name).read_bytes())==digest
    prep=read(source/'preparation.json');assert prep['manifest_sha256']==pin
    v=read(local/a.label/'observation.json');records={}
    import base64
    for r in v['files']:
        b=base64.b64decode(r['base64'],validate=True) if 'base64' in r else r['text'].encode()
        assert len(b)==r['bytes'] and sha(b)==r['sha256'] and r['path'] not in records;records[r['path']]=(r,b)
    root=PurePosixPath(plan['root'])
    def data(rel):return json.loads(records[str(root/rel)][1])
    cert=data('cpu-certificate.json');assert cert['status']=='passed' and cert['source_manifest_sha256']==pin
    for proof in cert['proofs']:assert {k:records[proof['path']][0][k] for k in ('path','bytes','sha256')}==proof
    def cases(name):
        rows=list(ET.fromstring(records[str(root/'cpu/payload'/name)][1]).iter('testcase'))
        assert rows and all(not list(r) for r in rows)
        return [(r.get('classname'),r.get('name')) for r in rows]
    merged=cases('tests.xml');raw=cases('tests-jax.xml')+cases('tests-torch.xml')
    assert len(set(merged))==len(merged) and Counter(merged)==Counter(raw)
    assert dict(Counter(k for k,_ in merged))==manifest['cpu_counts'] and len(merged)==sum(manifest['cpu_counts'].values())
    assert v['scheduler']['returncode']==0
    sched={r['JOBID']:r for r in json.loads(v['scheduler']['stdout'])['RECORDS']}
    sys.path.insert(0,str(source/'scripts'))
    spec=importlib.util.spec_from_file_location('_exact_cpu_metadata_ops',source/'scripts/precision_continuation_ops.py')
    o=importlib.util.module_from_spec(spec);sys.modules[spec.name]=o;spec.loader.exec_module(o)
    o.ROOT=root;o.SOURCE=PurePosixPath(plan['source_root'])
    for phase in ('cpu','after-cpu'):
        jid=v['jobs'][phase];sub=data(phase+'/submission.json');dep='ended('+v['jobs']['cpu']+')' if phase=='after-cpu' else None
        expected=o.argv(phase,pin,dependency=dep);assert sub['argv']==expected and sub['job_id']==jid and sub['status']=='submitted'
        assert sched[jid]['STAT']=='DONE' and sched[jid]['QUEUE']=='hpc'
        o.lsf_success(records[str(root/phase/('stdout.'+jid+'.log'))][1].decode(),jid,expected[expected.index('-J')+1],'hpc')
    native={}
    for arm,offset in (('protenix',20000),('dbmol',7500)):
        for phase in ('validate','after-validate'):
            name=o.label(phase,arm,offset);sub=data(name+'/submission.json')
            dep='ended('+v['jobs'][o.label('validate',arm,offset)]+')' if phase=='after-validate' else None
            assert sub['argv']==o.argv(phase,pin,arm,offset,dep) and sub['status']=='submitted' and sub['job_id']==v['jobs'][name]
            native[name]=sub['job_id']
    # The immutable controller authenticates the full existing control budget
    # at entry before submitting its native checks; no segment claim is made.
    gates=[json.loads(b) for name,(_,b) in records.items() if '/after-cpu/rolling-capacity-entry-' in name and name.endswith('/assessment.json')]
    assert len(gates)==1;gate=gates[0]
    assert gate['status']=='passed' and gate['reserve_bytes']==4*1024**3 and gate['file_reserve']>=100000
    assert gate['phase']=='entry'
    assert gate['byte_margin']>=0 and gate['file_margin']>=0 and gate['source_manifest_sha256']==pin
    proof=data('cpu/payload/assessment.json');assert proof['cases']==len(merged) and proof['status']=='passed'
    assert len(proof['actual_startup_checks'])==4 and len(proof['actual_worker_startup_checks'])==1
    verification={'status':'passed_independent_source_byte_XML_LSF_actual_submission_audit','observed_utc':v['observed_utc'],
        'files_verified':len(records),'source_files_verified':len(manifest['files']),'cases':len(merged),'modules':len(manifest['cpu_counts']),
        'failures_errors_skips':0,'CPU_job':v['jobs']['cpu'],'controller_job':v['jobs']['after-cpu'],
        'CPU_certificate':{k:records[str(root/'cpu-certificate.json')][0][k] for k in ('path','bytes','sha256')},
        'actual_native_jobs':native,'controller_entry_storage':{k:gate[k] for k in ('phase','required_bytes','available_bytes','reserve_bytes','file_reserve')},
        'native_pass_established':False,'optimizer_job_submitted':any('-segment-' in k for k in v['jobs']),
        'saved_results_changed':False,'local_scientific_evaluation':False}
    path=run/'cpu-verification.json'
    with path.open('x',encoding='utf8',newline='\n') as f:json.dump(verification,f,indent=2);f.write('\n')
    campaign=read(run/'campaign.json');campaign.update(status='CPU_and_controller_passed_native_jobs_submitted_pending_own_gates',jobs=v['jobs'],cpu_verification=verification)
    (run/'campaign.json').write_text(json.dumps(campaign,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
