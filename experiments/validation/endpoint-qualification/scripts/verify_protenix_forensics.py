"""Independent stdlib byte/XML/LSF audit; no scientific array computation."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

REPO=Path(__file__).resolve().parents[4]
RUN=REPO/'experiments/validation/endpoint-qualification/runs/20261025-protenix-endpoint-forensics-01'
LOCAL=REPO/'results/hpc/20261025-protenix-endpoint-forensics-01'
REMOTE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/protenix-endpoint-forensics-01'


def digest(raw):return hashlib.sha256(raw).hexdigest()


def verify(observation):
    obs=json.loads(Path(observation).read_bytes())
    prep=json.loads((RUN/'preparation.json').read_bytes())
    files={}
    for item in obs['files']:
        data=item['text'].encode()
        assert len(data)==item['bytes'] and digest(data)==item['sha256']
        assert item['path'].startswith(REMOTE+'/') and '..' not in Path(item['path']).parts
        assert item['path'] not in files or files[item['path']]['sha256']==item['sha256']
        files[item['path']]=item
    def row(relative):return files[REMOTE+'/'+relative]
    def read(relative):return json.loads(row(relative)['text'])
    def bound(proof):
        assert files[proof['path']]['sha256']==proof['sha256'] and files[proof['path']]['bytes']==proof['bytes']
    rawcpu=row('cpu/payload/tests.xml')
    cases=list(ET.fromstring(rawcpu['text']).iter('testcase'))
    assert cases and all(not list(c) and c.get('classname') and c.get('name') for c in cases)
    assert len({(c.get('classname'),c.get('name')) for c in cases})==len(cases)
    counts=dict(Counter(c.get('classname') for c in cases));assert counts==prep['cpu_counts']
    cert=read('cpu-certificate.json')
    assert cert['status']=='passed' and cert['cases']==len(cases) and cert['source_manifest_sha256']==prep['manifest_sha256']
    assert cert['plan_sha256']==prep['plan_sha256'] and cert['original_scientific_cpu_replaced'] is False
    for proof in cert['proofs']:bound(proof)
    sub=read('cpu/submission.json');job=sub['job_id'];assert sub['status']=='submitted'
    log=''.join(row('cpu/stdout.'+job+'.log')['text'].split())
    assert re.findall(r'Subject:Job(\d+):',log)==[job] and re.findall(r'inqueue<([^>]+)>',log)==['hpc']
    assert log.count('Successfullycompleted.')==1 and 'Exitedwithexit' not in log
    assert re.findall(r'Job<([^>]+)>',log)[:1]==['mosaic-pro-forensics-cpu']
    queries=[json.loads(item['text']) for path,item in files.items() if '/cpu/terminal-query-' in path]
    assert any(q['returncode']==0 and 'Status<DONE>' in ''.join(q['stdout'].split())
               and 'Job<'+job+'>' in ''.join(q['stdout'].split()) and 'Queue<hpc>' in ''.join(q['stdout'].split()) for q in queries)
    result={'status':'passed_cpu_byte_xml_lsf_schema','captured_utc':obs['captured_utc'],'verified_files':len(files),
            'cpu_job_id':job,'cpu_cases':len(cases),'cpu_counts':counts,
            'cpu_certificate_sha256':row('cpu-certificate.json')['sha256'],
            'source_manifest_sha256':prep['manifest_sha256'],'plan_sha256':prep['plan_sha256'],
            'native_findings_established':False,'original_results_changed':False,
            'optimization_settings_changed':False,'can_certify_or_restart_original_study':False,
            'scope':'Local stdlib bytes/XML/schema/LSF only; no numerical or array calculations'}
    if REMOTE+'/gpu/submission.json' in files:
        gpu=read('gpu/submission.json');argv=gpu['argv']
        assert gpu['status']=='submitted' and gpu['source_manifest_sha256']==prep['manifest_sha256']
        assert argv[argv.index('-q')+1]=='gpul40s' and argv[argv.index('-n')+1]=='4'
        assert argv[argv.index('-W')+1]=='02:00' and argv[argv.index('-gpu')+1]=='num=1:mode=exclusive_process'
        assert 'rusage[mem=8GB]' in argv and '-w' not in argv
        result['gpu_job_id']=gpu['job_id']
    if REMOTE+'/gpu/payload/assessment.json' in files:
        native=read('gpu/payload/assessment.json')
        result['native_assessment_status']=native['status']
        result['native_findings_established']=native['status']=='diagnostics_completed_original_failure_preserved'
        assert native['updates']==0 and native['optimization_settings_changed'] is False
        assert native['can_certify_or_restart_original_study'] is False
        if result['native_findings_established']:
            # Full native proof collection and raw NPZ audit are separate; this
            # compact assessment is metadata, never all-point certification.
            assert native['protected_inputs_unchanged'] is True and native['frozen_unchanged'] is True
            result['native_scope']='compact authenticated metadata; full raw point bundle not yet independently collected'
    out=LOCAL/('verification-'+Path(observation).parent.name+'.json')
    assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n')
    (RUN/'cpu-verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('observation',type=Path);a=p.parse_args();verify(a.observation)
