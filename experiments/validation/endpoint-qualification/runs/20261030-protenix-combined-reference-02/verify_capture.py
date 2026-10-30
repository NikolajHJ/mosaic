"""Independent byte/XML/actual-command audit; no array or model evaluation."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import xml.etree.ElementTree as ET

BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
SCOPE=BASE+'/nonchem-precision-continuation-26'
ROOT=SCOPE+'/chemistry-continuation-05/combined-reference-qualification-02'
SOURCE=SCOPE+'/combined-reference-qualification-release-02/source'
Q=SCOPE+'/runtime-qualification-release-01/source'
PIN='85a812d69c85860800e854763cee286962e059507e1e2853bc09c0efbc408e7b'


def require(value,message):
    if not value:raise ValueError(message)


def sha(value):return hashlib.sha256(value).hexdigest()


def verify(packet):
    files={}
    for name,row in packet['files'].items():
        b=base64.b64decode(row['base64'],validate=True)
        require(len(b)==row['bytes'] and sha(b)==row['sha256'],'Delivered byte hash differs')
        files[name]=b
    def read(name):return json.loads(files[name])
    require(sha(files[SOURCE+'/source_manifest.json'])==PIN,'Current sealed manifest differs')
    m=read(SOURCE+'/source_manifest.json');plan=read(SOURCE+'/plan.json')
    for n,h in m['files'].items():require(sha(files[SOURCE+'/'+n])==h,'Own sealed member differs')
    require(len(packet['parent_sources'])==len(m['parent_sources']),'Incomplete external sources')
    for row in m['parent_sources']:
        found=[s for s in packet['parent_sources'] if s['manifest']['path']==row['path']+'/source_manifest.json']
        require(len(found)==1 and found[0]['manifest']['sha256']==row['manifest_sha256']
                and len(found[0]['members'])==row['files'],'Parent manifest/count differs')
    require(packet['inputs_rehashed']==plan['inputs'],'Original input rehash set differs')
    local=Path(__file__).resolve().parents[5]/'results/hpc/20261030-protenix-runtime-qualification-01/source02'
    original=ast.parse((local/'scripts/failure_diagnostic_ops.py').read_bytes())
    ns={'ROOT':PurePosixPath(ROOT),'SOURCE':PurePosixPath(SOURCE),
        'CAPACITY':PurePosixPath(plan['capacity_source']['path']),'require':require}
    nodes=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='argv']
    require(len(nodes)==1,'Original command metadata missing')
    exec(compile(ast.Module(nodes,[]),'<authenticated-metadata-only-command>','exec'),ns)
    for phase,row in packet['submissions'].items():
        expected=ns['argv'](phase,PIN,row['dependency']);expected[expected.index('-J')+1]='mosaic-Prochem-reference02-'+phase;expected+=['--shared-capacity-source',BASE+'/runtime-production-release-05/source']
        parent=ROOT+'/'+phase;require(row['argv']==expected,'Actual allocation/source/dependency differs')
        require(packet['registry'][parent[len(SCOPE)+1:]]==row['job_id'],'Registry differs')
        raw=read(parent+'/submission-raw.json')
        require(raw['returncode']==0 and re.findall(r'Job <([0-9]+)> is submitted',raw['stdout'])==[row['job_id']],
                'Actual scheduler acknowledgement differs')
        held=[read(n) for n in files if n.startswith(parent+'/held-scheduler-')]
        require(any(r['returncode']==0 and r['stdout'].split()[:3]==[row['job_id'],'PSUSP','s234818'] for r in held),
                'Owned pre-release held proof missing')
        release=read(parent+'/release.json');raw=read(parent+'/release-raw.json')
        require(release['receipt_persisted_before_release'] and release['job_id']==row['job_id']
                and raw['returncode']==0 and raw['argv']==['bresume',row['job_id']],'Actual owned release differs')
    result={'status':'sealed_source_inputs_and_actual_submission_proofs_verified_CPU_pending',
        'source_manifest_sha256':PIN,'jobs':{k:v['job_id'] for k,v in packet['submissions'].items()},
        'observed_utc':packet['observed_utc'],'optimizer_admission':False,'scientific_GPU_pass':False}
    certificate=ROOT+'/cpu-certificate.json'
    if certificate not in files:
        failures={}
        for phase,row in packet['submissions'].items():
            path=ROOT+'/'+phase+'/stdout.'+row['job_id']+'.log'
            if path in files and b'Exited with exit code 1.' in files[path]:failures[phase]=row['job_id']
        if failures:
            result.update(status='source_and_submission_proofs_verified_actual_failed_CPU_or_controller_preserved',terminal_failures=failures)
            path=ROOT+'/cpu/payload/combined-reference.xml'
            if path in files:
                cases=list(ET.fromstring(files[path]).iter('testcase'))
                result['new_raw_cases']=len(cases)
                result['failed_new_cases']=[c.get('name') for c in cases if any(list(c.iter(t)) for t in ('failure','error'))]
        return result
    cases=list(ET.fromstring(files[ROOT+'/cpu/payload/tests.xml']).iter('testcase'))
    counts={k:0 for k in m['all_cpu_counts']}
    for case in cases:
        require(not any(list(case.iter(t)) for t in ('failure','error','skipped')),'Raw CPU case failed/skipped')
        matches=[k for k in counts if case.get('classname')==k or case.get('classname','').startswith(k+'.')]
        require(len(matches)==1,'Exact raw case attribution required');counts[matches[0]]+=1
    require(counts==m['all_cpu_counts'] and len(cases)==1044,'Allocated count/module set differs')
    for phase in ('cpu','after-cpu'):
        job=packet['submissions'][phase]['job_id'];text=' '.join(files[ROOT+'/'+phase+'/stdout.'+job+'.log'].decode().split())
        require(text.count('Subject: Job '+job+':')==1 and 'queue <hpc>' in text
                and text.count('Successfully completed.')==1 and 'Exited with exit code' not in text,
                'Actual same CPU/controller LSF success required')
    startup=read(ROOT+'/cpu/payload/startup/assessment.json')
    require(startup['status']=='passed_actual_full_validation_startups' and startup['model_constructed'] is False
            and [r['arm'] for r in startup['startups']]==['plain','chemistry']
            and [r['arm'] for r in startup['reference_metadata']]==['plain','chemistry'],
            'All four actual original CLI startup bodies required')
    gate=read(ROOT+'/cpu/payload/assessment.json')
    require(gate['cases']==1044 and gate['source_manifest_sha256']==PIN,'CPU assessment identity differs')
    dispatched=read(ROOT+'/after-cpu/payload/assessment.json')
    require(set(dispatched['jobs'])=={'chemistry'} and dispatched['optimizer_jobs']==[],
            'Only chemistry validator/capture dispatch allowed')
    result.update(status='independent_CPU1044_four_actual_startups_source_bytes_LSF_and_held_submission_verified',
        CPU_cases=1044,modules=counts,delivered_files=len(files))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('packet',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'Fresh independent proof output required')
    b=a.packet.read_bytes();result=verify(json.loads(b));result['packet_sha256']=sha(b)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
