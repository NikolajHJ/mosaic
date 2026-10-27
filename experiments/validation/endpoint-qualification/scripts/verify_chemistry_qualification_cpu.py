"""Stdlib-only verification of native-qualification04 CPU/dispatch delivery."""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
ROOT=BASE+'/chemistry-native-qualification-04'
SOURCE=BASE+'/chemistry-native-qualification-release-04/source'
PIN='5d8f4ef0f8bb820e76d2e92442492f604393e79b58725fd01f9bb042b3114863'
CPU='29834574'
CALLBACK='29834576'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('observation',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();v=json.loads(a.observation.read_bytes())
    assert v['root_exists'] and v['source_verified'] and v['source_manifest_sha256']==PIN
    jobs=v['jobs'];assert jobs.get('cpu')==CPU and jobs.get('after-cpu')==CALLBACK
    assert set(jobs)<= {'cpu','after-cpu','gpu-protenix','gpu-boltz'}
    scheduler=v['scheduler'];assert scheduler['returncode']==0
    observed=dict(re.findall(r'Job<([0-9]+)>.*?Status<([^>]+)>',''.join(scheduler['stdout'].split())))
    assert observed.get(CPU)=='DONE' and observed.get(CALLBACK)=='DONE'
    a.output.mkdir(exist_ok=False);bindings={}
    for path,r in v['files'].items():
        assert path.startswith(ROOT+'/') and r['binding']['path']==path
        rel=Path(path[len(ROOT)+1:]);assert not rel.is_absolute() and '..' not in rel.parts
        b=base64.b64decode(r['base64'],validate=True)
        assert len(b)==r['binding']['bytes'] and hashlib.sha256(b).hexdigest()==r['binding']['sha256']
        q=a.output/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b);bindings[path]=r['binding']
    def read(rel):return json.loads((a.output/rel).read_bytes())
    for phase,job in jobs.items():
        sub=read(phase+'/submission.json');assert sub['job_id']==job and sub['status']=='submitted'
        assert sub['source_manifest_sha256']==PIN
        argv=sub['argv'];assert argv[-6:]==['bash','-l',SOURCE+'/scripts/chemistry_endpoint_stage.lsf',SOURCE,PIN,phase]
        gpu=phase.startswith('gpu-');queue='gpul40s' if gpu else 'hpc'
        assert argv[argv.index('-q')+1]==queue
        assert argv[argv.index('-n')+1]==('1' if phase=='after-cpu' else '4')
        assert argv[argv.index('-W')+1]==('02:00' if gpu else '00:20' if phase=='after-cpu' else '01:00')
        assert argv[argv.index('-R',argv.index('-R')+1)+1]==('rusage[mem=8GB]' if gpu else 'rusage[mem=2GB]' if phase=='after-cpu' else 'rusage[mem=4GB]')
        assert (sub['dependency']=='ended('+CPU+')') if phase=='after-cpu' else sub['dependency'] is None
        if gpu:assert argv[argv.index('-gpu')+1]=='num=1:mode=exclusive_process' and '-w' not in argv
    for phase,job in [('cpu',CPU),('after-cpu',CALLBACK)]:
        text=''.join((a.output/phase/('stdout.'+job+'.log')).read_text().split())
        assert text.count('Subject:Job'+job+':')==1 and 'Successfullycompleted.' in text
        assert 'Exitedwithexit' not in text and 'inqueue<hpc>' in text
    cert=read('cpu-certificate.json');assert cert['status']=='passed' and cert['cases']==29
    assert cert['job_id']==CPU and cert['source_manifest_sha256']==PIN
    for row in cert['proofs']:assert bindings[row['path']]==row
    cpu_scheduler=[]
    for path in (a.output/'after-cpu').glob('cpu-terminal-*.json'):
        r=json.loads(path.read_bytes());compact=''.join(r['stdout'].split())
        if (r['job_id']==CPU and r['returncode']==0 and re.findall(r'Job<([^>]+)>',compact)==[CPU]
                and re.findall(r'Queue<([^>]+)>',compact)==['hpc']
                and re.findall(r'Status<([^>]+)>',compact)==['DONE']):cpu_scheduler.append(path)
    assert cpu_scheduler,'Actual persisted CPU DONE/hpc scheduler proof required'
    cases=list(ET.parse(a.output/'cpu/payload/tests.xml').iter('testcase'))
    assert len(cases)==29 and all(not list(x) for x in cases)
    assert len({(x.get('classname'),x.get('name')) for x in cases})==29
    assert Counter(x.get('classname') for x in cases)=={'tests.test_chemistry_native_qualification.ChemistryNativeQualificationTests':29}
    for arm in ('saved','protenix','boltz'):
        assert read('cpu/payload/startup-'+arm+'/startup.json')['status']=='passed_actual_startup'
    saved=read('cpu/payload/saved/assessment.json')
    assert saved['status']=='diagnostics_completed_original_failure_preserved' and saved['protected_inputs_unchanged']
    assert saved['optimizer_updates']==0 and saved['can_certify_or_restart_original_study'] is False
    after=read('after-cpu/payload/assessment.json')
    assert after['status']=='passed_CPU_and_two_native_component_checks_submitted' and after['optimizer_jobs_submitted'] is False
    assert set(jobs)=={'cpu','after-cpu','gpu-protenix','gpu-boltz'}
    storage=read('after-cpu/complete-storage-dispatch.json')
    gate=storage['whole_gate'];assert storage['status']=='passed_complete_inventory'
    assert storage['files_skipped'] is False and storage['partial_totals_used'] is False
    assert gate['status']=='passed' and gate['available_bytes']>=gate['required_bytes']
    assert gate['reserve_bytes']==4*1024**3 and gate['file_reserve']==100000
    assert gate['new_group_cap_bytes']==1088*1024**2 and gate['forecasts_removed'] is False
    result={'status':'passed_independent_CPU_dispatch_XML_LSF_byte_schema_audit','jobs':jobs,'cases':29,
        'files':len(bindings),'source_manifest_sha256':PIN,'native_pass_established':False,
        'optimizer_submission':False,'settings_changed':False,'saved_results_changed':False,
        'whole_storage':{k:gate[k] for k in ('status','required_bytes','available_bytes','reserve_bytes','file_reserve','new_group_cap_bytes')},
        'bindings':bindings}
    (a.output/'verification.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps({k:x for k,x in result.items() if k!='bindings'},indent=2))


if __name__=='__main__':main()
