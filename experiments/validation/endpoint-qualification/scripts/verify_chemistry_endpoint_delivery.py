"""Independent stdlib byte/XML/LSF audit of chemistry diagnostic02 delivery."""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


PIN='d3d8678542db3cf6100006f8f9dbd3ce761f169bd904ddcae3d4525a53252f51'
ROOT='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/chemistry-endpoint-recovery-02'
JOBS={'cpu':'29830696','after-cpu':'29830697','gpu-protenix':'29830750','gpu-boltz':'29830751'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('observation',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();v=json.loads(args.observation.read_bytes())
    assert v['jobs']==JOBS
    args.output.mkdir(parents=True,exist_ok=False);files={}
    for absolute,row in v['files'].items():
        assert absolute.startswith(ROOT+'/') and row['binding']['path']==absolute
        rel=absolute[len(ROOT)+1:];path=Path(rel)
        assert not path.is_absolute() and '..' not in path.parts
        b=base64.b64decode(row['base64'],validate=True)
        assert len(b)==row['binding']['bytes'] and hashlib.sha256(b).hexdigest()==row['binding']['sha256']
        q=args.output/path;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b);files[absolute]=row['binding']
    def read(rel):return json.loads((args.output/rel).read_bytes())
    for phase,job in JOBS.items():
        sub=read(phase+'/submission.json')
        assert sub['status']=='submitted' and sub['job_id']==job and sub['source_manifest_sha256']==PIN
        argv=sub['argv'];queue='gpul40s' if phase.startswith('gpu-') else 'hpc'
        assert argv[argv.index('-q')+1]==queue
        if phase.startswith('gpu-'):
            assert argv[argv.index('-n')+1]=='4' and argv[argv.index('-W')+1]=='02:00'
            assert argv[argv.index('-gpu')+1]=='num=1:mode=exclusive_process' and '-w' not in argv
        log=(args.output/phase/('stdout.'+job+'.log')).read_text();c=''.join(log.split())
        assert c.count('Subject:Job'+job+':')==1 and 'Successfullycompleted.' in c
        assert 'Exitedwithexit' not in c and 'inqueue<'+queue+'>' in c
    cases=list(ET.parse(args.output/'cpu/payload/tests.xml').iter('testcase'))
    assert len(cases)==33 and all(not list(c) for c in cases)
    assert dict(Counter(c.get('classname') for c in cases))=={
        'tests.test_chemistry_endpoint_diagnostic.ChemistryEndpointDiagnosticTests':16,
        'tests.test_chemistry_endpoint_ops.ChemistryEndpointOpsTests':17}
    assert len({(c.get('classname'),c.get('name')) for c in cases})==33
    cert=read('cpu-certificate.json');assert cert['status']=='passed' and cert['cases']==33 and cert['source_manifest_sha256']==PIN
    for b in cert['proofs']:assert files[b['path']]==b
    for a in ('saved','protenix','boltz'):
        assert read('cpu/payload/startup-'+a+'/startup.json')['status']=='passed_actual_startup'
    for arm in ('protenix','boltz'):
        top=read('gpu-'+arm+'/payload/assessment.json');r=read('gpu-'+arm+'/payload/native/assessment.json')
        assert top['status']=='diagnostic_evidence_complete_not_admission' and top['optimizer_updates']==0
        assert files[top['native_assessment']['path']]==top['native_assessment']
        assert r['status']=='diagnostics_completed_original_failure_preserved' and r['optimizer_updates']==0
        assert r['can_certify_or_restart_original_study'] is False
        assert r['protected_inputs_unchanged'] is True and r['frozen_unchanged'] is True
        assert r['original_results_changed'] is False and r['optimization_settings_changed'] is False
    rows=read('gpu-protenix/payload/native/assessment.json')['native']['variant_points']
    original=[{k:x for k,x in row.items() if k!='variant'} for row in rows if row['variant']=='original']
    accelerated=[{k:x for k,x in row.items() if k!='variant'} for row in rows if row['variant']=='accelerated']
    assert len(original)==len(accelerated)==11 and original==accelerated
    result={'status':'passed_independent_delivery_byte_XML_LSF_schema_audit',
        'source_manifest_sha256':PIN,'jobs':JOBS,'files':len(files),'cpu_cases':33,
        'original_accelerated_protenix_records_exactly_equal':True,
        'original_failure_status_preserved':True,'optimizer_continuations_submitted':False,
        'native_point_NPZ_files_downloaded':False,
        'scope':'Compact metadata only; raw point bytes require separate inventory authentication',
        'observation_sha256':hashlib.sha256(args.observation.read_bytes()).hexdigest(),
        'bindings':files}
    (args.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='bindings'},indent=2))


if __name__=='__main__':main()
