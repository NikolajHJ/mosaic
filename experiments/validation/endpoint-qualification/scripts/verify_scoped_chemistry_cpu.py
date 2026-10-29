"""Independent stdlib audit of the nested chemistry component CPU admission."""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import xml.etree.ElementTree as ET

OWNER=Path(__file__).resolve().parents[1]
REPO=OWNER.parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--label',required=True);a=p.parse_args()
    assert a.tag.replace('-','').isalnum() and a.label.replace('-','').isalnum()
    local=REPO/'results/hpc'/a.tag;run=OWNER/'runs'/a.tag;source=local/'source01'
    def read(p):return json.loads(Path(p).read_bytes())
    def sha(b):return hashlib.sha256(b).hexdigest()
    manifest=read(source/'source_manifest.json');pin=sha((source/'source_manifest.json').read_bytes());plan=read(source/'plan.json')
    assert len(manifest['files'])==13 and manifest['can_submit_continuation'] is False
    for name,digest in manifest['files'].items():assert sha((source/name).read_bytes())==digest
    value=read(local/a.label/'stdout.json');assert value['source_verified'] and value['source_manifest_sha256']==pin
    jobs=value['jobs'];assert set(jobs)=={'cpu','after-cpu','gpu-protenix'}
    root=PurePosixPath(plan['output_root']);records={}
    for path,r in value['files'].items():
        b=base64.b64decode(r['base64'],validate=True);assert len(b)==r['binding']['bytes'] and sha(b)==r['binding']['sha256']
        assert r['binding']['path']==path and PurePosixPath(path).is_relative_to(root);records[path]=(r['binding'],b)
    def data(rel):return json.loads(records[str(root/rel)][1])
    initial=read(run/'submission.json');remote=PurePosixPath(initial['cpu']['argv'][-5]);number=root.name.rsplit('-',1)[1]
    assert remote.parent.name=='chemistry-native-qualification-release-'+number
    source_capacity=plan['current_capacity_source'];assert source_capacity['files']==50
    assert root.parent==remote.parent.parent
    for phase,jid in jobs.items():
        sub=data(phase+'/submission.json');argv=sub['argv'];gpu=phase=='gpu-protenix';callback=phase=='after-cpu'
        assert sub['status']=='submitted' and sub['job_id']==jid and sub['source_manifest_sha256']==pin
        assert argv[-8:]==['bash','-l',str(remote/'scripts/chemistry_endpoint_stage.lsf'),str(remote),pin,phase,'--capacity-source',source_capacity['path']]
        assert argv[argv.index('-J')+1]=='mosaic-chem-qualification'+number+'-'+phase
        assert argv[argv.index('-q')+1]==('gpul40s' if gpu else 'hpc') and argv[argv.index('-n')+1]==('1' if callback else '4')
        assert argv[argv.index('-W')+1]==('02:00' if gpu else '00:20' if callback else '01:00')
        assert argv[argv.index('-R',argv.index('-R')+1)+1]==('rusage[mem=8GB]' if gpu else 'rusage[mem=2GB]' if callback else 'rusage[mem=4GB]')
        assert argv.count('-env')==1 and '~CUDA_VISIBLE_DEVICES' in argv[argv.index('-env')+1]
        assert (sub['dependency']=='ended('+jobs['cpu']+')' and argv[argv.index('-w')+1]==sub['dependency']) if callback else ('-w' not in argv and sub['dependency'] is None)
        if gpu:assert argv[argv.index('-gpu')+1]=='num=1:mode=exclusive_process'
    scheduler=value['scheduler'];assert scheduler['returncode']==0
    states=dict(re.findall(r'Job<([0-9]+)>.*?Status<([^>]+)>',''.join(scheduler['stdout'].split())))
    for phase in ('cpu','after-cpu'):
        jid=jobs[phase];assert states[jid]=='DONE'
        lsf=''.join(records[str(root/phase/('stdout.'+jid+'.log'))][1].decode().split())
        assert lsf.count('Subject:Job'+jid+':')==1 and 'Successfullycompleted.' in lsf and 'Exitedwithexit' not in lsf and lsf.count('inqueue<hpc>')==1
    cert=data('cpu-certificate.json');assert cert['status']=='passed' and cert['job_id']==jobs['cpu'] and cert['source_manifest_sha256']==pin
    for proof in cert['proofs']:assert records[proof['path']][0]==proof
    cases=list(ET.fromstring(records[str(root/'cpu/payload/tests.xml')][1]).iter('testcase'))
    assert all(not list(c) for c in cases) and len(cases)==cert['cases']==sum(manifest['cpu_counts'].values())==43
    assert dict(Counter(c.get('classname') for c in cases))==manifest['cpu_counts'] and len({(c.get('classname'),c.get('name')) for c in cases})==43
    for arm in ('saved','protenix','boltz'):assert data('cpu/payload/startup-'+arm+'/startup.json')['status']=='passed_actual_startup'
    saved=data('cpu/payload/saved/assessment.json')
    assert saved['status']=='diagnostics_completed_original_failure_preserved' and saved['protected_inputs_unchanged'] and saved['optimizer_updates']==0 and saved['can_certify_or_restart_original_study'] is False
    after=data('after-cpu/payload/assessment.json');assert after['status']=='passed_CPU_and_Pro_native_component_check_submitted' and after['optimizer_jobs_submitted'] is False
    gates=[json.loads(b) for path,(_,b) in records.items() if '/after-cpu/rolling-capacity-dispatch-' in path and path.endswith('/assessment.json')]
    assert len(gates)==1;gate=gates[0]
    assert gate['status']=='passed' and gate['reserve_bytes']==4*1024**3 and gate['file_reserve']>=100000 and gate['byte_margin']>=0 and gate['file_margin']>=0
    assert gate['source_manifest_sha256']==source_capacity['manifest_sha256']
    result={'status':'passed_independent_scoped_chemistry_CPU_source_byte_XML_LSF_startup_dispatch_audit','cases':43,'files_verified':len(records),
       'CPU_job':jobs['cpu'],'controller_job':jobs['after-cpu'],'actual_native_job':jobs['gpu-protenix'],'source_manifest_sha256':pin,
       'CPU_certificate_sha256':sha(records[str(root/'cpu-certificate.json')][1]),
       'whole_dispatch_storage':{k:gate[k] for k in ('required_bytes','available_bytes','reserve_bytes','file_reserve')},
       'native_pass_established':False,'optimizer_jobs_submitted':False,'saved_results_changed':False,'local_scientific_evaluation':False}
    with (run/'cpu-verification.json').open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
    campaign=read(run/'campaign.json');campaign.update(status='CPU43_controller_passed_native_component_job_submitted_pending_own_gates',jobs=jobs,cpu_verification=result)
    (run/'campaign.json').write_text(json.dumps(campaign,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
