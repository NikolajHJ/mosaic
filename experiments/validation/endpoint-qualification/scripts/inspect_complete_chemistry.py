"""Collect/audit complete chemistry CPU metadata; never submit or evaluate models."""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import xml.etree.ElementTree as ET

OWNER=Path(__file__).resolve().parents[1]
REPO=OWNER.parents[2]
HELPER=r'C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe'


def read(path):return json.loads(Path(path).read_bytes())
def sha(data):return hashlib.sha256(data).hexdigest()
def raw(value):return (json.dumps(value,indent=2,sort_keys=True)+'\n').encode()


def collect(local,label):
    prep=read(local/'source01/preparation.json');out=local/label;out.mkdir()
    program='ROOT='+repr(prep['root'])+'\nSOURCE='+repr(prep['source_root'])+'\n'+r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,subprocess
root=Path(ROOT);source=Path(SOURCE);files={};jobs={}
def keep(p):
 assert p.is_file() and not p.is_symlink() and p.stat().st_size<4000000
 b=p.read_bytes();files[str(p)]={'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}
for p in root.rglob('submission.json'):
 assert not p.is_symlink();v=json.loads(p.read_bytes());jobs[p.parent.relative_to(root).as_posix()]=v['job_id']
 for f in (p,p.parent/('stdout.'+v['job_id']+'.log'),p.parent/('stderr.'+v['job_id']+'.log'),p.parent/'payload/assessment.json'):
  if f.exists() and f.stat().st_size<4000000:keep(f)
cert=root/'cpu-certificate.json'
if cert.exists():
 keep(cert)
 for proof in json.loads(cert.read_bytes())['proofs']:
  keep(Path(proof['path']));assert all(files[proof['path']][k]==v for k,v in proof.items())
for p in (root/'cpu/payload').rglob('*'):
 if p.is_file() and p.suffix in ('.json','.xml','.log') and p.stat().st_size<4000000:keep(p)
for phase in ('cpu','after-cpu'):
 for p in (root/phase).glob('whole-capacity-*/chemistry-assessment.json'):keep(p)
manifest=(source/'source_manifest.json').read_bytes();m=json.loads(manifest)
for n,d in m['files'].items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==d
q=subprocess.run(['bjobs','-a','-json','-o','jobid stat queue job_name command',*jobs.values()],capture_output=True,text=True,timeout=40)
print(json.dumps({'observed_utc':datetime.now(timezone.utc).isoformat(),'jobs':jobs,'files':list(files.values()),
 'source_verified':True,'source_manifest_sha256':hashlib.sha256(manifest).hexdigest(),
 'scheduler':{'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr},'mutations':False}))
'''
    (out/'program.py').write_bytes(program.encode())
    run=subprocess.run([HELPER,'exec','--timeout','100','bash -lc '+shlex.quote(
        '/work3/s234818/mosaic/envs/protenix-tiny/bin/python -B -')],input=program.encode(),capture_output=True,timeout=120)
    (out/'stdout.json').write_bytes(run.stdout);(out/'stderr.txt').write_bytes(run.stderr)
    (out/'transport.json').write_bytes(raw({'returncode':run.returncode}))
    if run.returncode:raise RuntimeError('Collection failed; preserve partial evidence, never infer job failure')
    value=json.loads(run.stdout)
    print(json.dumps({'observed_utc':value['observed_utc'],'jobs':value['jobs'],
                     'states':{r['JOBID']:r['STAT'] for r in json.loads(value['scheduler']['stdout']).get('RECORDS',[])}}))


def verify(local,label,run):
    packet=local/'source01';source=packet/'source';prep=read(packet/'preparation.json')
    manifest=read(source/'source_manifest.json');pin=sha((source/'source_manifest.json').read_bytes())
    assert pin==prep['manifest_sha256'] and len(manifest['files'])==10
    for n,d in manifest['files'].items():assert sha((source/n).read_bytes())==d
    plan=read(source/'plan.json');root=PurePosixPath(plan['root']);remote=PurePosixPath(plan['source_root'])
    prior=REPO/'results/hpc/20261029-precision-continuation-26/source01'
    cap=manifest['capacity_source'];assert sha((prior/'source_manifest.json').read_bytes())==cap['manifest_sha256']
    inherited=read(prior/'source_manifest.json')['cpu_counts'];expected={**inherited,**manifest['cpu_counts']}
    value=read(local/label/'stdout.json');assert value['source_verified'] and value['source_manifest_sha256']==pin
    records={}
    for r in value['files']:
        b=base64.b64decode(r['base64'],validate=True)
        assert len(b)==r['bytes'] and sha(b)==r['sha256'] and r['path'] not in records
        assert PurePosixPath(r['path']).is_relative_to(root);records[r['path']]=(r,b)
    def data(rel):return json.loads(records[str(root/rel)][1])
    jobs=value['jobs'];cert=data('cpu-certificate.json')
    assert cert['status']=='passed' and cert['source_manifest_sha256']==pin
    for proof in cert['proofs']:assert {k:records[proof['path']][0][k] for k in ('path','bytes','sha256')}==proof
    def rows(name):
        xml=ET.fromstring(records[str(root/'cpu/payload'/name)][1])
        assert not any(list(xml.iter(k)) for k in ('failure','error','skipped'))
        return [(r.get('classname'),r.get('name')) for r in xml.iter('testcase')]
    merged=rows('tests.xml');original=rows('torch.xml')+rows('jax.xml')+rows('chemistry.xml')
    assert len(merged)==len(set(merged))==sum(expected.values())==cert['cases']==997
    assert Counter(merged)==Counter(original)
    attributed=Counter()
    for name,_ in merged:
        matches=[m for m in expected if m.rsplit('.',1)[-1] in name.split('.')]
        assert len(matches)==1;attributed[matches[0]]+=1
    assert dict(attributed)==expected and value['scheduler']['returncode']==0
    sched={r['JOBID']:r for r in json.loads(value['scheduler']['stdout'])['RECORDS']}
    for phase in ('cpu','after-cpu'):
        jid=jobs[phase];sub=data(phase+'/submission.json');argv=sub['argv'];callback=phase=='after-cpu'
        assert sub['status']=='submitted' and sub['source_manifest_sha256']==pin and sub['job_id']==jid
        assert sched[jid]['STAT']=='DONE' and sched[jid]['QUEUE']=='hpc'
        assert argv[-9:]==['bash','-l',str(remote/'scripts/chemistry_continuation_stage.lsf'),str(remote),pin,phase,'all','--capacity-source',cap['path']]
        assert argv[argv.index('-q')+1]=='hpc' and argv[argv.index('-n')+1]==('1' if callback else '4')
        assert argv[argv.index('-W')+1]==('00:30' if callback else '01:00')
        assert argv[argv.index('-J')+1]=='mosaic-chem-complete01-'+phase
        assert ('rusage[mem=2GB]' if callback else 'rusage[mem=4GB]') in argv and '-gpu' not in argv
        dep='ended('+jobs['cpu']+')' if callback else None
        assert sub['dependency']==dep and (argv[argv.index('-w')+1]==dep if callback else '-w' not in argv)
        text=' '.join(records[str(root/phase/('stdout.'+jid+'.log'))][1].decode().split())
        assert text.count('Subject: Job '+jid+':')==1 and 'queue <hpc>' in text
        assert text.count('Successfully completed.')==1 and 'Exited with exit code' not in text
    assessment=data('cpu/payload/assessment.json')
    assert assessment['status']=='passed' and assessment['cases']==997 and assessment['job_id']==jobs['cpu']
    assert data('cpu/payload/actual-startups.json')['status']=='passed_metadata_only'
    for arm in ('protenix','boltz'):
        for phase in ('validate','segment'):
            report=data('cpu/payload/startup-'+arm+'-'+phase+'/endpoint-startup.json')
            assert report['status']=='passed_metadata_only' and report['updates']==0 and report['numerical_setup_entered'] is False
    assert data('after-cpu/payload/assessment.json')=={'status':'combined_qualification_jobs_submitted','optimizer_submitted':False}
    for arm in ('protenix','boltz'):
        native=data(arm+'-validate/submission.json');callback=data(arm+'-after-validate/submission.json')
        assert native['job_id']==jobs[arm+'-validate'] and native['source_manifest_sha256']==pin
        argv=native['argv'];assert argv[argv.index('-q')+1]=='gpul40s' and argv[argv.index('-n')+1]=='8'
        assert 'rusage[mem=4GB]' in argv and argv[argv.index('-gpu')+1]=='num=1:mode=exclusive_process:mps=yes'
        dep='ended('+plan['components']['protenix']['job_id']+')' if arm=='protenix' else None
        assert native['dependency']==dep and callback['dependency']=='ended('+native['job_id']+')'
        assert argv[argv.index('-W')+1]=='02:00'
        assert argv[-9:]==['bash','-l',str(remote/'scripts/chemistry_continuation_stage.lsf'),str(remote),pin,'validate',arm,'--capacity-source',cap['path']]
        assert argv[argv.index('-J')+1]=='mosaic-chem-complete01-'+arm+'-validate'
        assert callback['source_manifest_sha256']==pin and callback['job_id']==jobs[arm+'-after-validate']
    gates=[json.loads(b) for name,(_,b) in records.items() if '/after-cpu/whole-capacity-dispatch-' in name and name.endswith('/chemistry-assessment.json')]
    assert len(gates)==1 and gates[0]['status']=='passed' and gates[0]['reserve_bytes']==4*1024**3
    assert gates[0]['available_bytes']>=gates[0]['required_bytes'] and gates[0]['available_files']>=gates[0]['required_files']
    result={'status':'passed_independent_complete_chemistry_CPU_byte_XML_LSF_startup_submission_audit',
        'observed_utc':value['observed_utc'],'cases':997,'modules':len(expected),'files_verified':len(records),
        'source_manifest_sha256':pin,'CPU_certificate_sha256':sha(records[str(root/'cpu-certificate.json')][1]),
        'jobs':jobs,'native_pass_established':False,'optimizer_jobs_submitted':False,
        'saved_results_changed':False,'local_scientific_evaluation':False}
    with (run/'cpu-verification.json').open('xb') as f:f.write(raw(result))
    campaign=read(run/'campaign.json');campaign.update(status='CPU997_controller_passed_combined_validators_submitted_not_passed',jobs=jobs,cpu_verification=result)
    (run/'campaign.json').write_bytes(raw(campaign));print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('collect','verify'))
    p.add_argument('--tag',default='20261029-chemistry-continuation-04');p.add_argument('--label',required=True);a=p.parse_args()
    assert a.tag.replace('-','').isalnum() and a.label.replace('-','').isalnum()
    local=REPO/'results/hpc'/a.tag
    if a.action=='collect':collect(local,a.label)
    else:verify(local,a.label,OWNER/'runs'/a.tag)


if __name__=='__main__':main()
