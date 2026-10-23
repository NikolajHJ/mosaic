"""Independent stdlib byte/JSON/XML proof audit; no scientific frame arithmetic."""
import ast,hashlib,json,re,sys,tarfile,xml.etree.ElementTree as ET
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
ROOT=BASE+'/lrreset-analysis-03'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def actual(raw,job,queue='hpc'):
    compact=re.sub(r'\s+','',raw)
    assert re.findall(r'Job<([^>]+)>',compact)==[job] and re.findall(r'Status<([^>]+)>',compact)==['DONE'] and re.findall(r'Queue<([^>]+)>',compact)==[queue]
def lsf(raw,job):
    assert re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]* Done\s*$',raw,re.M)==[job] and 'Successfully completed.' in raw and 'Exited with exit' not in raw
def main():
    name=sys.argv[1];folder=REPO/'results/hpc'/HERE.name
    delivery=folder/('delivery-'+name);evidence=read(delivery/'evidence.json')
    for n,p in evidence['files'].items():
        f=delivery/n;assert f.resolve().is_relative_to(delivery.resolve()) and f.stat().st_size==p['bytes'] and sha(f)==p['sha256']
    package=read(HERE/'package.json');manifest=package['manifest_sha256']
    assert sha(delivery/'source/source_manifest.json')==manifest
    localmeta=read(folder/'source01/source_manifest.json');assert localmeta==read(delivery/'source/source_manifest.json')
    for n,h in localmeta['files'].items():assert sha(HERE/('core-'+n.replace('/','__')))==h
    archive=folder/(name+'.tar.gz');arc=evidence['report_archive'];assert arc['path']==ROOT+'/'+name+'.tar.gz' and archive.stat().st_size==arc['bytes'] and sha(archive)==arc['sha256']
    extracted=folder/name;download=read(folder/(name+'-download.json'));assert download['sha256']==arc['sha256']
    with tarfile.open(archive) as t:
        members=t.getmembers();names=set()
        for m in members:
            p=(extracted/m.name).resolve();assert p.is_relative_to(extracted.resolve()) and m.name not in names and (m.isdir() or m.isfile());names.add(m.name)
        assert t.extractfile('evidence_manifest.json').read()==(extracted/'evidence_manifest.json').read_bytes()
    em=read(extracted/'evidence_manifest.json')['files'];assert download['verified_files']==len(em)
    assert {f.relative_to(extracted).as_posix() for f in extracted.rglob('*') if f.is_file()}==set(em)|{'evidence_manifest.json'}
    for n,p in em.items():assert (extracted/n).stat().st_size==p['bytes'] and sha(extracted/n)==p['sha256']
    payload=extracted/'20261022-brd4-jq1-lrboost-analysis-01/lrreset-analysis-03'/name/'payload'
    report=payload/'report';receipt=read(report/'report_receipt.json');assessment=read(payload/'assessment.json')
    job=evidence['report_job'];actual(evidence['actual_report_scheduler'],job)
    sub=read(delivery/('analysis/'+name+'/submission.json'));assert sub['job_id']==job and sub['phase']==('final' if name=='combined-report' else 'interim') and sub['arm']=='none' and sub['outcome']=='submitted'
    assert sub['args'][-6:]==[BASE+'/lrreset-analysis-release-03/source',ROOT,manifest,sub['phase'],'none',ROOT+'/'+name]
    lsf((delivery/('analysis/'+name+'/stdout.'+job+'.log')).read_text(encoding='utf-8'),job)
    assert assessment==read(delivery/('analysis/'+name+'/payload/assessment.json')) and receipt==read(delivery/('analysis/'+name+'/payload/report/report_receipt.json'))
    assert assessment['status']=='passed' and assessment['job_id']==job and assessment['report_source_manifest_sha256']==manifest and assessment['report_receipt_sha256']==sha(report/'report_receipt.json')
    assert receipt['version']=='jq1-25-lrreset-saved-report-v1' and receipt['input_jobs']==['29792670','29792671','29792672']
    assert receipt['states'][:2]==[10001,10001] and receipt['included_updates'][:2]==[10000,10000] and not receipt['scientific_passed'] and receipt['campaign_status']=='failed'
    html=(report/'report.html').read_text(encoding='utf-8');assert sha(report/'report.html')==receipt['html_sha256'];data=json.loads(re.search(r'<script\s+id="manifest"\s+type="application/json">(.*?)</script>',html,re.S).group(1))
    bindings=receipt['arm_bindings'];pins=read(delivery/'source/analysis-input-pins.json')['arms']
    for i,arm in enumerate(['protenix','boltz2','boltz2_chemistry']):
        b=bindings[i];s=data['studies'][i];p=pins[arm]
        assert b['arm']==s['arm']==arm and b['job_id']==s['job_id']==p['job'] and b['learning_rate']==p['rate'] and b['config_sha256']==p['config']['sha256'] and b['readiness_sha256']==p['ready']['sha256']
        assert b['source_manifest_sha256']=='c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057' and b['readiness_status']=='passed'
        if i<2:
            expected='passed' if i==0 else 'failed';assert b['overall_status']==b['terminal_status']==expected and b['scientific_passed']==(i==0) and b['all_terminal'] and b['budget_status']=='full'
            c=evidence['collectors'][arm];actual(c['actual_scheduler'],c['job_id']);cc=read(delivery/('analysis/complete-'+arm+'/payload/assessment.json'))
            assert cc==read(payload/('completion-'+arm+'/assessment.json')) and cc['collector_job']==c['job_id'] and cc['native_job']==p['job']
            assert cc['status']=='passed' and cc['scientific_status']==expected and cc['updates']==10000 and cc['states']==10001 and cc['all_probability_frames_verified'] and cc['report_source_manifest_sha256']==manifest
            inv=read(delivery/('analysis/complete-'+arm+'/payload/snapshots.json'));assert [f['step'] for f in inv['frames']]==list(range(10001))
            for field,ending in [('summary','/summary.json'),('case','/case-0/report.json')]:
                match=[q for q in cc['proofs'] if q['path']==p[field]['path']];assert len(match)==1 and match[0]['sha256']==p[field]['sha256']
            log=(delivery/('analysis/complete-'+arm+'/payload/native-stdout.txt')).read_text(encoding='utf-8')
            if i==0:lsf(log,p['job'])
            else:assert 'Subject: Job '+p['job']+':' in log and 'Exited with exit code 1.' in log and b['terminal_error']=='AssertionError: Native affinity E_upper finite differences failed'
        elif s['states']==0:
            assert s['latest_update'] is None and s['chunks']==[] and s['native_status']=='PEND' and b['overall_status']=='queued' and not b['all_terminal']
    cpu=read(payload/'cpu-proof/assessment.json');assert cpu==read(delivery/'analysis/cpu-certificate.json') and cpu['status']=='passed' and cpu['tests']==284 and cpu['source_manifest_sha256']==manifest
    assert sha(payload/'cpu-proof/assessment.json')==assessment['cpu_certificate_sha256']
    x=ET.parse(payload/'cpu-proof/report-tests.xml');cases=list(x.iter('testcase'));assert len(cases)==284 and not any(list(x.iter(k)) for k in ['failure','error','skipped'])
    assert len(cpu['tests_by_module'])==8 and sum(cpu['tests_by_module'].values())==284
    for k,n in cpu['tests_by_module'].items():assert sum(k in c.get('classname','').split('.') for c in cases)==n
    for n,p in evidence['files'].items():
        if n.endswith('/gate.json'):
            gate=read(delivery/n);assert gate['status']=='passed' and gate['reserve_bytes']==4294967296 and gate['file_reserve']==100000 and gate['work3']['available_bytes_conservative']>=gate['work3']['required_bytes']
    assert "byId('stepInput').addEventListener('focus',stop);" in html
    value={'status':'passed','report_job':job,'manifest':manifest,'archive_sha256':arc['sha256'],'report_receipt_sha256':sha(report/'report_receipt.json'),'html_sha256':sha(report/'report.html'),'files':len(em),'cpu_cases':284,'states':receipt['states'],'updates':receipt['included_updates'],'scientific_statuses':[b['overall_status'] for b in bindings],'scope':'Independent stdlib bytes/JSON/XML/actual scheduler/LSF/source/input/capture proof audit; original big scientific files remain authenticated remote references. No local scientific frame calculations.'}
    output=delivery/'independent-verification.json';output.write_text(json.dumps(value,indent=2)+'\n');(HERE/('delivery-verification-'+name+'.json')).write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
if __name__=='__main__':main()
