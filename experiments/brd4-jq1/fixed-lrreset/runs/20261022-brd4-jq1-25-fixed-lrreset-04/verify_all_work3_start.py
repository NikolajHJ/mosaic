"""Independent stdlib byte/XML/JSON/resource/start audit; no scientific array calculations."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
ROOT='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-04'
SOURCE=ROOT.rsplit('/',1)[0]+'/restart-lrreset-release-04/source'
SCI='c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057'
LAUNCH='40c5055150c06e0aeb9989d33f869aecb6642c3823f6bdc9330e66f41a3bcbbe'
ARMS=['protenix','boltz2','boltz2_chemistry']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--name',required=True);parser.add_argument('--submitted-only',action='store_true');args=parser.parse_args()
    assert re.fullmatch('[a-zA-Z0-9-]+',args.name)
    folder=REPO/'results/hpc/20261022-brd4-jq1-25-fixed-lrreset-04'/args.name
    index=json.loads((folder/'evidence-index.json').read_bytes());by={}
    for row in index['files']:
        p=folder/'payload'/row['path'];assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'];by[row['remote_path']]=p
    def read(p):return json.loads(by[p].read_bytes())
    def digest(p):return hashlib.sha256(by[p].read_bytes()).hexdigest()
    def bound(v):assert digest(v['path'])==v['sha256'];return by[v['path']]
    def terminal(rows,job,status,queue):
        scheduler=lsf=False
        for row in rows:
            p=bound(row)
            if p.suffix not in ['.txt','.log']:continue
            raw=p.read_text();c=re.sub(r'\s+','',raw)
            scheduler|=re.findall(r'Job<([^>]+)>',c)==[job] and re.findall(r'Status<([^>]+)>',c)==[status] and re.findall(r'Queue<([^>]+)>',c)==[queue]
            lsf|=('Subject: Job '+job+':' in raw and ('Successfully completed.' if status=='DONE' else 'Exited with exit code 1.') in raw)
        assert scheduler and lsf,(job,status)
    prior=json.loads((folder/'verification.json').read_bytes());assert prior['status']=='passed' and len(prior['native_passes_verified'])==3 and prior['source_files']==625 and prior['cpu_cases']==728
    ctl=ROOT+'/all-work3-launch-02';launch_source=SOURCE.replace('restart-lrreset-release-04','restart-lrreset-allwork3-launch-02')
    assert digest(launch_source+'/source_manifest.json')==LAUNCH
    package=json.loads((HERE/'all-work3-launch-02/package.json').read_bytes());assert package['manifest_sha256']==LAUNCH
    archive=REPO/'results/hpc/20261022-brd4-jq1-25-fixed-lrreset-04/all-work3-launch-02/source.tar.gz'
    assert archive.stat().st_size==package['archive_bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==package['archive_sha256']
    manifest=read(launch_source+'/source_manifest.json')
    with tarfile.open(archive) as tar:
        rows=tar.getmembers();assert len(rows)==5 and all(x.isfile() and x.name.startswith('source/') and '..' not in Path(x.name).parts for x in rows)
        for name,h in manifest['files'].items():
            assert digest(launch_source+'/'+name)==h==hashlib.sha256(tar.extractfile('source/'+name).read()).hexdigest()
    own=read(ctl+'/cpu-certificate.json');assert own['status']=='passed' and own['tests']==26 and own['manifest']==LAUNCH and own['job']=='29792667'
    terminal(own['proofs'],'29792667','DONE','hpc')
    t=ET.parse(by[ctl+'/cpu/payload/tests.xml']);assert len(list(t.iter('testcase')))==26 and not any(list(t.iter(s)) for s in ['failure','error','skipped'])
    assert all('fixture_all_work3_launch' in x.get('classname','') for x in t.iter('testcase'))
    adoption=read(ctl+'/adoption.json');assert adoption['scientific_source_manifest_sha256']==SCI and adoption['launcher_manifest']==LAUNCH and adoption['launcher_cpu']['sha256']==digest(ctl+'/cpu-certificate.json')
    for p in adoption['readiness'].values():bound(p)
    failure=adoption['failed_home_dispatch'];assert failure['status']=='EXIT' and read(failure['gate']['path'])['status']=='failed' and 'TimeoutExpired' in read(failure['gate']['path'])['error']
    terminal(failure['proofs'],'29792418','EXIT','hpc')
    prev=adoption['previous_launcher_failure'];assert prev['cpu_cases']==20 and prev['callback_status']=='EXIT'
    terminal(prev['proofs'],'29792659','DONE','hpc');terminal(prev['proofs'],'29792660','EXIT','hpc')
    marker=read(ROOT+'/long-dispatch-started.json');assert marker['placement']==dict.fromkeys(ARMS,'work3') and marker['launch_manifest_sha256']==LAUNCH and marker['source_manifest_sha256']==SCI
    bound(marker['adoption'])
    expected={a:read(ROOT+'/validate-'+a+'/payload/study/assessment.json')['storage_projection']['projected_bytes'] for a in ARMS}
    assert adoption['projections']==expected
    def gate(g):
        assert g['status']=='passed' and g['placement']==dict.fromkeys(ARMS,'work3') and g['projections']==expected and g['home_output_writes'] is False
        assert set(g['filesystems'])=={'work3'}
        w=g['filesystems']['work3'];assert w['status']=='passed' and w['reserve_bytes']==4294967296 and w['file_reserve']==100000 and w['available_files_lower']>=100000
        assert g['outstanding_native_bytes']==sum(expected[a]-g['current_outputs'][a] for a in ARMS)
        assert w['required_bytes']==g['outstanding_native_bytes']+2684354560+4294967296
        assert w['available_bytes_conservative']>=w['required_bytes']
        assert w['filesystem_inode_reporting']=='unreported_all_zero' and w['filesystem_inode_observation']=={'f_files':0,'f_ffree':0,'f_favail':0}
    gate(marker['storage_gate'])
    submissions=read(ROOT+'/long-submissions.json');assert submissions['all_three_submitted'] and len(submissions['jobs'])==3
    starts={}
    for row in submissions['jobs']:
        a=row['arm'];assert a in ARMS and row['outcome']=='submitted' and row['source_manifest_sha256']==SCI and row['launch_manifest_sha256']==LAUNCH
        assert row['learning_rate']==(.3 if a=='protenix' else .005) and row['launch_cpu_certificate_sha256']==digest(ctl+'/cpu-certificate.json')
        p=ROOT+'/run-'+a;assert row['output_parent']==p and read(p+'/submission.json')==row
        v=row['args'];assert v[:5]==['bsub','-q','gpul40s','-n','4'] and v[v.index('-W')+1]=='23:00' and v[v.index('-gpu')+1]=='num=1:mode=exclusive_process' and '-w' not in v and 'rusage[mem=8GB]' in v
        assert v[-5:]==[launch_source,LAUNCH,'run',a,digest(ctl+'/cpu-certificate.json')]
        config=read(SOURCE+'/examples/configs/brd4_jq1_25_fixed_lrreset_'+a+'_10000.json')
        assert config['initial_step_size']==row['learning_rate'] and config['steps']==10000 and config['max_backtracks']==1 and config['acceptance_policy']=='fixed_step'
        if a=='boltz2_chemistry':assert config['repair_schedule']=={'version':'delayed-geometric-v1','initial':.001,'final':100.,'hold_updates':5000,'total_updates':10000}
        if args.submitted_only:
            starts[a]={'job':row['job_id'],'scope':'Actual submission verified; entry/optimization start not certified by this audit'}
            continue
        gate(read(p+'/entry-quota/lrreset-storage-gate.json'))
        d=read(p+'/payload/driver-invocation.json');assert d['source_manifest_sha256']==SCI and d['launch_manifest_sha256']==LAUNCH and d['launch_cpu_sha256']==digest(ctl+'/cpu-certificate.json')
        v=d['args'];assert v[1]==SOURCE+'/examples/brd4_25_atoms.py' and v[v.index('--mode')+1]=='run' and v[v.index('--manifest')+1]==SCI and v[v.index('--output')+1]==p+'/payload/study'
        assert v[v.index('--ready-sha256')+1]==digest(ROOT+'/validate-'+a+'/payload/study/assessment.json')
        config=read(v[v.index('--config')+1]);assert config['initial_step_size']==row['learning_rate'] and config['steps']==10000 and config['max_backtracks']==1 and config['acceptance_policy']=='fixed_step'
        prefix=index['committed_prefixes'][a];assert prefix['states']['contiguous'] and prefix['updates']['contiguous'] and prefix['states']['committed_rows']>=2 and prefix['updates']['committed_rows']>=1
        starts[a]={'job':row['job_id'],'states':prefix['states']['committed_rows'],'updates':prefix['updates']['committed_rows'],'entry_gate':'passed','driver_source_manifest':SCI}
    result={'status':'passed','scope':'Independent stdlib bytes/XML/JSON/source/resource/proof audit; no local scientific calculations','all_scientifically_started_verified':not args.submitted_only,'captured_utc':index['captured_utc'],'scientific_cpu_cases':728,'launcher_cpu_cases':26,'source_manifest_sha256':SCI,'launcher_manifest_sha256':LAUNCH,'starts':starts,'saved_results_changed':False,'failed_home_gate_preserved':True,'full_work3_dispatch_gate':marker['storage_gate'],'evidence_index_sha256':hashlib.sha256((folder/'evidence-index.json').read_bytes()).hexdigest()}
    with (folder/('launcher-submission-verification.json' if args.submitted_only else 'launcher-start-verification.json')).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='full_work3_dispatch_gate'},indent=2))


if __name__=='__main__':main()
