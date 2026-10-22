"""Independent stdlib byte/XML/schema/LSF audit; no scientific array calculation."""
from pathlib import Path
import hashlib,json,re,xml.etree.ElementTree as ET

H=Path(__file__).resolve().parent
REPO=H.parents[3]
D=REPO/'results/hpc'/H.name/'verification-evidence-01'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
manifest=json.loads((D/'evidence_manifest.json').read_text())
assert len(manifest['files'])==61
for name,v in manifest['files'].items():
    p=D/name
    assert p.resolve().is_relative_to(D.resolve()) and p.is_file() and not p.is_symlink()
    assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256']
origins=json.loads((D/'origin-map.json').read_text())
lookup={remote:D/name for name,remote in origins.items()}
cert=json.loads((D/'02/cpu-certificate.json').read_text())
assert cert['status']=='passed' and cert['producer_job_id']=='29781363' and cert['tests']==723
assert cert['source_manifest_sha256']=='832f833dd486c2ff305f7e3eb33fc775c87810a2e12a8bc597c5df744699c5d3'
for proof in cert['proofs']:assert sha(lookup[proof['path']])==proof['sha256']
groups={}
for name,count in [('hpc-contract',391),('legacy',239),('native-mapping',34),('chemistry',59)]:
    tree=ET.parse(D/'02/cpu/payload'/(name+'.xml'))
    rows=tree.findall('.//testcase')
    assert len(rows)==count and not any(list(tree.iter(t)) for t in ('failure','error','skipped'))
    log=(D/'02/cpu/payload'/(name+'.log')).read_text()
    assert re.search(r'\b'+str(count)+r' passed\b',log)
    groups[name]=count
modules={s for row in ET.parse(D/'02/cpu/payload/hpc-contract.xml').findall('.//testcase') for s in row.get('classname','').split('.')}
assert {'test_fixed25_lrreset_contract','test_fixed25_lrreset_ops','test_fixed25_lrreset_storage','test_fixed25_lrboost_contract','test_fixed25_lrboost_ops','test_fixed25_lrboost_storage'}<=modules
raw=(D/'02/cpu/scheduler-final-0.txt').read_text();compact=re.sub(r'\s+','',raw)
assert re.findall(r'Job<([^>]+)>',compact)==['29781363']
assert re.findall(r'Status<([^>]+)>',compact)==['DONE'] and re.findall(r'Queue<([^>]+)>',compact)==['hpc']
for attempt,role,job,success in [('01','cpu','29781350',False),('01','after-cpu','29781351',False),('02','cpu','29781363',True),('02','after-cpu','29781364',False)]:
    log=(D/attempt/role/f'stdout.{job}.log').read_text()
    assert f'Subject: Job {job}:' in log
    assert ('Successfully completed.' in log) if success else ('Exited with exit code 1.' in log)
    submission=json.loads((D/attempt/role/'submission.json').read_text())
    assert submission['outcome']=='submitted' and submission['job_id']==job and submission['returncode']==0
    args=submission['args']
    assert args[:5]==['bsub','-q','hpc','-n','4' if role=='cpu' else '1']
old=ET.parse(D/'01/cpu/payload/hpc-contract.xml')
assert len(old.findall('.//testcase'))==391 and len(list(old.iter('failure')))==3
assert not list(old.iter('error')) and not list(old.iter('skipped'))
gate=json.loads((D/'02/native-validation-dispatch-quota/lrreset-storage-gate.json').read_text())
g=gate['filesystems']['work3']
assert gate['status']==g['status']=='failed' and gate['phase']=='validate'
assert gate['current_validation_outputs']=={'protenix':0,'boltz2':0,'boltz2_chemistry':0}
assert gate['full_validation_allowance_bytes']==3*1024**3 and gate['concurrent_analysis_allowance_bytes']==2*1024**3+512*1024**2
assert g['required_bytes']==10200547328 and g['available_bytes_conservative']==7836349422
assert g['reserve_bytes']==4*1024**3 and g['file_reserve']==100000
assert g['available_files_lower']>=100000 and g['filesystem_inode_reporting']=='unreported_all_zero'
for attempt in ('01','02'):
    source=json.loads((D/attempt/'source/source_manifest.json').read_text())
    assert len(source['files'])==622
    for arm,rate in [('protenix',.3),('boltz2',.005),('boltz2_chemistry',.005)]:
        name=f'examples/configs/brd4_jq1_25_fixed_lrreset_{arm}_10000.json'
        p=D/attempt/'source'/name
        assert sha(p)==source['files'][name]
        config=json.loads(p.read_text())
        assert config['version']=='mosaic-brd4-jq1-25-fixed-step-lrreset-v1' and config['initial_step_size']==rate
        assert config['steps']==10000 and config['seed']==17 and config['n_atoms']==25
        assert config['acceptance_policy']=='fixed_step' and config['max_backtracks']==1
        if arm=='boltz2_chemistry':assert config['repair_schedule']=={'version':'delayed-geometric-v1','initial':.001,'final':100.,'hold_updates':5000,'total_updates':10000}
record={'status':'passed','scope':'Independent local stdlib byte/XML/schema/actual LSF and persisted CPU scheduler proof verification; no model/numerical execution',
        'files':61,'actual_cpu_job':'29781363','actual_cpu_status':'DONE','callback_job':'29781364','callback_status':'EXIT','callback_failure':'Mandatory native-dispatch storage admission; CPU scientific/model checks not performed',
        'tests':723,'groups':groups,'failures':0,'errors':0,'skips':0,'legacy_deliberately_deselected':3,
        'cpu_certificate_sha256':sha(D/'02/cpu-certificate.json'),'storage_gate_sha256':sha(D/'02/native-validation-dispatch-quota/lrreset-storage-gate.json'),
        'required_bytes':g['required_bytes'],'available_bytes':g['available_bytes_conservative'],'storage_shortfall_bytes':g['required_bytes']-g['available_bytes_conservative'],
        'source01_failure_preserved':True,'new_native_or_long_pass_established':False}
with (H/'cpu-delivery-verification.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps({'status':'passed','tests':723,'verification_sha256':sha(H/'cpu-delivery-verification.json'),'storage_shortfall_bytes':record['storage_shortfall_bytes']}))
