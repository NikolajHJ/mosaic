"""Independent local stdlib byte/XML/schema audit; no scientific calculations."""
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
CONTROL = BASE + '/nonchem-stride20-02/startup-fix-03'
LAUNCHER = BASE + '/nonchem-stride20-startup-release-03/source'
MANIFEST = '704fab033d97342a14dc12b58dc32df44874232c5c30aad47c4d02041a502aab'
CPU_PINS = {'boltz30k':'06efbb027f1008a0921494af0e226c3fc7a78c0a8f4ab01b93e8b50f2c042836',
            'dbmol10k':'387070e8fe56b17ddbd7c8e0b82d2d6a44d9aa99e7015d3fbe4757ce82c0819b',
            'protenix50k40k':'ab3148ea6dd8d25361f85e9d8283578e0cf4429844d820a4dab6b03673ef577b'}
SCIENCE = {'boltz30k':('boltz-lr0025-03','c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6'),
           'dbmol10k':('boltz-dbmol-pilot-02','600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed'),
           'protenix50k40k':('protenix-extended-03','da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562')}


def main():
    evidence = ROOT/'results/hpc/20261025-nonchem-stride20-startup-03'/('evidence-'+sys.argv[1])
    data = json.loads((evidence/'metadata.json').read_text())
    assert data['manifest_sha256']==MANIFEST and data['source_seal_verified'] and not data['mutation']
    files=data['files']
    def raw(path):
        rec=files[path];content=(evidence/rec['local_path']).read_bytes()
        assert len(content)==rec['bytes'] and hashlib.sha256(content).hexdigest()==rec['sha256']
        return content
    def read(path):return json.loads(raw(path))
    for path in files:raw(path)
    def lsf(parent,job):
        text=raw(parent+'/stdout.'+job+'.log').decode()
        assert re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$',text,re.M)==[(job,'Done')]
        assert 'Successfully completed.' in text and 'Exited with exit' not in text
    def command_check(parent,sub):
        assert sub['outcome']=='submitted' and sub['launcher_manifest_sha256']==MANIFEST
        args=sub['args'];phase=sub['phase'];profile=sub['profile']
        assert args[0]=='bsub' and args[args.index('-q')+1]==('gpul40s' if phase in ['native','segment','noise-gpu'] else 'hpc')
        assert args[-8:]==['bash','-l',LAUNCHER+'/stage.lsf',LAUNCHER,MANIFEST,profile,phase,parent]
        assert args[args.index('-n')+1]==('4' if phase in ['native','segment','noise-cpu','noise-gpu'] else '2')
        assert args[args.index('-W')+1]==('23:00' if phase=='segment' else '02:00' if phase in ['native','noise-gpu'] else '01:00' if phase=='noise-cpu' else '00:30')
        assert ('num=1:mode=exclusive_process' in args)==(phase in ['native','segment','noise-gpu'])
        if sub['dependency']:
            assert re.fullmatch(r'ended\(\d+\)',sub['dependency']) and args[args.index('-w')+1]==sub['dependency']
        else:assert '-w' not in args
    actual=[]
    for path in files:
        if path.endswith('/submission.json'):
            sub=read(path)
            if sub.get('launcher_manifest_sha256')==MANIFEST:
                command_check(path.removesuffix('/submission.json'),sub)
                actual.append({'job_id':sub['job_id'],'profile':sub['profile'],'phase':sub['phase'],'parent':path.removesuffix('/submission.json')})
    for name,(root,source) in SCIENCE.items():
        units=data['native_units'][name]
        assert units['status']=='passed'
        proofs={p['path']:p['sha256'] for p in units['proofs']}
        assert proofs[BASE+'/'+root+'/cpu-certificate.json']==CPU_PINS[name]
        assert source in proofs.values()
        total,segments={'boltz30k':(30000,3),'dbmol10k':(10000,4),'protenix50k40k':(40000,4)}[name]
        assert units['full']==2*units['unit_bytes']*(total//20+3*segments)+segments*1024**3
        assert units['original_dense_full']==2*units['unit_bytes']*total+segments*1024**3
    gates=[]
    for path in files:
        if '/selected-storage-' not in path or not path.endswith('/gate.json'):continue
        gate=read(path);g=gate['work3']
        assert gate['status']==g['status']=='passed'
        assert gate['matrix_stride']==20 and gate['loss_stride']==1
        assert g['reserve_bytes']==4*1024**3 and g['file_reserve']==100000
        assert g['required_bytes']==g['projected_bytes']+4*1024**3<=g['available_bytes_conservative']
        assert g['available_files_lower']>=100000
        assert set(gate['selected'])==set(SCIENCE)
        assert all(v['full']==data['native_units'][k]['full'] and v['remaining']==v['full']-v['current'] for k,v in gate['selected'].items())
        assert gate['retained_learned_prior_allowance_bytes']==gate['new_metadata_release_allowance_bytes']==1024**3
        assert all(not row['future_admission'] and row['production_absent'] for row in gate['deferred_arms'])
        if g['filesystem_inode_reporting']=='unreported_all_zero':
            assert set(g['filesystem_inode_observation'].values())=={0}
        gates.append(path)
    cert_path=CONTROL+'/cpu-certificate.json'
    cpu_passed=cert_path in files
    startup=[]
    if cpu_passed:
        cert=read(cert_path)
        assert cert['status']=='passed' and cert['cases']==78 and cert['suite']=='selected-nonchemistry-stride20-metadata-v2' and cert['launcher_manifest_sha256']==MANIFEST
        for proof in cert['proofs']:
            assert files[proof['path']]['sha256']==proof['sha256']
        xml=raw(CONTROL+'/cpu/payload/selected.xml')
        cases=list(ET.fromstring(xml).iter('testcase'))
        assert len(cases)==78
        assert sum('test_selected_launch' in c.get('classname','') for c in cases)==55
        assert sum('test_retention' in c.get('classname','') for c in cases)==23
        assert not any(c.find(k) is not None for c in cases for k in ['failure','error','skipped'])
        assert re.search(r'\b78 passed\b',raw(CONTROL+'/cpu/payload/selected.log').decode())
        cpu=read(CONTROL+'/cpu/submission.json');lsf(CONTROL+'/cpu',cpu['job_id'])
        scheduler=''.join(raw(p['path']).decode() for p in cert['proofs'] if 'scheduler-final-' in p['path'])
        compact=re.sub(r'\s+','',scheduler)
        assert 'Job<'+cpu['job_id']+'>' in compact and 'Status<DONE>' in compact and 'Queue<hpc>' in compact
    for name,(root,source) in SCIENCE.items():
        base=BASE+'/'+root
        marker=base+'/nonchem-stride20-v3-owner.json'
        if marker in files:
            value=read(marker)
            assert value['status']=='passed' and value['kind']=='selected_owner' and value['profile']==name and value['launcher_manifest_sha256']==MANIFEST
        if name=='protenix50k40k':path=base+'/continuation-stride20-v3-command-segment-0000.json'
        elif name=='boltz30k':path=base+'/boltz2-stride20-v3-segment-0000-certificate.json'
        else:path=base+'/pilot-stride20-v3-segment-0000-startup.json'
        if path not in files:continue
        value=read(path)
        assert value['source_manifest_sha256']==source and value['phase']=='segment' and value['index']==0
        if name=='boltz30k':result=read(base+'/boltz2-stride20-v3-segment-0000.json')
        else:result=value['result']
        assert result['model_loaded'] is False and result['scientific_readiness_passed'] is False and result['optimizer_updates']==0
        if name=='protenix50k40k':
            assert result['kind']=='cpu_generated_command_only' and result['actual_driver_executed'] is False
        else:assert result['status']=='startup_only_passed'
        dispatch=read(base+'/dispatch-0004/submission.json');lsf(base+'/dispatch-0004',dispatch['job_id'])
        startup.append({'profile':name,'scope':'CPU_command_metadata_only_GPU_probe_pending' if name=='protenix50k40k' else 'actual_CPU_premodel_startup_only',
                        'proof':path,'sha256':files[path]['sha256'],'dispatch_job_id':dispatch['job_id']})
    result={'status':'passed','scope':'independent_stdlib_bytes_XML_schema_resources_scope_and_reserve_audit',
            'utc':data['utc'],'compact_files':len(files),'source_manifest_sha256':MANIFEST,
            'selected_metadata_cpu_passed':cpu_passed,'cpu_cases':78 if cpu_passed else None,
            'policy':{'matrix_stride':20,'loss_stride':1,'keeps_initial_terminal_boundaries_rolling_latest_and_best':True},
            'actual_submissions':actual,'passed_capacity_gates':gates,'prefixes':data['prefixes'],
            'original_startup_checks':startup,'preserved_independent_noisy_gpu_job':'29809228',
            'scientific_original_sources_settings_results_changed':False,'all_original_arrays_downloaded':False,
            'original_scientific_proofs': 'Exact source/CPU/native proof bytes rehashed remotely; CPU pins compared independently here',
            'monitor_remains_paused':True}
    out=evidence/'verification.json'
    assert not out.exists()
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':result['status'],'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'cpu_passed':cpu_passed,'submissions':actual},indent=2))


if __name__=='__main__':main()
