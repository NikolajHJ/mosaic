"""Deterministic owned-controller editing; local stdlib source transformation."""
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent


def replace_function(text, name, replacement):
    node = next(item for item in ast.parse(text).body if isinstance(item, ast.FunctionDef) and item.name == name)
    lines = text.splitlines(keepends=True)
    return ''.join(lines[:node.lineno - 1]) + replacement.strip() + '\n' + ''.join(lines[node.end_lineno:])


def main():
    path = HERE / 'extended_ops.py'
    text = path.read_text()
    marker = "POCKET='9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43'"
    assert text.count(marker) == 1
    text = text.replace(marker, marker + '''
GIB=1024**3
REVISION='c74c659344123937608077e3e4078f75f09cd743'
STARTUP_VERSION='protenix-actual-command-gpu-premodel-v1'
STARTUP_CASES=28
BASELINE=BASE/'boltz-lr0025-03'
BASELINE_SOURCE=BASE/'boltz-lr0025-release-03/source'
DBMOL=BASE/'boltz-dbmol-pilot-02'
DBMOL_SOURCE=BASE/'boltz-dbmol-pilot-release-02/source'
CONTEXT_READY=OLD/'validate-protenix/payload/study/assessment.json'
CONTEXT_READY_SHA='9674aacd7c8601f755fcad31df6efb9877dc39cd5114d9b1fe3ee867c1dc55dd'
NOISY_ROOT=Path('/work3/s234818/mosaic/runs/20261023-protenix-noisy-01')
NOISY_RELEASE=Path('/work3/s234818/mosaic/releases/20261023-protenix-noisy-01')
NOISY_UPLOAD=Path('/work3/s234818/mosaic/releases/20261023-protenix-noisy-01.tar.gz')
NOISY_MANIFEST='7bb970f48f6f907b7dc05e74a9bc2645a313addf814f08a0eeca11582f3b84ef'
NOISY_STAGE='experiments/brd4-jq1/protenix-noisy-1000/scripts/stage.py'
NOISY_STAGE_SHA='48e3adc68cdb4aed0268b83e9f09508ac7836e0aec14c0113b4b441f947dc6fc'
''')
    text = text.replace("('after-cpu','after-native','after-segment')", "('after-cpu','after-native','after-segment','dispatch')")
    text = replace_function(text, 'command', '''
def command(label,phase,manifest,dependency=None):
    if phase=='dispatch':require(re.fullmatch(r'dispatch-\\d{4}',label) is not None,'Dispatch requires a new unique parent')
    elif phase not in ('cpu','after-cpu'):
        arm,actual,index=decode(label);require(actual==phase,'Parent/phase mismatch')
    else:require(label==phase,'CPU parent mismatch')
    parent=ROOT/label;args=['bsub',*resources(phase),'-J','mosaic-pro50k-'+label,'-oo',str(parent/'stdout.%J.log'),'-eo',str(parent/'stderr.%J.log')]
    if dependency:args+=['-w',dependency]
    wrapper=SOURCE/'scripts/hpc'/('extended_cpu.lsf' if phase=='cpu' else 'extended_stage.lsf')
    args+=['bash','-l',str(wrapper),str(SOURCE),str(ROOT),manifest]
    if phase!='cpu':args += [phase,str(parent)]
    return args
''')
    extra = (HERE / 'controller_additions.py.txt').read_text()
    text = text.replace('def tree(path):', extra + '\n\ndef tree(path):', 1)
    text = replace_function(text, 'storage_gate', '''
def storage_gate(parent,phase,full=None,*,existing_full=False):
    import fixed25_lrreset_storage as storage
    p=Path(parent)/('storage-'+phase);p.mkdir(exist_ok=False)
    r={'status':'failed','phase':phase,'reserve_bytes':4*GIB,'file_reserve':100000,
       'scope':'All original remaining +report2GiB/diagnostic512MiB +both Pro02 +current Boltz30k +DBMol10k +bounded noisyPro; no reservations',
       'stale_scope':'Entire analysis BASE includes original/new Pro/Boltz/DBMol roots/releases/uploads; external noise paths additionally charged'}
    try:
        old={a:tree(OLD/('run-'+a)) for a in OLD_FULL};require(all(0<=old[a]<=OLD_FULL[a] for a in OLD_FULL),'Original projection exceeded')
        own={a:sum(tree(item) for item in ROOT.glob(a+'-*')) for a in ARMS}
        budgets={a:GIB for a in ARMS} if full is None else full
        require(set(budgets)==set(ARMS) and all(type(budgets[a]) is int and budgets[a]>0 and 0<=own[a]<=budgets[a] for a in ARMS),'Extended output exceeded full projection')
        peers,missing=external_budgets(existing_full)
        noisy=tree(NOISY_ROOT);require(0<=noisy<=GIB,'Noisy Pro output exceeds bound')
        noise_mode='pending_validation_allowance_only'
        if NOISY_ROOT.exists():
            manifest=NOISY_RELEASE/'source/source_manifest.json';stage=NOISY_RELEASE/'source'/NOISY_STAGE
            bound=manifest.is_file() and sha(manifest)==NOISY_MANIFEST and stage.is_file() and sha(stage)==NOISY_STAGE_SHA and read(manifest)['files'].get(NOISY_STAGE)==NOISY_STAGE_SHA
            require(bound or not existing_full,'Await authenticated noisy Pro persistent-output cap')
            if bound:noise_mode='authenticated_immutable_one_GiB_cap'
        projected=sum(OLD_FULL[a]-old[a] for a in OLD_FULL)+sum(budgets[a]-own[a] for a in ARMS)+sum(v['remaining'] for v in peers.values())+(GIB-noisy)+2*GIB+512*1024**2
        raw=subprocess.run(['getquota_work3.sh'],capture_output=True,text=True,check=True,timeout=60).stdout
        stale=storage.stale_work_bytes()+tree(NOISY_ROOT)+tree(NOISY_RELEASE)+tree(NOISY_UPLOAD)
        gate=storage.gate_record('work3',raw,projected,stale,shutil.disk_usage(ROOT).free,storage.inode_observation(os.statvfs(ROOT)))
        r.update(status=gate['status'],old_current=old,own_current=own,full_extended=budgets,
                 peers=peers,missing_full_projections=missing,full_long_admission=existing_full,
                 noisy_protenix={'full':GIB,'current':noisy,'remaining':GIB-noisy,'scope':noise_mode},projected_bytes=projected,work3=gate)
    except Exception as e:r['error']=repr(e);raise
    finally:exclusive(p/'gate.json',r)
    require(r['status']=='passed','Whole extended admission failed');return r
''')
    text = replace_function(text, 'all_budgets', '''
def all_budgets():
    values=read(ROOT/'projection.json');require(set(values)==set(ARMS),'Both full Pro native projections required')
    manifest=sha(SOURCE/'source_manifest.json');budgets={}
    for arm in ARMS:
        observed=[]
        for path in sorted(ROOT.glob(arm+'-native-*-certificate.json')):
            cert=read(path);verify_proofs(cert);require(cert['source_manifest_sha256']==manifest,'Own native source differs')
            _,_,index=decode(path.name.removesuffix('-certificate.json'))
            ready_path,ready=native_assessment(arm,index,manifest)
            require(any(proof['path']==str(ready_path) and proof['sha256']==sha(ready_path) for proof in cert['proofs']),'Readiness not certificate-bound')
            observed.append(ready['storage_projection']['maximum_pilot_update_bytes'])
        require(observed and values[arm]['unit_bytes']==max(observed),'Own projection differs from all authenticated pilot maxima')
        budgets[arm]=projection(max(observed),COUNTS[arm]*10000,COUNTS[arm])
    return budgets
''')
    text = replace_function(text, 'queue_native', '''
def queue_native(arm,index,manifest,owner):
    ensure_command_capture(arm,'native',index,manifest,owner)
    native=name(arm,'native',index);callback=name(arm,'after-native',index)
    sub=submit(native,'native',manifest);submit(callback,'after-native',manifest,'ended('+sub['job_id']+')')
''')
    text = text.replace("for arm in ARMS:queue_native(arm,0,manifest)", "for arm in ARMS:queue_native(arm,0,manifest,ROOT/'after-cpu')")
    marker = "    require(counts['legacy']==239 and counts['native-mapping']==34 and counts['chemistry']==59,'Changed inherited suite')"
    assert text.count(marker) == 1
    text = text.replace(marker, marker + '''
    require(counts['hpc-contract']==500 and sum(counts.values())==832,'Exact original832 CPU rows changed')
    xml=ROOT/'cpu/payload/startup.xml';log=xml.with_suffix('.log');parsed=ET.parse(xml);cases=list(parsed.iter('testcase'))
    require(len(cases)==STARTUP_CASES and not any(list(parsed.iter(k)) for k in ['failure','error','skipped'])
            and re.search(r'\\b'+str(STARTUP_CASES)+r'\\s+passed\\b',log.read_text())
            and all('test_extended_startup' in c.get('classname','').split('.') for c in cases),'Allocated startup fixture group incomplete')
    counts['startup']=len(cases)
    proofs += [{'path':str(v),'sha256':sha(v)} for v in [xml,log]]
''')
    text = replace_function(text, 'after_native', '''
def after_native(arm,index,manifest):
    producer=ROOT/name(arm,'native',index)
    proofs=final_job(producer,'gpul40s');p,a=native_assessment(arm,index,manifest)
    verify_gpu_startup(arm,'native',index,manifest,producer)
    proofs += [{'path':str(v),'sha256':sha(v)} for v in [p,producer/'startup-gpu.json',command_path(arm,'native',index),producer/'payload/driver-invocation.json']]
    exclusive(ROOT/(name(arm,'native',index)+'-certificate.json'),{'status':'passed','proofs':proofs,'source_manifest_sha256':manifest,'start':offset(arm,index)})
    path=ROOT/'projection.json';values=read(path) if path.exists() else {}
    values[arm]={'unit_bytes':max(values.get(arm,{}).get('unit_bytes',0),a['storage_projection']['maximum_pilot_update_bytes'])}
    atomic(path,values)
    dispatch(manifest,ROOT/name(arm,'after-native',index))
''')
    text = text.replace("storage_gate(ROOT/name(arm,'after-segment',index),'next-native',all_budgets());queue_native(arm,index+1,manifest)", "storage_gate(ROOT/name(arm,'after-segment',index),'next-native',all_budgets(),existing_full=True);queue_native(arm,index+1,manifest,ROOT/name(arm,'after-segment',index))")
    marker = "    for path in [p/'summary.json',p/'case-0/report.json',p/'terminal_gradient_checks/report.json']:proofs.append({'path':str(path),'sha256':sha(path)})"
    assert text.count(marker) == 1
    text = text.replace(marker, marker + '''
    verify_gpu_startup(arm,'segment',index,manifest,parent)
    for state in case['states']:
        path=p/'case-0'/state['snapshot']
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to((p/'case-0').resolve()) and sha(path)==state['snapshot_sha256'],'Saved state snapshot changed')
        proofs.append({'path':str(path),'sha256':sha(path)})
    proofs += [{'path':str(v),'sha256':sha(v)} for v in [parent/'startup-gpu.json',command_path(arm,'segment',index),parent/'payload/driver-invocation.json']]
''')
    text = text.replace("storage_gate(parent,'entry',all_budgets() if (ROOT/'projection.json').exists() and set(read(ROOT/'projection.json'))==set(ARMS) else None)", "storage_gate(parent,'entry',all_budgets() if (ROOT/'projection.json').exists() and set(read(ROOT/'projection.json'))==set(ARMS) else None,existing_full=phase=='segment')")
    text = replace_function(text, 'main', '''
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--manifest',required=True);p.add_argument('--parent',type=Path);a=p.parse_args()
    require(a.root.resolve()==ROOT and a.source.resolve()==SOURCE and os.environ.get('LSB_JOBID'),'Wrong/unallocated execution')
    parent=ROOT/'cpu' if a.phase=='cpu-entry' else a.parent;require(parent is not None and parent.resolve().is_relative_to(ROOT),'Parent escaped')
    executing(parent,'cpu' if a.phase=='cpu-entry' else a.phase,a.manifest)
    if a.phase=='cpu-entry':storage_gate(parent,'cpu-entry');return
    if a.phase in ('after-cpu','dispatch'):
        with (ROOT/'dispatch.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            if a.phase=='after-cpu':certify_cpu(a.manifest)
            else:cpu_certificate(a.manifest);dispatch(a.manifest,parent)
        return
    arm,phase,index=decode(parent)
    if phase in ('native','segment'):
        verify_command_capture(arm,phase,index,a.manifest)
        ensure_gpu_startup(arm,phase,index,a.manifest,parent)
        produce(parent,phase,arm,index,a.manifest)
    else:
        cpu_certificate(a.manifest)
        with (ROOT/'dispatch.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            if phase=='after-native':after_native(arm,index,a.manifest)
            elif phase=='after-segment':certify_segment(arm,index,a.manifest)
            else:raise ValueError('Unknown phase')
''')
    ast.parse(text)
    path.write_text(text, encoding='utf8', newline='\n')
    cpu = HERE / 'extended_cpu.lsf'
    wrapper = cpu.read_text().replace('set -euo pipefail', 'set -euo pipefail\nunset PYTEST_ADDOPTS', 1)
    wrapper += '''
# Additive CPU command/startup fixtures; original832 rows remain unchanged.
export JAX_ENABLE_X64=0
"$python" -m pytest tests/test_extended_startup.py --rootdir "$source" --import-mode=importlib -o addopts= -o cache_dir="$run/cache-startup" --basetemp "$TMPDIR/startup-contracts" --junitxml="$run/startup.xml" > "$run/startup.log" 2>&1
'''
    cpu.write_text(wrapper, encoding='utf8', newline='\n')


if __name__ == '__main__':
    main()
