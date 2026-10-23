"""Injected only into the new analysis contract; original scientific bytes stay sealed."""


def _science_ops():
    _release(SCIENCE, SCIENCE_MANIFEST)
    previous = list(sys.path)
    try:
        sys.path[:0] = [str(SCIENCE / 'scripts')]
        spec = importlib.util.spec_from_file_location('_lrreset_original_science', SCIENCE / 'scripts/fixed25_lrreset_ops.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous


def _cpu(ops):
    cert = ops.verify_cpu(SCIENCE_MANIFEST, CPU_SHA)
    require(cert['producer_job_id'] == '29792389' and cert['tests'] == 728, 'Wrong original CPU job/count')
    required = {'hpc-contract.xml': 396, 'legacy.xml': 239, 'native-mapping.xml': 34, 'chemistry.xml': 59}
    proofs = {p['path']: p['sha256'] for p in cert['proofs']}
    for name, count in required.items():
        path = ROOT / 'cpu/payload' / name
        require(str(path) in proofs, 'Original CPU omitted raw XML')
        tree = ET.parse(path)
        require(len(list(tree.iter('testcase'))) == count and not any(list(tree.iter(t)) for t in ('failure','error','skipped')), 'Original raw CPU cases failed/changed')
    scheduler=ROOT/'cpu/scheduler-final-0.txt';stdout=ROOT/'cpu/stdout.29792389.log'
    require(str(scheduler) in proofs and str(stdout) in proofs and scheduler_status(scheduler.read_text(),'29792389','hpc')=='DONE','Original CPU final proof missing')
    lsf_success(stdout,'29792389')
    return cert


def _launcher():
    _release(LAUNCH_SOURCE, LAUNCH_MANIFEST)
    previous=list(sys.path)
    try:
        sys.path[:0]=[str(SCIENCE/'scripts'),str(SCIENCE/'src')]
        spec=importlib.util.spec_from_file_location('_lrreset_original_launch',LAUNCH_SOURCE/'all_work3_launch.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module
    finally:sys.path[:]=previous


def _storage(gate, phase):
    projections={a:p['projected_bytes'] for a,p in ARMS.items()}
    require(gate.get('version')=='jq1-lrreset-all-work3-launch-adoption-v1' and gate.get('status')=='passed'
            and gate.get('phase')==phase and gate.get('projections')==projections
            and gate.get('placement')==dict.fromkeys(ARMS,'work3') and gate.get('home_output_writes') is False,
            'Original whole all-work3 storage scope/gate changed')
    require(set(gate['filesystems'])=={'work3'},'Wrong assigned output filesystem')
    budget=gate['filesystems']['work3']
    outstanding=sum(projections[a]-gate['current_outputs'][a] for a in ARMS)
    require(all(type(gate['current_outputs'][a]) is int and 0<=gate['current_outputs'][a]<=projections[a] for a in ARMS)
            and gate['outstanding_native_bytes']==outstanding
            and gate['concurrent_analysis_allowance_bytes']==2684354560
            and budget['status']=='passed' and budget['projected_bytes']==outstanding+2684354560
            and budget['reserve_bytes']==4294967296 and budget['file_reserve']==100000
            and budget['required_bytes']==outstanding+2684354560+4294967296
            and budget['available_bytes_conservative']>=budget['required_bytes']
            and budget['available_files_lower']>=100000,'Original combined budget/reserves differ')
    require(budget['filesystem_inode_reporting'] in ('unreported_all_zero','reported_physical_filesystem_only'), 'Wrong physical inode reporting')


def authenticate_inputs(arm):
    require(arm in ARMS,'Unknown report arm')
    pin=ARMS[arm];ops=_science_ops();_cpu(ops)
    launcher=_launcher();cpu=launcher.verify_cpu(LAUNCH_MANIFEST)
    require(cpu['sha256']==LAUNCH_CPU_SHA,'Original launcher CPU pin differs')
    marker_path,jobs_path=ROOT/'long-dispatch-started.json',ROOT/'long-submissions.json'
    hashed(marker_path,DISPATCH_SHA);hashed(jobs_path,JOBS_SHA)
    marker,jobs=read(marker_path),read(jobs_path)
    inherited=launcher.inherited()
    require(inherited=={k:marker[k] for k in inherited},'Original scientific/native/proof adoption changed')
    require(marker['source_manifest_sha256']==SCIENCE_MANIFEST and marker['launch_manifest_sha256']==LAUNCH_MANIFEST
            and marker['launcher_cpu']['sha256']==LAUNCH_CPU_SHA and jobs['all_three_submitted'] is True
            and {j['job_id'] for j in jobs['jobs']}=={p['job'] for p in ARMS.values()},'Wrong original dispatch/cohort')
    hashed(marker['adoption']['path'],marker['adoption']['sha256']);_storage(marker['storage_gate'],'dispatch')
    for selected,bound in ARMS.items():
        decision_path=ROOT/('decision-'+selected+'.json');hashed(decision_path,bound['decision'])
        decision=read(decision_path);ready=ops.authenticate_ready(selected,SCIENCE_MANIFEST,decision)
        require(decision['native_job']==bound['native_job'] and decision['assessment_sha256']==bound['ready']
                and ready['model_digest']==bound['model'] and ready['context_digest']==bound['context']
                and ready['pocket_definition_sha256']==POCKET and ready['diffusion_gradient'] is False
                and decision['projected_bytes']==bound['projected_bytes'],'Original passed readiness pins differ')
    parent=native_parent(arm);path=parent/'submission.json';hashed(path,pin['submission']);sub=read(path)
    require(sub in jobs['jobs'] and sub['outcome']=='submitted' and sub['job_id']==pin['job']
            and sub['source_manifest_sha256']==SCIENCE_MANIFEST and sub['launch_manifest_sha256']==LAUNCH_MANIFEST
            and sub['launch_cpu_certificate_sha256']==LAUNCH_CPU_SHA and sub['cpu_certificate_sha256']==CPU_SHA
            and sub['config_sha256']==pin['config'] and sub['learning_rate']==pin['rate']
            and sub['output_parent']==str(parent)
            and sub['args']==launcher.long_command(arm,LAUNCH_MANIFEST,LAUNCH_CPU_SHA),'Original actual submission differs')
    hashed(config_path(arm),pin['config'])
    invocation=parent/'payload/driver-invocation.json';gate=parent/'entry-quota/lrreset-storage-gate.json'
    started=invocation.exists() or study_path(arm).exists()
    if started:
        require(pin.get('entry') and pin.get('driver'),'New arm entry requires a fresh immutable report pin')
        hashed(gate,pin['entry']);_storage(read(gate),'entry');hashed(invocation,pin['driver'])
        expected={'args':launcher.driver_command(arm),'source_manifest_sha256':SCIENCE_MANIFEST,
                  'launch_manifest_sha256':LAUNCH_MANIFEST,'launch_cpu_sha256':LAUNCH_CPU_SHA,'scientific_cpu_sha256':CPU_SHA}
        require(read(invocation)==expected,'Original scientific driver/config/readiness/output changed')
    require(sha(ROOT/'cpu/payload/composition_reference.json')==REFERENCE
            and sha(ROOT/'cpu/payload/pocket_definition.json')==POCKET,'Original pocket/reference changed')
    if arm!='protenix':require(sha(PREPARED/'context.pt')==BOLTZ_CONTEXT,'Original detached context bytes changed')
    return {'version':VERSION,'arm':arm,'native_job':pin['job'],'source_manifest_sha256':SCIENCE_MANIFEST,
            'cpu_certificate_sha256':CPU_SHA,'launcher_manifest_sha256':LAUNCH_MANIFEST,
            'launcher_cpu_certificate_sha256':LAUNCH_CPU_SHA,'config_sha256':pin['config'],'readiness_sha256':pin['ready'],
            'submission':hashed(path),'entry':hashed(gate) if started else None,
            'driver_invocation':hashed(invocation) if started else None,'scientific_execution_started':started}
