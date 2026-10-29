"""Metadata-only adapter for a new native-component qualification capsule.

Reuse authenticated diagnostic02 controller machinery in a fresh namespace.
Explicit root/command/storage routing is private; its sealed files stay exact.
There is deliberately no optimizer or combined-chemistry admission branch.
"""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

BASE=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
PARENT=BASE/'chemistry-endpoint-recovery-release-02/source'
PARENT_PIN='d3d8678542db3cf6100006f8f9dbd3ce761f169bd904ddcae3d4525a53252f51'
ROOT=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-precision-continuation-25/chemistry-native-qualification-08')
SOURCE=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-precision-continuation-25/chemistry-native-qualification-release-08/source')
CAP=2*512*1024**2+64*1024**2
NATIVE_ARMS=('protenix',)
# Bound only after current cross-campaign resource16 is sealed and05's writer
# descriptor is validated. Never stage/admit with qualification04's old scope.
CURRENT_CAPACITY_SOURCE={'path':str(BASE/'nonchem-precision-release-25/source'),'manifest_sha256':'62a3db1b97b81dd0ef6911ee93f11eb1379da794e01f86f035c0c036e1822f8b','files':50}
ATOMIC_REFERENCE=BASE/'nonchem-endpoint-continuation-09/boltz-segment-10000/payload/study/case-0/references/latest-reference.json.tmp'
INVENTORY_ATTEMPTS=3


def capacity_operator():
    """Authenticate the already sealed/passed current ledger, never its old gates."""
    if CURRENT_CAPACITY_SOURCE is None:raise ValueError('Current scoped capacity source missing')
    source=Path(CURRENT_CAPACITY_SOURCE['path'])
    spec=importlib.util.spec_from_file_location('_chem06_current_capacity',source/'scripts/precision_continuation_ops.py')
    op=importlib.util.module_from_spec(spec);sys.modules[spec.name]=op;spec.loader.exec_module(op)
    manifest=op.verify(source,CURRENT_CAPACITY_SOURCE['manifest_sha256'])
    op.require(len(manifest['files'])==50,'Current resource source50 required')
    op.certificate(CURRENT_CAPACITY_SOURCE['manifest_sha256'],manifest)
    op.require(ROOT.parent==op.ROOT and SOURCE.parent.parent==op.ROOT,
               'Chemistry controls/source must be inside the funded2GiB current control tree')
    return op


def current_capacity(parent,phase):
    op=capacity_operator()
    value=op.storage(parent,phase)
    op.require(value['status']=='passed' and value['reserve_bytes']==4*1024**3,
               'Current shared VAE/embedding/own controls/remaining segment budgets must pass')
    return value


def register_shared_job(op,phase,record):
    """Exact nested writer under existing group cap; no new scientific allocation."""
    import fcntl,json
    scope=capacity_operator();name=(ROOT/phase).relative_to(scope.ROOT).as_posix()
    op.require(record['status']=='submitted' and record['argv']==op.command(phase,record['source_manifest_sha256'],record.get('dependency')),
               'Only exact authenticated chemistry commands may enter current group registry')
    with (scope.ROOT/'jobs.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=scope.ROOT/'jobs.json';jobs=scope.read(path)
        op.require(name not in jobs,'Existing/unknown chemistry registry cannot be replaced')
        jobs[name]=record['job_id'];temporary=path.with_suffix('.partial')
        scope.write(temporary,jobs);temporary.replace(path)


def complete_inventory(operation,parent,phase,write):
    """Retry the WHOLE unchanged gate after one authenticated atomic-file race.

    Never skip a file, reuse an incomplete total, overwrite a failed attempt,
    retry capacity rejection, or suppress a missing protected input. Each new
    attempt obtains a fresh quota and complete inventory through the original
    operation. Only the exact observed owned atomic temp path is retryable.
    """
    for number in range(1,INVENTORY_ATTEMPTS+1):
        attempt=Path(parent)/f'complete-inventory-{phase}-{number:02d}'
        attempt.mkdir(exist_ok=False)
        try:
            value=operation(attempt,phase)
        except FileNotFoundError as error:
            if error.filename is None or Path(error.filename)!=ATOMIC_REFERENCE:raise
            write(attempt/'incomplete-inventory.json',{'status':'failed_incomplete_inventory_not_admission',
                'missing_path':error.filename,'error':repr(error),'attempt':number,
                'next_action':'repeat_entire_original_gate' if number<INVENTORY_ATTEMPTS else 'fail_closed',
                'files_skipped':False,'partial_totals_used':False})
            if number==INVENTORY_ATTEMPTS:raise
        else:
            if not isinstance(value,dict) or value.get('status')!='passed':
                raise ValueError('A complete passed original storage gate is required')
            write(Path(parent)/('complete-storage-'+phase+'.json'),{'status':'passed_complete_inventory',
                'successful_attempt':number,'scope':str(attempt),'whole_gate':value,
                'files_skipped':False,'partial_totals_used':False})
            return value


def inherited():
    with (PARENT/'source_manifest.json').open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==PARENT_PIN
    spec=importlib.util.spec_from_file_location('_qualification02_metadata',PARENT/'scripts/chemistry_endpoint_ops.py')
    op=importlib.util.module_from_spec(spec);sys.modules[spec.name]=op;spec.loader.exec_module(op)
    op.verify(PARENT,PARENT_PIN,10)
    op.ROOT=ROOT;op.SOURCE=SOURCE;op.CAP=CAP
    original_command=op.command
    def command(phase,pin,dependency=None):
        argv=original_command(phase,pin,dependency)
        argv[argv.index('-J')+1]='mosaic-chem-qualification08-'+phase
        argv[1:1]=['-env','all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']
        argv+=['--capacity-source',str(BASE/'nonchem-precision-release-25/source')]
        return argv
    op.command=command
    original_submit=op.submit
    def submit(phase,pin,dependency=None):
        record=original_submit(phase,pin,dependency)
        register_shared_job(op,phase,record)
        return record
    op.submit=submit
    op.storage=lambda parent,phase:current_capacity(parent,phase)

    return op


def execute(phase,pin):
    op=inherited();manifest=op.verify(SOURCE,pin);plan=op.external(manifest);parent=op.own(phase,pin)
    op.require(phase in ('cpu','after-cpu','gpu-protenix'),'05 may only requalify the failed Protenix native component')
    op.require(CURRENT_CAPACITY_SOURCE is not None,'Current cross-campaign capacity source and05 writer are not yet declared')
    op.storage(parent,'entry');payload=parent/'payload';payload.mkdir(exist_ok=False)
    if phase=='after-cpu':
        certificate=op.cpu_pass(pin,manifest);op.write(ROOT/'cpu-certificate.json',certificate)
        op.storage(parent,'dispatch')
        for arm in NATIVE_ARMS:op.submit('gpu-'+arm,pin)
        op.write(payload/'assessment.json',{'status':'passed_CPU_and_Pro_native_component_check_submitted',
            'native_arms':list(NATIVE_ARMS),'Boltz04_qualified_evidence_preserved':True,
            'optimizer_jobs_submitted':False});return
    base=[op.PYTHON,str(SOURCE/'scripts/chemistry_native_qualification.py'),'--plan',str(SOURCE/'plan.json'),
          '--plan-sha256',op.sha(SOURCE/'plan.json')]
    if phase=='cpu':
        for arm in ('saved','protenix','boltz'):
            argv=base+['--arm',arm,'--output',str(payload/('startup-'+arm)),'--startup-only']
            op.write(payload/('startup-invocation-'+arm+'.json'),{'argv':argv});subprocess.run(argv,check=True,timeout=180)
        argv=[op.PYTHON,'-m','pytest',str(SOURCE/'tests'),'-q','-o','addopts=','-p','no:cacheprovider',
              '--junitxml='+str(payload/'tests.xml')]
        with (payload/'tests.log').open('x') as stream:subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=900)
        cases=op.xml_cases(payload/'tests.xml',manifest['cpu_counts'])
        argv=base+['--arm','saved','--output',str(payload/'saved')]
        with (payload/'saved.log').open('x') as stream:subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=900)
        op.write(payload/'assessment.json',{'status':'passed','cases':cases,'job_id':os.environ['LSB_JOBID'],
            'source_manifest_sha256':pin,'optimizer_updates':0});return
    cert=op.read(ROOT/'cpu-certificate.json')
    op.require(cert['status']=='passed' and cert['source_manifest_sha256']==pin,'New allocated CPU required')
    for b in cert['proofs']:op.require(op.binding(b['path'])==b,'CPU evidence changed')
    scope=capacity_operator()
    probe_spec=importlib.util.spec_from_file_location('_chem06_gpu_mapping',scope.SOURCE/'scripts/cuda_ownership.py')
    routing=importlib.util.module_from_spec(probe_spec);probe_spec.loader.exec_module(routing)
    before=routing.allocation_environment(os.environ)
    try:
        resolved=routing.resolve_visibility(os.environ)
        os.environ['CUDA_VISIBLE_DEVICES']=resolved['selected_cuda_visible_devices']
    except BaseException as error:
        op.write(payload/'cuda-routing-failure.json',{'status':'failed','allocation_environment':before,'error':repr(error)})
        raise
    op.write(payload/'cuda-routing.json',{'status':'resolved_explicit_single_mapping','allocation_environment':before,'resolution':resolved})
    arm=phase.removeprefix('gpu-');argv=base+['--arm',arm,'--output',str(payload/'native')]
    op.write(payload/'native-invocation.json',{'argv':argv})
    with (payload/'native.log').open('x') as stream:
        subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=5600)
    result=op.read(payload/'native/assessment.json')
    op.require(result['status']=='native_component_qualified_not_continuation_admission'
        and result['can_submit_continuation'] is False,'Native component evidence incomplete')
    op.write(payload/'assessment.json',{'status':'native_component_qualified_not_continuation_admission',
        'job_id':os.environ['LSB_JOBID'],'source_manifest_sha256':pin,'optimizer_updates':0,
        'native_assessment':op.binding(payload/'native/assessment.json'),'can_submit_continuation':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pin',required=True)
    p.add_argument('--phase',choices=('cpu','after-cpu','gpu-protenix'),required=True)
    a=p.parse_args();execute(a.phase,a.pin)


if __name__=='__main__':main()
