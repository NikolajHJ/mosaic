"""Boltz-only retry ownership; existing Protenix03 is read-only budget scope."""
from pathlib import Path
import re
import stat

BASE=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
OLD_ROOT=BASE/'chemistry-production-03'
OLD_SOURCE=BASE/'chemistry-production-release-03/source'
OLD_PIN='c73796b041d8432ef1634e8c56920006170a094d3bdc74867c08f3b5415126d7'
VERSION='boltz-chemistry-startup-only-recovery-v1'
DISPATCH_ARMS=('boltz',)


def execution_scope(phase,arm):
    if phase in ('cpu','after-cpu'):
        if arm!='control':raise ValueError('Recovery CPU cannot impersonate a scientific arm')
    elif arm not in DISPATCH_ARMS:
        raise ValueError('This recovery cannot execute or submit the existing Protenix chain')


def budget_root(arm,new_root):
    if arm=='protenix':return OLD_ROOT/'protenix'
    if arm=='boltz':return Path(new_root)/'boltz'
    raise ValueError('Undeclared recovery budget arm')


def _old_control_bytes_once(op):
    total=op.tree_bytes(OLD_SOURCE.parent)
    for path in OLD_ROOT.iterdir():
        if path.name in ('protenix','boltz'):continue
        if path.is_dir():total+=op.tree_bytes(path)
        else:
            info=path.lstat()
            op.require(stat.S_ISREG(info.st_mode) and not path.is_symlink(),'Existing control proof is not ordinary')
            total+=info.st_size
    op.require(total<=64*1024**2,'Existing chemistry03 control cap exhausted')
    return total


def old_control_bytes(op):
    # The independent Pro03 callback may atomically replace root metadata while
    # it is being listed. Retry the complete scan; never omit a vanished entry.
    for attempt in range(3):
        try:return _old_control_bytes_once(op)
        except FileNotFoundError:
            if attempt==2:raise
    raise AssertionError('Unreachable bounded old-control scan')


def authenticate(op,manifest):
    scope=manifest.get('recovery')
    op.require(isinstance(scope,dict) and scope.get('version')==VERSION
               and scope.get('execution_arms')==list(DISPATCH_ARMS)
               and scope.get('preserved_protenix_root')==str(OLD_ROOT/'protenix'),
               'Exact Boltz-only recovery scope required')
    op.require(len(op.verify(OLD_SOURCE,OLD_PIN)['files'])==14,'Original production03 source differs')
    expected={
        str(OLD_ROOT/'boltz/boltz2_chemistry-segment-0000/submission.json'),
        str(OLD_ROOT/'boltz/boltz2_chemistry-segment-0000/stdout.29820257.log'),
        str(OLD_ROOT/'boltz/boltz2_chemistry-segment-0000/driver.json'),
        str(OLD_ROOT/'boltz/boltz2_chemistry-after-segment-0000/submission.json'),
        str(OLD_ROOT/'boltz/boltz2_chemistry-after-segment-0000/stdout.29820258.log'),
        str(OLD_ROOT/'protenix/random-segment-0000/submission.json'),
        str(OLD_ROOT/'protenix/random-segment-0000-certificate.json')}
    proofs=scope.get('proofs')
    op.require(isinstance(proofs,list) and {r['path'] for r in proofs}==expected
               and len(proofs)==len(expected),'Exact unchanged failed/predecessor proofs required')
    for row in proofs:op.check_binding(row)
    jobs=op.read(OLD_ROOT/'jobs.json')
    allowed={'boltz/boltz2_chemistry-segment-0000':'29820257',
             'boltz/boltz2_chemistry-after-segment-0000':'29820258'}
    op.require({k:v for k,v in jobs.items() if k.startswith('boltz/')}==allowed,
               'Existing/unknown original Boltz successor prevents replacement')
    for label,job in allowed.items():
        parent=OLD_ROOT/label;sub=op.read(parent/'submission.json');args=sub['argv']
        op.require(sub['status']=='submitted' and sub['job_id']==job and sub['source_manifest_sha256']==OLD_PIN,
                   'Failed original producer identity differs')
        raw=''.join((parent/('stdout.'+job+'.log')).read_text().split())
        op.require(raw.count('Subject:Job')==1
            and re.findall(r'Subject:Job(\d+):<([^>]+)>incluster<([^>]+)>Exited',raw)==[(job,args[args.index('-J')+1],'dcc')]
            and re.findall(r'Exitedwithexitcode(\d+)\.',raw)==['1'], 'Original failure is not authenticated')
    parent=OLD_ROOT/'boltz/boltz2_chemistry-segment-0000'
    child=op.read(parent/'driver.json')
    op.require(child['returncode']==1 and 'FileNotFoundError' in child['stderr']
               and '/payload/acceleration/gpu-chemistry' in child['stderr']
               and not (parent/'payload/study').exists(),'Only the scientifically unstarted startup failure may be replaced')
    sub=op.read(OLD_ROOT/'protenix/random-segment-0000/submission.json')
    certificate=op.read(OLD_ROOT/'protenix/random-segment-0000-certificate.json')
    op.require(sub['job_id']=='29820255' and sub['source_manifest_sha256']==OLD_PIN
               and certificate['status']=='passed' and certificate['start']==0 and certificate['end']==10000,
               'Existing successful Protenix chain identity changed')
    return {'status':'authenticated_Boltz_startup_failure_and_preserved_Protenix',
            'execution_arms':list(DISPATCH_ARMS),'original_results_changed':False}
