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
ROOT=BASE/'chemistry-native-qualification-03'
SOURCE=BASE/'chemistry-native-qualification-release-03/source'
CAP=2*512*1024**2+64*1024**2


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
        argv[argv.index('-J')+1]='mosaic-chem-qualification03-'+phase
        return argv
    op.command=command
    oldstorage=op.storage
    def storage(parent,phase):
        value=oldstorage(parent,phase)
        prior=0
        for root in (BASE/'chemistry-endpoint-recovery-02',PARENT.parent):
            for p in root.rglob('*'):
                op.require(not p.is_symlink() and (p.is_file() or p.is_dir()),'Prior diagnostic linked/special output')
                if p.is_file():prior+=p.stat().st_size
        result={'status':'failed','inherited_gate':value,'required_bytes':value['required_bytes'],
            'available_bytes':value['available_bytes']-prior,'entire_diagnostic02_stale_charge':prior,
            'reserve_bytes':4*1024**3,'file_reserve':100000,'new_group_cap_bytes':CAP,
            'forecasts_removed':False,'original_gate_changed':False}
        try:
            op.require(result['available_bytes']>=result['required_bytes'],'Whole native qualification does not fit')
            result['status']='passed'
        finally:op.write(parent/('qualification-storage-'+phase+'.json'),result)
        return result
    op.storage=storage
    return op


def execute(phase,pin):
    op=inherited();manifest=op.verify(SOURCE,pin);plan=op.external(manifest);parent=op.own(phase,pin)
    op.storage(parent,'entry');payload=parent/'payload';payload.mkdir(exist_ok=False)
    if phase=='after-cpu':
        certificate=op.cpu_pass(pin,manifest);op.write(ROOT/'cpu-certificate.json',certificate)
        op.storage(parent,'dispatch')
        for arm in ('protenix','boltz'):op.submit('gpu-'+arm,pin)
        op.write(payload/'assessment.json',{'status':'passed_CPU_and_two_native_component_checks_submitted',
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
    p.add_argument('--phase',choices=('cpu','after-cpu','gpu-protenix','gpu-boltz'),required=True)
    a=p.parse_args();execute(a.phase,a.pin)


if __name__=='__main__':main()
