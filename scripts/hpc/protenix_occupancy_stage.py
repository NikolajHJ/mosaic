"""Fail-closed allocated CPU preparation and first native GPU gate controller.

Only the C/Z native-parity/gradient phase is queued. This controller cannot
select a new objective, train a model or run occupancy optimization.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def command_run(command,path,env,cwd,timeout):
    done=subprocess.run(command,text=True,capture_output=True,env=env,cwd=cwd,timeout=timeout)
    value=dict(command=command,returncode=done.returncode,stdout=done.stdout,stderr=done.stderr)
    write(path,value)
    if done.returncode:raise RuntimeError('Allocated stage command failed: '+str(path))
    return value


def verify_release(root,expected):
    record=json.loads((root/'release_manifest.json').read_text())
    if sha(root/'release_manifest.json')!=expected:raise ValueError('Release manifest hash mismatch')
    for name,digest in record['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=digest:
            raise ValueError('Release source byte identity mismatch: '+name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--mode',choices=['cpu','gpu'],required=True)
    args=parser.parse_args()
    if not os.environ.get('LSB_JOBID'):raise RuntimeError('Must execute in allocated DTU job')
    if sha(args.plan)!=args.plan_sha256:raise ValueError('Plan hash mismatch')
    plan=json.loads(Path(args.plan).read_text())
    release=Path(plan['release']);campaign=Path(plan['campaign'])
    verify_release(release,plan['manifest_sha256'])
    if plan['phase']!='protenix-occupancy-native-parity-gradient-v1' or plan['optimization']!='forbidden':
        raise ValueError('Controller scope differs from initial native check phase')
    if (plan.get('checkpoint_compatibility')!='tiny-default-dormant-esm-v1'
            or plan.get('actual_checkpoint_cpu_readiness') is not True):
        raise ValueError('Require declared compatibility and actual CPU checkpoint loading')
    if plan.get('attached_execution')!='native-eval-grad-context-v1':
        raise ValueError('Require declared source-derived native attached execution')
    if (plan.get('element_boundary')!='shared-fp32-element-boundary-v1'
            or plan.get('cpu_native_feature_probe') is not True):
        raise ValueError('Require common element boundary and actual CPU feature preflight')
    if plan.get('native_repeat_policy')!='native-repeat-loss-tv-v1':
        raise ValueError('Require declared native-repeat loss/TV policy')
    if plan.get('output_parity_policy')!='native-output-loss-tv-v1':
        raise ValueError('Require declared output loss/TV policy')
    campaign.mkdir(parents=True,exist_ok=True)
    write(campaign/(args.mode+'-claim.json'),dict(jobid=os.environ['LSB_JOBID'],
        plan_sha256=args.plan_sha256,mode=args.mode,claimed=time.time()))
    deps=Path(plan['environment'])
    outcome=json.loads((deps/'environment-result.json').read_text())
    if outcome['status']!='passed' or sha(deps/'environment-result.json')!=plan['environment_sha256']:
        raise ValueError('Allocated dependency readiness is not authenticated/passed')
    env=dict(os.environ,PYTHONPATH=str(release/'src')+':'+str(deps/'deps')+':'+plan['source_root'],
        PROTENIX_ROOT_DIR=plan['data_root'],LAYERNORM_TYPE='torch',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',
        MPLCONFIGDIR=str(campaign/'matplotlib'),PYTHONDONTWRITEBYTECODE='1')
    state=dict(status='running',mode=args.mode,jobid=os.environ['LSB_JOBID'],
        plan_sha256=args.plan_sha256,release_manifest_sha256=plan['manifest_sha256'],
        phase=plan['phase'],optimization='not_run',started=time.time())
    try:
        if args.mode=='cpu':
            junit=campaign/'cpu-fixtures.xml'
            command_run([sys.executable,'-m','pytest','tests/test_protenix_occupancy.py',
                '-q','--junitxml',str(junit),'-o','cache_dir='+str(campaign/'pytest-cache')],
                campaign/'cpu-fixtures-command.json',env,release,600)
            suites=ET.parse(junit).getroot()
            totals={name:sum(int(s.get(name,'0')) for s in suites.iter('testsuite'))
                    for name in ['tests','failures','errors','skipped']}
            if not totals['tests'] or any(totals[k] for k in ['failures','errors','skipped']):
                raise RuntimeError('Mandatory occupancy fixture suite did not fully pass')
            state['junit']=dict(path=str(junit),sha256=sha(junit),**totals)
            command_run([sys.executable,'-m','latentmol.protenix_occupancy.prepare',
                '--cases-json',str(release/'configs/protenix_occupancy_cases.json'),
                '--input-root',str(release/'inputs'),
                '--output-root',str(campaign/'prepared'),
                '--native-source-root',plan['source_root'],
                '--native-source-sha256',plan['source_sha256'],
                '--ccd-root',str(Path(plan['data_root'])/'common')],
                campaign/'cpu-prepare-command.json',env,release,1800)
            prepared=json.loads((campaign/'prepared/preparation.json').read_text())
            if prepared['status']!='prepared_not_native_validated' or not prepared['all_three_inputs_supported']:
                raise RuntimeError('All three score-blind native inputs must pass preparation')
            command_run([sys.executable,str(release/'scripts/hpc/protenix_occupancy_load_check.py'),
                '--plan',args.plan,'--plan-sha256',args.plan_sha256,
                '--output',str(campaign/'cpu-checkpoint-load.json'),
                '--prepared-manifest',str(campaign/'prepared/preparation.json'),
                '--prepared-manifest-sha256',sha(campaign/'prepared/preparation.json'),
                '--feature-probe-output',str(campaign/'cpu-native-feature-probe.json')],
                campaign/'cpu-checkpoint-load-command.json',env,release,1200)
            loaded=json.loads((campaign/'cpu-checkpoint-load.json').read_text())
            if loaded['status']!='strict_checkpoint_load_passed_not_native_validated':
                raise RuntimeError('Actual checkpoint strict-load readiness did not pass')
            if loaded['provenance']['attached_execution']['version']!=plan['attached_execution']:
                raise RuntimeError('Loaded attached implementation differs from declared plan')
            probe=json.loads((campaign/'cpu-native-feature-probe.json').read_text())
            if probe['status']!='cpu_native_feature_probe_passed_not_cuda_validated':
                raise RuntimeError('Actual CPU native feature preflight failed')
            state['checkpoint_load_sha256']=sha(campaign/'cpu-checkpoint-load.json')
            state['cpu_native_feature_probe_sha256']=sha(campaign/'cpu-native-feature-probe.json')
            config=dict(source_root=plan['source_root'],source_sha256=plan['source_sha256'],
                weights_path=plan['weights_path'],weights_sha256=plan['weights_sha256'],
                element_boundary=plan['element_boundary'],
                native_repeat_policy=plan['native_repeat_policy'],
                output_parity_policy=plan['output_parity_policy'],
                cpu_threads=4,internal_seconds=3300,max_model_calls=3000,
                cases=prepared['cases'])
            for item in config['cases']:
                if sha(item['bundle_path'])!=item['bundle_sha256']:raise ValueError('Prepared bundle checksum mismatch')
            config_path=campaign/'native-config.json';write(config_path,config)
            state.update(status='passed',prepared_manifest_sha256=sha(campaign/'prepared/preparation.json'),
                         native_config_sha256=sha(config_path))
            # Bind the actual completed CPU readiness before submitting a single GPU job.
            write(campaign/'cpu-ready.json',state)
            p=subprocess.run(['getquota_work3.sh'],capture_output=True,text=True,timeout=30)
            write(campaign/'fresh-quota-before-gpu.json',dict(command=['getquota_work3.sh'],
                returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
            matches=re.findall(r'([0-9.]+) GiB\|\s*([0-9.]+) GiB',p.stdout)
            if p.returncode or not matches or any(float(hard)-float(used)<2 for used,hard in matches):
                raise RuntimeError('Fresh work3 quota cannot authenticate at least2GiB headroom')
            # Persist the GPU's gate bindings before bsub can dispatch it.
            # The later submission receipt is historical evidence, not a startup dependency.
            dispatch_binding=dict(cpu_ready_sha256=sha(campaign/'cpu-ready.json'),
                native_config_sha256=sha(config_path),plan_sha256=args.plan_sha256)
            write(campaign/'native-dispatch-binding.json',dispatch_binding)
            command=['bsub','-J',plan['name']+'-native','-q',plan['gpu_queue'],'-n','4',
                '-R','span[hosts=1]','-R','rusage[mem=4GB]','-gpu','num=1:mode=exclusive_process',
                '-W','01:00','-o',str(campaign/'native.%J.out'),'-e',str(campaign/'native.%J.err'),
                sys.executable,str(release/'scripts/hpc/protenix_occupancy_stage.py'),
                '--plan',args.plan,'--plan-sha256',args.plan_sha256,'--mode','gpu']
            done=subprocess.run(command,capture_output=True,text=True,timeout=30)
            submission=dict(command=command,returncode=done.returncode,stdout=done.stdout,stderr=done.stderr,
                cpu_ready_sha256=sha(campaign/'cpu-ready.json'),native_config_sha256=sha(config_path))
            write(campaign/'native-submission.json',submission)
            if done.returncode:raise RuntimeError('GPU submission failed')
        else:
            ready=json.loads((campaign/'cpu-ready.json').read_text())
            config_path=campaign/'native-config.json'
            submitted=json.loads((campaign/'native-dispatch-binding.json').read_text())
            if submitted['plan_sha256']!=args.plan_sha256:raise ValueError('Dispatch plan changed')
            if ready['status']!='passed' or sha(campaign/'cpu-ready.json')!=submitted['cpu_ready_sha256']:
                raise ValueError('GPU lacks authenticated CPU readiness')
            if sha(config_path)!=ready['native_config_sha256'] or sha(config_path)!=submitted['native_config_sha256']:
                raise ValueError('GPU native config changed after CPU stage')
            command_run([sys.executable,'-m','latentmol.protenix_occupancy.study',
                '--config',str(config_path),'--config-sha256',sha(config_path),
                '--output',str(campaign/'native')],
                campaign/'native-command.json',env,release,3420)
            gate=json.loads((campaign/'native/native-gate-result.json').read_text())
            if gate['status']!='native_parity_and_gradient_passed':raise RuntimeError('Native gate not passed')
            state.update(status='passed',gate_sha256=sha(campaign/'native/native-gate-result.json'))
    except BaseException as exc:
        state.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        state['elapsed_seconds']=time.time()-state['started']
        write(campaign/(args.mode+'-terminal.json'),state)
    print(json.dumps(state,indent=2))


if __name__=='__main__':main()
