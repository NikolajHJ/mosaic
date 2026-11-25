"""Allocated readiness and conditional GPU deletion/selection diagnostic."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

from protenix_occupancy_stage import sha, write, command_run, verify_release


def literal_policy(release, module, name):
    return ast.literal_eval(next(node.value for node in ast.parse((release/module).read_bytes()).body
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--mode',choices=['cpu','gpu'],required=True)
    args=parser.parse_args()
    if not os.environ.get('LSB_JOBID','').isdigit() or sha(args.plan)!=args.plan_sha256:
        raise RuntimeError('Require allocated job and authenticated immutable plan')
    plan=json.loads(Path(args.plan).read_text())
    release=Path(plan['release']);campaign=Path(plan['campaign'])
    verify_release(release,plan['manifest_sha256'])
    if (plan['phase']!='protenix-deletion-selection-diagnostic-v1'
            or plan['optimization']!='direct_q_diagnostic_only'
            or plan['maximum_model_calls']!=3000 or plan['gpu_internal_seconds']!=3300
            or plan['gpu_wall_minutes']!=60 or plan['maximum_cpus']!=4):
        raise ValueError('Sealed stage scope/resources differ')
    if plan['finite_difference_policy']!=literal_policy(release,'src/latentmol/protenix_occupancy/study.py','FD_POLICY'):
        raise ValueError('Finite-difference policy source/plan mismatch')
    if plan['selection_policy']!=literal_policy(release,'src/latentmol/protenix_occupancy/deletion_study.py','SELECTION_POLICY'):
        raise ValueError('Selection policy source/plan mismatch')
    campaign.mkdir(parents=True,exist_ok=True)
    write(campaign/(args.mode+'-claim.json'),dict(jobid=os.environ['LSB_JOBID'],plan_sha256=args.plan_sha256,mode=args.mode,claimed=time.time()))
    deps=Path(plan['environment'])
    if sha(deps/'environment-result.json')!=plan['environment_sha256'] or json.loads((deps/'environment-result.json').read_text())['status']!='passed':
        raise ValueError('Authenticated dependency outcome is not passed')
    env=dict(os.environ,PYTHONPATH=str(release/'src')+':'+str(deps/'deps')+':'+plan['source_root'],
        PROTENIX_ROOT_DIR=plan['data_root'],LAYERNORM_TYPE='torch',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(campaign/'matplotlib'))
    state=dict(status='running',mode=args.mode,jobid=os.environ['LSB_JOBID'],plan_sha256=args.plan_sha256,
        release_manifest_sha256=plan['manifest_sha256'],phase=plan['phase'],started=time.time())
    try:
        if args.mode=='cpu':
            junit=campaign/'cpu-fixtures.xml'
            command_run([sys.executable,'-m','pytest','tests/test_protenix_occupancy.py',
                'tests/test_protenix_deletion_prepare.py','tests/test_protenix_selection.py',
                '-q','--junitxml',str(junit),'-o','cache_dir='+str(campaign/'pytest-cache')],
                campaign/'cpu-fixtures-command.json',env,release,600)
            suites=ET.parse(junit).getroot()
            totals={k:sum(int(s.get(k,'0')) for s in suites.iter('testsuite')) for k in ['tests','failures','errors','skipped']}
            if totals['tests']<287 or any(totals[k] for k in ['failures','errors','skipped']):
                raise RuntimeError('Required CPU fixture suite did not fully pass')
            state['junit']=dict(path=str(junit),sha256=sha(junit),**totals)
            command_run([sys.executable,'-m','latentmol.protenix_occupancy.prepare',
                '--cases-json',str(release/'configs/protenix_occupancy_cases.json'),
                '--input-root',str(release/'inputs'),'--output-root',str(campaign/'prepared'),
                '--native-source-root',plan['source_root'],'--native-source-sha256',plan['source_sha256'],
                '--ccd-root',str(Path(plan['data_root'])/'common')],campaign/'cpu-prepare-command.json',env,release,900)
            parent=campaign/'prepared/preparation.json'
            command_run([sys.executable,'-m','latentmol.protenix_occupancy.deletion_prepare',
                '--parent-manifest',str(parent),'--parent-manifest-sha256',sha(parent),
                '--cases-json',str(release/'configs/protenix_occupancy_cases.json'),
                '--input-root',str(release/'inputs'),'--output-root',str(campaign/'deleted-prepared'),
                '--native-source-root',plan['source_root'],'--native-source-sha256',plan['source_sha256'],
                '--ccd-root',str(Path(plan['data_root'])/'common')],campaign/'cpu-deletion-prepare-command.json',env,release,900)
            prepared_path=campaign/'deleted-prepared/preparation.json'
            prepared=json.loads(prepared_path.read_text())
            if prepared['status']!='deletion_inputs_prepared_not_native_validated' or not prepared['all_three_inputs_supported']:
                raise RuntimeError('All score-blind deletion endpoints must prepare before scoring')
            command_run([sys.executable,str(release/'scripts/hpc/protenix_occupancy_load_check.py'),
                '--plan',args.plan,'--plan-sha256',args.plan_sha256,
                '--output',str(campaign/'cpu-checkpoint-load.json'),'--prepared-manifest',str(parent),
                '--prepared-manifest-sha256',sha(parent),'--feature-probe-output',str(campaign/'cpu-native-feature-probe.json')],
                campaign/'cpu-checkpoint-load-command.json',env,release,900)
            loaded=json.loads((campaign/'cpu-checkpoint-load.json').read_text())
            probe=json.loads((campaign/'cpu-native-feature-probe.json').read_text())
            if loaded['status']!='strict_checkpoint_load_passed_not_native_validated' or probe['status']!='cpu_native_feature_probe_passed_not_cuda_validated':
                raise RuntimeError('Actual full native input/strict load CPU readiness failed')
            config={k:plan[k] for k in ['phase','source_root','source_sha256','weights_path','weights_sha256',
                'element_boundary','native_repeat_policy','output_parity_policy','finite_difference_policy','selection_policy']}
            config.update(cpu_threads=4,internal_seconds=3300,max_model_calls=3000,cases=prepared['cases'])
            for row in config['cases']:
                if sha(row['bundle_path'])!=row['bundle_sha256']:
                    raise ValueError('Prepared deletion bundle differs')
            config_path=campaign/'native-config.json';write(config_path,config)
            state.update(status='passed',native_config_sha256=sha(config_path),
                checkpoint_load_sha256=sha(campaign/'cpu-checkpoint-load.json'),
                cpu_native_feature_probe_sha256=sha(campaign/'cpu-native-feature-probe.json'),
                deletion_preparation_sha256=sha(prepared_path))
            write(campaign/'cpu-ready.json',state)
            quota=subprocess.run(['getquota_work3.sh'],capture_output=True,text=True,timeout=30)
            write(campaign/'fresh-quota-before-gpu.json',dict(returncode=quota.returncode,stdout=quota.stdout,stderr=quota.stderr))
            matches=re.findall(r'([0-9.]+) GiB\|\s*([0-9.]+) GiB',quota.stdout)
            if quota.returncode or not matches or any(float(hard)-float(used)<4 for used,hard in matches):
                raise RuntimeError('Fresh quota lacks authenticated4GiB headroom')
            binding=dict(cpu_ready_sha256=sha(campaign/'cpu-ready.json'),native_config_sha256=sha(config_path),plan_sha256=args.plan_sha256)
            write(campaign/'native-dispatch-binding.json',binding)
            command=['bsub','-J',plan['name']+'-native','-q',plan['gpu_queue'],'-n','4','-R','span[hosts=1]',
                '-R','rusage[mem=4GB]','-gpu','num=1:mode=exclusive_process','-W','01:00',
                '-o',str(campaign/'native.%J.out'),'-e',str(campaign/'native.%J.err'),sys.executable,
                str(release/'scripts/hpc/protenix_deletion_stage.py'),'--plan',args.plan,'--plan-sha256',args.plan_sha256,'--mode','gpu']
            submitted=subprocess.run(command,capture_output=True,text=True,timeout=30)
            write(campaign/'native-submission.json',dict(command=command,returncode=submitted.returncode,
                stdout=submitted.stdout,stderr=submitted.stderr,**binding))
            if submitted.returncode:
                raise RuntimeError('GPU submission failed')
        else:
            config_path=campaign/'native-config.json'
            binding=json.loads((campaign/'native-dispatch-binding.json').read_text())
            ready=json.loads((campaign/'cpu-ready.json').read_text())
            if (ready['status']!='passed' or binding['plan_sha256']!=args.plan_sha256
                    or sha(campaign/'cpu-ready.json')!=binding['cpu_ready_sha256']
                    or sha(config_path)!=binding['native_config_sha256']
                    or sha(config_path)!=ready['native_config_sha256']):
                raise ValueError('GPU dispatch lacks authenticated CPU readiness/configuration')
            command_run([sys.executable,'-m','latentmol.protenix_occupancy.deletion_study',
                '--config',str(config_path),'--config-sha256',sha(config_path),'--output',str(campaign/'native')],
                campaign/'native-command.json',env,release,3420)
            gate=campaign/'native/deletion-selection-result.json'
            if json.loads(gate.read_text())['status']!='deletion_selection_diagnostic_completed':
                raise RuntimeError('Diagnostic phase incomplete')
            state.update(status='completed_diagnostic_not_G_validated',gate_sha256=sha(gate))
    except BaseException as exc:
        state.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        state['elapsed_seconds']=time.time()-state['started']
        write(campaign/(args.mode+'-terminal.json'),state)
    print(json.dumps(state,indent=2))


if __name__=='__main__':
    main()
