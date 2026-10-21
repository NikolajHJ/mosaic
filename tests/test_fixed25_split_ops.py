"""Allocated split launcher contracts; original numerical/model gates are reused unchanged."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import fixed25_split_ops as ops

REPO = Path(__file__).resolve().parents[1]


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if not isinstance(value, str) else value)
    return path


@pytest.mark.parametrize('status,queue,job,success', [('DONE','gpul40s','29776117',True),
    ('RUN','gpul40s','29776117',True), ('EXIT','gpul40s','29776117',True),
    ('DONE','hpc','29776117',True), ('DONE','gpul40s','29776119',True), ('DONE','gpul40s','29776117',False)])
def test_persisted_actual_job_done_queue_and_lsf_success_all_required(tmp_path,status,queue,job,success):
    scheduler=put(tmp_path/'scheduler', f'Job <{job}>, Status <{status}>, Queue <{queue}>')
    log=put(tmp_path/'stdout', 'Subject: Job 29776117: test\n'+('Successfully completed.' if success else 'Exited with exit code 1.'))
    if status=='DONE' and queue=='gpul40s' and job=='29776117' and success:
        ops.persisted_success('29776117','gpul40s',scheduler,log)
    else:
        with pytest.raises(ValueError): ops.persisted_success('29776117','gpul40s',scheduler,log)


def test_persisted_byte_mutation_rejected(tmp_path):
    path=put(tmp_path/'proof','a'); old=ops.sha(path); path.write_text('b')
    with pytest.raises(ValueError,match='bytes changed'): ops.hashed(path,old)


@pytest.fixture
def original(tmp_path,monkeypatch):
    science=tmp_path/'science'; old=tmp_path/'original'; root=tmp_path/'split'; root.mkdir()
    monkeypatch.setattr(ops,'SCIENCE',science); monkeypatch.setattr(ops,'SCIENCE_ROOT',old)
    monkeypatch.setattr(ops,'ROOT',root); monkeypatch.setattr(ops,'check_release',lambda *a:None)
    cert_proofs=[]
    for name,text in [('scheduler-final-0.txt','Job <29776103>, Status <DONE>, Queue <hpc>'),
                      ('stdout.29776103.log','Subject: Job 29776103: tests\nSuccessfully completed.')]:
        p=put(old/'cpu'/name,text);cert_proofs.append({'path':str(p),'sha256':ops.sha(p)})
    assessment=put(old/'cpu/payload/assessment.json',{'passed':482})
    monkeypatch.setattr(ops,'CPU_ASSESSMENT_SHA',ops.sha(assessment))
    cert=put(old/'cpu-certificate.json',{'status':'passed','source_manifest_sha256':ops.SCIENCE_MANIFEST,
         'producer_job_id':'29776103','tests':482,'proofs':cert_proofs})
    monkeypatch.setattr(ops,'CPU_CERT_SHA',ops.sha(cert))
    monkeypatch.setitem(sys.modules,'check_boltz_cpu_proof',SimpleNamespace(verify=lambda *a:{'passed':482}))
    pins={}
    for index,arm in enumerate(ops.ARMS):
        p=old/('validate-'+arm);job=ops.NATIVE_PINS[arm]['job']
        config=ops.config_path(science,arm);config.parent.mkdir(parents=True,exist_ok=True)
        config.write_bytes((REPO/f'examples/configs/brd4_jq1_25_fixed_{arm}_10000.json').read_bytes())
        scheduler=put(p/'scheduler-final-1.txt',f'Job <{job}>, Status <DONE>, Queue <gpul40s>')
        stdout=put(p/('stdout.'+job+'.log'),f'Subject: Job {job}: readiness\nSuccessfully completed.')
        a=put(p/'payload/study/assessment.json',{'context_digest':ops.CONTEXTS[arm],
              'model_digest':'f'*64,'pocket_definition_sha256':ops.POCKET,'diffusion_gradient':False,
              'storage_projection':{'projected_bytes':100+index}})
        pins[arm]={'job':job,'scheduler':ops.sha(scheduler),'stdout':ops.sha(stdout),
                   'assessment':ops.sha(a),'config':ops.sha(config)}
        put(old/('decision-'+arm+'.json'),{'status':'passed','native_job':job,'assessment_sha256':ops.sha(a),
             'projected_bytes':100+index,'scheduler_proofs':[{'path':str(scheduler),'sha256':ops.sha(scheduler)},
                                                         {'path':str(stdout),'sha256':ops.sha(stdout)}]})
    monkeypatch.setattr(ops,'NATIVE_PINS',pins)
    monkeypatch.setattr(ops,'verify_readiness',lambda path,*a:ops.read(path))
    put(old/'shared-dispatch-quota/work3-storage-gate.json',{'status':'failed','projected_bytes':303,
        'reserve_bytes':ops.storage.RESERVE,'required_bytes':303+ops.storage.RESERVE})
    return old,root


def test_adoption_rehashes_original_proofs_without_live_scheduler_and_preserves_failed_quota(original,monkeypatch):
    old,root=original
    monkeypatch.setattr(ops.subprocess,'run',lambda *a,**k:pytest.fail('No live bjobs or model call'))
    record=ops.capture_adoption();path=root/'adoption.json'
    assert record['original']['historical_failed_storage']['path'].endswith('work3-storage-gate.json')
    assert record['original']['arms']['boltz2']['native_job']=='29776119'
    assert ops.verify_adoption(ops.sha(path))==record
    assert ops.read(old/'shared-dispatch-quota/work3-storage-gate.json')['status']=='failed'


@pytest.mark.parametrize('mutation',['bytes','status','projection','context','prior_long'])
def test_original_mutations_or_competing_submission_block_adoption(original,mutation):
    old,root=original
    decision=old/'decision-protenix.json'
    if mutation=='bytes': (old/'validate-protenix/scheduler-final-1.txt').write_text('Job not found')
    elif mutation=='prior_long': put(old/'long-submissions.json',{'jobs':[]})
    elif mutation=='context':
        a=old/'validate-protenix/payload/study/assessment.json';v=ops.read(a);v['context_digest']='0'*64;put(a,v)
        ops.NATIVE_PINS['protenix']['assessment']=ops.sha(a)
        v=ops.read(decision);v['assessment_sha256']=ops.sha(a);put(decision,v)
    else:
        v=ops.read(decision);v['status' if mutation=='status' else 'projected_bytes']='failed' if mutation=='status' else 1;put(decision,v)
    with pytest.raises(ValueError): ops.capture_adoption()
    assert not (root/'adoption.json').exists()


@pytest.fixture
def tool_cpu(tmp_path,monkeypatch):
    root=tmp_path/'split';monkeypatch.setattr(ops,'ROOT',root);job='29799901';proofs=[]
    raw='<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0"><testcase classname="tests.test_fixed25_split_ops"/><testcase classname="test_fixed25_split_storage"/></testsuite></testsuites>'
    for name,text in [('scheduler-final-1.txt',f'Job <{job}>, Status <DONE>, Queue <hpc>'),
        ('stdout.'+job+'.log',f'Subject: Job {job}: CPU\nSuccessfully completed.'),('payload/contracts.xml',raw)]:
        p=put(root/'cpu'/name,text);proofs.append({'path':str(p),'sha256':ops.sha(p)})
    cert=put(root/'cpu-certificate.json',{'status':'passed','source_manifest_sha256':'a'*64,
         'producer_job_id':job,'tests':2,'suite':ops.SUITE,'proofs':proofs})
    return root,cert


def test_new_launcher_requires_actual_done_cpu_lsf_xml_and_both_test_modules(tool_cpu):
    root,cert=tool_cpu
    assert ops.verify_tool_cpu('a'*64,ops.sha(cert))['tests']==2


@pytest.mark.parametrize('mutation',['xml_failure','skip','count','missing_module','wrong_job','proof_bytes','wrong_manifest'])
def test_cpu_proof_mutations_block_scientific_dispatch(tool_cpu,mutation):
    root,cert=tool_cpu;v=ops.read(cert)
    if mutation in ('xml_failure','skip','missing_module'):
        p=root/'cpu/payload/contracts.xml';text=p.read_text()
        text=text.replace('/>', '><failure/></testcase>',1) if mutation=='xml_failure' else text
        text=text.replace('/>', '><skipped/></testcase>',1) if mutation=='skip' else text
        text=text.replace('test_fixed25_split_storage','other') if mutation=='missing_module' else text
        p.write_text(text)
        for proof in v['proofs']:
            if proof['path']==str(p):proof['sha256']=ops.sha(p)
    elif mutation=='count':v['tests']=3
    elif mutation=='wrong_job':v['producer_job_id']='29799902'
    elif mutation=='wrong_manifest':v['source_manifest_sha256']='b'*64
    else:(root/'cpu/scheduler-final-1.txt').write_text('Job not found')
    put(cert,v)
    with pytest.raises(ValueError):ops.verify_tool_cpu('a'*64,ops.sha(cert))


@pytest.mark.parametrize('arm',ops.ARMS)
def test_unchanged_scientific_driver_configs_ready_and_fixed_step_rate_pins(arm):
    command=ops.driver_command(arm,'/python')
    assert command[1]==str(ops.SCIENCE/'examples/brd4_25_atoms.py')
    assert command[command.index('--manifest')+1]==ops.SCIENCE_MANIFEST
    assert command[command.index('--ready-sha256')+1]==ops.NATIVE_PINS[arm]['assessment']
    assert command[command.index('--output')+1]==str(ops.storage.output_parent(arm)/'payload/study')
    assert '--mode' in command and command[command.index('--mode')+1]=='run'
    assert '--prepared' in command if arm!='protenix' else '--prepared' not in command
    assert not any(x in command for x in ('--step-size','--gradient-clip','--temperature','--resume'))


@pytest.mark.parametrize('arm',ops.ARMS)
def test_actual_new_independent_job_arguments_and_separate_tool_science_pins(tmp_path,monkeypatch,arm):
    monkeypatch.setattr(ops.storage,'output_parent',lambda a:tmp_path/('run-'+a));calls=[]
    def run(command,**kwargs):
        calls.append(command);return SimpleNamespace(returncode=0,stdout='Job <29799909> is submitted' if command[0]=='bsub' else 'PEND',stderr='')
    monkeypatch.setattr(ops.subprocess,'run',run)
    result=ops.submit(arm,'a'*64,'b'*64,'c'*64);command=calls[0]
    assert result['job_id']=='29799909' and result['science_manifest_sha256']==ops.SCIENCE_MANIFEST
    assert command[command.index('-q')+1]=='gpul40s' and command[command.index('-n')+1]=='4'
    assert command[command.index('-W')+1]=='23:00' and '-w' not in command
    assert 'rusage[mem=8GB]' in command and 'num=1:mode=exclusive_process' in command
    assert str(ops.TOOL_SOURCE/'scripts/hpc/fixed25_split_stage.lsf') in command
    assert command[-6:]==[str(ops.TOOL_SOURCE),'a'*64,str(ops.ROOT),arm,'b'*64,'c'*64]


def test_unknown_bsub_outcome_preserves_receipt_and_never_invents_job(tmp_path,monkeypatch):
    monkeypatch.setattr(ops.storage,'output_parent',lambda a:tmp_path/('run-'+a))
    monkeypatch.setattr(ops.subprocess,'run',lambda *a,**k:(_ for _ in ()).throw(TimeoutError('unknown')))
    with pytest.raises(RuntimeError,match='Unknown'):ops.submit('protenix','a'*64,'b'*64,'c'*64)
    receipt=ops.read(tmp_path/'run-protenix/submission.json')
    assert receipt['outcome']=='unknown' and 'job_id' not in receipt
    with pytest.raises(FileExistsError):ops.submit('protenix','a'*64,'b'*64,'c'*64)


def test_all_native_tool_cpu_and_both_storage_gates_precede_marker_or_submissions(tmp_path,monkeypatch):
    monkeypatch.setattr(ops,'ROOT',tmp_path)
    monkeypatch.setattr(ops,'check_release',lambda *a:None)
    adopted={'original':{'arms':{a:{'projected_bytes':10} for a in ops.ARMS}}}
    monkeypatch.setattr(ops,'verify_adoption',lambda *a:adopted)
    monkeypatch.setattr(ops,'verify_tool_cpu',lambda *a:{})
    monkeypatch.setattr(ops.storage,'output_parent',lambda a:tmp_path/('run-'+a))
    calls=[]
    monkeypatch.setattr(ops.storage,'check_campaign_storage',lambda p,*a,**k:calls.append(('quota',p)) or {'status':'passed'})
    monkeypatch.setattr(ops,'submit',lambda arm,*a:calls.append(('submit',arm)) or {'arm':arm,'job_id':'29799999'})
    jobs=ops.dispatch('a'*64,'b'*64,'c'*64)
    assert calls[0]==('quota',dict.fromkeys(ops.ARMS,10)) and len(jobs)==3
    with pytest.raises(RuntimeError,match='Previous'):ops.dispatch('a'*64,'b'*64,'c'*64)
    assert len(calls)==4


@pytest.mark.parametrize('blocker', ['storage', 'prior_parent', 'prior_receipt'])
def test_resource_or_unknown_output_blocker_never_creates_dispatch_marker(tmp_path,monkeypatch,blocker):
    monkeypatch.setattr(ops,'ROOT',tmp_path)
    monkeypatch.setattr(ops,'check_release',lambda *a:None)
    monkeypatch.setattr(ops,'verify_adoption',lambda *a:{'original':{'arms':{a:{'projected_bytes':10} for a in ops.ARMS}}})
    monkeypatch.setattr(ops,'verify_tool_cpu',lambda *a:{})
    monkeypatch.setattr(ops.storage,'output_parent',lambda a:tmp_path/('run-'+a))
    monkeypatch.setattr(ops,'submit',lambda *a,**k:pytest.fail('No production submission'))
    def quota(*a,**k):
        raise ValueError('Storage failed honestly')
    monkeypatch.setattr(ops.storage,'check_campaign_storage',quota)
    if blocker=='prior_parent':(tmp_path/'run-boltz2_chemistry').mkdir()
    elif blocker=='prior_receipt':put(tmp_path/'long-submissions.json',{'jobs':[]})
    with pytest.raises((ValueError,RuntimeError)):ops.dispatch('a'*64,'b'*64,'c'*64)
    assert not (tmp_path/'long-dispatch-started.json').exists()


def test_wrapper_cli_and_environment_call_the_authentic_source_without_diffusion_gradients():
    stage=(REPO/'scripts/hpc/fixed25_split_stage.lsf').read_text()
    for flag in ('--phase run','--tool-source','--tool-manifest','--root','--arm','--cpu-certificate-sha256','--adoption-sha256'):
        assert flag in stage
    assert 'cd "$science"' in stage and 'JAX_PLATFORMS=cpu' in stage and 'JAX_PLATFORMS=cuda' in stage
    assert 'boltz_node_tmpdir.sh' in stage
    after=(REPO/'scripts/hpc/fixed25_split_after_cpu.lsf').read_text()
    assert "final_job(producer,root/'cpu')" in after and "sha(root/'adoption.json')" in after


def test_cpu_contract_fixtures_use_owned_node_tmpdir_and_never_campaign_storage():
    cpu=(REPO/'scripts/hpc/fixed25_split_cpu.lsf').read_text()
    assert 'source "$science/scripts/hpc/boltz_node_tmpdir.sh"' in cpu
    assert '--basetemp "$TMPDIR/storage-contracts"' in cpu
    assert '--basetemp "$run/tmp"' not in cpu
