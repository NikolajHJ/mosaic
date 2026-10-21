"""Allocated stdlib contracts for report scheduling, CPU proof and storage scope."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
import jq1_fixed_report_ops as ops


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def xml(path, *, omit=None, failure=None):
    names=[Path(n).stem for n in ops.TESTS+ops.LEGACY_TESTS if n!=omit]
    root=ET.Element('testsuites',tests=str(len(names)),failures='0',errors='0',skipped='0')
    suite=ET.SubElement(root,'testsuite',tests=str(len(names)),failures='0',errors='0',skipped='0')
    for name in names:
        c=ET.SubElement(suite,'testcase',classname='tests.'+name+'.TestGroup',name='fixture')
        if failure==name:ET.SubElement(c,'failure')
    path.parent.mkdir(parents=True,exist_ok=True);ET.ElementTree(root).write(path)


def test_actual_done_identity_is_required():
    ops.actual_scheduler('Job <7>, Status <DONE>, Queue <hpc>', '7')
    for raw in ('Job <8>, Status <DONE>, Queue <hpc>', 'Job <7>, Status <RUN>, Queue <hpc>',
                'Job <7>, Status <DONE>, Queue <gpul40s>', 'Job <7> Job <7> Status <DONE> Queue <hpc>'):
        with pytest.raises(ValueError):ops.actual_scheduler(raw,'7')


def test_same_job_lsf_success_is_required():
    ops.actual_lsf('Subject: Job 7: <report> in cluster <dcc> Done\nSuccessfully completed.\n','7')
    for text in ('Subject: Job 8: <report> Done\nSuccessfully completed.\n',
                 'Subject: Job 7: <report> Exited\nExited with exit code 1.\n',
                 'Subject: Job 7: <report> Done\nSuccessfully completed.\nExited by signal 9.\n'):
        with pytest.raises(ValueError):ops.actual_lsf(text,'7')


def test_all_required_fixture_modules_with_classes(tmp_path):
    q=tmp_path/'tests.xml';xml(q);assert len(ops.xml_cases(q))==len(ops.TESTS+ops.LEGACY_TESTS)


@pytest.mark.parametrize('omit',ops.TESTS+ops.LEGACY_TESTS)
def test_missing_fixture_module_rejected(tmp_path,omit):
    q=tmp_path/'tests.xml';xml(q,omit=omit)
    with pytest.raises(ValueError):ops.xml_cases(q)


def test_raw_failure_cannot_be_hidden_by_suite_totals(tmp_path):
    q=tmp_path/'tests.xml';xml(q,failure=Path(ops.TESTS[0]).stem)
    with pytest.raises(ValueError):ops.xml_cases(q)


def test_storage_retains_all_remaining_work3_native_exposure():
    current={'protenix':100,'boltz2':200};b=ops.report_budget(current,20*1024**3,19*1024**3,300,'interim')
    assert b['outstanding_native_bytes']==1493301824+1770121824-300
    assert b['required_bytes']==b['outstanding_native_bytes']+6*1024**3
    assert b['available_bytes_conservative']==19*1024**3-300
    cpu=ops.report_budget(current,20*1024**3,19*1024**3,300,'cpu')
    assert b['required_bytes']-cpu['required_bytes']==1024**3


@pytest.mark.parametrize('current',({'protenix':1493301825,'boltz2':0}, {'protenix':0}, {'protenix':0,'boltz2':True}))
def test_malformed_or_exceeded_native_projection_fails(current):
    with pytest.raises(ValueError):ops.report_budget(current,20*1024**3,20*1024**3,0,'interim')


def test_negative_stale_or_unknown_phase_fails():
    for stale,phase in [(-1,'interim'),(0,'unlimited')]:
        with pytest.raises(ValueError):ops.report_budget({'protenix':0,'boltz2':0},1,1,stale,phase)


@pytest.fixture
def certificate(tmp_path,monkeypatch):
    root=tmp_path/'report';source=tmp_path/'source';(root/'cpu/payload').mkdir(parents=True);source.mkdir()
    monkeypatch.setattr(ops,'ROOT',root);monkeypatch.setattr(ops,'SOURCE',source)
    (source/'source_manifest.json').write_text('{"files":{}}\n')
    x=root/'cpu/payload/report-tests.xml';xml(x)
    sched=root/'cpu/scheduler-final-1.txt';sched.write_text('Job <7>, Status <DONE>, Queue <hpc>')
    log=root/'cpu/stdout.7.log';log.write_text('Subject: Job 7: <report> in cluster <dcc> Done\nSuccessfully completed.\n')
    testlog=root/'cpu/payload/report-tests.log';testlog.write_text('All required modules passed.\n')
    proofs=[{'path':str(q),'sha256':ops.sha(q)} for q in (x,sched,log,testlog)]
    c={'status':'passed','producer_job_id':'7','job_id':'7','tests':len(ops.TESTS+ops.LEGACY_TESTS),
       'suite':ops.SUITE,'source_manifest_sha256':'m','proofs':proofs}
    dump(root/'cpu-certificate.json',c)
    return root,c


def test_cpu_proof_and_bundle_preserve_actual_paths(certificate):
    root,c=certificate;assert ops.verify_cpu('m')==c
    run=root/'edition/payload';run.mkdir(parents=True)
    ops.bundle_cpu(run,'m')
    links=json.loads((run/'cpu-proof/bundled-proofs.json').read_text())
    assert len(links)==len(c['proofs'])
    for link in links:
        assert ops.sha(run/link['local_path'])==link['sha256']==ops.sha(Path(link['original_path']))
    assert (run/'cpu-proof/assessment.json').read_bytes()==(root/'cpu-certificate.json').read_bytes()


def test_wrong_cpu_manifest_and_modified_proof_fail(certificate):
    root,c=certificate
    with pytest.raises(ValueError):ops.verify_cpu('other')
    (root/'cpu/payload/report-tests.log').write_text('changed')
    with pytest.raises(ValueError):ops.verify_cpu('m')


def test_cpu_proof_cannot_escape_cpu_scope(certificate):
    root,c=certificate;q=root/'outside.txt';q.write_text('opaque')
    c['proofs'].append({'path':str(q),'sha256':ops.sha(q)});dump(root/'cpu-certificate.json',c)
    with pytest.raises(ValueError):ops.verify_cpu('m')


def test_unknown_report_submission_is_persisted_and_not_duplicated(tmp_path,monkeypatch):
    root=tmp_path/'report';root.mkdir();monkeypatch.setattr(ops,'ROOT',root)
    calls=[]
    def unknown(args,**kwargs):
        calls.append(args);return SimpleNamespace(returncode=0,stdout='unknown outcome',stderr='')
    monkeypatch.setattr(ops.subprocess,'run',unknown)
    with pytest.raises(ValueError):ops.submit('interim-0001','interim','m')
    r=json.loads((root/'interim-0001/submission.json').read_text());assert r['outcome']=='unknown'
    with pytest.raises(FileExistsError):ops.submit('interim-0001','interim','m')
    assert len(calls)==1 and calls[0][:5]==['bsub','-q','hpc','-n','2'] and '-gpu' not in calls[0]
    assert (root/'interim-0001/submission-started.json').is_file()


def test_report_cannot_submit_native_job_name(tmp_path,monkeypatch):
    root=tmp_path/'report';root.mkdir();monkeypatch.setattr(ops,'ROOT',root)
    with pytest.raises(ValueError):ops.submit('run-boltz2','interim','m')
