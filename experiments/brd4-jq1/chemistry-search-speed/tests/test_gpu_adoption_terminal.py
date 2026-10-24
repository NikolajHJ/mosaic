"""Allocated stdlib raw-string fixtures; no scheduler or numerical execution."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gpu_adoption_terminal import wait_terminal, lsf_success


JOB = '29809477'


def scheduler(status='DONE', job=JOB, queue='hpc'):
    return {
        'returncode': 0,
        'stdout': ('Job <' + job + '>, Job Name <mosaic-chem-speed02-cpu>, '
                   'User <s234818>, Status <' + status + '>, Queue <' + queue + '>\n'),
        'stderr': '',
    }


def lsf(job=JOB, name='mosaic-chem-speed02-cpu', queue='hpc', state='Done'):
    return (
        'Sender: LSF System <lsfadmin@hpc.dtu.dk>\n'
        'Subject: Job ' + job + ': <' + name + '> in cluster <dcc> ' + state + '\n'
        'Job was executed on host(s) <16*n-62-30-23>, in queue <' + queue + '>, '
        'as user <s234818> in cluster <dcc>\n'
        'Successfully completed.\n'
    )


def test_actual_wrapped_done_receipt_returned_unchanged_and_recorded():
    raw = scheduler()
    raw['stdout'] = raw['stdout'].replace('DONE', 'DO\n       NE').replace('hpc', 'hp\n       c')
    recorded, queried, pauses = [], [], []
    def query(job):
        queried.append(job)
        return raw
    result = wait_terminal(query, JOB, record=recorded.append, pause=lambda: pauses.append(True))
    assert result is raw and recorded == [raw] and queried == [JOB] and not pauses


def test_live_queries_wait_and_preserve_every_unmodified_observation():
    raw = [scheduler('PEND'), scheduler('RUN'), scheduler('DONE')]
    pending = iter(raw)
    recorded, pauses = [], []
    result = wait_terminal(lambda job: next(pending), JOB, max_queries=3,
                           record=recorded.append, pause=lambda: pauses.append(True))
    assert result is raw[-1] and recorded == raw and len(pauses) == 2
    assert [r['stdout'] for r in recorded] == [r['stdout'] for r in raw]


def test_wait_exhaustion_is_not_invented_success_or_an_extra_query():
    recorded, pauses = [], []
    def query(job):
        return scheduler('RUN')
    with pytest.raises(TimeoutError, match='did not establish DONE'):
        wait_terminal(query, JOB, max_queries=3, record=recorded.append,
                      pause=lambda: pauses.append(True))
    assert len(recorded) == 3 and len(pauses) == 2
    assert all('Status <RUN>' in r['stdout'] for r in recorded)


@pytest.mark.parametrize('fault', ('wrong_job', 'wrong_queue', 'exit', 'query_error', 'multiple_headers'))
def test_actual_bad_terminal_evidence_fails_immediately_and_is_preserved(fault):
    raw = scheduler()
    if fault == 'wrong_job':
        raw = scheduler(job='29809478')
    elif fault == 'wrong_queue':
        raw = scheduler(queue='gpul40s')
    elif fault == 'exit':
        raw = scheduler('EXIT')
    elif fault == 'query_error':
        raw['returncode'], raw['stderr'] = 255, 'Authentication failed'
    else:
        raw['stdout'] += scheduler()['stdout']
    recorded, pauses = [], []
    with pytest.raises(RuntimeError):
        wait_terminal(lambda job: raw, JOB, record=recorded.append, pause=lambda: pauses.append(True))
    assert recorded == [raw] and not pauses


def test_query_bound_and_type_rejected_before_query_or_pause():
    for bad in (True, 0, 14, 1.5):
        calls = []
        with pytest.raises(ValueError):
            wait_terminal(lambda job: calls.append(job), JOB, max_queries=bad,
                          pause=lambda: calls.append('pause'))
        assert calls == []


def test_original_lsf_wrapping_retains_unique_same_job_success():
    text = lsf().replace('speed02', 'speed\n     02').replace('hpc', 'hp\n     c')
    assert lsf_success(text, JOB) is True


def test_new_adoption_cpu_has_its_own_exact_name_and_identity():
    name = 'mosaic-chem-speed-gpu-adoption03-cpu'
    raw = scheduler(); raw['stdout'] = raw['stdout'].replace('mosaic-chem-speed02-cpu', name)
    assert wait_terminal(lambda job: raw, JOB, job_name=name) is raw
    assert lsf_success(lsf(name=name), JOB, name) is True


def test_original_name_cannot_authenticate_the_new_adoption_cpu():
    name = 'mosaic-chem-speed-gpu-adoption03-cpu'
    with pytest.raises(RuntimeError):
        wait_terminal(lambda job: scheduler(), JOB, job_name=name)
    with pytest.raises(RuntimeError):
        lsf_success(lsf(), JOB, name)


def test_undeclared_cpu_names_fail_before_query():
    calls = []
    with pytest.raises(ValueError):
        wait_terminal(lambda job: calls.append(job), JOB, job_name='other')
    assert not calls
    with pytest.raises(ValueError):
        lsf_success(lsf(), JOB, 'other')


@pytest.mark.parametrize('fault', ('wrong_job', 'wrong_name', 'wrong_queue', 'exited', 'duplicate_subject', 'missing_success'))
def test_lsf_ambiguity_wrong_identity_or_failed_footer_never_pass(fault):
    if fault == 'wrong_job':
        text = lsf(job='29809478')
    elif fault == 'wrong_name':
        text = lsf(name='another-experiment-cpu')
    elif fault == 'wrong_queue':
        text = lsf(queue='gpul40s')
    elif fault == 'exited':
        text = lsf(state='Exited')
    elif fault == 'duplicate_subject':
        text = lsf() + lsf()
    else:
        text = lsf().replace('Successfully completed.', 'Exited with exit code 1.')
    with pytest.raises(RuntimeError):
        lsf_success(text, JOB)
