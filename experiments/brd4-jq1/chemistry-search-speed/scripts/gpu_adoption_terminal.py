"""Bounded raw terminal evidence checks for the separate metadata adoption.

An ended dependency may become eligible before a live bjobs query shows DONE.
Only actual repeated RUN/PEND observations may wait; no state is manufactured.
The caller's record callback must persist every raw query receipt immutably.
"""

import re
import time


EXPECTED_CPU_NAME = 'mosaic-chem-speed02-cpu'
MAX_QUERIES = 13


def _job(job_id):
    if type(job_id) is not str or re.fullmatch(r'[1-9]\d*', job_id) is None:
        raise ValueError('Require one explicit positive numeric job ID string')
    return job_id


def _compact(text):
    if not isinstance(text, str):
        raise TypeError('Raw scheduler/LSF evidence must be text')
    return ''.join(text.split())


def _scheduler_status(raw, job_id, job_name):
    if not isinstance(raw, dict):
        raise TypeError('Scheduler query must return a raw receipt dictionary')
    if type(raw.get('returncode')) is not int or raw['returncode'] != 0:
        raise RuntimeError('Actual terminal query failed; no terminal state inferred')
    if not isinstance(raw.get('stderr'), str):
        raise TypeError('Raw scheduler stderr must be retained as text')
    if 'job_id' in raw and raw['job_id'] != job_id:
        raise RuntimeError('Scheduler receipt belongs to a different job')
    text = _compact(raw.get('stdout'))
    jobs = re.findall(r'Job<(\d+)>', text)
    statuses = re.findall(r'Status<([^>]+)>', text)
    queues = re.findall(r'Queue<([^>]+)>', text)
    names = re.findall(r'JobName<([^>]+)>', text)
    if jobs != [job_id] or len(statuses) != 1 or queues != ['hpc']:
        raise RuntimeError('Require unique actual same-job scheduler state in hpc')
    if names != [job_name]:
        raise RuntimeError('Actual CPU scheduler job name differs from its sealed submission')
    status = statuses[0]
    if status not in ('DONE', 'RUN', 'PEND'):
        raise RuntimeError('Actual CPU state is not success or a bounded live wait: ' + status)
    return status


def wait_terminal(query, job_id, max_queries=MAX_QUERIES,
                  pause=lambda: time.sleep(5), record=lambda raw: None,
                  job_name=EXPECTED_CPU_NAME):
    """Return the unchanged actual DONE receipt or fail without submitting jobs.

    ``query(job_id)`` returns ``{returncode, stdout, stderr}`` and may include
    additional raw provenance fields. ``record(raw)`` is called before validating
    every query, including a failing one. ``pause()`` runs only between observed
    RUN/PEND queries, at most 12 times by default (60 seconds). The caller owns
    durable recording; defaults are useful for isolated dependency-free fixtures.
    """
    job_id = _job(job_id)
    if job_name not in (EXPECTED_CPU_NAME, 'mosaic-chem-speed-gpu-adoption03-cpu'):
        raise ValueError('Require an exact declared original or adoption CPU job name')
    if type(max_queries) is not int or not 1 <= max_queries <= MAX_QUERIES:
        raise ValueError('Require an explicit query bound from 1 through 13')
    if not all(callable(value) for value in (query, pause, record)):
        raise TypeError('Require explicit query, pause and recording callbacks')
    for index in range(max_queries):
        raw = query(job_id)
        record(raw)
        status = _scheduler_status(raw, job_id, job_name)
        if status == 'DONE':
            return raw
        if index + 1 < max_queries:
            pause()
    raise TimeoutError('Bounded actual scheduler observations did not establish DONE')


def lsf_success(text, job_id, job_name=EXPECTED_CPU_NAME):
    """Authenticate the unique original CPU LSF success summary, verbatim input.

    Wrapping whitespace is ignored only for field matching. The caller retains
    and hashes the unchanged original LSF text; no synthetic summary is returned.
    """
    job_id = _job(job_id)
    if job_name not in (EXPECTED_CPU_NAME, 'mosaic-chem-speed-gpu-adoption03-cpu'):
        raise ValueError('Require an exact declared original or adoption CPU job name')
    value = _compact(text)
    # Keep the line boundary after the terminal word. Compacting the whole log
    # first would join "Done\nJob was executed..." into "DoneJobwasexecuted".
    subjects = re.findall(
        r'Subject:\s*Job\s*(\d+)\s*:\s*<([^>]+)>\s*in\s*cluster\s*<([^>]+)>\s*([A-Za-z]+)[ \t]*(?=\r?\n|$)',
        text,
    )
    subjects = [tuple(_compact(field) for field in row) for row in subjects]
    if value.count('Subject:Job') != 1 or subjects != [
        (job_id, job_name, 'dcc', 'Done')
    ]:
        raise RuntimeError('Require unique same-job/name/dcc DONE LSF subject')
    if re.findall(r'inqueue<([^>]+)>', value) != ['hpc']:
        raise RuntimeError('Require unique original LSF execution in hpc')
    if value.count('Successfullycompleted.') != 1:
        raise RuntimeError('Original same-job LSF success footer is absent or ambiguous')
    return True
