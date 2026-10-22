"""Allocated metadata contracts for one corrected Protenix diagnostic only."""
import os
from pathlib import Path
import pytest
import launcher_resource_recovery as ops


@pytest.fixture(autouse=True)
def allocated_only():
    assert os.environ.get('LSB_JOBID'), 'Allocated CPU fixtures only'


@pytest.mark.parametrize('phase', ['cpu', 'dispatch', 'diagnostic', 'diagnostic-capture'])
def test_declared_resources(phase):
    r = ops.resources(phase)
    assert r[r.index('-q') + 1] == ('gpul40s' if phase == 'diagnostic' else 'hpc')
    assert ('-gpu' in r) == (phase == 'diagnostic')
    assert r[r.index('-W') + 1] == ('02:00' if phase == 'diagnostic' else '00:30' if phase == 'dispatch' else '00:45')


@pytest.mark.parametrize('phase', ['run', 'training', 'boltz2'])
def test_no_scientific_or_second_diagnostic_phase(phase):
    with pytest.raises(ValueError): ops.resources(phase)


def test_raw_command_is_unchanged_certified_core_and_saved_steps():
    args = ops.raw_command()
    assert args[args.index('--manifest') + 1] == ops.CORE_MANIFEST
    assert args[args.index('--arm') + 1] == 'protenix'
    assert args[args.index('--nearby-step') + 1] == '9900'
    assert args[args.index('--max-output-bytes') + 1] == str(256 * 1024**2)
    assert '--learning-rate' not in args and '--steps' not in args


def test_capture_depends_only_on_own_ended_producer():
    args = ops.command('diagnostic-capture', 'a' * 64, 'ended(123)')
    assert args[args.index('-w') + 1] == 'ended(123)'
    with pytest.raises(ValueError): ops.command('diagnostic-capture', 'a' * 64)
    with pytest.raises(ValueError): ops.command('diagnostic-capture', 'a' * 64, 'done(123)')


def test_no_dependency_on_independent_gpu_job():
    with pytest.raises(ValueError): ops.command('diagnostic', 'a' * 64, 'ended(123)')
    assert '-w' not in ops.command('diagnostic', 'a' * 64)


def test_fresh_output_does_not_overlap_original_studies():
    assert ops.parent('diagnostic') == ops.CORE_ROOT / 'diagnostic-protenix'
    assert '20261021-brd4-jq1-25-fixed-lrboost-01' not in str(ops.parent('diagnostic'))
    assert ops.ROOT.is_relative_to(ops.CORE_ROOT)


def test_exclusive_record_prohibits_blind_retry(tmp_path):
    p = tmp_path / 'unknown.json'; ops.exclusive(p, {'outcome': 'unknown'})
    with pytest.raises(FileExistsError): ops.exclusive(p, {'outcome': 'submitted'})
