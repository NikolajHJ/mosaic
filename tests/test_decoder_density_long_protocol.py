"""Allocated metadata/orchestration checks for100epoch serial continuation."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from latentmol.decoder_density_long_protocol import validate_long_protocol, segments


def protocol():
    return json.loads((Path(__file__).parents[1] / 'configs/decoder_density_100_study.json').read_text())


def base():
    return json.loads((Path(__file__).parents[1] / 'configs/decoder_density_study.json').read_text())


def submission():
    path = Path(__file__).parents[1] / 'scripts/hpc/submit_decoder_density_long.py'
    spec = importlib.util.spec_from_file_location('long_density_submit_fixture', path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.mark.parametrize('change', ['horizon', 'milestones', 'float_epoch', 'budget', 'source', 'path',
                                  'prefix', 'retention', 'compression', 'unknown'])
def test_long_protocol_rejects_scientific_or_allocation_drift(change):
    value = deepcopy(protocol())
    if change == 'horizon': value['endpoint_epochs'] = 90
    elif change == 'milestones': value['milestones'] = [25, 50, 100]
    elif change == 'float_epoch': value['milestones'][0] = 25.
    elif change == 'budget': value['segment_budget_seconds'] = 23 * 3600
    elif change == 'source': value['base_archive_sha256'] = 'latest'
    elif change == 'path': value['base_release'] = '/tmp/anything'
    elif change == 'prefix': value['prefix_policy'] = 'reuse_any_partial'
    elif change == 'retention': value['checkpoint_keep_recent'] = True
    elif change == 'compression': value['archive_evaluation_rows'] = 1
    else: value['unknown'] = True
    with pytest.raises(ValueError): validate_long_protocol(value)


def test_long_grid_is_four_preserved_trajectories_and_four_advance_only_segments():
    value = protocol()
    assert validate_long_protocol(value) is value
    rows = segments(base(), value)
    assert len(rows) == len({row['label'] for row in rows}) == 16
    for start in range(0, 16, 4):
        cohort = rows[start:start + 4]
        assert [row['start_epoch'] for row in cohort] == [10, 25, 50, 75]
        assert [row['stop_epoch'] for row in cohort] == [25, 50, 75, 100]
        assert sum(row['stop_epoch'] - row['start_epoch'] for row in cohort) == 90
        assert len({(row['seed'], row['coefficient']) for row in cohort}) == 1


def test_serial_plan_protects_both_gates_and_all_old_reservations():
    module = submission()
    work = Path('/work3/user/latentmol')
    release = work / 'releases/unique'
    rows = module.workflow(release, work, protocol(), base(), 'a' * 64, 'b' * 64, 'c' * 64,
        'segments', cpu_job='123', gate_job='456', base_job='789')
    for index, row in enumerate(rows):
        argv = row['arguments']
        assert argv[argv.index('-W') + 1] == '23:00'
        assert argv[argv.index('-w') + 1] == 'done(123) && done(456) && ended(789)'
        assert row['serial_after_stage'] == (rows[index - 1]['stage'] if index else None)
        assert row['job_command'][-2:] == ['--segment', str(index)]
    second = module.serial_after(rows[1], '987')
    assert second['arguments'].count('-w') == 1
    assert second['arguments'][second['arguments'].index('-w') + 1].endswith('&& ended(987)')
    assert rows[1]['arguments'][rows[1]['arguments'].index('-w') + 1] == 'done(123) && done(456) && ended(789)'
    with pytest.raises(ValueError): module.serial_after(rows[1], 'unsafe')
