import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

spec = importlib.util.spec_from_file_location('single_continuation', Path(__file__).with_name('continue_v100.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_build_input_preserves_committed_cache_and_rejects_corruption(tmp_path):
    source = tmp_path / 'source'
    for directory in ('states', 'hard-cache', 'best', 'observations'):
        (source / directory).mkdir(parents=True)
    for name in ('best/best.pt', 'observations/obs.json', 'observations/obs.pt'):
        (source / name).write_text(name)
    def asset(name):
        p = source / name
        return {'path': str(p), 'sha256': module.sha(p)}
    graph = 'a' * 64
    cache = source / 'hard-cache' / (graph + '.json')
    cache.write_text('{}')
    row = {'key': graph, 'sha256': module.sha(cache), 'previous': module.hashlib.sha256(b'').hexdigest()}
    (source / 'hard-cache/index.jsonl').write_text(json.dumps(row) + '\n' + '{"uncommitted": true}\n')
    signature = {'case_id': 'case-001'}
    sig = module.digest(signature)
    (source / 'manifest.json').write_text(json.dumps({'signature': signature, 'sha256': sig}))
    state = {'case_id': 'case-001', 'update': 2902, 'gate_binding_verified': True,
             'counts': {'iterations': 2902, 'consecutive_noop': 0}, 'z': torch.zeros(1, 3), 'z0': torch.zeros(1, 3),
             'signature_sha256': sig, 'best_valid': {'tensors': asset('best/best.pt')},
             'last_observation': {'record': asset('observations/obs.json'), 'tensors': asset('observations/obs.pt')},
             'hard_cache_entries': 1, 'hard_cache_chain': module.digest(row), 'last_hard_graph_sha256': graph}
    checkpoint = source / 'states/state-002902.pt'
    torch.save(state, checkpoint)
    target = tmp_path / 'isolated'
    result = module.build_input(source, 2902, module.sha(checkpoint), target)
    assert module.sha(result['path']) == module.sha(checkpoint)
    assert len((target / 'hard-cache/index.jsonl').read_text().splitlines()) == 1
    assert len((source / 'hard-cache/index.jsonl').read_text().splitlines()) == 2
    cache.write_text('corrupt')
    with pytest.raises(ValueError, match='Hard-cache chain'):
        module.build_input(source, 2902, module.sha(checkpoint), tmp_path / 'bad')

def test_completed_segment_schedules_only_same_case(monkeypatch, tmp_path):
    monkeypatch.setattr(module, 'WORK', tmp_path)
    monkeypatch.setattr(module, 'JOB_ROOT', tmp_path / 'jobs')
    monkeypatch.setenv('LSB_JOBID', '123')
    monkeypatch.setattr(sys, 'argv', ['continue_v100.py', '--run', 'testrun', '--after', '1', '--job', '456'])
    root = tmp_path / 'jobs/recovery-testrun'
    control = root / 'control'
    control.mkdir(parents=True)
    (control / 'segment-000001-worker.json').write_text(json.dumps({'job_id': '456'}))
    (control / 'segment-000001-request.json').write_text(json.dumps({'start_update': 2902}))
    segment = root / 'segment-000001'
    segment.mkdir()
    (segment / 'result.json').write_text(json.dumps({'status': 'completed_v100_segment', 'job_id': '456',
        'result': {'completed_update': 3402, 'scientific_stall': False}}))
    monkeypatch.setattr(module, 'committed', lambda source, start: {'update': 3402, 'current_state': {'sha256': 'statehash'}})
    monkeypatch.setattr(module, 'build_input', lambda *args: {'path': '/valid/state.pt', 'sha256': 'statehash'})
    calls = []
    def submit(args, command, release, intent, receipt):
        calls.append(args)
        return {'job_id': '789' if len(calls) == 1 else '790', 'status': 'submitted'}
    monkeypatch.setattr(module, 'submit', submit)
    module.main()
    request = json.loads((control / 'segment-000002-request.json').read_text())
    assert request['case'] == 'case-001' and request['start_update'] == 3402 and request['max_updates'] == 500
    assert calls[0][calls[0].index('-q') + 1] == 'gpuv100'
    assert 'ended(789)' in calls[1]
