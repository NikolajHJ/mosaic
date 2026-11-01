"""Portable report capacity/cache metadata; no arrays, model, scheduler or quota."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

OWNER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OWNER / 'scripts'))
shared = OWNER.parents[2] / 'scripts/hpc'
if shared.exists(): sys.path.insert(0, str(shared))
import quick_runtime as runtime
import storage_accounting as storage


def profile(root):
    return {'remote_root': str(root), 'allowed_roots': [str(root.parent)],
            'research_profile': 'exploratory', 'output_filesystem': 'work3',
            'request_cap_bytes': 2048, 'validation_cap_bytes': 64, 'report_file_cap': 4000}


def test_work3_delegates_only_phase_cap_and_does_not_load_legacy_guard(tmp_path):
    root = tmp_path/'reports'; parent = root/'requests'/'current'; parent.mkdir(parents=True)
    p = profile(root)
    with patch.object(storage, 'assess_output', side_effect=lambda *a: {'status':'passed'}) as assess, \
            patch.object(runtime.subprocess, 'check_output', side_effect=AssertionError('No HOME quota')):
        assert runtime.capacity(p, parent, 'render')['scientific_campaign_admission_granted'] is False
        assert runtime.capacity(p, parent, 'validate')['status'] == 'passed'
        assert assess.call_args_list[0].args == (parent, 2048, 4000)
        assert assess.call_args_list[1].args == (parent, 64, 4000)
    assert json.loads((parent/'capacity-render.json').read_text())['output_filesystem'] == 'work3'


def test_work3_preserves_shortfall_receipt_and_rejects_scope_escape(tmp_path):
    root = tmp_path/'reports'; parent = root/'requests'/'current'; parent.mkdir(parents=True)
    p = profile(root)
    with patch.object(storage, 'assess_output', return_value={'status':'failed','reason':'real_shortfall'}) as assess:
        with pytest.raises(ValueError, match='does not fit'): runtime.capacity(p, parent, 'render')
        assert json.loads((parent/'capacity-render.json').read_text())['reason'] == 'real_shortfall'
        with pytest.raises(ValueError, match='escaped'): runtime.capacity(p, tmp_path/'outside', 'render')
        with pytest.raises(ValueError, match='escaped'):
            runtime.capacity({**p,'allowed_roots':[str(tmp_path/'elsewhere')]}, parent, 'render')
        assert assess.call_count == 1


def test_external_cache_reuses_exact_terminal_proofs_read_only(tmp_path):
    root = tmp_path/'new'; root.mkdir()
    old = tmp_path/'old'; parent = old/'requests'/'previous'; viewer = parent/'viewer'; viewer.mkdir(parents=True)
    source = old/'engines'/('f'*64)/'source'; (source/'scripts').mkdir(parents=True)
    (source/'scripts/report_data.py').write_text('# original renderer\n')
    runtime.write(parent/'terminal-scheduler.json', {'returncode':0,'stdout':'Job <123> Status <DONE>'})
    (parent/'stdout.123.log').write_text('Subject: Job 123: fixture\nSuccessfully completed.\n')
    cert = {'cpu_job_id':'88'}
    for name,value in [('cpu-certificate.json',cert),('manifest.json',{'arms':[]}),('input-plan.json',{'arms':[]})]:
        runtime.write(viewer/name,value)
    receipt = {'status':'passed_saved_only_presentation','engine_pin':'f'*64,'cpu_job_id':'88',
        'cpu_certificate_sha256':runtime.inputs.sha(viewer/'cpu-certificate.json'),
        'files':{n:runtime.inputs.binding(viewer/n) for n in ('manifest.json','input-plan.json')}}
    runtime.write(viewer/'report-receipt.json',receipt)
    row = {'parent':str(parent),'source':str(source),'engine_pin':'f'*64,'job_id':'123',
        'receipt_sha256':runtime.inputs.sha(viewer/'report-receipt.json'),
        'terminal_proofs':[runtime.inputs.binding(parent/'terminal-scheduler.json')]}
    index = old/'cache-index.jsonl'; index.write_text(json.dumps(row)+'\n'); before = index.read_bytes()
    p = {**profile(root),'external_cache_roots':[str(old)]}
    with patch.object(runtime,'verify_source') as verify, patch.object(runtime,'certificate',return_value=cert):
        seeds = runtime.registered_seeds(p)
        assert len(seeds) == 1 and seeds[0]['root'] == viewer
        verify.assert_called_once_with(source,'f'*64)
        assert index.read_bytes() == before and not (root/'cache-index.jsonl').exists()
        (viewer/'input-plan.json').write_text('changed')
        with pytest.raises(ValueError,match='plan/manifest changed'): runtime.registered_seeds(p)
    with pytest.raises(ValueError,match='Unauthorized'):
        runtime.registered_seeds({**p,'external_cache_roots':[str(tmp_path.parent/'outside')]})
