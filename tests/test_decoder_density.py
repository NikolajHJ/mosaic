"""HPC CPU fixtures: exact1:1 tails, legacy quotas, paired RNG and planning."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from latentmol.decoder_density_protocol import read_protocol, validate_protocol, cases
from latentmol.decoder_finetune import FineTuneConfig, fit_case, make_panel
from latentmol.experiments import file_sha256
from latentmol.losses import LossConfig
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import GraphSpec
from test_decoder_finetune import TinyDataset


def protocol():
    return json.loads((Path(__file__).parents[1] / 'configs/decoder_density_study.json').read_text())


def test_density_contract_and_historical_default_field_shape(tmp_path):
    value = protocol()
    assert validate_protocol(value) is value
    assert FineTuneConfig(**value['fine_tune']).prior_count(24) == 24
    original = json.loads((Path(__file__).parents[1] / 'configs/decoder_validity_study.json').read_text())
    assert asdict(FineTuneConfig(**original['fine_tune'])) == original['fine_tune']
    assert FineTuneConfig().prior_count(24) == 16
    assert [case['label'] for case in cases(value)] == ['seed17-a0', 'seed17-a1', 'seed23-a0', 'seed23-a1']
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(value))
    assert read_protocol(path, file_sha256(path)) == value
    with pytest.raises(ValueError, match='hash mismatch'): read_protocol(path, 'a' * 64)


@pytest.mark.parametrize('change', ['prior', 'chemistry_subset', 'reconstruction', 'epochs', 'seeds', 'coefficient', 'budget', 'unknown'])
def test_density_rejects_changed_scientific_budget(change):
    value = deepcopy(protocol())
    if change == 'prior': value['fine_tune']['prior_batch_size'] = 64
    elif change == 'chemistry_subset': value['fine_tune']['chemistry_reconstruction_batch'] = 16
    elif change == 'reconstruction': value['fine_tune']['batch_size'] = 32
    elif change == 'epochs': value['fine_tune']['epochs'] = 3
    elif change == 'seeds': value['seeds'] = [17]
    elif change == 'coefficient': value['coefficients'] = [0., .1]
    elif change == 'budget': value['case_budget_seconds'] = 23 * 3600
    else: value['extra'] = True
    with pytest.raises(ValueError): validate_protocol(value)


@pytest.mark.parametrize('prior', [None, 2])
def test_actual_tail_coverage_and_matched_streams(tmp_path, prior):
    spec = GraphSpec(max_nodes=3)
    config = ModelConfig(hidden=8, latent=4, layers=1)
    model = GraphAutoencoder(spec, config)
    initial = tmp_path / 'original.pt'
    torch.save({'format_version': 1, 'graph_spec': spec.to_dict(), 'model_config': asdict(config),
        'model_state': model.state_dict(), 'loss_config': asdict(LossConfig()),
        'dataset_sha256': 'a' * 64, 'train_config': {'mode': 'vae', 'beta': .001}, 'epoch': 89}, initial)
    data = TinyDataset(spec)
    fine = FineTuneConfig(epochs=1, batch_size=2, chemistry_reconstruction_batch=2, prior_batch_size=prior)
    outputs = [fit_case(initial, data, data, tmp_path / 'runs', fine, seed=17, coefficient=alpha,
        train_limit=3, val_limit=2, validation_panel=make_panel(2, 4, 81000), device='cpu',
        source_root=tmp_path, budget_seconds=120, profile_updates=2) for alpha in (0., 1.)]
    assert outputs[0]['terminal_rng'] == outputs[1]['terminal_rng']
    assert outputs[0]['encoder_sha256'] == outputs[1]['encoder_sha256']
    for output in outputs:
        rows = output['update_profile']
        assert [r['real_points'] for r in rows] == [2, 1]
        assert [r['posterior_chemistry_points'] for r in rows] == [2, 1]
        assert [r['fresh_prior_points'] for r in rows] == ([2, 1] if prior is None else [2, 2])
        metrics = [json.loads(line) for line in (Path(output['run_dir']) / 'metrics.jsonl').read_text().splitlines()]
        assert len(metrics) == 1  # Profile rows must not corrupt resume receipts.
        assert metrics[0]['training_point_counts'] == {'reconstruction': 3, 'posterior_chemistry': 3,
            'fresh_prior': 3 if prior is None else 4, 'connectivity_graph_draws': 24 if prior is None else 28}


def submission():
    path = Path(__file__).parents[1] / 'scripts/hpc/submit_decoder_density.py'
    spec = importlib.util.spec_from_file_location('density_submit_fixture', path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try: spec.loader.exec_module(module)
    finally: sys.path.pop(0)
    return module


def test_workflow_uses_distinct_single_case_allocations_and_gate_dependencies():
    module = submission()
    work = Path('/work3/user/latentmol')
    release = work / 'releases/unique'
    rows = module.workflow(release, work, protocol(), release / 'configs/case.json', 'a'*64, 'b'*64,
                           'c'*64, 'cases', '123', '456')
    assert len(rows) == 4
    assert len({row['stage'] for row in rows}) == 4
    for index, row in enumerate(rows):
        assert row['arguments'][row['arguments'].index('-q') + 1] == 'gpul40s'
        assert row['arguments'][row['arguments'].index('-W') + 1] == '23:00'
        assert row['arguments'][row['arguments'].index('-w') + 1] == 'done(123) && done(456)'
        assert row['job_command'][-2:] == ['--case', str(index)]
    second = module.following_case(rows[1], '123', '456', '789')
    assert second['arguments'].count('-w') == 1
    assert second['arguments'][second['arguments'].index('-w') + 1] == 'done(123) && done(456) && ended(789)'
    assert rows[1]['arguments'][rows[1]['arguments'].index('-w') + 1] == 'done(123) && done(456)'


@pytest.mark.parametrize('change', ['none', 'seed', 'coefficient', 'train_count', 'split',
                                  'record_encoder', 'fit_encoder', 'checkpoint'])
def test_historical_reference_authenticates_actual_training_identity(tmp_path, change):
    """Case-row labels alone cannot authenticate a same-seed old endpoint."""
    from latentmol.decoder_density_study import historical

    value = protocol()
    value['historical_root'] = str(tmp_path / 'historical')
    segment = Path(value['historical_root']) / 'segment-1'
    run_dir = segment / 'old-training'
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / 'epoch-010.pt'
    # Metadata fixture only: the helper authenticates bytes without torch.load.
    checkpoint.write_bytes(b'synthetic completed-checkpoint identity fixture')
    encoder_sha = 'e' * 64
    fit = {'run_dir': str(run_dir), 'checkpoint': str(checkpoint),
           'sha256': file_sha256(checkpoint), 'encoder_sha256': encoder_sha}
    identity = {'initializer_sha256': value['initializer_sha256'],
                'dataset_sha256': value['dataset_sha256'], 'split_sha256': value['split_sha256'],
                'seed': 23, 'coefficient': 1.0, 'train_count': value['train_limit'],
                'frozen_encoder_sha256': encoder_sha,
                'config': {'prior_batch_size': 16, 'chemistry_reconstruction_batch': 16}}
    if change == 'seed': identity['seed'] = 17
    elif change == 'coefficient': identity['coefficient'] = 0.0
    elif change == 'train_count': identity['train_count'] -= 1
    elif change == 'split': identity['split_sha256'] = 'f' * 64
    elif change == 'record_encoder': identity['frozen_encoder_sha256'] = 'f' * 64
    elif change == 'fit_encoder': fit['encoder_sha256'] = 'f' * 64
    elif change == 'checkpoint': checkpoint.write_bytes(b'changed after receipt')
    record = {'source_archive_sha256': value['historical_archive_sha256'],
              'status': 'completed', 'completed_epochs': 10, 'config': identity}
    (run_dir / 'run.json').write_text(json.dumps(record))
    row = {'seed': 23, 'coefficient': 1.0, 'status': 'completed', 'fit': fit,
           'evaluation_dir': str(run_dir / 'final-prior'), 'reconstruction': {'validity': .99}}
    (segment / 'cases.json').write_text(json.dumps([row]))
    if change == 'none':
        result = historical(value, 23, encoder_sha)
        assert result['checkpoint_sha256'] == fit['sha256']
        assert result['record_sha256'] == file_sha256(run_dir / 'run.json')
        assert result['evaluation_dir'] == row['evaluation_dir']
    else:
        with pytest.raises(ValueError, match='Historical density reference identity'):
            historical(value, 23, encoder_sha)


@pytest.mark.parametrize('failure_phase', ['final_evaluation', 'paired_comparison'])
def test_post_fit_failure_preserves_completed_training_receipt(tmp_path, monkeypatch, failure_phase):
    """Mock stage failures must retain the completed fit, not claim full success.

    This checks orchestration only; no pretrained-model or endpoint quality is
    established by the synthetic assets and checkpoint below.
    """
    from latentmol import decoder_density_study as study

    value = protocol()
    root = tmp_path / 'density-study'
    gate_dir = root / 'gate'
    gate_dir.mkdir(parents=True)
    source_sha = 'c' * 64
    monkeypatch.setenv('LATENTMOL_SOURCE_SHA256', source_sha)
    (gate_dir / 'passed.json').write_text(json.dumps(
        {'status': 'passed', 'protocol': value, 'source_archive_sha256': source_sha}))
    (gate_dir / 'original.json').write_text(json.dumps({'reconstruction': {'validity': .99}}))
    run_dir = tmp_path / 'completed-fit'
    run_dir.mkdir()
    checkpoint = run_dir / 'epoch-010.pt'
    checkpoint.write_bytes(b'synthetic completed fit retained on evaluation failure')
    fit = {'run_dir': str(run_dir), 'checkpoint': str(checkpoint), 'sha256': file_sha256(checkpoint),
           'encoder_sha256': 'e' * 64, 'terminal_rng': {'loader': 'd' * 64}}
    data = [SimpleNamespace(spec='synthetic-spec')] * 3
    reference = SimpleNamespace(config=SimpleNamespace(latent=4))
    monkeypatch.setattr(study, 'assets', lambda *args: (data, reference, {}))
    monkeypatch.setattr(study, 'make_panel', lambda *args: 'synthetic-validation-panel')
    monkeypatch.setattr(study, 'fit_case', lambda *args, **kwargs: deepcopy(fit))
    monkeypatch.setattr(study, 'load_model', lambda *args, **kwargs: (SimpleNamespace(), {}))
    reconstruction, prior = {'validity': .98}, {'status': 'completed', 'latent_count': 4096}

    def evaluate(*args):
        if failure_phase == 'final_evaluation':
            raise RuntimeError('injected final evaluation failure')
        return reconstruction, prior

    def compare(*args, **kwargs):
        raise RuntimeError('injected paired comparison failure')

    monkeypatch.setattr(study, 'evaluate', evaluate)
    monkeypatch.setattr(study, 'paired_compare', compare)
    with pytest.raises(RuntimeError, match='injected'):
        study.run_case(value, root, tmp_path, 0, 'cpu')
    saved = json.loads((root / 'seed17-a0/summary.json').read_text())
    assert saved['status'] == 'failed_or_time_censored'
    assert saved['failure_phase'] == failure_phase
    assert saved['training_completed'] is True
    assert saved['fit'] == fit and saved['run_dir'] == str(run_dir)
    assert file_sha256(checkpoint) == saved['fit']['sha256']
    assert saved['reconstruction'] == (reconstruction if failure_phase == 'paired_comparison' else None)
    assert saved['prior'] == (prior if failure_phase == 'paired_comparison' else None)
