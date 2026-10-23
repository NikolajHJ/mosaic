"""HPC fixture tests for the new panel; no pretrained-weight validation here."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from latentmol.boltz.active_set import make_branch, score_at, ContextBank, select_slots
from latentmol.boltz.protein_panel import (panel_summary, elemental_trajectory,
                                           seal_case, verify_terminal, child_config, next_attempt)
from latentmol.boltz.prepare import make_target_chain
from latentmol.multimodel.engine import Counters
from test_boltz_active_set_runtime import _config, _Decoder, _Scorer, SPEC


def test_element_diagnostics_keep_latent_gradient_and_capture_element_signal(tmp_path):
    cfg = _config(tmp_path)
    decoder, scorer = _Decoder(), _Scorer()
    point = torch.zeros(1, 1, dtype=torch.float64)
    selected = select_slots(decoder.decode(point), temperature=1, threshold=.5)
    branch = make_branch(scorer, ContextBank(cfg, SPEC), selected, cfg, 'c' * 64)
    original, old_gradient = score_at(decoder, point, branch, cfg, Counters(), 'fixture', gradients=True)
    cfg = replace(cfg, element_diagnostics=True, arms=('fixed_initial_mask',))
    observed, gradient = score_at(decoder, point, branch, cfg, Counters(), 'fixture', gradients=True)
    torch.testing.assert_close(gradient, old_gradient, rtol=0, atol=0)
    assert observed['loss'] == original['loss']
    elements = observed['elements']
    assert elements['atom_logit_gradient_norm'] > 0
    # This frozen arithmetic fixture rewards nitrogen; its signed loss contrasts
    # must retain the favorable nitrogen signal rather than merely report a norm.
    assert elements['probability_gradient_contrast_to_carbon'][0][2] < 0


def test_invalid_hard_graph_still_counts_heteroatom_emergence(tmp_path):
    folder = tmp_path / 'fixed_initial_mask'
    folder.mkdir()
    for step, atoms in [(0, [6, 6]), (1, [6, 7]), (2, [6, 6])]:
        row = {'step': step, 'argmax_smiles': None, 'latent_displacement': .2,
               'natural_selection': {'count': 2}, 'score': {'raw': .1, 'gradient_norm': .01,
               'elements': {'active_slots': [0, 1], 'hard_atomic_numbers': atoms,
               'conditional_expected_heteroatom_count': .4,
               'raw_expected_heteroatom_count_all_slots': .5,
               'carbon_logit_margin_over_best_heteroatom': [1, -1]}}}
        (folder / f'metrics-{step:04d}.json').write_text(json.dumps(row))
    (folder / 'metrics-0003.json').write_text('{')
    result = elemental_trajectory(tmp_path)
    assert result['ever_noncarbon'] and result['first_noncarbon_step'] == 1
    assert result['last_observed_noncarbon_count'] == 0
    assert result['unreadable_artifacts']


def test_terminal_seal_detects_table_and_file_mutation(tmp_path):
    attempt = tmp_path / 'cases' / 'ca2-c06' / 'attempt-001'
    attempt.mkdir(parents=True)
    evidence = attempt / 'failed.log'
    evidence.write_text('Retained failure')
    row = {'id': 'ca2-c06', 'status': 'preparation_failed', 'attempt_dir': str(attempt)}
    seal_case(row, tmp_path)
    verify_terminal(row, tmp_path)
    row['status'] = 'completed'
    with pytest.raises(ValueError, match='table'):
        verify_terminal(row, tmp_path)
    row['status'] = 'preparation_failed'
    evidence.write_text('changed')
    with pytest.raises(ValueError):
        verify_terminal(row, tmp_path)


def test_summary_retains_all_planned_cases():
    rows = [{'status': 'not_attempted'} for _ in range(100)]
    rows[0] = {'status': 'optimization_failed', 'elements': {
        'ever_noncarbon': True, 'last_observed_noncarbon_count': 1}}
    summary = panel_summary(rows)
    assert summary['planned_trajectories'] == 100
    assert summary['completed'] == 0 and summary['unstarted'] == 99
    assert summary['ever_hard_noncarbon_observed'] == 1


def test_interrupted_attempt_drops_stale_observations_and_retains_orphan(tmp_path):
    row = {'id': 'ca2-c06', 'target': 'ca2', 'accession': 'P00918',
           'sequence_sha256': 'a' * 64, 'msa_sha256': 'b' * 64,
           'count': 6, 'pose_seed': 17, 'status': 'optimizing', 'attempts': [],
           'elements': {'ever_noncarbon': True}, 'pilot': 'old-pilot'}
    orphan = tmp_path / 'cases' / row['id'] / 'attempt-002'
    orphan.mkdir(parents=True)
    path = next_attempt(row, tmp_path)
    assert path.name == 'attempt-003'
    assert 'elements' not in row and 'pilot' not in row
    assert row['attempts'][0]['elements']['ever_noncarbon']
    assert row['attempts'][1]['status'] == 'orphan_interrupted'


def test_generic_target_builder_checks_official_sequence_and_query(tmp_path):
    raw = {'primaryAccession': 'TEST', 'sequence': {'value': 'ACDE'}}
    (tmp_path / 'raw.json').write_text(json.dumps(raw))
    (tmp_path / 'msa.a3m').write_text('>TEST\nACDE\n')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    target = {'id': 'test', 'accession': 'TEST', 'sequence': 'ACDE',
              'sequence_sha256': hashlib.sha256(b'ACDE').hexdigest(),
              'raw_record': {'path': 'raw.json', 'sha256': digest(tmp_path / 'raw.json')},
              'msa': {'path': 'msa.a3m', 'sha256': digest(tmp_path / 'msa.a3m')}}
    manifest = tmp_path / 'targets.json'
    manifest.write_text(json.dumps({'version': 1, 'targets': [target],
                        'msa_policy': 'single_query', 'context_policy': 'test'}))
    path = make_target_chain(tmp_path / 'input', 6, manifest, 'test')
    assert path.exists()
    assert json.loads((path.parent / 'input_manifest.json').read_text())['seed_smiles'] == 'CCCCCC'
    target['sequence'] = 'AAAA'
    manifest.write_text(json.dumps({'version': 1, 'targets': [target]}))
    with pytest.raises(ValueError, match='hash'):
        make_target_chain(tmp_path / 'invalid', 6, manifest, 'test')
