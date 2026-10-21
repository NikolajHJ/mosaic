"""Saved-artifact contracts for the separate three-arm JQ1 25-slot report.

This module uses only the standard library. It authenticates persisted native
outcomes and never invokes a scorer, molecular conversion, repair or sampler.
"""

import hashlib
import json
from pathlib import Path
import re


POCKET = '9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43'
HOME = Path('/zhome/95/e/206121/mosaic-runs')
RELEASES = Path('/work3/s234818/mosaic/releases')
ARMS = {
    'protenix': {'job': '29770179', 'tag': '20261021-brd4-jq1-25-protenix-02',
        'metric': 'contact', 'color': '#246bb2', 'label': 'Protenix · JQ1 pocket contact',
        'manifest': '40ab2d4532c477b5074d6631587ae15188941d5167016b7421f993257a064438',
        'config': '4faaa44f4ee4adca2c3f1e0f7bdc33b0e3d939a9be312afe9b2f1c736f18252e',
        'ready': '70fa96f9009a0e7316fff341e8c091bfd5339f9c9e2f4456f39d35b6a2e13cf8',
        'model': '8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2',
        'context': 'b28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4'},
    'boltz2': {'job': '29770093', 'tag': '20261021-brd4-jq1-25-01',
        'metric': 'affinity', 'color': '#9a4dba', 'label': 'Boltz-2 · fixed JQ1-pocket pose',
        'manifest': '2503c2382b429ec6b79d7a256b2a2fccd09f7739a609e97a876d604d00dc063c',
        'config': 'cf8d582b79da2c6c65e032a938e9acabcc69a83ef2f490306deb060e7a03c851',
        'ready': '41901b3ad903c5d14b0344a1876564ee2481ace6a3adc8be5a33245214b4f387',
        'model': '395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16',
        'context': 'fd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4'},
    'boltz2_chemistry': {'job': '29770124', 'tag': '20261021-brd4-jq1-25-01',
        'metric': 'affinity', 'color': '#008979', 'label': 'Boltz-2 + delayed chemistry',
        'manifest': '2503c2382b429ec6b79d7a256b2a2fccd09f7739a609e97a876d604d00dc063c',
        'config': 'd6408d2512a6db8cff802369de019f6e878d2c12cbc3421996dff7eb7f3aeee3',
        'ready': '17ce0869f0a9888292e9017e03d407bf80d979a9cbfbed1367ae0785eb532394',
        'model': '395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16',
        'context': 'fd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4'},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def study_path(arm):
    return HOME / ARMS[arm]['tag'] / ('run-' + arm) / 'payload/study'


def verify_case(study, summary, *, steps=10000):
    """Full production budgets have no partial-mode override in the CLI."""
    case = read(study / 'case-0/report.json')
    require(summary['status'] == case['status'] == 'passed', 'Native study/case failed or partial')
    require(len(summary.get('cases', [])) == 1 and all(summary['cases'][0].get(k) == v for k, v in case.items()), 'Summary/case records differ')
    require(case['requested_updates'] == steps and len(case['updates']) == steps and len(case['states']) == steps + 1, 'Incomplete update/state budget')
    require([r['step'] for r in case['updates']] == list(range(steps)) and [r['step'] for r in case['states']] == list(range(steps + 1)), 'Noncontiguous saved records')
    for name in ('updates', 'states'):
        rows = [json.loads(line) for line in (study / f'case-0/{name}.jsonl').read_text().splitlines()]
        require(rows == case[name], 'Saved JSONL differs from terminal case')
    coupled = read(study / 'case-0/combined_initial_checks/report.json')
    require(coupled['status'] == 'passed' and coupled['directions'] == case['initial_checks'], 'Initial coupled derivative check differs/failed')
    counts = case['counts']
    require(counts['applied'] == steps and counts['forced_fallback'] == sum(bool(u['fallback_used']) for u in case['updates'])
            and counts['no_op'] == sum(bool(u['no_op']) for u in case['updates'])
            and counts['strict_improving'] == steps - counts['forced_fallback'], 'Saved update counts differ')
    return case


def authenticate_study(arm):
    """Exact per-arm pins permit different native metrics and scientific sources."""
    pin, study = ARMS[arm], study_path(arm)
    summary = read(study / 'summary.json')
    require(summary['job_id'] == pin['job'] and summary['mode'] == 'run', 'Wrong native job or mode')
    require(summary['source_manifest_sha256'] == pin['manifest'] and summary['config_sha256'] == pin['config'], 'Scientific source/configuration changed')
    config = RELEASES / pin['tag'] / 'source/examples/configs' / f'brd4_jq1_25_{arm}_10000.json'
    require(sha(config) == pin['config'] and summary['config'] == read(config), 'Native embedded/sealed configuration differs')
    source_manifest = config.parents[2] / 'source_manifest.json'
    require(sha(source_manifest) == pin['manifest'], 'Native source manifest bytes differ')
    source_files = summary['source'].get('files')
    manifest = read(source_manifest)
    require(bool(source_files) and all(manifest['files'].get(name) == value for name, value in source_files.items()), 'Native execution file map differs')
    require(summary['config']['n_atoms'] == 25 and summary['config']['steps'] == 10000 and summary['config']['arm'] == arm, 'Wrong atom/update/arm scope')
    require(summary.get('pocket_definition_sha256') == POCKET and summary.get('diffusion_gradient') is False, 'Pocket/no-diffusion identity changed')
    require(summary['pocket_validation_status'] == 'passed' and summary['native_hard_parity']['status'] == 'passed', 'Native hard parity/pocket gate failed')
    for field in ('model', 'context'):
        require(summary[field + '_digest'] == summary['frozen_identity_after'][field + '_digest'] == pin[field], 'Frozen identity differs')
    for field in ('native_gradient_checks', 'terminal_gradient_checks'):
        require(summary[field]['status'] == 'passed', 'Initial/terminal native derivative gate failed')
        require(read(study / field / 'report.json') == summary[field], 'Native derivative report/summary differ')
    ready = HOME / pin['tag'] / ('validate-' + arm) / 'payload/study/assessment.json'
    require(sha(ready) == pin['ready'], 'Passed readiness bytes changed')
    from campaign25_contract import verify_readiness
    verify_readiness(ready, pin['ready'], config, pin['manifest'])
    require(read(ready)['runtime_gate']['status'] == 'passed', 'Native readiness runtime gate failed')
    if arm != 'protenix':
        require(summary['context_file_sha256'] == 'a18a331269170aac3281b0a2ecad313aab81756c9b77e048d7e2cbaa4d8f1093', 'Fixed Boltz pose file differs')
    if arm == 'boltz2_chemistry':
        require(read(ready.with_name('summary.json'))['chemistry_assay']['status'] == 'passed', 'Native chemistry assay failed')
        for label in ('low', 'high'):
            require(read(ready.parent / (label + '_coupled_checks/report.json'))['status'] == 'passed', 'Native coupled readiness failed')
    case = verify_case(study, summary)
    if arm == 'boltz2_chemistry':
        require(case['states'][-1]['repair_weight'] == 50.0, 'Chemistry ramp did not reach 50')
    else:
        require(all(s['repair_weight'] == 0 for s in case['states']), 'Scorer-only arm acquired chemistry guidance')
    return study, summary, case


def scheduler_outcome(raw, job):
    normalized = re.sub(r'\s+', '', raw)
    require(f'Job<{job}>' in normalized and 'Queue<gpul40s>' in normalized, 'Scheduler identity/queue missing')
    match = re.search(r'Status<([^>]+)>', normalized)
    require(match is not None and match[1] in ('DONE', 'EXIT'), 'Actual final scheduler outcome unavailable')
    return match[1]
