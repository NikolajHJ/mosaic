"""Stdlib contracts for separate unchanged-setting full-model speed pilots.

All model/gradient/FD evaluations run on allocated HPC. These helpers validate
their declared metadata, complete pilot records and preserved outcome scopes;
they never manufacture native evidence or promote production runs.
"""

import math
from pathlib import PurePosixPath
import re


VERSION = 'chemistry-full-model-speed-pilot-v1'
BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/'
MODEL_PRO = '8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2'
CONTEXT_PRO = 'b28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4'
MODEL_BOL = '395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16'
CONTEXT_BOL = 'fd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4'
POCKET_SHA = '9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43'
REFERENCE_SHA = '37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb'
CONTEXT_FILE_SHA = 'a18a331269170aac3281b0a2ecad313aab81756c9b77e048d7e2cbaa4d8f1093'
POLICY = {'workers': 4, 'matrix': 'matrix', 'early_updates': 10, 'high_updates': 10,
          'paired_atol': 2e-5, 'epsilons': [.01, .003, .001]}
PROFILES = {
    'protenix-random50k': {
        'source': BASE + 'protenix-extended-release-03/source',
        'manifest': 'da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562',
        'files': 643, 'model': MODEL_PRO, 'context': CONTEXT_PRO,
        'config': 'extended_random.json', 'arm': 'protenix', 'rate': .3, 'updates': 50000,
        'config_version': 'mosaic-brd4-jq1-protenix-random-chem-50000-v1',
        'extension_version': 'protenix-serial50k-v1',
        'initialization': {'kind': 'dirichlet', 'concentration': 1.0, 'seed': 17},
    },
    'boltz-chem30k': {
        'source': BASE + 'boltz-lr0025-release-03/source',
        'manifest': 'c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6',
        'files': 635, 'model': MODEL_BOL, 'context': CONTEXT_BOL,
        'config': 'boltz_serial_boltz2_chemistry.json', 'arm': 'boltz2_chemistry', 'rate': .0025, 'updates': 30000,
        'config_version': 'mosaic-brd4-jq1-boltz-lr0025-30000-v1',
        'extension_version': 'boltz-serial30k-v1',
        'initialization': {'kind': 'chain_uniform', 'mixing': .5},
    },
}
LOSS = {'version': 'repair-failure-guidance-v1', 'extra_valid_weight': 1.0,
        'partial_weight': 1.0, 'local_bond_weight': .1,
        'component_cut_weight': .1, 'nonempty_weight': .1}
SEARCH = {'max_edits': 3, 'checks_per_depth': 32, 'beam_width': 4,
          'probability_weight': 1.0, 'probability_floor': 1e-12}
PROGRESS = {'version': 'repair-progress-v1', 'max_edits': 6, 'checks_per_depth': 32,
            'beam_width': 4, 'probability_weight': 1.0, 'allow_atom_edits': False,
            'atom_edit_penalty': 3.0, 'max_atom_edits': 1}
POCKET = {'version': 'brd4-jq1-heavy-neighborhood-v1', 'pdb_id': '3MXF',
          'radius_angstrom': 6.0, 'anchor_author_residues': [81, 140],
          'protenix_contact_cutoff_angstrom': 8.0, 'boltz_anchor_max_distance_angstrom': 6.0,
          'minimum_ligand_fraction_within_pocket': .5, 'diffusion_gradient': False}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _binding(value, name):
    require(isinstance(value, dict), 'Missing explicit file binding: ' + name)
    path = value.get('path')
    require(isinstance(path, str) and PurePosixPath(path).is_absolute()
            and '..' not in PurePosixPath(path).parts and '\\' not in path,
            'Noncanonical file path: ' + name)
    require(type(value.get('bytes')) is int and value['bytes'] > 0,
            'Missing positive file length: ' + name)
    require(isinstance(value.get('sha256'), str) and re.fullmatch('[a-f0-9]{64}', value['sha256']) is not None,
            'Missing exact file SHA: ' + name)


def validate_plan(plan):
    require(isinstance(plan, dict) and plan.get('version') == VERSION, 'Wrong full-model pilot plan version')
    require(plan.get('profile') in PROFILES, 'Undeclared full-model pilot profile')
    required = {'version', 'profile', 'source', 'driver', 'inputs', 'expected', 'policy', 'timeout_seconds'}
    if plan['profile'] == 'boltz-chem30k':
        required.add('worker_python')
    require(required <= set(plan) <= required | {'python'}, 'Unknown or missing full-model pilot plan field')
    require('python' not in plan or plan['python'] == '/work3/s234818/mosaic/envs/protenix-tiny/bin/python',
            'Full-model coordinator interpreter changed')
    profile = PROFILES[plan['profile']]
    require(plan.get('source') == {'path': profile['source'], 'manifest_sha256': profile['manifest'],
                                  'files': profile['files']}, 'Original scientific source identity changed')
    _binding(plan.get('driver'), 'driver')
    require(plan['driver']['path'] == profile['source'] + '/examples/brd4_25_atoms.py',
            'Require the actual original validation driver')
    inputs = plan.get('inputs', {})
    names = {'config', 'cpu_certificate', 'reference', 'pocket'}
    if plan['profile'] == 'boltz-chem30k':
        names.add('prepared_context')
    require(set(inputs) == names, 'Exact original input binding scope differs')
    for name, binding in inputs.items():
        _binding(binding, name)
    require(inputs['config']['path'] == profile['source'] + '/examples/configs/' + profile['config'],
            'Original configuration path changed')
    require(inputs['reference']['sha256'] == REFERENCE_SHA and inputs['pocket']['sha256'] == POCKET_SHA,
            'Original reference or pocket changed')
    if plan['profile'] == 'boltz-chem30k':
        require(inputs['prepared_context']['sha256'] == CONTEXT_FILE_SHA
                and PurePosixPath(inputs['prepared_context']['path']).name == 'context.pt',
                'Detached original Boltz context changed')
        require(plan.get('worker_python') == '/work3/s234818/latentmol/env/boltz311/bin/python',
                'Original native Boltz worker interpreter changed')
    require(plan.get('expected') == {'model_digest': profile['model'], 'context_digest': profile['context'],
                                     'pocket_sha256': POCKET_SHA}, 'Frozen native identities changed')
    require(plan.get('policy') == POLICY, 'Candidate acceleration/pilot/FD policy changed')
    require(type(plan.get('timeout_seconds')) is int and 1 <= plan['timeout_seconds'] <= 6600,
            'Require a bounded pilot process within the two-hour allocation')
    return plan


def validate_config(config, profile):
    require(profile in PROFILES and isinstance(config, dict), 'Unknown original scientific configuration')
    p = PROFILES[profile]
    expected = {
        'version': p['config_version'], 'arm': p['arm'], 'seed': 17, 'steps': p['updates'],
        'n_atoms': 25, 'initial_step_size': p['rate'], 'max_backtracks': 1, 'pilot_steps': 10,
        'allocation_seconds': 82800, 'acceptance_policy': 'fixed_step', 'chemistry_enabled': True,
        'element_js_weight': 0.0, 'initialization': p['initialization'], 'loss': LOSS,
        'search': SEARCH, 'progress_search': PROGRESS, 'pocket': POCKET,
        'fixture': 'tests/data/brd4_3mxf/fixture_25.json',
        'repair_schedule': {'version': 'delayed-geometric-v1', 'initial': .001, 'final': 100.0,
                            'hold_updates': 5000, 'total_updates': p['updates']},
        'extension': {'version': p['extension_version'], 'start_global_update': 0,
                      'target_global_update': p['updates'], 'segment_updates': 10000, 'parent_job': None},
    }
    for name, value in expected.items():
        require(config.get(name) == value, 'Original scientific setting changed: ' + name)
    require(isinstance(config.get('label'), str) and bool(config['label']), 'Original label missing')
    require(set(config) == set(expected) | {'label'}, 'Unknown or removed scientific configuration field')
    return config


def require_allocation(environment, workers=4):
    require(type(workers) is int and workers == POLICY['workers'], 'Undeclared worker count')
    require(re.fullmatch(r'[1-9]\d*', environment.get('LSB_JOBID', '')) is not None,
            'Full-model evaluation requires actual allocated DTU HPC')
    require(environment.get('LSB_QUEUE') == 'gpul40s', 'Full-model pilot requires L40S queue')
    require(environment.get('LSB_DJOB_NUMPROC') == '8' and workers + 1 <= 8,
            'Require eight allocated CPU slots for the declared worker/coordinator scope')
    return {'queue': 'gpul40s', 'cpu_slots': 8, 'worker_processes': workers,
            'gpu_count': 1, 'allocation_seconds': 7200,
            'scope': 'Resource metadata only; actual L40S/backend evidence is separate'}


def _pilot(record, config, weight):
    require(record.get('status') == 'passed' and record.get('requested_updates') == 10,
            'Both complete disposable ten-update pilots are mandatory')
    require(record.get('acceptance_policy') == 'fixed_step'
            and record.get('nominal_step_size') == config['initial_step_size'], 'Pilot optimizer policy changed')
    updates, states = record.get('updates', []), record.get('states', [])
    require(len(updates) == 10 and len(states) == 11
            and [u.get('step') for u in updates] == list(range(10))
            and [s.get('step') for s in states] == list(range(11)), 'Incomplete or noncontiguous disposable pilot')
    for update in updates:
        attempts = update.get('attempts', [])
        require(len(attempts) == 1 and update.get('selected_trial') == 0
                and type(update['selected_trial']) is int and update.get('fallback_used') is False,
                'Pilot changed single-candidate/no-fallback policy')
        attempt = attempts[0]
        require(attempt.get('trial') == 0 and type(attempt['trial']) is int
                and attempt.get('rate') == update.get('accepted_step_size') == config['initial_step_size']
                and update.get('selection_policy') == 'fixed_step', 'Pilot fixed rate changed')
        require(update.get('repair_weight') == attempt.get('repair_weight') == weight,
                'Declared low/high chemistry coefficient changed')
        for value in [update.get('fixed_before'), update.get('fixed_after'), update.get('elapsed_seconds'),
                      attempt.get('autodiff_surrogate'), attempt.get('independent_primal_surrogate')]:
            require(_finite(value), 'Nonfinite or missing pilot scalar')
        require(update['elapsed_seconds'] >= 0
                and update['fixed_after'] == attempt['independent_primal_surrogate'], 'Pilot primal/timing mismatch')
        require(type(attempt.get('strict')) is bool and type(update.get('no_op')) is bool
                and type(update.get('fixed_increased')) is bool
                and update['fixed_increased'] is (update['fixed_after'] > update['fixed_before']),
                'Finite uphill/equal/noop bookkeeping is not authenticated')
        require(isinstance(attempt.get('components'), dict) and attempt['components']
                and all(_finite(v) for v in attempt['components'].values()), 'Finite component evidence missing')


def assess_result(result, plan, config):
    """Classify complete allocated evidence without waiving any science gate."""
    validate_plan(plan); validate_config(config, plan['profile'])
    variant = result.get('variant')
    require(variant in ('baseline', 'combined'), 'Unknown full-model pilot variant')
    require(result.get('original_results_changed') is False, 'Saved-result preservation is not authenticated')
    summary = result.get('driver_summary', {})
    require(summary.get('config') == config and summary.get('mode') == 'validate', 'Driver scientific scope changed')
    require(summary.get('source_manifest_sha256') == plan['source']['manifest_sha256']
            and summary.get('config_sha256') == plan['inputs']['config']['sha256'], 'Driver source/config binding changed')
    expected = plan['expected']
    require(summary.get('model_digest') == expected['model_digest']
            and summary.get('context_digest') == expected['context_digest']
            and summary.get('pocket_definition_sha256') == expected['pocket_sha256'], 'Native identity changed')
    frozen = summary.get('frozen_identity_after', {})
    require(frozen.get('model_digest') == expected['model_digest']
            and frozen.get('context_digest') == expected['context_digest'],
            'Frozen model/context before/after check missing')
    if plan['profile'] == 'protenix-random50k':
        require(set(frozen) == {'model_digest', 'context_digest'}, 'Unexpected Protenix frozen-state fields')
    else:
        # Preserve the original Boltz worker's authenticated transport metadata.
        # Its id/status/elapsed fields are not extra model/context state.
        require(frozen.get('status') == 'passed', 'Boltz frozen worker response failed')
    require(summary.get('diffusion_gradient') is False and summary.get('pocket_validation_status') == 'passed',
            'Pocket/detached-context gate missing')
    for key in ('native_hard_parity', 'native_gradient_checks', 'chemistry_assay'):
        require(summary.get(key, {}).get('status') == 'passed', 'Mandatory native/chemistry gate failed: ' + key)
    require(_finite(summary['chemistry_assay'].get('loss')) and summary['chemistry_assay']['loss'] > 0
            and _finite(summary['chemistry_assay'].get('gradient_norm'))
            and summary['chemistry_assay']['gradient_norm'] > 1e-10, 'Nonzero finite chemistry signal is required')
    coupled = result.get('coupled_checks', {})
    require(set(coupled) == {'low', 'high'} and all(v.get('status') == 'passed' for v in coupled.values()),
            'Both original coupled gradient checks must pass')
    pilots = result.get('pilot_reports', {})
    require(set(pilots) == {'early', 'high'}, 'Both low/high disposable pilots are required')
    _pilot(pilots['early'], config, .001); _pilot(pilots['high'], config, 100.0)
    runtime = summary.get('runtime_gate', {})
    means = runtime.get('mean_update_seconds', [])
    require(len(means) == 2 and all(_finite(v) and v >= 0 for v in means), 'Full runtime observations missing')
    projected = 1.5 * max(means) * 10000 + 1800
    require(runtime.get('allocation_seconds') == 82800
            and runtime.get('conservative_projected_seconds') == projected
            and runtime.get('status') == ('passed' if projected < 82800 else 'failed'), 'Original runtime gate formula changed')
    paired = result.get('paired_checks', {})
    if variant == 'baseline':
        require(paired.get('status') == 'not_applicable_original_baseline', 'Baseline cannot claim accelerated paired parity')
        if runtime['status'] == 'failed':
            error = 'RuntimeError: 10,000-update allocation-fit gate failed'
            require(summary.get('status') == 'failed' and summary.get('error') == error
                    and result.get('driver_error') == error, 'Only the exact original runtime failure is admissible')
            return {'status': 'passed_original_runtime_failure_preserved', 'runtime_failure_preserved': True,
                    'scientific_gates_passed': True, 'production_readiness_passed': False}
        require(summary.get('status') == 'passed' and result.get('driver_error') in (None, ''), 'Original baseline failed')
        return {'status': 'passed_original_baseline', 'runtime_failure_preserved': False,
                'scientific_gates_passed': True, 'production_readiness_passed': False}
    require(summary.get('status') == runtime.get('status') == 'passed'
            and result.get('driver_error') in (None, ''), 'Combined native/runtime checks must all pass')
    require(paired.get('status') == 'passed' and paired.get('atol') == 2e-5
            and paired.get('epsilons') == [.01, .003, .001], 'Original paired tolerance/FD policy changed')
    points = paired.get('points', [])
    require(len(points) == 4 and {p.get('label') for p in points} == {'early-0', 'early-10', 'high-0', 'high-10'},
            'Require all four paired low/high start/end checkpoints')
    require(all(p.get('status') == 'passed' and isinstance(p.get('preparation_identity'), str)
                and re.fullmatch('[a-f0-9]{64}', p['preparation_identity']) for p in points),
            'Paired fixed-reference checks missing or failed')
    return {'status': 'passed_combined_full_model_pilot', 'runtime_failure_preserved': False,
            'scientific_gates_passed': True, 'production_readiness_passed': False,
            'scope': 'Disposable pilot only; no production adoption or saved-run change'}
