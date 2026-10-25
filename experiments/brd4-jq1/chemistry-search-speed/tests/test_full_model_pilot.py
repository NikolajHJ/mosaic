"""Allocated fixtures for immutable full-model pilot scope and composition.

Metadata fixtures below are deliberately synthetic and never native proofs.
The small CPU loss/composition fixtures verify engineering only. Actual native
models, original finite-difference gates and complete pilots run on allocated GPU.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import full_model_contract as contract
import full_model_pilot as pilot


def plan(profile='protenix-random50k'):
    p = contract.PROFILES[profile]
    def binding(path, digest='a' * 64):
        return {'path': path, 'bytes': 123, 'sha256': digest}
    inputs = {
        'config': binding(p['source'] + '/examples/configs/' + p['config']),
        'cpu_certificate': binding(contract.BASE + 'fixture/cpu-certificate.json'),
        'reference': binding(contract.BASE + 'fixture/reference.json', contract.REFERENCE_SHA),
        'pocket': binding(contract.BASE + 'fixture/pocket.json', contract.POCKET_SHA),
    }
    result = {
        'version': contract.VERSION, 'profile': profile,
        'source': {'path': p['source'], 'manifest_sha256': p['manifest'], 'files': p['files']},
        'driver': binding(p['source'] + '/examples/brd4_25_atoms.py'),
        'inputs': inputs, 'expected': {'model_digest': p['model'], 'context_digest': p['context'],
                                     'pocket_sha256': contract.POCKET_SHA},
        'policy': deepcopy(contract.POLICY), 'timeout_seconds': 6600,
    }
    if profile == 'boltz-chem30k':
        inputs['prepared_context'] = binding('/zhome/95/e/206121/fixture/prepared/context.pt', contract.CONTEXT_FILE_SHA)
        result['worker_python'] = '/work3/s234818/latentmol/env/boltz311/bin/python'
    return result


def configuration(profile='protenix-random50k'):
    p = contract.PROFILES[profile]
    return {
        'seed': 17, 'steps': p['updates'], 'initial_step_size': p['rate'], 'max_backtracks': 1,
        'pilot_steps': 10, 'allocation_seconds': 82800, 'loss': deepcopy(contract.LOSS),
        'search': deepcopy(contract.SEARCH), 'progress_search': deepcopy(contract.PROGRESS),
        'initialization': deepcopy(p['initialization']), 'version': p['config_version'],
        'arm': p['arm'], 'n_atoms': 25, 'element_js_weight': 0.0, 'chemistry_enabled': True,
        'repair_schedule': {'version': 'delayed-geometric-v1', 'initial': .001, 'final': 100.0,
                            'hold_updates': 5000, 'total_updates': p['updates']},
        'fixture': 'tests/data/brd4_3mxf/fixture_25.json', 'label': 'Synthetic metadata fixture only',
        'pocket': deepcopy(contract.POCKET), 'acceptance_policy': 'fixed_step',
        'extension': {'version': p['extension_version'], 'start_global_update': 0,
                      'target_global_update': p['updates'], 'segment_updates': 10000, 'parent_job': None},
    }


def pilot_report(config, weight):
    updates = []
    for step in range(10):
        updates.append({
            'step': step, 'attempts': [{'trial': 0, 'rate': config['initial_step_size'],
                'repair_weight': weight, 'autodiff_surrogate': 2.0,
                'independent_primal_surrogate': 2.0, 'strict': False,
                'components': {'native': 1.0, 'repair': 1.0}}],
            'selected_trial': 0, 'accepted_step_size': config['initial_step_size'],
            'selection_policy': 'fixed_step', 'fallback_used': False, 'repair_weight': weight,
            'fixed_before': 1.0, 'fixed_after': 2.0, 'fixed_increased': True,
            'no_op': False, 'elapsed_seconds': 1.0,
        })
    return {'status': 'passed', 'requested_updates': 10, 'acceptance_policy': 'fixed_step',
            'nominal_step_size': config['initial_step_size'], 'updates': updates,
            'states': [{'step': step} for step in range(11)]}


def result(profile='protenix-random50k', variant='combined', runtime_failure=False):
    p, c = plan(profile), configuration(profile)
    means = [6.0, 4.0] if runtime_failure else [1.0, 2.0]
    error = 'RuntimeError: 10,000-update allocation-fit gate failed' if runtime_failure else None
    frozen = {'model_digest': p['expected']['model_digest'], 'context_digest': p['expected']['context_digest']}
    if profile == 'boltz-chem30k':
        frozen.update(status='passed', id=13, elapsed_seconds=.01)
    return {
        'variant': variant, 'original_results_changed': False, 'driver_error': error,
        'driver_summary': {
            'status': 'failed' if runtime_failure else 'passed', 'error': error,
            'mode': 'validate', 'config': c, 'config_sha256': p['inputs']['config']['sha256'],
            'source_manifest_sha256': p['source']['manifest_sha256'],
            'model_digest': p['expected']['model_digest'], 'context_digest': p['expected']['context_digest'],
            'pocket_definition_sha256': contract.POCKET_SHA, 'pocket_validation_status': 'passed',
            'diffusion_gradient': False, 'frozen_identity_after': frozen,
            'native_hard_parity': {'status': 'passed'}, 'native_gradient_checks': {'status': 'passed'},
            'chemistry_assay': {'status': 'passed', 'loss': 1.0, 'gradient_norm': 1.0},
            'runtime_gate': {'status': 'failed' if runtime_failure else 'passed', 'mean_update_seconds': means,
                'allocation_seconds': 82800, 'conservative_projected_seconds': 1.5 * max(means) * 10000 + 1800},
        },
        'coupled_checks': {'low': {'status': 'passed'}, 'high': {'status': 'passed'}},
        'pilot_reports': {'early': pilot_report(c, .001), 'high': pilot_report(c, 100.0)},
        'paired_checks': ({'status': 'not_applicable_original_baseline'} if variant == 'baseline' else {
            'status': 'passed', 'atol': 2e-5, 'epsilons': [.01, .003, .001],
            'points': [{'label': label, 'status': 'passed', 'preparation_identity': 'b' * 64}
                       for label in ('early-0', 'early-10', 'high-0', 'high-10')]}),
    }


@pytest.mark.parametrize('profile', tuple(contract.PROFILES))
def test_original_scientific_config_and_plan_validate_without_mutation(profile):
    p, c = plan(profile), configuration(profile)
    before = deepcopy((p, c))
    assert contract.validate_plan(p) is p
    assert contract.validate_config(c, profile) is c
    assert (p, c) == before


@pytest.mark.parametrize('profile', tuple(contract.PROFILES))
def test_actual_generated_original_cli_is_validate_only_with_all_provenance(profile):
    p = plan(profile)
    argv = pilot.build_driver_argv(p, Path('/work3/s234818/mosaic/fixture/disposable/study'))
    assert argv[1] == p['driver']['path']
    assert argv[argv.index('--mode') + 1] == 'validate'
    assert argv[argv.index('--offset') + 1] == '0'
    assert argv[argv.index('--segment-steps') + 1] == '10000'
    for flag, key in (('--config', 'config'), ('--cpu-certificate', 'cpu_certificate'),
                      ('--reference', 'reference'), ('--pocket', 'pocket')):
        assert argv[argv.index(flag) + 1] == p['inputs'][key]['path']
    assert argv[argv.index('--cpu-certificate-sha256') + 1] == p['inputs']['cpu_certificate']['sha256']
    assert argv[argv.index('--manifest') + 1] == p['source']['manifest_sha256']
    assert '--ready' not in argv and '--resume-snapshot' not in argv
    if profile == 'boltz-chem30k':
        assert argv[argv.index('--context-sha256') + 1] == contract.CONTEXT_FILE_SHA
        assert argv[argv.index('--worker-python') + 1] == p['worker_python']


def test_wrapper_real_argument_parser_accepts_only_declared_child_scopes():
    args = pilot.parse_args(['--plan', '/work3/fixture/plan.json', '--output', '/work3/fixture/out'])
    assert args.child is None and args.baseline is None
    combined = pilot.parse_args(['--plan', '/work3/fixture/plan.json', '--output', '/work3/fixture/out',
                                '--child', 'combined', '--baseline', '/work3/fixture/baseline'])
    assert combined.child == 'combined'
    with pytest.raises(SystemExit):
        pilot.parse_args(['--plan', '/work3/fixture/plan.json', '--output', '/work3/fixture/out', '--child', 'run'])


ORIGINAL_COMMAND_CHECK = r'''
import json,os,runpy,sys
from pathlib import Path
plan_path=Path(sys.argv[1]);output=Path(sys.argv[2]);fault=sys.argv[3]
sys.path.insert(0,str(plan_path.parent.parent/'scripts'))
from full_model_pilot import build_driver_argv
from full_model_contract import validate_plan,validate_config
plan=json.loads(plan_path.read_text());validate_plan(plan)
config=json.loads(Path(plan['inputs']['config']['path']).read_text())
validate_config(config,plan['profile'])
source=Path(plan['source']['path'])
sys.path[:0]=[str(source/'scripts'),str(source/'tests'),str(source/'src')]
import campaign25_contract
argv=build_driver_argv(plan,output)
if fault=='missing_certificate_sha':
 k=argv.index('--cpu-certificate-sha256');del argv[k:k+2]
class BeforeModelSentinel(Exception):pass
called=[]
def stop_at_actual_certificate_call(path,digest,manifest):
 assert str(path)==plan['inputs']['cpu_certificate']['path']
 assert digest==plan['inputs']['cpu_certificate']['sha256']
 assert manifest==plan['source']['manifest_sha256']
 called.append(True)
 raise BeforeModelSentinel('actual_original_certificate_dispatch_reached')
campaign25_contract.verify_cpu_certificate=stop_at_actual_certificate_call
sys.argv=argv[1:]
try:
 runpy.run_path(argv[1],run_name='__main__')
except BeforeModelSentinel:
 assert called==[True] and not output.exists()
 print('MOSAIC_ORIGINAL_COMMAND_ONLY:'+json.dumps({'status':'actual_parser_config_scope_reached_before_model',
  'argv':argv,'model_setup_entered':False,'native_authentication_passed':False,'optimizer_updates':0}))
else:
 raise RuntimeError('Original command did not reach its exact pre-model sentinel')
'''


@pytest.mark.parametrize('profile, filename', (('protenix-random50k', 'protenix.json'),
                                              ('boltz-chem30k', 'boltz.json')))
@pytest.mark.parametrize('fault', ('none', 'missing_certificate_sha'))
def test_allocated_actual_original_entrypoint_parser_config_scope_dispatch(profile, filename, fault, tmp_path):
    # Run the unchanged original main in an isolated allocated CPU process. Only
    # its certificate call is a deliberate stop, BEFORE any model setup/native
    # proof/numerical work. This is command dispatch evidence, not readiness.
    plan_path = Path(__file__).resolve().parents[1] / 'plans' / filename
    actual_plan = json.loads(plan_path.read_text())
    assert actual_plan['profile'] == profile
    output = tmp_path / ('pre-model-' + fault)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='')
    completed = subprocess.run([sys.executable, '-c', ORIGINAL_COMMAND_CHECK, str(plan_path), str(output), fault],
                               env=env, capture_output=True, text=True, timeout=90)
    marker = 'MOSAIC_ORIGINAL_COMMAND_ONLY:'
    lines = [line[len(marker):] for line in completed.stdout.splitlines() if line.startswith(marker)]
    if fault == 'none':
        assert completed.returncode == 0, completed.stderr
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row['status'] == 'actual_parser_config_scope_reached_before_model'
        assert row['model_setup_entered'] is False and row['native_authentication_passed'] is False
        assert row['optimizer_updates'] == 0
        assert row['argv'] == pilot.build_driver_argv(actual_plan, output)
    else:
        assert completed.returncode == 2 and not lines
        assert '--cpu-certificate-sha256' in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize('field, value', (
    ('initial_step_size', .03), ('max_backtracks', 2), ('acceptance_policy', 'best_primal'),
    ('seed', 18), ('n_atoms', 6), ('element_js_weight', .1), ('chemistry_enabled', False),
))
def test_scientific_setting_changes_rejected(field, value):
    c = configuration(); c[field] = value
    with pytest.raises(ValueError, match='scientific setting'):
        contract.validate_config(c, 'protenix-random50k')


@pytest.mark.parametrize('field', ('loss', 'search', 'progress_search', 'repair_schedule', 'initialization', 'pocket', 'extension'))
def test_nested_original_protocol_cannot_change_or_gain_fields(field):
    c = configuration(); c[field]['new_override'] = True
    with pytest.raises(ValueError):
        contract.validate_config(c, 'protenix-random50k')


@pytest.mark.parametrize('fault', ('source', 'config_path', 'file_sha', 'context', 'policy', 'timeout'))
def test_plan_input_source_and_execution_bounds_fail_closed(fault):
    p = plan('boltz-chem30k')
    if fault == 'source': p['source']['manifest_sha256'] = 'c' * 64
    elif fault == 'config_path': p['inputs']['config']['path'] += '.changed'
    elif fault == 'file_sha': p['inputs']['reference']['sha256'] = 'c' * 64
    elif fault == 'context': p['expected']['context_digest'] = 'c' * 64
    elif fault == 'policy': p['policy']['early_updates'] = 9
    else: p['timeout_seconds'] = 7201
    with pytest.raises(ValueError):
        contract.validate_plan(p)


@pytest.mark.parametrize('profile', tuple(contract.PROFILES))
def test_only_original_baseline_runtime_failure_is_preserved_without_success_relabel(profile):
    answer = contract.assess_result(result(profile, 'baseline', True), plan(profile), configuration(profile))
    assert answer['status'] == 'passed_original_runtime_failure_preserved'
    assert answer['runtime_failure_preserved'] is True and answer['production_readiness_passed'] is False


@pytest.mark.parametrize('profile', tuple(contract.PROFILES))
def test_combined_complete_native_pilot_pass_does_not_authorize_production(profile):
    answer = contract.assess_result(result(profile), plan(profile), configuration(profile))
    assert answer['status'] == 'passed_combined_full_model_pilot'
    assert answer['scientific_gates_passed'] is True and answer['production_readiness_passed'] is False


@pytest.mark.parametrize('fault', ('raw_gradient', 'hard_parity', 'assay_zero', 'coupled_high', 'frozen', 'pilot_short',
                                  'pilot_rate', 'pilot_weight', 'fallback', 'paired_missing', 'runtime'))
def test_combined_never_waives_original_scientific_or_completion_gates(fault):
    r = result()
    if fault == 'raw_gradient': r['driver_summary']['native_gradient_checks']['status'] = 'failed'
    elif fault == 'hard_parity': r['driver_summary']['native_hard_parity']['status'] = 'failed'
    elif fault == 'assay_zero': r['driver_summary']['chemistry_assay']['gradient_norm'] = 0.0
    elif fault == 'coupled_high': r['coupled_checks']['high']['status'] = 'failed'
    elif fault == 'frozen': r['driver_summary']['frozen_identity_after']['model_digest'] = 'c' * 64
    elif fault == 'pilot_short': r['pilot_reports']['high']['updates'].pop()
    elif fault == 'pilot_rate': r['pilot_reports']['early']['updates'][0]['attempts'][0]['rate'] /= 2
    elif fault == 'pilot_weight': r['pilot_reports']['high']['updates'][0]['repair_weight'] = 50.0
    elif fault == 'fallback': r['pilot_reports']['early']['updates'][0]['fallback_used'] = True
    elif fault == 'paired_missing': r['paired_checks']['points'].pop()
    else: r = result(runtime_failure=True)
    with pytest.raises(ValueError):
        contract.assess_result(r, plan(), configuration())


def test_baseline_non_runtime_failure_cannot_be_hidden_by_runtime_exception():
    r = result(variant='baseline', runtime_failure=True)
    r['driver_summary']['native_gradient_checks']['status'] = 'failed'
    with pytest.raises(ValueError):
        contract.assess_result(r, plan(), configuration())
    r = result(variant='baseline', runtime_failure=True)
    r['driver_error'] = 'RuntimeError: unrelated execution failure'
    with pytest.raises(ValueError):
        contract.assess_result(r, plan(), configuration())


def test_nonfinite_pilot_scalars_rejected_and_finite_uphill_equal_noop_allowed():
    r = result(); update = r['pilot_reports']['early']['updates'][0]
    update['fixed_after'] = update['attempts'][0]['independent_primal_surrogate'] = update['fixed_before']
    update['fixed_increased'], update['no_op'] = False, True
    assert contract.assess_result(r, plan(), configuration())['status'] == 'passed_combined_full_model_pilot'
    update['attempts'][0]['components']['repair'] = float('nan')
    with pytest.raises(ValueError, match='component'):
        contract.assess_result(r, plan(), configuration())


@pytest.mark.parametrize('bad', ('queue', 'slots', 'job'))
def test_worker_and_native_processes_must_fit_actual_declared_allocation(bad):
    env = {'LSB_JOBID': '29809999', 'LSB_QUEUE': 'gpul40s', 'LSB_DJOB_NUMPROC': '8'}
    assert contract.require_allocation(env)['worker_processes'] == 4
    if bad == 'queue': env['LSB_QUEUE'] = 'hpc'
    elif bad == 'slots': env['LSB_DJOB_NUMPROC'] = '4'
    else: env['LSB_JOBID'] = ''
    with pytest.raises(ValueError):
        contract.require_allocation(env)


def test_real_adapter_composition_restores_on_success_and_exception():
    # Allocated CPU-only engineering check: matrix JAX kernel is portable here;
    # actual GPU device/native correctness remains the separate full-pilot gate.
    from hybrid_search import install_hybrid
    from gpu_matrix import install_matrix_loss
    from mosaic.small_molecule.chemistry import preparation, loss
    original = (preparation.search_atom_preserving_repair, preparation.search_progress_repair,
                loss._local, loss._compiled_terms)
    for fail in (False, True):
        try:
            with install_hybrid(workers=2, max_entries=16):
                with install_matrix_loss('matrix'):
                    assert loss._local is not original[2]
                    assert preparation.search_atom_preserving_repair is not original[0]
                    if fail:
                        raise LookupError('fixture composition error')
        except LookupError:
            assert fail
        actual = (preparation.search_atom_preserving_repair, preparation.search_progress_repair,
                  loss._local, loss._compiled_terms)
        assert all(left is right for left, right in zip(original, actual))


def test_paired_read_only_original_math_temporarily_restores_then_selected_math_returns():
    from gpu_matrix import install_matrix_loss
    from mosaic.small_molecule.chemistry import loss
    original = loss._local, loss._compiled_terms
    with install_matrix_loss('matrix'):
        selected = loss._local, loss._compiled_terms
        with pytest.raises(LookupError):
            with pilot._original_loss_bindings(original):
                assert (loss._local, loss._compiled_terms) == original
                raise LookupError('paired fixture')
        assert (loss._local, loss._compiled_terms) == selected
    assert (loss._local, loss._compiled_terms) == original


def test_optional_coordinator_interpreter_is_the_exact_declared_environment():
    p = plan(); p['python'] = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
    assert contract.validate_plan(p) is p
    assert pilot.build_driver_argv(p, Path('/work3/fixture/study'))[0] == p['python']


@pytest.mark.parametrize('fault', ('python', 'unknown', 'pro_worker'))
def test_unknown_plan_fields_or_arbitrary_interpreters_cannot_change_scope(fault):
    p = plan()
    if fault == 'python': p['python'] = '/work3/fixture/other-python'
    elif fault == 'unknown': p['resume'] = True
    else: p['worker_python'] = '/work3/s234818/latentmol/env/boltz311/bin/python'
    with pytest.raises(ValueError):
        contract.validate_plan(p)


@pytest.mark.parametrize('turnovers', (1, 2))
def test_live_output_atomic_rename_retries_whole_scan_without_omitting_file(monkeypatch, tmp_path, turnovers):
    leaf = tmp_path / 'snapshot.npz'; leaf.write_bytes(b'fixture')
    original = Path.lstat
    calls = []
    def transient(path, *args, **kwargs):
        if path == leaf:
            calls.append(True)
            if len(calls) <= turnovers:
                raise FileNotFoundError('Fixture atomic partial rename')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'lstat', transient)
    assert pilot.output_bytes(tmp_path) == len(b'fixture')
    assert len(calls) == turnovers + 1


def test_continuously_unstable_output_tree_is_not_a_partial_inventory_pass(monkeypatch, tmp_path):
    leaf = tmp_path / 'snapshot.npz'; leaf.write_bytes(b'fixture')
    original = Path.lstat
    calls = []
    def unstable(path, *args, **kwargs):
        if path == leaf:
            calls.append(True)
            raise FileNotFoundError('Fixture continued atomic turnover')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'lstat', unstable)
    with pytest.raises(FileNotFoundError):
        pilot.output_bytes(tmp_path)
    assert len(calls) == 3


def _record_spawn(monkeypatch):
    original = subprocess.Popen
    records = []
    def recorded(*args, **kwargs):
        process = original(*args, **kwargs)
        records.append((process, kwargs['env']['MOSAIC_FULL_MODEL_PILOT_OWNER']))
        return process
    monkeypatch.setattr(pilot.subprocess, 'Popen', recorded)
    return records


def test_timeout_stops_authenticated_child_and_grandchild_not_just_leader(monkeypatch, tmp_path):
    # Allocated Linux CPU engineering fixture only: no scientific/model imports.
    records = _record_spawn(monkeypatch)
    comparison = tmp_path / 'comparison'; comparison.mkdir()
    ready = comparison / 'ready.json'
    code = ('import json,os,subprocess,sys,time; '
            'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]); '
            'open(sys.argv[1],"w").write(json.dumps({"parent":os.getpid(),"grandchild":p.pid})); '
            'time.sleep(30)')
    try:
        with (tmp_path / 'child.log').open('xb') as stream:
            with pytest.raises(subprocess.TimeoutExpired):
                pilot.run_isolated_child([sys.executable, '-c', code, str(ready)], tmp_path,
                    dict(os.environ), stream, 2.0, comparison)
        process, owner = records[0]
        row = json.loads(ready.read_text())
        assert row['parent'] == process.pid and row['grandchild'] != process.pid
        assert process.poll() is not None
        assert pilot._owned_group_members(process.pid, owner) == []
    finally:
        for process, owner in records:
            if process.poll() is None or pilot._owned_group_members(process.pid, owner):
                pilot._stop_own_group(process, process.pid, owner)


def test_output_cap_failure_stops_own_child_without_reducing_production_cap(monkeypatch, tmp_path):
    assert pilot.OUTPUT_CAP == 2 * 1024**3
    comparison = tmp_path / 'comparison'; comparison.mkdir()
    (comparison / 'fixture.bin').write_bytes(b'xx')
    records = _record_spawn(monkeypatch)
    monkeypatch.setattr(pilot, 'OUTPUT_CAP', 1)  # Two-byte fixture, never a production guard.
    try:
        with (tmp_path / 'child.log').open('xb') as stream:
            with pytest.raises(RuntimeError, match='output cap'):
                pilot.run_isolated_child([sys.executable, '-c', 'import time;time.sleep(30)'],
                    tmp_path, dict(os.environ), stream, 2.0, comparison)
        process, owner = records[0]
        assert process.poll() is not None and pilot._owned_group_members(process.pid, owner) == []
    finally:
        for process, owner in records:
            if process.poll() is None or pilot._owned_group_members(process.pid, owner):
                pilot._stop_own_group(process, process.pid, owner)


@pytest.mark.parametrize('exit_code', (0, 7))
def test_fast_child_exit_is_reaped_and_cannot_be_fabricated_as_success(monkeypatch, tmp_path, exit_code):
    records = _record_spawn(monkeypatch)
    with (tmp_path / 'child.log').open('xb') as stream:
        if exit_code:
            with pytest.raises(subprocess.CalledProcessError) as error:
                pilot.run_isolated_child([sys.executable, '-c', 'raise SystemExit(7)'],
                    tmp_path, dict(os.environ), stream, 5.0, tmp_path)
            assert error.value.returncode == 7
        else:
            row = pilot.run_isolated_child([sys.executable, '-c', 'pass'],
                tmp_path, dict(os.environ), stream, 5.0, tmp_path)
            assert row['returncode'] == 0 and row['new_session'] is True
    process, owner = records[0]
    assert process.poll() == exit_code and pilot._owned_group_members(process.pid, owner) == []


def test_parent_termination_handler_restored_after_cleanup_exception():
    original = signal.getsignal(signal.SIGTERM)
    with pytest.raises(InterruptedError):
        with pilot._parent_sigterm_cleanup():
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    assert signal.getsignal(signal.SIGTERM) == original


def _actual_preparation_record():
    # Real tiny original preparation, allocated CPU only. Use its actual flattened
    # record schema rather than a synthetic nested-receipt substitute.
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.chemistry import GraphLogits, mosaic_spec, prepare_failure_guidance
    from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig
    spec = mosaic_spec(6)
    atoms = np.full((6, 6), -1.0, dtype=np.float32); atoms[:, 0] = 2.0
    bonds = np.full((15, 5), -1.0, dtype=np.float32)
    for row, (i, j) in enumerate(zip(*np.triu_indices(6, 1))):
        bonds[row, 1 if j == i + 1 else 0] = 2.0
    logits = GraphLogits(jnp.asarray(atoms), jnp.asarray(bonds))
    prepared = prepare_failure_guidance(logits.decoded(), spec, update=7, candidate_ids=(31,),
        search=ProbabilityRepairConfig(max_edits=2, checks_per_depth=3, beam_width=2))
    return prepared, spec, json.loads(json.dumps(prepared.record()))


def test_actual_flattened_preparation_record_restores_every_original_array_receipt_identity_and_timing():
    import numpy as np
    prepared, spec, record = _actual_preparation_record()
    assert 'receipt' not in record
    restored = pilot.restore_preparation(record, spec)
    # JSON storage represents the original candidate_ids tuple as a list; the
    # exact saved JSON/identity is authoritative for a file roundtrip.
    assert restored.record() == record
    assert restored.receipt == json.loads(json.dumps(prepared.receipt))
    assert restored.identity == prepared.identity
    for field in prepared.arrays:
        expected, actual = np.asarray(prepared.arrays[field]), np.asarray(restored.arrays[field])
        assert expected.dtype == actual.dtype and expected.shape == actual.shape
        assert expected.tobytes() == actual.tobytes()


@pytest.mark.parametrize('fault', ('identity', 'receipt', 'array'))
def test_saved_preparation_identity_verification_rejects_mutated_record(fault):
    _, spec, record = _actual_preparation_record()
    if fault == 'identity': record['identity'] = '0' * 64
    elif fault == 'receipt': record['source_update'] = 8
    else: record['arrays']['active'][0] = [False] * len(record['arrays']['active'][0])
    with pytest.raises(ValueError, match='identity changed'):
        pilot.restore_preparation(record, spec)
