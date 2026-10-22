"""Prepare declared source text/configuration and pinned predecessor metadata only."""
from pathlib import Path
import ast
import hashlib
import json

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
TAG = HERE.name
ANALYSIS = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
NEW_ROOT = ANALYSIS + '/restart-lrreset-01'
NEW_SOURCE = ANALYSIS + '/restart-lrreset-release-01/source'

def write(name, value):
    path = REPO / name
    if path.exists():
        raise FileExistsError(path)
    path.write_bytes(value.encode('utf-8'))

def jtext(value):
    return json.dumps(value, indent=2, allow_nan=False) + '\n'

rates = {'protenix': .3, 'boltz2': .005, 'boltz2_chemistry': .005}
for arm, rate in rates.items():
    config = json.loads((REPO / f'examples/configs/brd4_jq1_25_fixed_lrboost_{arm}_10000.json').read_text())
    config.update(version='mosaic-brd4-jq1-25-fixed-step-lrreset-v1', initial_step_size=rate,
                  label=f'JQ1 pocket / 25 occupied slots / {arm} / fixed LR{rate} / 10000 fresh updates / LR reset')
    if arm == 'boltz2_chemistry':
        config['repair_schedule']['final'] = 100.0
    write(f'examples/configs/brd4_jq1_25_fixed_lrreset_{arm}_10000.json', jtext(config))

# Exact byte pins from the independently verified second delivery. These are
# terminal-history guards, never native readiness for the new scientific settings.
delivery = REPO / 'results/hpc/20261022-brd4-jq1-lrboost-analysis-01/interim-delivery-evidence-02/analysis'
pins = {'version': 'jq1-lrreset-finished-predecessors-v1', 'arms': {}}
for arm, native, collector in [('protenix','29779177','29780825'),('boltz2','29779178','29780826'),('boltz2_chemistry','29779179','29780827')]:
    parent = delivery / ('complete-' + arm)
    assessment = parent / 'payload/assessment.json'
    data = json.loads(assessment.read_bytes())
    def ref(local, remote):
        b = local.read_bytes()
        return {'path': remote, 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
    remote = ANALYSIS + '/complete-' + arm
    pins['arms'][arm] = {'native_job': native, 'collector_job': collector,
        'scientific_status': data['scientific_status'], 'config_sha256': data['config_sha256'],
        'assessment': ref(assessment, remote + '/payload/assessment.json'),
        'collector_lsf': ref(parent / f'stdout.{collector}.log', remote + f'/stdout.{collector}.log')}
write('scripts/hpc/lrreset_predecessors.json', jtext(pins))

ops = (REPO / 'scripts/fixed25_lrboost_ops.py').read_text()
ops = ops.replace('20261021-brd4-jq1-25-fixed-lrboost-01', TAG).replace('lrboost', 'lrreset').replace('LRBOOST', 'LRRESET')
ops = ops.replace("SOURCE = Path('/work3/s234818/mosaic/releases') / TAG / 'source'", f"SOURCE = Path('{NEW_SOURCE}')")
ops = ops.replace("RATES = {'protenix': 3.0, 'boltz2': 0.05, 'boltz2_chemistry': 0.05}", "RATES = {'protenix': 0.3, 'boltz2': 0.005, 'boltz2_chemistry': 0.005}")
start = ops.index('STOP_ROOT = ')
end = ops.index('\n\n\ndef require', start)
ops = ops[:start] + '''PREDECESSOR_PINS = SOURCE / 'scripts/hpc/lrreset_predecessors.json'
AUTHORIZATION = {'version': VERSION, 'fresh_updates_per_arm': 10000, 'rates': RATES,
    'acceptance_policy': 'fixed_step', 'one_candidate': True, 'diffusion_gradient': False,
    'repair_schedule': {'initial': 0.001, 'final': 100.0, 'hold_updates': 5000, 'total_updates': 10000},
    'finished_predecessors_preserved': ['29779177', '29779178', '29779179']}
''' + ops[end:]
start = ops.index('def verify_stopped_predecessors():')
end = ops.index('\n\ndef dispatch(', start)
ops = ops[:start] + '''def verify_finished_predecessors():
    """Authenticate finished history without changing either failed outcome."""
    pin = read(PREDECESSOR_PINS)
    require(pin['version'] == 'jq1-lrreset-finished-predecessors-v1'
            and set(pin['arms']) == set(ARMS), 'Wrong finished predecessor cohort')
    for arm in ARMS:
        item = pin['arms'][arm]
        for name in ('assessment', 'collector_lsf'):
            proof = item[name]
            hashed(proof['path'], proof['sha256'])
            require(Path(proof['path']).stat().st_size == proof['bytes'], 'Predecessor proof length changed')
        assessment = read(item['assessment']['path'])
        require(assessment['status'] == 'passed' and assessment['presentation_complete'] is True
                and assessment['native_job'] == item['native_job']
                and assessment['collector_job'] == item['collector_job'] and assessment['arm'] == arm
                and assessment['states'] == 10001 and assessment['updates'] == 10000
                and assessment['scientific_status'] == item['scientific_status']
                and assessment['outcome']['all_terminal'] is True
                and assessment['config_sha256'] == item['config_sha256']
                and assessment['all_probability_frames_verified'] is True,
                'Predecessor is not the authenticated finished trajectory')
        lsf = Path(item['collector_lsf']['path']).read_text()
        require('Subject: Job ' + item['collector_job'] + ':' in lsf
                and 'Successfully completed.' in lsf
                and not re.search(r'Exited\\s+(?:with exit|by signal)', lsf, re.I),
                'Actual same-job completion collector LSF success missing')
        native_lsf = []
        for proof in assessment['proofs']:
            hashed(proof['path'], proof['sha256'])
            if Path(proof['path']).name == 'native-stdout.txt':
                native_lsf.append(Path(proof['path']).read_text())
        require(len(native_lsf) == 1 and 'Subject: Job ' + item['native_job'] + ':' in native_lsf[0],
                'Actual original native terminal LSF missing')
        expected = 'passed' if arm == 'boltz2_chemistry' else 'failed'
        require(item['scientific_status'] == expected, 'Historical scientific failure cannot be relabelled')
        if expected == 'passed':
            require('Successfully completed.' in native_lsf[0], 'Original chemistry completion not established')
        else:
            require(re.search(r'Exited with exit code\\s+1\\.', native_lsf[0]), 'Original failed native exit missing')
    return hashed(PREDECESSOR_PINS)
''' + ops[end:]
ops = ops.replace('verify_stopped_predecessors()', 'verify_finished_predecessors()').replace("'predecessors_stopped': stop", "'finished_predecessors': stop")
ops = ops.replace("{'test_fixed25_lrreset_contract', 'test_fixed25_lrreset_ops', 'test_fixed25_lrreset_storage'} <= modules", "{'test_fixed25_lrreset_contract', 'test_fixed25_lrreset_ops', 'test_fixed25_lrreset_storage',\n             'test_fixed25_lrboost_contract', 'test_fixed25_lrboost_ops', 'test_fixed25_lrboost_storage'} <= modules")
ops = ops.replace('Fresh high-rate', 'Fresh lower-rate').replace('high-rate', 'lower-rate')
write('scripts/fixed25_lrreset_ops.py', ops)

storage = (REPO / 'scripts/fixed25_lrboost_storage.py').read_text()
storage = storage.replace('20261021-brd4-jq1-25-fixed-lrboost-01', TAG).replace('lrboost','lrreset')
storage = storage.replace("ROOT = Path('/work3/s234818/mosaic/runs') / TAG", f"ANALYSIS_ROOT = Path('{ANALYSIS}')\nROOT = ANALYSIS_ROOT / 'restart-lrreset-01'")
storage = storage.replace("RELEASE = Path('/work3/s234818/mosaic/releases') / TAG", "RELEASE = ANALYSIS_ROOT / 'restart-lrreset-release-01'")
storage = storage.replace("UPLOAD = Path('/work3/s234818/mosaic') / (TAG + '.tar.gz')", "UPLOAD = ANALYSIS_ROOT / (TAG + '.tar.gz')\nCONCURRENT_ANALYSIS_ALLOWANCE = 2 * 1024**3 + 512 * 1024**2")
storage = storage.replace("'20261021-brd4-jq1-25-fixed-report-04', '20261021-brd4-jq1-25-fixed-report-05'))", "'20261021-brd4-jq1-25-fixed-report-04', '20261021-brd4-jq1-25-fixed-report-05',\n    '20261021-brd4-jq1-25-fixed-lrboost-01', '20261022-brd4-jq1-lrboost-analysis-01'))")
storage = storage.replace("return sum(tree_bytes(p) for p in (*HISTORICAL_WORK_ROOTS, ROOT, RELEASE, UPLOAD))", "# The containing analysis root charges new run/release/upload bytes exactly once.\n    roots = (*HISTORICAL_WORK_ROOTS, ROOT, RELEASE, UPLOAD)\n    return sum(tree_bytes(p) for p in roots if not any(p != q and p.is_relative_to(q) for q in roots))")
storage = storage.replace("'scope': 'All three fresh high-rate trajectories; no existing trajectory resumed or replaced',", "'scope': 'Three fresh lower-rate trajectories; immutable finished studies retained',\n            'concurrent_analysis_allowance_bytes': CONCURRENT_ANALYSIS_ALLOWANCE,")
storage = storage.replace("raw, projected, stale_work_bytes()", "raw, projected + CONCURRENT_ANALYSIS_ALLOWANCE, stale_work_bytes()")
storage = storage.replace("gate_record(pool, raw, budgets[pool]['outstanding_projected_bytes'], stale,", "gate_record(pool, raw, budgets[pool]['outstanding_projected_bytes'] + (CONCURRENT_ANALYSIS_ALLOWANCE if pool == 'work3' else 0), stale,")
write('scripts/fixed25_lrreset_storage.py', storage)

for kind in ('cpu','stage'):
    text = (REPO / f'scripts/hpc/fixed25_lrboost_{kind}.lsf').read_text().replace('lrboost','lrreset')
    if kind == 'cpu':
        text = text.replace('tests/test_fixed25_lrreset_contract.py', 'tests/test_fixed25_lrboost_contract.py tests/test_fixed25_lrboost_ops.py tests/test_fixed25_lrboost_storage.py tests/test_fixed25_lrreset_contract.py')
    write(f'scripts/hpc/fixed25_lrreset_{kind}.lsf', text)

test = (REPO / 'tests/test_fixed25_lrboost_contract.py').read_text().replace('lrboost','lrreset').replace('LRBOOST','LRRESET')
test = test.replace("('protenix', 3.0), ('boltz2', 0.05), ('boltz2_chemistry', 0.05)", "('protenix', 0.3), ('boltz2', 0.005), ('boltz2_chemistry', 0.005)")
test = test.replace("old_path = ROOT / f'examples/configs/brd4_jq1_25_fixed_{arm}_10000.json'", "old_path = ROOT / f'examples/configs/brd4_jq1_25_fixed_lrboost_{arm}_10000.json'")
test = test.replace("assert {k for k in new if new[k] != old[k]} == {'version', 'initial_step_size', 'label'}", "assert {k for k in new if new[k] != old[k]} == ({'version', 'initial_step_size', 'label', 'repair_schedule'} if arm == 'boltz2_chemistry' else {'version', 'initial_step_size', 'label'})")
test = test.replace('update = row(3.0)', 'update = row(0.3)').replace("'accepted_step_size': 3.0", "'accepted_step_size': 0.3").replace("'step_size': 3.0", "'step_size': 0.3")
test = test.replace("pilot(value['initial_step_size']))", "pilot(value['initial_step_size'], 0.001 if arm == 'boltz2_chemistry' else 0.0))")
test = test.replace("pilot(value['initial_step_size'], 50.0)", "pilot(value['initial_step_size'], 100.0)")
test = test.replace("['accepted_step_size'] = 0.3", "['accepted_step_size'] = 3.0")
test += '''

def test_new_chemistry_endpoints_are_both_native_checked_and_high_pilot_is_100():
    value = config('boltz2_chemistry')
    assert contract.chemistry_validation_weights(value) == (('low', 0.001), ('high', 100.0))
    old = json.loads((ROOT / 'examples/configs/brd4_jq1_25_fixed_lrboost_boltz2_chemistry_10000.json').read_text())
    assert contract.chemistry_validation_weights(old) == (('low', 0.001), ('high', 50.0))
    from mosaic.small_molecule.chemistry.schedule import RepairWeightSchedule
    schedule = RepairWeightSchedule(**value['repair_schedule'])
    assert schedule.weight(0) == 0.001 and schedule.weight(5000) == 0.001
    assert schedule.weight(10000) == 100.0


def test_driver_routes_declared_endpoints_to_actual_coupled_and_pilot_calls():
    import ast
    tree = ast.parse((ROOT / 'examples/brd4_25_atoms.py').read_text())
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.iter, ast.Call)
             and isinstance(n.iter.func, ast.Name) and n.iter.func.id == 'chemistry_validation_weights']
    assert len(loops) == 1
    calls = [n.func.id for n in ast.walk(loops[0]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert 'evaluate_combined' in calls and 'check_combined_directions' in calls
'''
write('tests/test_fixed25_lrreset_contract.py', test)

test = (REPO / 'tests/test_fixed25_lrboost_ops.py').read_text().replace('lrboost','lrreset').replace('LRBOOST','LRRESET')
test = test.replace('verify_stopped_predecessors', 'verify_finished_predecessors')
start = test.index('def test_predecessor_stop_guard_')
test = test[:start]  # The predecessor schema is intentionally different; test it explicitly below.
test += '''def test_finished_predecessor_guard_preserves_failures_and_cannot_adopt_a_running_study(tmp_path, monkeypatch):
    cohort = {'version': 'jq1-lrreset-finished-predecessors-v1', 'arms': {}}
    for arm, native, collector in [('protenix','29779177','29780825'),('boltz2','29779178','29780826'),('boltz2_chemistry','29779179','29780827')]:
        expected = 'passed' if arm == 'boltz2_chemistry' else 'failed'
        native_log = tmp_path / (arm + '-native-stdout.txt')
        # Give all three identical native-stdout basename as production does.
        native_log = tmp_path / arm / 'native-stdout.txt'
        native_log.parent.mkdir()
        native_log.write_text('Subject: Job ' + native + ': Finished\\n' + ('Successfully completed.' if expected == 'passed' else 'Exited with exit code 1.'))
        assessment = {'status': 'passed', 'presentation_complete': True, 'native_job': native,
            'collector_job': collector, 'arm': arm, 'states': 10001, 'updates': 10000,
            'scientific_status': expected, 'config_sha256': 'a' * 64, 'outcome': {'all_terminal': True},
            'all_probability_frames_verified': True, 'proofs': [ops.hashed(native_log)]}
        a = put(tmp_path / arm / 'assessment.json', assessment)
        log = tmp_path / arm / 'collector.log'
        log.write_text('Subject: Job ' + collector + ': Done\\nSuccessfully completed.')
        def proof(p): return {**ops.hashed(p), 'bytes': p.stat().st_size}
        cohort['arms'][arm] = {'native_job': native, 'collector_job': collector, 'scientific_status': expected,
            'config_sha256': 'a' * 64, 'assessment': proof(a), 'collector_lsf': proof(log)}
    path = put(tmp_path / 'pins.json', cohort)
    monkeypatch.setattr(ops, 'PREDECESSOR_PINS', path)
    assert ops.verify_finished_predecessors() == ops.hashed(path)
    bad = tmp_path / 'protenix/assessment.json'
    v = ops.read(bad); v['outcome']['all_terminal'] = False; put(bad, v)
    cohort['arms']['protenix']['assessment'] = {**ops.hashed(bad), 'bytes': bad.stat().st_size}
    put(path, cohort)
    with pytest.raises(ValueError, match='finished trajectory'):
        ops.verify_finished_predecessors()
'''
write('tests/test_fixed25_lrreset_ops.py', test)
test = (REPO / 'tests/test_fixed25_lrboost_storage.py').read_text().replace('lrboost','lrreset')
test += '''

def test_old_analysis_observers_charge_all_nested_new_source_and_output_bytes():
    assert storage.ROOT.is_relative_to(storage.ANALYSIS_ROOT)
    assert storage.RELEASE.is_relative_to(storage.ANALYSIS_ROOT)
    assert storage.UPLOAD.is_relative_to(storage.ANALYSIS_ROOT)
    assert storage.ANALYSIS_ROOT in storage.HISTORICAL_WORK_ROOTS
    assert storage.CONCURRENT_ANALYSIS_ALLOWANCE == 2 * 1024**3 + 512 * 1024**2
'''
write('tests/test_fixed25_lrreset_storage.py', test)

for path in [REPO / 'scripts/fixed25_lrreset_ops.py', REPO / 'scripts/fixed25_lrreset_storage.py', *REPO.glob('tests/test_fixed25_lrreset_*.py')]:
    ast.parse(path.read_text(), filename=str(path))
print('Prepared new source text/configuration and parsed ASTs only; no tests or numerical execution.')
