"""Saved chemistry-screen inputs for the existing report assembler and viewer.

Metadata discovery is standard-library only. Array checks/display geometry run
on allocated report CPU. No model, chemistry, repair or sampling is evaluated.
"""
from copy import deepcopy
from pathlib import Path
import json
import types

import quick_inputs as inputs

VERSION = 'protenix-chemistry-screen-v1'
SCHEMA = 'terminal-chemistry-screen-v1'
PROOFS = ('producer_plan', 'producer_submission', 'producer_driver',
          'producer_checksums', 'science_manifest', 'sampled_validity')


def header(summary):
    inputs.require(summary.get('version') == VERSION, 'Wrong chemistry screen version')
    config = summary['configuration']
    steps = config['updates']
    inputs.require(type(steps) is int and steps > 0 and
                   summary.get('status') == 'completed_exploratory_updates' and
                   summary.get('applied_updates') == summary.get('actual_steps') == steps and
                   summary.get('full_budget_complete') is True and
                   summary.get('probe_not_adopted') is False,
                   'Only complete production chemistry screens are reportable')
    inputs.require(summary.get('derivative_qualified') is False and
                   summary.get('original_results_changed') is False,
                   'Screen qualification/original-result scope changed')
    inputs.require(summary.get('frozen_before') is not None and
                   summary['frozen_before'] == summary.get('frozen_after') and
                   'preservation_error' not in summary, 'Screen frozen inputs changed')
    return config


def snapshot_path(study, state):
    """This producer stores absolute paths, constrained to its own snapshots."""
    path = inputs.regular(state['snapshot'])
    inputs.require(path.parent == Path(study) / 'snapshots', 'Screen snapshot escaped its original study')
    inputs.require(inputs.sha(path) == state['snapshot_sha256'], 'Screen snapshot SHA changed')
    return path


def capture_segment(spec, profile, parent, summary):
    config = header(summary)
    roots = profile['allowed_roots']
    parent = Path(parent)
    study = parent / 'study'
    root = parent.parent.parent
    arm = summary['arm']
    inputs.require(parent.parent.name == arm and Path(summary['report_path']) == study / 'report.json',
                   'Screen output identity differs')
    submission = inputs.binding(root / 'production' / arm / 'submission.json', roots)
    submitted = inputs.load(submission['path'])
    job = str(summary['job_id'])
    inputs.require(job.isdigit() and submitted['job_id'] == job and submitted['kind'] == 'run'
                   and submitted['arm'] == arm, 'Screen production submission differs')
    plan = inputs.binding(submitted['plan']['path'], roots)
    inputs.require(plan == submitted['plan'], 'Screen submitted plan changed')
    planned = inputs.load(plan['path'])
    row = planned['arms'][arm]
    inputs.require(Path(planned['root']) == root and row['screen'] == config and
                   row['frozen_identity'] == summary['frozen_before'], 'Screen planned inputs differ')
    source = Path(planned['source_root'])
    scientific = inputs.binding(Path(row['science_source']['path']) / 'source_manifest.json', roots)
    inputs.require(scientific['sha256'] == row['science_source']['manifest_sha256'], 'Native source changed')
    original_config = inputs.binding(row['config']['path'], roots)
    inputs.require(original_config == row['config'], 'Native source configuration changed')
    driver = inputs.binding(source / 'protenix_chemistry_screen.py', roots)
    checksums = inputs.binding(source / 'checksums.json', roots)
    declared = inputs.load(checksums['path'])
    inputs.require(declared.get('protenix_chemistry_screen.py') == driver['sha256'] and
                   declared.get('plan.json') == plan['sha256'], 'Screen source manifest differs')
    result = {'id': spec.get('id', arm), 'input_schema': SCHEMA, 'case_directory': str(study),
              'anchor_directory': str(study), 'global_offset': 0, 'requested_updates': config['updates'],
              'source_manifest_sha256': checksums['sha256'], 'job_id': job, 'scientific_status': 'unqualified',
              'config': original_config, 'summary': inputs.binding(parent / 'summary.json', roots),
              'producer_plan': plan, 'producer_submission': submission, 'producer_driver': driver,
              'producer_checksums': checksums, 'science_manifest': scientific}
    for key, name in (('states', 'states.jsonl'), ('updates', 'updates.jsonl'),
                      ('case_report', 'report.json'), ('sampled_validity', 'sampled-validity.jsonl')):
        result[key] = inputs.binding(study / name, roots)
    for key in ('stdout', 'stderr'):
        result[key] = inputs.binding(root / 'production' / arm / (key + '.' + job + '.log'), roots)
    inputs.check_lsf(result['stdout']['path'], job, 'passed')
    result['expected_counts'] = inputs.counts(result['states']['path'], result['updates']['path'], 0)
    inputs.require(result['expected_counts']['committed_updates'] == config['updates'], 'Incomplete screen prefix')
    with Path(result['states']['path']).open() as stream:
        for line in stream:
            state = json.loads(line)
            if state['snapshot_retained']:
                inputs.regular(snapshot_path(study, state), roots)
    # A metadata view for the common selection builder, never a replacement
    # configuration on disk. The bound submitted plan contains every variant.
    return result, {'initial_step_size': row['rate'], 'steps': config['updates'],
                    'arm': 'protenix', 'version': VERSION}, 'fixed'


def collect_segment(segment, arm):
    import report_data as data
    for key in (*PROOFS, 'config', 'summary', 'case_report'):
        data.bound_file(segment[key])
    summary = data.load(segment['summary']['path'])
    config = header(summary)
    plan = data.load(segment['producer_plan']['path'])
    row = plan['arms'][summary['arm']]
    submitted = data.load(segment['producer_submission']['path'])
    data.require(row['screen'] == config and row['rate'] == arm['rate'] and
                 row['frozen_identity'] == summary['frozen_before'] and
                 submitted['plan'] == segment['producer_plan'] and
                 submitted['job_id'] == summary['job_id'] == segment['job_id'], 'Screen provenance differs')
    states, states_proof = data.committed_jsonl(segment['states'])
    updates, updates_proof = data.committed_jsonl(segment['updates'])
    steps = config['updates']
    data.require(len(states) == steps + 1 and len(updates) == steps and
                 states_proof['uncommitted_suffix_bytes'] == updates_proof['uncommitted_suffix_bytes'] == 0,
                 'Screen committed budget is incomplete')
    report = data.load(segment['case_report']['path'])
    data.require(report['status'] == summary['status'] and report['configuration'] == config and
                 report['applied_updates'] == steps and report['initial_state'] == states[0] and
                 report['last_state'] == states[-1], 'Screen case report differs from committed records')
    retained = []
    first_arrays = last_arrays = None
    for index, state in enumerate(states):
        data.require(state['step'] == state['global_step'] == index, 'Screen states are not contiguous')
        expected = index % 20 == 0 or index == steps
        data.require(type(state['snapshot_retained']) is bool and state['snapshot_retained'] == expected,
                     'Screen retained cadence differs')
        if not expected:
            data.require(state['snapshot'] is None and state['snapshot_sha256'] is None, 'Invented screen frame')
            continue
        path = snapshot_path(segment['anchor_directory'], state)
        view = deepcopy(state)
        # These remain exact saved JSON scalars. This schema does not duplicate
        # them in NPZ, so do not claim NPZ correspondence for those fields.
        for field in ('refreshed_surrogate', 'autodiff_surrogate', 'repair_weight'):
            view.pop(field, None)
        frame, arrays, proof = data.validate_frame(path, view, index)
        data.require(arrays['step'].shape == () and int(arrays['step']) == index and
                     arrays['temperature'].shape == () and float(arrays['temperature']) == state['temperature'] == config['temperature'],
                     'Screen saved step/temperature differs')
        data.require(arrays['key_data'].tolist() == [0, row['model_key']], 'Screen saved model key differs')
        frame['losses'] = data._losses(state)
        proof['scalar_scope'] = 'Exact contact primal/temperature NPZ; other losses are original JSON observations'
        retained.append((frame, proof))
        if index == 0: first_arrays = arrays
        if index == steps: last_arrays = arrays
    for index, update in enumerate(updates):
        before, after = states[index:index + 2]
        data.require(update['step'] == update['global_step'] == index and update['candidate_count'] == 1
                     and update['rate'] == arm['rate'], 'Screen update identity/rate/candidate count differs')
        data.require(update['fixed_before'] == before['refreshed_surrogate'] and
                     update['contact_before'] == before['contact_loss'] and
                     update['contact_after'] == after['contact_loss'], 'Screen scalar transition differs')
        for field in ('preparation_identity', 'repair_weight', 'binding_weight'):
            data.require(update[field] == before[field], 'Within-step screen target/weight changed: ' + field)
        data.require(type(update['fixed_increased']) is bool and
                     update['fixed_increased'] == (update['fixed_after'] > update['fixed_before']) and
                     type(update['no_op']) is bool, 'Screen update bookkeeping differs')
    samples, sample_proof = data.committed_jsonl(segment['sampled_validity'])
    sample_steps = list(range(0, steps + 1, config['sampling_interval']))
    if sample_steps[-1] != steps: sample_steps.append(steps)
    data.require([r['step'] for r in samples] == sample_steps and not sample_proof['uncommitted_suffix_bytes'],
                 'Saved sample cadence differs')
    for record in samples:
        raw = data.bound_file(record['raw_samples'])
        data.require(raw.parent == Path(segment['anchor_directory']) / 'samples' and
                     record['draws'] == config['sampling_draws'] and
                     record['binding_scores_evaluated_for_samples'] is False and
                     record['native_model_key_changed'] is False, 'Saved sampling scope differs')
    return {'states': states, 'updates': updates, 'retained': retained,
            'first_arrays': first_arrays, 'last_arrays': last_arrays,
            'proof': {'id': segment['id'], 'global_offset': 0, 'requested_updates': steps,
                      'committed_updates': steps, 'states': states_proof, 'updates': updates_proof,
                      'summary': segment['summary'], 'config': segment['config'],
                      'source_manifest_sha256': segment['source_manifest_sha256'],
                      'scientific_status': 'unqualified', 'producer_status': summary['status'],
                      'input_schema': SCHEMA, 'derivative_qualified': False,
                      'retained_frames': len(retained), 'configuration': config,
                      'sampled_validity': samples, 'sampled_validity_proof': sample_proof,
                      'sampling_scope': 'Original saved classifications and raw-draw hashes; no new chemistry evaluation',
                      'producer_proofs': {key: segment[key] for key in PROOFS}}}


def collect_arm(arm, output):
    import report_data as data
    namespace = {**data.__dict__, 'collect_segment': collect_segment}
    assembler = types.FunctionType(data.collect_arm.__code__, namespace, 'assemble_chemistry_screen')
    compatible = deepcopy(arm)
    compatible['scientific_status'] = 'partial'
    result = assembler(compatible, output)
    result['scientific_status'] = 'unqualified'
    result['optimization_complete'] = result['latest_global_step'] == result['global_target']
    result['qualification_scope'] = 'Exploratory finite screen; no derivative or binding certification'
    result['chemistry_scope'] = 'Targets/weights may change between updates; refreshed total is not one fixed objective'
    return result
