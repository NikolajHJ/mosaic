"""Stdlib qualification of isolated replay metadata, never native acceptance.

The complete original and controlled cohorts must exist. Selection uses a
predeclared least-change order and every saved point/repeat, never whichever
rounded probability happens to match one old failed process. Any candidate
still requires a new full-model comparison and all original scientific gates.
"""
import re

FIELDS = ('atom_logits', 'bond_logits', 'X', 'E_upper', 'key_data')
POINTS = [(variant, label) for variant in ('baseline', 'combined')
          for label in ('early-0', 'early-10', 'high-0', 'high-10')]
BASE_CONTROLS = ('plain', 'matrix_import', 'matrix_installed', 'combined', 'native_backend_import')
POLICIES = ('hash0', 'autotune0', 'hash0_autotune0')
CONTROLS = (*BASE_CONTROLS, *(name + '_' + policy for policy in POLICIES for name in ('plain', 'combined')))
EVALUATIONS = [(placement, cache, repeat)
               for placement in ('declared_committed', 'default_uncommitted')
               for cache in ('warm_repeats', 'cleared_cache') for repeat in range(3)]


def require(value, message):
    if not value:
        raise ValueError(message)


def field_schema(value):
    require(type(value) is dict and type(value.get('dtype')) is str
            and type(value.get('shape')) is list
            and all(type(n) is int and n >= 0 for n in value['shape'])
            and type(value.get('sha256')) is str
            and re.fullmatch('[a-f0-9]{64}', value['sha256']) is not None,
            'Incomplete field byte/schema fingerprint')
    return (value['dtype'], tuple(value['shape']), value['sha256'])


def environment(policy):
    require(policy in POLICIES, 'Undeclared compiler context')
    result = {}
    if 'hash0' in policy:
        result['PYTHONHASHSEED'] = '0'
    if 'autotune0' in policy:
        result['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
    return result


def qualify(records):
    """Read only authenticated JSON fingerprints; do not evaluate any array."""
    expected = [(control, repeat) for control in CONTROLS for repeat in (0, 1)]
    require(type(records) is list and [(r.get('control'), r.get('repeat')) for r in records] == expected,
            'All22 ordered unique successful control receipts are required')
    indexed = {}
    original_class = None
    original_inputs = None
    for record in records:
        require(record.get('version') == 'chemistry-softmax-replay-diagnostic-v2'
                and record.get('status') == 'diagnostic_control_completed_original_failure_preserved'
                and record.get('optimizer_updates') == 0 and record.get('model_evaluations') == 0
                and record.get('original_gate_reclassified') is False
                and record.get('can_certify_or_restart_original_study') is False
                and record.get('probability_substitution') is False
                and type(record.get('input_before')) is dict and bool(record['input_before'])
                and record.get('input_before') == record.get('input_after')
                and record.get('integrity_hash_scope') == 'exact_manifest_member_readonly_hashlib_only',
                'Incomplete replay control or changed original scientific scope')
        require(type(record.get('original_class')) is dict and record['original_class'],
                'Original class fingerprint missing')
        runtime = record['runtime_after_control']
        require(runtime['jax_config_repr']['jax_enable_x64'] == 'False'
                and runtime['jax_config_repr']['jax_default_matmul_precision'] == "'highest'",
                'Original FP32/highest precision policy differs')
        if 'hash0' in record['control']:
            require(runtime['python_hash_seed'] == '0', 'Fixed hash control not actually applied')
        if 'autotune0' in record['control']:
            require(re.findall(r'--xla_gpu_autotune_level=([^ ]+)',
                               runtime['environment'].get('XLA_FLAGS', '')) == ['0'],
                    'Autotune control not actually applied or ambiguous')
        if original_class is None:
            original_class = record['original_class']
            original_inputs = record['input_before']
        require(record['original_class'] == original_class and record['input_before'] == original_inputs,
                'Cross-process original class/input identity differs')
        points = record.get('points')
        require(type(points) is list and [(p.get('variant'), p.get('label')) for p in points] == POINTS,
                'Every original saved point is required in declared order')
        outputs = {}
        for point in points:
            require(set(point['input_fields']) == set(FIELDS), 'Original input fields incomplete')
            inputs = {name: field_schema(point['input_fields'][name]) for name in FIELDS}
            require(all(inputs[name][0] == ('uint32' if name == 'key_data' else 'float32') for name in FIELDS),
                    'Original field precision differs')
            evaluations = point.get('evaluations')
            require(type(evaluations) is list
                    and [(r.get('placement_control'), r.get('cache_control'), r.get('repeat'))
                         for r in evaluations] == EVALUATIONS,
                    'All12 commitment/cache/repeat observations are required')
            observations = []
            for row in evaluations:
                require(row.get('original_class_graph') is True and set(row['outputs']) == set(FIELDS),
                        'Graph was substituted or output fields are missing')
                actual = tuple(field_schema(row['outputs'][name]) for name in FIELDS)
                require(all(actual[index][:2] == inputs[name][:2] for index, name in enumerate(FIELDS))
                        and all(actual[FIELDS.index(name)] == inputs[name]
                                for name in ('atom_logits', 'bond_logits', 'key_data')),
                        'Raw saved logit/key transport or field schema differs')
                observations.append(actual)
            outputs[(point['variant'], point['label'])] = {
                'inputs': tuple(inputs[name] for name in ('atom_logits', 'bond_logits', 'key_data')),
                'observations': observations,
            }
        indexed[(record['control'], record['repeat'])] = outputs
    candidates = {}
    for policy in POLICIES:
        cohort = [(name + '_' + policy, repeat) for name in ('plain', 'combined') for repeat in (0, 1)]
        by_input = {}
        point_rows = []
        for point in POINTS:
            rows = [indexed[key][point] for key in cohort]
            require(all(row['inputs'] == rows[0]['inputs'] for row in rows),
                    'Compiler cohort has different raw saved inputs')
            unique = set(value for row in rows for value in row['observations'])
            by_input.setdefault(rows[0]['inputs'], set()).update(unique)
            point_rows.append({'variant': point[0], 'label': point[1], 'fresh_processes': 4,
                               'observations': 48, 'distinct_five_field_fingerprints': len(unique)})
        stable = all(row['distinct_five_field_fingerprints'] == 1 for row in point_rows)
        stable = stable and all(len(values) == 1 for values in by_input.values())
        candidates[policy] = {'all_eight_points_stable': stable, 'points': point_rows,
                              'declared_environment': environment(policy)}
    selected = next((policy for policy in POLICIES if candidates[policy]['all_eight_points_stable']), None)
    return {'version': 'isolated-replay-context-qualification-v1',
            'status': 'isolated_candidate_requires_new_full_model_gates' if selected else 'no_isolated_candidate',
            'selection_order': list(POLICIES), 'candidates': candidates, 'selected_policy': selected,
            'selected_environment': environment(selected) if selected else None,
            'original_scientific_gate_reclassified': False, 'saved_probability_substitution': False,
            'can_certify_or_restart_original_study': False, 'native_gradient_passed': False,
            'fresh_full_model_validation_required': True}
