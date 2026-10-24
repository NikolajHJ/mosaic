"""Versioned 50k scope and exact checkpoint replay; numerical imports are lazy."""
import json
from pathlib import Path

RANDOM = 'mosaic-brd4-jq1-protenix-random-chem-50000-v1'
CONTINUATION = 'mosaic-brd4-jq1-protenix-10000-plus40000-v1'
VERSIONS = (RANDOM, CONTINUATION)


def validate_config(config, legacy):
    random = config['version'] == RANDOM
    expected = {'version': 'protenix-serial50k-v1',
                'start_global_update': 0 if random else 10000,
                'target_global_update': 50000, 'segment_updates': 10000,
                'parent_job': None if random else '29792670'}
    if config.get('extension') != expected or config.get('steps') != 50000:
        raise ValueError('Changed explicit extended scope')
    alias = dict(config)
    del alias['extension']
    alias['version'] = ('mosaic-brd4-jq1-protenix-random-delayed-chem-v1'
                        if random else 'mosaic-brd4-jq1-25-fixed-step-lrreset-v1')
    alias['steps'] = 10000
    if random:
        schedule = dict(config['repair_schedule'])
        if schedule != {'version': 'delayed-geometric-v1', 'initial': .001,
                        'final': 100., 'hold_updates': 5000, 'total_updates': 50000}:
            raise ValueError('User-selected stretched chemistry ramp differs')
        alias['repair_schedule'] = {**schedule, 'total_updates': 10000}
    legacy(alias)
    if config['arm'] != 'protenix' or config['initial_step_size'] != .3:
        raise ValueError('Extended authorization covers original LR0.3 Protenix only')
    return config


def segment_scope(config, offset, steps, snapshot, digest):
    start = config['extension']['start_global_update']
    if (type(offset) is not int or type(steps) is not int or steps != 10000
            or offset not in range(start, 50000, 10000) or offset + steps > 50000):
        raise ValueError('Undeclared global segment range')
    if (offset > 0) != bool(snapshot and digest):
        raise ValueError('Every continuation needs exact predecessor checkpoint')
    if (snapshot is None) != (digest is None):
        raise ValueError('Incomplete checkpoint identity')
    return {'version': 'protenix-exact-logits-segment-v1', 'start': offset,
            'end': offset + steps, 'target': 50000, 'updates': steps,
            'input_snapshot': None if snapshot is None else str(snapshot),
            'input_snapshot_sha256': digest}


def restore_logits(path, expected_sha256, logit_type, key, sha):
    import jax
    import jax.numpy as jnp
    import numpy as np
    path = Path(path)
    if sha(path) != expected_sha256:
        raise ValueError('Predecessor checkpoint bytes changed')
    with np.load(path, allow_pickle=False) as saved:
        required = {'X': (25, 6), 'E_upper': (300, 5),
                    'atom_logits': (25, 6), 'bond_logits': (300, 5)}
        for field, shape in required.items():
            if saved[field].shape != shape or saved[field].dtype != np.float32:
                raise ValueError('Wrong predecessor dimensions/FP32 precision')
            if not np.isfinite(saved[field]).all():
                raise ValueError('Nonfinite predecessor checkpoint')
        np.testing.assert_array_equal(saved['key_data'], np.asarray(jax.random.key_data(key)))
        if saved['key_data'].dtype != np.uint32:
            raise ValueError('Wrong predecessor native key type')
        logits = logit_type(jnp.asarray(saved['atom_logits']), jnp.asarray(saved['bond_logits']))
        for name, value in [('X', logits.graph().X), ('E_upper', logits.graph().E_upper),
                            ('atom_logits', logits.atoms), ('bond_logits', logits.bonds)]:
            np.testing.assert_array_equal(saved[name], np.asarray(value))
    return logits


def require_boundary(summary, assessment=None):
    execution = summary.get('extended_execution', {})
    if (execution.get('version') != 'protenix-exact-logits-segment-v1'
            or execution.get('updates') != 10000 or execution.get('target') != 50000
            or summary.get('initialization_execution', {}).get('version') !=
            'protenix-fp32-cpu-atom-softmax-v1'
            or summary.get('exact_boundary_replay') != 'passed'):
        raise ValueError('Extended boundary/placement replay is not authenticated')
    if assessment is not None and assessment.get('extended_execution') != execution:
        raise ValueError('Readiness global scope differs from actual execution')
    return execution


def projection(unit_bytes, new_updates, segments):
    if (type(unit_bytes) is not int or unit_bytes <= 0
            or type(new_updates) is not int or new_updates not in (40000, 50000)
            or segments != new_updates // 10000):
        raise ValueError('Invalid full extended output projection')
    return 2 * unit_bytes * new_updates + segments * 1024**3


def verify_parent(pins, sha, read):
    for value in [pins['snapshot'], *pins['proofs'].values()]:
        path = Path(value['path'])
        if path.stat().st_size != value['bytes'] or sha(path) != value['sha256']:
            raise ValueError('Original completed Protenix evidence changed')
    log = Path(pins['proofs']['stdout']['path']).read_text()
    if ('Subject: Job 29792670:' not in log or 'Successfully completed.' not in log
            or 'Exited with exit' in log):
        raise ValueError('Original same-job successful LSF proof missing')
    summary = read(pins['proofs']['summary']['path'])
    if (summary['status'] != 'passed' or summary['job_id'] != '29792670'
            or summary['terminal_gradient_checks']['status'] != 'passed'
            or summary['native_hard_parity']['status'] != 'passed'
            or summary['native_gradient_checks']['status'] != 'passed'
            or summary['source_manifest_sha256'] != 'c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057'
            or summary['frozen_identity_after'] != {
                'model_digest': summary['model_digest'], 'context_digest': summary['context_digest']}):
        raise ValueError('Original scientific completion not passed')
    case = read(pins['proofs']['case']['path'])
    wrapped = summary['cases'][0]
    if {k: v for k, v in wrapped.items() if k not in ('label', 'initialization')} != case:
        raise ValueError('Original summary/case mismatch')
    if case['status'] != 'passed' or len(case['updates']) != 10000 or len(case['states']) != 10001:
        raise ValueError('Original full budget missing')
    from campaign25_contract import verify_attempt_rates
    verify_attempt_rates(case, summary['config'])
    collector = read(pins['proofs']['collector']['path'])
    if collector['status'] != 'passed' or collector['scientific_status'] != 'passed':
        raise ValueError('Original complete-frame collector did not accept the study')
    for name, expected in [('states', case['states']), ('updates', case['updates'])]:
        with Path(pins['proofs'][name]['path']).open() as stream:
            for index, line in enumerate(stream):
                if not line.endswith('\n') or json.loads(line) != expected[index] or expected[index]['step'] != index:
                    raise ValueError('Original committed JSONL differs')
            if index + 1 != len(expected):
                raise ValueError('Original committed JSONL incomplete')
    return {'status': 'passed', 'job': pins['job'], 'global_state': 10000,
            'snapshot': pins['snapshot'], 'scope': 'Original terminal scientific pass preserved; new exact child only'}
