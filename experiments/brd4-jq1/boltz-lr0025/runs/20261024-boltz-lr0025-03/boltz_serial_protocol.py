"""Declared Boltz30k fixed-step scope and exact checkpoint replay.

Imports used by packaging/controller validation are standard-library only;
numerical checkpoint validation is lazy and allocated-HPC only.
"""
from pathlib import Path

VERSION = 'mosaic-brd4-jq1-boltz-lr0025-30000-v1'
VERSIONS = (VERSION,)
EXECUTION = 'boltz-fp32-cpu-exact-logits-segment-v1'
TARGET = 30000
SEGMENT = 10000
ARMS = ('boltz2', 'boltz2_chemistry')


def validate_config(config, legacy):
    if config.get('arm') not in ARMS:
        raise ValueError('Only two fresh Boltz arms are authorized')
    if config.get('extension') != {
        'version': 'boltz-serial30k-v1', 'start_global_update': 0,
        'target_global_update': TARGET, 'segment_updates': SEGMENT,
        'parent_job': None,
    } or config.get('steps') != TARGET or config.get('initial_step_size') != .0025:
        raise ValueError('Changed declared LR0.0025/fresh30k scope')
    if config.get('chemistry_enabled') != (config['arm'] == 'boltz2_chemistry'):
        raise ValueError('Declared native/chemistry arm mismatch')
    alias = dict(config)
    del alias['extension']
    alias.update(version='mosaic-brd4-jq1-25-fixed-step-lrreset-v1',
                 steps=SEGMENT, initial_step_size=.005)
    if config['chemistry_enabled']:
        schedule = config.get('repair_schedule')
        if schedule != {'version': 'delayed-geometric-v1', 'initial': .001,
                        'final': 100., 'hold_updates': 5000,
                        'total_updates': TARGET}:
            raise ValueError('User-selected stretched chemistry ramp differs')
        alias['repair_schedule'] = {**schedule, 'total_updates': SEGMENT}
    legacy(alias)
    return config


def segment_scope(config, offset, steps, snapshot, digest):
    if type(offset) is not int or type(steps) is not int or steps != SEGMENT or offset not in (0, 10000, 20000):
        raise ValueError('Undeclared Boltz segment range')
    if (offset > 0) != bool(snapshot and digest) or (snapshot is None) != (digest is None):
        raise ValueError('Every continuation needs its exact predecessor checkpoint')
    return {'version': EXECUTION, 'start': offset, 'end': offset + steps,
            'target': TARGET, 'updates': steps,
            'input_snapshot': None if snapshot is None else str(snapshot),
            'input_snapshot_sha256': digest}


def restore_logits(path, expected_sha256, logit_type, key, sha):
    import os
    if not os.environ.get('LSB_JOBID'):
        raise RuntimeError('Numerical checkpoint replay requires allocated HPC')
    import jax
    import jax.numpy as jnp
    import numpy as np
    path = Path(path)
    if sha(path) != expected_sha256:
        raise ValueError('Predecessor checkpoint bytes changed')
    with np.load(path, allow_pickle=False) as saved:
        for field, shape in {'X': (25, 6), 'E_upper': (300, 5),
                             'atom_logits': (25, 6), 'bond_logits': (300, 5)}.items():
            if saved[field].shape != shape or saved[field].dtype != np.float32 or not np.isfinite(saved[field]).all():
                raise ValueError('Wrong/nonfinite predecessor dimensions or FP32 type')
        if saved['key_data'].dtype != np.uint32:
            raise ValueError('Wrong predecessor native key type')
        np.testing.assert_array_equal(saved['key_data'], np.asarray(jax.random.key_data(key)))
        # Preserve exact logits, never re-log or renormalize saved probabilities.
        logits = logit_type(jnp.asarray(saved['atom_logits']), jnp.asarray(saved['bond_logits']))
        for name, value in [('X', logits.graph().X), ('E_upper', logits.graph().E_upper),
                            ('atom_logits', logits.atoms), ('bond_logits', logits.bonds)]:
            np.testing.assert_array_equal(saved[name], np.asarray(value))
    return logits


def require_boundary(summary, assessment=None):
    execution = summary.get('serial_execution', {})
    if (execution.get('version') != EXECUTION or execution.get('updates') != SEGMENT
            or execution.get('target') != TARGET
            or summary.get('initialization_execution') != {
                'version': EXECUTION, 'jax_platform': 'cpu', 'dtype': 'float32',
                'parameterization': 'categorical_logits_v1', 'model_key': 17}
            or summary.get('exact_boundary_replay') != 'passed'):
        raise ValueError('Boltz exact boundary/backend/key execution not authenticated')
    if execution.get('start') not in (0, 10000, 20000) or execution.get('end') != execution['start'] + SEGMENT:
        raise ValueError('Boltz global range differs')
    if assessment is not None and assessment.get('serial_execution') != execution:
        raise ValueError('Readiness global scope differs from actual execution')
    return execution


def projection(unit_bytes, new_updates=TARGET, segments=3):
    if type(unit_bytes) is not int or unit_bytes <= 0 or new_updates != TARGET or segments != 3:
        raise ValueError('Full three-segment30k projection required')
    return 2 * unit_bytes * new_updates + segments * 1024**3
