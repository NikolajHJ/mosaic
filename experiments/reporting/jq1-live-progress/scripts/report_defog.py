"""Allocated-CPU validation of original stopped-DeFoG trajectory artifacts.

No model, chemistry toolkit, repair, sampler or new native score is executed.
Every saved NPZ is validated; only stride20/endpoints are selected for display.
"""
from copy import deepcopy
from pathlib import Path
import json
import hashlib
import math

import quick_inputs as inputs
import report_data as data

STRIDE = 20


def transport_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def schedule(config, index):
    ramp = config['teacher']['time_ramp']
    fraction = min(index, ramp['last_update']) / ramp['last_update']
    return ramp['initial'] + (ramp['final'] - ramp['initial']) * fraction


def coefficient(config, index):
    ramp = config['weight_ramp']
    fraction = min(index, ramp['last_update']) / ramp['last_update']
    return ramp['initial'] + (ramp['final'] - ramp['initial']) * fraction


def selected(index, requested):
    return index == 0 or index == requested or index % STRIDE == 0


def validate_snapshot(path, state, config, display):
    import numpy as np
    data.require(data.sha(path) == state['snapshot_sha256'], 'DeFoG saved NPZ SHA changed')
    shapes = {'X': (25, 6), 'E_upper': (300, 5), 'atom_logits': (25, 6), 'bond_logits': (300, 5),
              'teacher_X': (25, 6), 'teacher_E': (300, 5), 'full_native_teacher_X': (25, 12),
              'native_gradient_X': (25, 6), 'native_gradient_E_upper': (300, 5),
              'native_gradient_atom_logits': (25, 6), 'native_gradient_bond_logits': (300, 5),
              'prior_gradient_atom_logits': (25, 6), 'prior_gradient_bond_logits': (300, 5),
              'combined_gradient_atoms': (25, 6), 'combined_gradient_bonds': (300, 5)}
    with np.load(path, allow_pickle=False) as archive:
        data.require(set(archive.files) == set(shapes) | {'key_data', 'teacher_time', 'teacher_state_index'},
                     'DeFoG NPZ schema differs')
        arrays = {name: archive[name] for name in archive.files}
    for name, shape in shapes.items():
        value = arrays[name]
        data.require(value.dtype == np.float32 and value.shape == shape and np.isfinite(value).all(),
                     'DeFoG FP32 shape/finiteness differs: ' + name)
    for name in ('X', 'E_upper', 'teacher_X', 'teacher_E', 'full_native_teacher_X'):
        value = arrays[name]
        data.require((value >= 0).all() and (value <= 1).all() and np.allclose(value.sum(-1), 1, atol=1e-6, rtol=0),
                     'DeFoG probability simplex differs: ' + name)
    data.require(arrays['key_data'].dtype == np.uint32 and arrays['key_data'].shape == (2,)
                 and np.array_equal(arrays['key_data'], np.asarray([0, config['seed']], np.uint32)), 'DeFoG native key differs')
    index = state['state']
    data.require(arrays['teacher_time'].shape == () and arrays['teacher_time'].dtype == np.float64
                 and float(arrays['teacher_time']) == state['teacher_time'] == schedule(config, index), 'DeFoG saved time differs')
    data.require(arrays['teacher_state_index'].shape == () and arrays['teacher_state_index'].dtype == np.int64
                 and int(arrays['teacher_state_index']) == state['teacher_state_index'] == index, 'DeFoG saved schedule index differs')
    graph = {'X': arrays['X'].tolist(), 'E': arrays['E_upper'].tolist()}
    teacher = {'X': arrays['teacher_X'].tolist(), 'E': arrays['teacher_E'].tolist()}
    data.require(transport_digest(graph) == state['graph_sha256'], 'DeFoG graph transport digest differs')
    data.require(transport_digest(teacher) == state['teacher_sha256'], 'DeFoG teacher transport digest differs')
    query = {'graph_sha256': state['graph_sha256'], 'state_index': index, 'teacher_time': state['teacher_time']}
    data.require(transport_digest(query) == state['teacher_query_sha256'], 'DeFoG scheduled query digest differs')
    raw = state['raw_readout']
    atom = arrays['X'].argmax(-1).tolist()
    bond = arrays['E_upper'].argmax(-1).tolist()
    pairs = [[i, j] for i in range(25) for j in range(i + 1, 25)]
    for key, expected in (('element_indices', atom), ('bond_upper_indices', bond), ('pairs', pairs),
                          ('elements', [data.ATOMS[k] for k in atom])):
        data.require(raw.get(key) == expected, 'DeFoG raw readout/NPZ argmax differs: ' + key)
    for name in ('atoms', 'bonds'):
        native = arrays['native_gradient_atom_logits' if name == 'atoms' else 'native_gradient_bond_logits']
        prior = arrays['prior_gradient_atom_logits' if name == 'atoms' else 'prior_gradient_bond_logits']
        data.require(np.allclose(arrays['combined_gradient_' + name], native + state['weight'] * prior, rtol=2e-6, atol=2e-8),
                     'DeFoG saved combined gradient differs')
    frame = None
    if display:
        x, e = graph['X'], graph['E']
        data.require(np.array_equal(np.asarray(json.loads(json.dumps(x)), dtype=np.float32), arrays['X'])
                     and np.array_equal(np.asarray(json.loads(json.dumps(e)), dtype=np.float32), arrays['E_upper']),
                     'DeFoG FP32 JSON roundtrip differs')
        frame = {'global_step': index, 'atoms': x, 'bonds': e, 'matrix_dtype': 'float32',
                 'atom_labels': [data.ATOMS[k] for k in atom], 'bond_labels': [data.BONDS[k] for k in bond],
                 'graph': data.layout_raw_graph([{'id': i, 'element': data.ATOMS[k]} for i, k in enumerate(atom)],
                     [{'source': pair[0], 'target': pair[1], 'label': data.BONDS[k]} for pair, k in zip(pairs, bond) if k]),
                 'readout': deepcopy(raw), 'losses': losses(state),
                 'source_snapshot': {'path': str(path), 'bytes': path.stat().st_size, 'sha256': state['snapshot_sha256']},
                 'matrix_sha256': {n: __import__('hashlib').sha256(arrays[n].tobytes(order='C')).hexdigest() for n in ('X', 'E_upper')},
                 'validation': {'FP32_roundtrip': True, 'argmax_correspondence': True, 'scheduled_teacher_identity': True},
                 'interpretation': 'Original raw graph/readout; display-selected original matrix; no repair or new scoring'}
    proof = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': state['snapshot_sha256']}
    return frame, proof


def losses(state):
    result = {key: value for key, value in state.items() if type(value) in (int, float)}
    result['contact_loss'] = state['native_loss']
    result['weighted_prior_kl'] = state['weight'] * state['prior_kl']
    result['raw_valid'] = int(state['raw_readout']['sanitized'])
    return result


def collect_arm(arm, output):
    data.require(arm['kind'] == 'defog' and len(arm['segments']) == 1 and arm['global_start'] == 0, 'Unsupported DeFoG grouping')
    segment = arm['segments'][0]
    config = data.load(data.bound_file(segment['config']))
    summary = data.load(data.bound_file(segment['summary']))
    data.require(config['n_atoms'] == 25 and config['learning_rate'] == arm['rate'] and config['steps'] == arm['global_target'],
                 'DeFoG graph/rate/budget differs')
    data.require(summary['source_manifest_sha256'] == segment['source_manifest_sha256'] and summary['config_sha256'] == segment['config']['sha256']
                 and str(summary['job']) == segment['job_id'] and summary['status'] == arm['scientific_status'] == segment['scientific_status'],
                 'DeFoG original summary/provenance/status differs')
    states, state_proof = data.committed_jsonl(segment['states'])
    updates, update_proof = data.committed_jsonl(segment['updates'])
    requested = config['steps']
    data.require(not state_proof['uncommitted_suffix_bytes'] and not update_proof['uncommitted_suffix_bytes'], 'Incomplete terminal DeFoG JSONL')
    data.require(len(states) == len(updates) + 1 == requested + 1 and summary['completed_updates'] == requested,
                 'DeFoG complete update budget differs')
    scalar_rows, frames, snapshots = [], [], []
    counts = {'uphill': 0, 'equal': 0, 'noop': 0, 'forced_fallback': 0}
    valid = connected = 0
    identities = set()
    for index, state in enumerate(states):
        data.require(state['state'] == index and state['teacher_state_index'] == index
                     and state['teacher_time'] == schedule(config, index) and state['weight'] == coefficient(config, index),
                     'DeFoG state index/time/weight schedule differs')
        data.require(type(state['raw_readout']['sanitized']) is bool, 'Missing original DeFoG validity')
        if state['raw_readout']['sanitized']:
            valid += 1
            connected += state['raw_readout'].get('disconnected') is False
            data.require(isinstance(state['raw_readout'].get('canonical_smiles'), str), 'Valid DeFoG readout lacks identity')
            identities.add(state['raw_readout']['canonical_smiles'])
        path = data._saved_path(segment['anchor_directory'], state['snapshot'])
        frame, proof = validate_snapshot(path, state, config, selected(index, requested))
        snapshots.append(proof)
        if frame is not None:
            frames.append(frame)
        update = None if index == 0 else updates[index - 1]
        if update is not None:
            before = states[index - 1]
            data.require(update['update'] == index and update['teacher_state_index'] == index - 1
                         and update['teacher_time'] == before['teacher_time'] and update['weight'] == before['weight']
                         and update['teacher_sha256'] == before['teacher_sha256']
                         and update['teacher_query_sha256'] == before['teacher_query_sha256'], 'DeFoG update teacher binding differs')
            data.require(update['native_before'] == before['native_loss'] and update['native_after'] == state['native_loss']
                         and update['fixed_teacher_before'] == before['refreshed_surrogate']
                         and math.isclose(update['fixed_teacher_delta'], update['fixed_teacher_after'] - update['fixed_teacher_before'], abs_tol=1e-14),
                         'DeFoG saved native/fixed-teacher transition differs')
            data.require(type(update['halvings']) is int and 0 <= update['halvings'] <= config['max_step_halvings']
                         and update['rate'] == config['learning_rate'] / 2**update['halvings']
                         and 0 <= update['max_graph_cell_tv'] <= config['max_graph_cell_tv'], 'DeFoG local step/rate differs')
            counts['uphill'] += update['fixed_teacher_delta'] > 0
            counts['equal'] += update['fixed_teacher_delta'] == 0
            counts['noop'] += update['max_graph_cell_tv'] == 0
        scalar_rows.append({'global_step': index, 'losses': losses(state), 'state': state, 'update': update, 'segment_id': segment['id']})
    output = Path(output)
    scalar_path = output / (arm['id'] + '-scalars.json')
    data.write(scalar_path, {'rows': scalar_rows, 'boundary_checks': [], 'scope': 'Every original saved DeFoG scalar/update and raw readout; no synthetic states'})
    chunks = []
    for number, first in enumerate(range(0, len(frames), data.CHUNK_FRAMES)):
        chunk = frames[first:first + data.CHUNK_FRAMES]
        path = output / (arm['id'] + '-frames-' + str(number).zfill(4) + '.json')
        data.write(path, {'frames': chunk})
        chunks.append({**data.asset_binding(path, output), 'first': chunk[0]['global_step'], 'last': chunk[-1]['global_step'],
                       'count': len(chunk), 'steps': [f['global_step'] for f in chunk]})
    proof = {'id': segment['id'], 'global_offset': 0, 'requested_updates': requested, 'committed_updates': len(updates),
             'states': state_proof, 'updates': update_proof, 'summary': segment['summary'], 'config': segment['config'],
             'source_manifest_sha256': segment['source_manifest_sha256'], 'scientific_status': summary['status'],
             'scientific_error': summary.get('error'), 'retained_frames': len(frames), 'saved_frames_verified': len(snapshots),
             'all_saved_snapshots': snapshots, 'native_scalar_evidence': 'Original JSON/update scalars; NPZ contains no native-primal scalar'}
    return {'id': arm['id'], 'label': arm['label'], 'kind': 'defog', 'rate': arm['rate'],
            'target_updates': requested, 'global_target': requested, 'global_start': 0, 'committed_updates': len(updates),
            'latest_global_step': len(updates), 'scientific_status': summary['status'], 'presentation_status': 'saved_prefix_verified',
            'status_detail': arm['status_detail'], 'continuation_status': arm['continuation_status'], 'metric_labels': {},
            'atom_categories': list(data.ATOMS), 'bond_categories': list(data.BONDS),
            'scalars': {**data.asset_binding(scalar_path, output), 'count': len(scalar_rows)},
            'frames': {'count': len(frames), 'steps': [f['global_step'] for f in frames], 'chunks': chunks},
            'latest': {'global_step': frames[-1]['global_step'], 'chunk': chunks[-1]['path'], 'index': (len(frames)-1) % data.CHUNK_FRAMES},
            'update_counts': counts, 'segment_proofs': [proof], 'matrix_stride': STRIDE, 'loss_stride': 1,
            'saved_frames_count': len(snapshots), 'validity': {'saved_states': len(states), 'sanitized': valid,
                'valid_connected': connected, 'distinct_valid_identities': len(identities)},
            'missing_frame_policy': 'All matrices were saved and verified; report selects every20th+endpoint for display. No interpolation.',
            'interpretation': 'Changing stopped teacher/time/weight; contact proxy and validity do not establish binding activity'}
