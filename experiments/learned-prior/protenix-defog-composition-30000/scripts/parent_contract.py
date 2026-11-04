"""Immutable pilot metadata and transport identities; stdlib only."""
import hashlib
import json
import math
from pathlib import Path

VERSION = "protenix-defog-moving-teacher-balanced-30000-v1"
ELEMENT_NATIVE_INDICES = (0, 1, 2, 9, 3, 6)  # C,N,O,S,F,Cl in official GuacaMol order.


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate(c):
    if (c['version'] != VERSION or c['steps'] != 30000 or c['n_atoms'] != 25 or c['seed'] != 17
        or c['initialization'] != {'kind': 'dirichlet', 'concentration': 1.0, 'seed': 17}
        or c['learning_rate'] != 0.3 or c['max_graph_cell_tv'] != 0.02 or c['max_step_halvings'] != 12
        or c['allocation_seconds'] != 28800 or c['internal_seconds'] != 28200):
        raise ValueError('Changed declared scientific budget/update policy')
    if c['teacher'] != {
        'policy': 'hard_aux_soft_core', 'time_ramp': {'version': 'linear_updates_v1', 'initial': 0.2, 'final': 0.95, 'first_update': 0, 'last_update': 29999}, 'refresh_every': 1,
        'normalization': 'mean_over_atoms_and_unordered_bonds', 'loss': 'KL_teacher_to_current',
        'support': 'condition_native_elements_on_C_N_O_S_F_Cl',
        'source_commit': '365bda9affadd5c2307014a0532ddaa244399441',
        'checkpoint_sha256': 'bc97d52ccf33a20d991f98349a1d37662273ab1c3e3f9777b938f9f144bc61f5',
        'max_queries': 30200}:
        raise ValueError('Changed teacher policy/assets')
    if c['weight_ramp'] != {'version': 'constant_initial_gradient_match_v1', 'initial': 0.5364198835347113, 'final': 0.5364198835347113, 'first_update': 0, 'last_update': 29999}:
        raise ValueError('Changed ramp')
    if (c['pilot_updates_per_endpoint'] != 10 or c['max_output_bytes'] != 4294967296
        or c['minimum_free_work_gib'] != 5 or c['native_parity_atol'] != 2e-5):
        raise ValueError('Changed validation/resource guards')
    if (c['native_model_digest'] != '8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2'
        or c['fixture'] != 'tests/data/brd4_3mxf/fixture_25.json' or c['pdb'] != 'tests/data/brd4_3mxf/3MXF.pdb'
        or c['contact_cutoff_angstrom'] != 8.0):
        raise ValueError('Changed native objective/model/input identity')
    if c.get('raw_probe_policy') != 'fixed_scales_largest_feasible_contrast_v1':
        raise ValueError('Changed random-simplex raw-probe policy')
    if c.get('calibration') != {'policy': 'initial_total_logit_gradient_l2_match_v1', 'initial_native_norm': 0.027899492688150188, 'initial_unweighted_prior_norm': 0.05201054909506321, 'constant_weight': 0.5364198835347113, 'relative_tolerance': 0.001, 'parent_source_manifest_sha256': 'a15c8bb2a8ac7e9e12a65c54079e14c994e53b119b9a59e1a9788b5044d8686f', 'parent_run': '/work3/s234818/mosaic/runs/20261028-protenix-defog-30000-time-ramp-01/pilot', 'parent_context_digest': 'b28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4', 'parent_context_files': {'prepared_context.json': '694234feda0b6abaa7e7ed8a7f4b8367165e4326392041457236e994689df9e3', 'prepared_context.npz': 'a0a52d66ba86619474ceed50f2fc380238ed7b4227599d71a4b50a75c50513dc'}, 'parent_initial_snapshot_sha256': 'f9a22d5cb4a3113b6fdf222dbb95d5000ce225ae8014d7d2180a9e6a48fd2eff', 'CPU_job': '29846712', 'saved_analysis_sha256': '0d28528e9c233eb0ffc3f825f1f14eca829979bd356afda7ada773c8c65799cc'}:
        raise ValueError('Changed declared gradient calibration or parent context')
    return c


def weight(c, index):
    if type(index) is not int or not 0 <= index <= c['steps']:
        raise ValueError('Invalid state/update index')
    r = c['weight_ramp']
    f = min(index, r['last_update']) / r['last_update']
    return r['initial'] + (r['final'] - r['initial']) * f


def teacher_time(c, index):
    if type(index) is not int or not 0 <= index <= c['steps']:
        raise ValueError('Invalid teacher schedule index')
    r = c['teacher']['time_ramp']
    f = min(index, r['last_update']) / r['last_update']
    return r['initial'] + (r['final'] - r['initial']) * f


def query_identity(graph_sha, index, active_time):
    return canonical_sha({'graph_sha256': graph_sha, 'state_index': index, 'teacher_time': active_time})


def verify_source(root, manifest_sha):
    root = Path(root).resolve()
    path = root / 'source_manifest.json'
    if sha(path) != manifest_sha:
        raise ValueError('Source manifest mismatch')
    m = json.loads(path.read_text())
    for name, expected in m['files'].items():
        file = (root / name).resolve()
        if not file.is_relative_to(root) or sha(file) != expected:
            raise ValueError('Source member changed: ' + name)
    return m
