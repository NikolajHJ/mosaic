"""Freeze saved-candidate coverage before observing any fresh native outcome.

This is an HPC-only selection stage: hard-label chemistry is recomputed by the
existing audited reader. Failed/unstarted parent roles and invalid categorical
draws retain their original denominators. Selection never ranks proxy scores.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile

import yaml

from latentmol.experiments import file_sha256
from latentmol.config_identity import verify_config_identity
from .active_set_rescore import _read_candidates, _target_identity, select_endpoints


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verified(path, sha256):
    path = Path(path)
    if file_sha256(path) != sha256:
        raise ValueError(f'Artifact identity changed: {path}')
    return path


def artifact_tree(root):
    """Full file-set seal; additions/removals matter, not only modified files."""
    return {str(p.relative_to(root)): file_sha256(p)
            for p in sorted(root.rglob('*')) if p.is_file()}


def verify_tree(root, seal):
    if artifact_tree(root) != seal:
        raise ValueError(f'Sealed file set changed: {root}')


def _controller(root, expected_archive, table):
    root = Path(root)
    if table == 'cases.json':
        folder = root if (root / 'identity.json').exists() else None
        if folder:
            record = read(folder / 'identity.json')
            if record['source_archive_sha256'] != expected_archive:
                raise ValueError('Panel controller has a different source archive')
    else:
        folders = [p.parent for p in root.glob('*/run.json')]
        if len(folders) > 1:
            raise ValueError('Random study root is ambiguous; do not choose by result')
        folder = folders[0] if folders else None
        if folder and read(folder / 'run.json')['source_archive_sha256'] != expected_archive:
            raise ValueError('Random controller has a different source archive')
    if folder is None:
        return None, []
    path = folder / table
    return folder, read(path) if path.exists() else []


def planned_roles(protocol):
    """The protocol, rather than successful outputs, defines the cohort."""
    parents = protocol['parents']
    panel = yaml.safe_load(verified(**parents['panel']['protocol']).read_text())
    targets = read(verified(**panel['targets']))['targets']
    # The target manifest is a dictionary keyed by stable target name.
    if isinstance(targets, list):
        targets = {r['id']: r for r in targets}
    random = yaml.safe_load(verified(**parents['random']['protocol']).read_text())
    roles = []
    for count in panel['counts']:
        for target in targets:
            roles.append({'id': f'panel/{target}-c{count:02d}/fixed_initial_mask',
                          'cohort': 'panel', 'case': f'{target}-c{count:02d}',
                          'target': target, 'count': count, 'arm': 'fixed_initial_mask',
                          'draws': protocol['selection']['panel_sample_draws']
                          if count in protocol['selection']['panel_sample_counts'] else 0})
    for seed in random['seeds']:
        for arm in ('active_set', 'fixed_initial_mask'):
            roles.append({'id': f'random/seed-{seed:04d}/{arm}', 'cohort': 'random',
                          'seed': seed, 'arm': arm, 'draws': protocol['selection']['random_sample_draws']})
    for arm in ('active_set', 'fixed_initial_mask'):
        roles.append({'id': f'c13/{arm}', 'cohort': 'c13', 'arm': arm,
                      'draws': protocol['selection']['c13_sample_draws']})
    return roles


def _pilot_candidates(pilot, expected_archive, expected_config_file=None):
    record = read(pilot / 'run.json')
    if record.get('error') == 'Frozen decoder or Boltz model state changed':
        raise ValueError('Parent reports frozen-model integrity failure')
    if record['source_archive_sha256'] != expected_archive:
        raise ValueError('Pilot source archive differs from declared parent')
    config = record['config']
    config_identity = verify_config_identity(record, allow_legacy_active_set_contexts=True)
    if expected_config_file and config['config_file_sha256'] != expected_config_file:
        raise ValueError('Parent config file identity differs from controller')
    if config['optimizer']['steps'] != 100 or config['diagnostics']['hard_samples'] < 8:
        raise ValueError('Unexpected parent optimization/sampling budget')
    if record.get('status') == 'completed':
        select_endpoints(pilot)  # All states, frozen-state check and passed gates.
    elif record.get('status') not in ('failed', 'running'):
        raise ValueError('Unsupported parent status')
    # An ended allocation can leave a running record. Its available states are
    # labeled partial, never upgraded to a completed/frozen-state-verified run.
    rows, proxies, warnings = {}, {}, []
    if config_identity['historical_context_keys_reconstructed']:
        warnings.append({'configuration_identity': config_identity})
    for arm in config.get('arms', ('active_set', 'fixed_initial_mask')):
        path = pilot / arm / 'candidates.jsonl'
        if not path.exists():
            continue
        gates_path = pilot / arm / 'gates.json'
        gates = read(gates_path) if gates_path.exists() else {}
        failed = [key for key, gate in gates.items() if not gate.get('passed') or not gate.get('native_seed_parity')]
        if failed:
            warnings.append({'arm': arm, 'failed_gates': failed})
        try:
            rows[arm] = _read_candidates(path, record['graph_spec_sha256'], 100)
        except json.JSONDecodeError:
            # Only an incomplete FINAL line is compatible with an interrupted
            # append. Complete malformed rows and all completed runs stay fatal.
            lines = path.read_text().splitlines(keepends=True)
            if record['status'] == 'completed' or not lines or lines[-1].endswith('\n'):
                raise
            for line in lines[:-1]:
                json.loads(line)
            with tempfile.TemporaryDirectory(prefix='latentmol-candidate-tail-') as tmp:
                copy = Path(tmp) / arm / path.name
                copy.parent.mkdir(); copy.write_text(''.join(lines[:-1]))
                rows[arm] = _read_candidates(copy, record['graph_spec_sha256'], 100)
            warnings.append({'arm': arm, 'excluded_truncated_final_candidate_line': True,
                             'original_file_sha256': file_sha256(path)})
        proxies[arm] = {}
        for metric in sorted((pilot / arm).glob('metrics-*.json')):
            row = read(metric)
            step = row['step']
            applicable = [(int(key[5:]), g) for key, g in gates.items() if int(key[5:]) <= step]
            latest = max(applicable, key=lambda pair: pair[0])[1] if applicable else None
            if not latest or not latest.get('passed') or not latest.get('native_seed_parity') or row['context_sha256'] != latest['context_sha256']:
                # JSON keys are sorted by write_json; step keys have equal width.
                if record['status'] == 'completed':
                    raise ValueError('Completed state has no corresponding passed native context gate')
                warnings.append({'arm': arm, 'unverified_step': step})
                continue
            proxies[arm][step] = {'binary': row['score']['raw'],
                                 'context_sha256': row['context_sha256'],
                                 'natural_count': row['natural_selection']['count'],
                                 'active_count': row['selection']['count']}
    return record, rows, proxies, warnings


def freeze_selection(protocol):
    """Discover once, verify source/config/raw graphs, then content-address requests.

    A missing parent is an explicit unavailable role. An existing malformed or
    changed parent is an integrity failure, never quietly called unavailable.
    """
    parents = protocol['parents']
    panel_root, cases = _controller(parents['panel']['root'], parents['panel']['source_sha256'], 'cases.json')
    random_root, trials = _controller(parents['random']['root'], parents['random']['source_sha256'], 'trials.json')
    if panel_root and read(panel_root / 'identity.json')['protocol_sha256'] != parents['panel']['protocol']['sha256']:
        raise ValueError('Panel controller protocol changed')
    if random_root and read(random_root / 'run.json')['config']['protocol_sha256'] != parents['random']['protocol']['sha256']:
        raise ValueError('Random controller protocol changed')
    roles = planned_roles(protocol)
    cases = {r['id']: r for r in cases}
    trials = {r['seed']: r for r in trials}
    parent_seals, targets, pilot_cache, requests, members = {}, {}, {}, {}, []
    for controller in (panel_root, random_root):
        if controller:
            # Full parent trees can contain native structures/checkpoints; hashes
            # stream from disk and do not load models into the selection process.
            parent_seals[str(controller)] = artifact_tree(controller)
    for role in roles:
        parent = parents[role['cohort']]
        row = cases.get(role.get('case'), {}) if role['cohort'] == 'panel' else trials.get(role.get('seed'), {})
        if role['cohort'] == 'panel' and row.get('sealed_artifacts'):
            from .protein_panel import verify_terminal
            verify_terminal(row, panel_root)
        pilot = Path(parent['pilot']) if role['cohort'] == 'c13' else Path(row['pilot']) if row.get('pilot') else None
        role['parent_status'] = row.get('status', 'unavailable') if role['cohort'] != 'c13' else 'unavailable'
        role['pilot'] = str(pilot) if pilot else None
        by_key, proxy, target_key = {}, {}, None
        if pilot and (pilot / 'run.json').exists():
            if role['cohort'] != 'c13' and not pilot.resolve().is_relative_to(Path(parent['root']).resolve()):
                raise ValueError('Pilot escaped the declared parent root')
            if str(pilot) not in pilot_cache:
                result = _pilot_candidates(pilot, parent['source_sha256'],
                                           parent['config_sha256'] if role['cohort'] == 'c13' else row.get('config_sha256'))
                pilot_cache[str(pilot)] = result
                parent_seals[str(pilot)] = artifact_tree(pilot)
            record, rows, proxies, warnings = pilot_cache[str(pilot)]
            if record['config']['affinity_checkpoint']['sha256'] != protocol['checkpoints']['affinity']:
                raise ValueError('Parent optimizer uses another affinity checkpoint')
            role['warnings'] = warnings
            if role['cohort'] == 'panel' and (row['target'] != role['target'] or row['count'] != role['count']
                    or record['target_sequence_sha256'] != row['sequence_sha256']
                    or record['msa_sha256'] != row['msa_sha256']):
                raise ValueError('Panel target/count identity differs from planned role')
            if role['cohort'] == 'random' and record['config']['start'] != {'kind': 'standard_normal', 'seed': role['seed']}:
                raise ValueError('Random initialization differs from planned seed')
            role['pilot_status'] = record['status']
            role['objective'] = record['config']['objective']
            role['full_budget_verified'] = record['status'] == 'completed'
            role['parent_status'] = record['status'] if role['cohort'] == 'c13' else role['parent_status']
            by_key = {(r['step'], r['draw']): r for r in rows.get(role['arm'], [])}
            proxy = proxies.get(role['arm'], {})
            template_path = (Path(row['attempt_dir']) / 'input' / f"carbon_chain_{role['count']:02d}.yaml"
                             if role['cohort'] == 'panel' else Path(parent['template']['path']))
            if role['cohort'] != 'panel':
                verified(**parent['template'])
            if not rows.get(role['arm']):
                # A gate/start failure before recording any graph needs no
                # native template and remains wholly unavailable below.
                target = None
            else:
                target = _target_identity(template_path, record['target_sequence_sha256'], record['msa_sha256'])
            if target is not None:
                template = target.pop('template')
                protein = next(v['protein'] for v in template['sequences'] if 'protein' in v)
                target.update(template=template, protein_sequence=protein['sequence'], msa_path=protein['msa'])
                context = deepcopy(template)
                next(v['protein'] for v in context['sequences'] if 'protein' in v)['msa'] = record['msa_sha256']
                next(v['ligand'] for v in context['sequences'] if 'ligand' in v)['smiles'] = '<shared-candidate>'
                target_key = digest(context)
                if target_key in targets and targets[target_key]['protein_sequence'] != target['protein_sequence']:
                    raise ValueError('Target identity collision')
                targets[target_key] = target
        role['target_key'] = target_key
        for step in protocol['selection']['checkpoints']:
            for draw in range(role['draws'] + 1):
                member = {'id': f"{role['id']}/s{step:04d}/d{draw:03d}", 'role': role['id'],
                          'cohort': role['cohort'], 'target': role.get('target', 'C13-reference-target'),
                          'step': step, 'draw': draw, 'kind': 'argmax' if draw == 0 else 'categorical',
                          'target_key': target_key, 'proxy': proxy.get(step),
                          'status': 'unavailable', 'failure': 'Parent/checkpoint not available', 'requests': {}}
                candidate = by_key.get((step, draw))
                if candidate:
                    member.update(candidate=deepcopy(candidate), status='valid' if candidate['smiles'] else 'invalid',
                                  failure=candidate['failure'])
                    if step not in proxy:
                        member.update(status='unverified', failure='Checkpoint lacks a passed matching native context gate')
                    if candidate['smiles'] and step in proxy:
                        for seed in protocol['native']['pose_seeds']:
                            request = {'target_key': target_key, 'smiles': candidate['smiles'], 'seed': seed,
                                       'native_settings': protocol['native'], 'checkpoints': protocol['checkpoints']}
                            key = digest(request)
                            requests[key] = {'id': key, **request}
                            member['requests'][str(seed)] = key
                members.append(member)
    for root, seal in parent_seals.items():
        verify_tree(Path(root), seal)
    # Stable order first scores starts across all roles, then increasing step.
    # Never prioritize by proxy, chemistry alerts or a native prediction.
    ordered = []
    seen = set()
    for member in sorted(members, key=lambda r: (r['step'], r['draw'], r['role'])):
        for seed in protocol['native']['pose_seeds']:
            key = member['requests'].get(str(seed))
            if key and key not in seen:
                ordered.append(requests[key]); seen.add(key)
    return {'version': 1, 'roles': roles, 'members': members, 'requests': ordered,
            'targets': targets, 'parent_seals': parent_seals,
            'policy': 'Predefined roles/checkpoints/draw indices; no outcome ranking or replacements',
            'maximum_memberships': len(members), 'logical_native_memberships': len(members) * len(protocol['native']['pose_seeds'])}
