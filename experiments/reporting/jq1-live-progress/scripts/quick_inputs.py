"""Standard-library run discovery and provenance for the reusable report command.

This module never imports a model or evaluates a scientific array. Terminal
fixed-step/noisy JQ1 saved-study schemas are discovered from original records.
Unknown schemas and incomplete/unstable inputs fail explicitly.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat

VERSION = 'mosaic-report-selection-v1'
MAX_RUNS = 24
METRICS = {'contact_loss', 'affinity_loss', 'dbmol_total_loss'}


def require(value, message):
    if not value:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def load(path):
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_bytes(), parse_constant=reject)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def regular(path, roots=None):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'Require an absolute canonical saved path')
    require(stat.S_ISREG(path.lstat().st_mode), 'Saved file is not regular: ' + str(path))
    require(all(not p.is_symlink() for p in (path, *path.parents)), 'Linked saved input is not supported')
    if roots is not None:
        require(any(path.is_relative_to(Path(root)) for root in roots), 'Path is outside authorized report roots')
    return path


def binding(path, roots=None):
    path = regular(path, roots)
    before = path.stat()
    result = {'path': str(path), 'bytes': before.st_size, 'sha256': sha(path)}
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            'Saved input changed during capture; retry with a fresh request')
    return result


def verify_binding(row, roots=None):
    require(binding(row['path'], roots) == row, 'Saved proof length/SHA changed: ' + row['path'])
    return Path(row['path'])


def study_path(path):
    """Accept a study, case, trajectory, job parent or summary path."""
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'Run path must be absolute')
    if path.name == 'summary.json':
        return path.parent
    candidates = [path, path / 'payload/study', path / 'study', path.parent]
    matches = [p for p in candidates if (p / 'summary.json').is_file()]
    require(matches, 'No saved summary.json found. This version reports terminal saved studies; '
            'a queued/running study without a terminal summary is not yet supported: ' + str(path))
    return matches[0]


def science_key(arm, renderer_sha):
    """Display wording is mutable; all scientific input/settings identities are not."""
    fields = ('global_offset', 'requested_updates', 'source_manifest_sha256', 'job_id',
              'scientific_status', 'case_directory', 'anchor_directory', 'states', 'updates',
              'summary', 'config', 'case_report', 'retained_index', 'stdout', 'stderr')
    return digest({'renderer_sha256': renderer_sha,
                   'arm': {k: arm[k] for k in ('id', 'kind', 'rate', 'global_start', 'global_target', 'scientific_status')},
                   'segments': [{k: s[k] for k in fields if k in s} for s in arm['segments']]})


def selection_key(arms, renderer_sha):
    rows = []
    for a in arms:
        display = {k: a.get(k) for k in ('label', 'status_detail', 'continuation_status')}
        display['primary_metric'] = a.get('primary_metric', 'dbmol_total_loss' if a['id'] == 'dbmol'
                                           else 'affinity_loss' if a['id'] in ('boltz', 'boltz_chemistry') else 'contact_loss')
        rows.append({'science': science_key(a, renderer_sha), 'display': display})
    return digest(rows)


def check_lsf(path, job, status):
    text = ''.join(Path(path).read_text().split())
    require(text.count('Subject:Job') == 1 and 'Subject:Job' + job + ':' in text,
            'Original same-job LSF terminal identity missing')
    if status == 'passed':
        require(text.count('Successfullycompleted.') == 1, 'Passed summary lacks original LSF success')
    else:
        require(re.search(r'Exitedwithexitcode[1-9][0-9]*\.', text) is not None,
                'Failed summary lacks original nonzero LSF exit')


def counts(states, updates, offset, anchor=None, roots=None):
    result = {}
    for field, path in (('states', states), ('updates', updates)):
        n = frames = 0
        with Path(path).open('rb') as stream:
            for line in stream:
                require(line.endswith(b'\n'), 'Terminal saved input has an incomplete JSONL suffix')
                row = json.loads(line)
                require(isinstance(row, dict), 'Saved JSONL row is not an object')
                index = row.get('step', row.get('update', -1) - 1)
                require(type(index) is int and index == n, 'Saved prefix index is not contiguous')
                if field == 'states':
                    require(row.get('global_step', offset + n) == offset + n, 'Saved global index differs')
                    require(type(row.get('snapshot_retained')) is bool, 'Missing retained-frame marker')
                    frames += int(row['snapshot_retained'])
                    if anchor is not None and row['snapshot_retained']:
                        part = PurePosixPath(row['snapshot'])
                        require(not part.is_absolute() and '..' not in part.parts and '\\' not in row['snapshot'],
                                'Unsafe original retained snapshot path')
                        snapshot = regular(Path(anchor).joinpath(*part.parts), roots)
                        require(sha(snapshot) == row['snapshot_sha256'], 'Original retained snapshot bytes changed')
                n += 1
        result[field] = n
        if field == 'states':
            result['retained_frames'] = frames
    require(result['states'] == result['updates'] + 1, 'Saved state/update prefix is incomplete')
    return {'committed_updates': result['updates'], 'scalar_states': result['states'],
            'retained_frames': result['retained_frames']}


def verified_seed(row, roots):
    """Authenticate an already allocated/validated report, not a guessed cache."""
    root = Path(row['path'])
    require(sha(regular(root / 'report-receipt.json', roots)) == row['receipt_sha256'], 'Seed receipt changed')
    receipt = load(root / 'report-receipt.json')
    require(receipt['status'] == 'passed_saved_only_presentation'
            and receipt['manifest_sha256'] == row['source_manifest_sha256'], 'Seed was not a validated render')
    require(sha(root / 'cpu-certificate.json') == receipt['cpu_certificate_sha256'] == row['cpu_certificate_sha256'],
            'Seed CPU certificate changed')
    cpu = load(root / 'cpu-certificate.json')
    require(cpu['status'] == 'passed' and cpu['manifest_sha256'] == row['source_manifest_sha256'], 'Seed CPU validation failed')
    verify_binding(row['render_lsf'], roots)
    check_lsf(row['render_lsf']['path'], receipt['job_id'], 'passed')
    source = Path(row['source'])
    require(sha(regular(source / 'source_manifest.json', roots)) == row['source_manifest_sha256'], 'Seed source manifest changed')
    original_source = load(source / 'source_manifest.json')
    renderer_sha = sha(regular(source / 'scripts/report_data.py', roots))
    require(original_source['files']['scripts/report_data.py'] == renderer_sha, 'Seed scientific renderer changed')
    for name in ('manifest.json', 'input-plan.json', 'report.html'):
        expected = receipt['files'][name]
        require(binding(root / name, roots) == {'path': str(root / name), **expected}, 'Seed metadata bytes changed')
    return {'root': root, 'receipt': receipt, 'manifest': load(root / 'manifest.json'),
            'plan': load(root / 'input-plan.json'), 'renderer_sha256': renderer_sha}


def seed_catalog(profile):
    known = {}
    seeds = [verified_seed(row, profile['allowed_roots']) for row in profile.get('seed_reports', [])]
    for seed in seeds:
        for arm in seed['plan']['arms']:
            for segment in arm['segments']:
                known[str(Path(segment['summary']['path']).parent)] = (arm, segment)
    return seeds, known


def source_and_config(summary, hint, profile, study=None):
    pin = summary['source_manifest_sha256']
    if hint.get('config'):
        path = Path(hint['config']['path'] if isinstance(hint['config'], dict) else hint['config'])
        rec = binding(path, profile['allowed_roots'])
        require(rec['sha256'] == summary['config_sha256'], 'Original config does not match summary')
        return rec
    if study is not None:
        drivers = [Path(study).parent / 'driver-invocation.json',
                   Path(study).parent / 'endpoint-driver-invocation.json',
                   Path(study).parent.parent / 'driver-invocation.json']
        for path in drivers:
            if not path.exists():
                continue
            value = load(regular(path, profile['allowed_roots']))
            args = value.get('argv', value.get('args', []))
            if not isinstance(args, list) or '--config' not in args or '--manifest' not in args:
                continue
            if args[args.index('--manifest') + 1] != pin:
                continue
            rec = binding(args[args.index('--config') + 1], profile['allowed_roots'])
            require(rec['sha256'] == summary['config_sha256'], 'Actual driver config/summary pins differ')
            return rec
    candidates = []
    sources = list(profile.get('source_catalog', []))
    if not any(s['manifest_sha256'] == pin for s in sources):
        manifests = [p for root in profile.get('source_roots', [])
                     for p in Path(root).glob('*/source/source_manifest.json')]
        require(len(manifests) <= 1000, 'Source discovery is too broad; declare the exact config path')
        for path in manifests:
            if sha(regular(path, profile['allowed_roots'])) == pin:
                sources.append({'path': str(path.parent), 'manifest_sha256': pin})
    for source in sources:
        if source['manifest_sha256'] != pin:
            continue
        root = Path(source['path'])
        require(sha(regular(root / 'source_manifest.json', profile['allowed_roots'])) == pin, 'Source manifest changed')
        for name, value in load(root / 'source_manifest.json')['files'].items():
            if value == summary['config_sha256'] and name.endswith('.json'):
                part = PurePosixPath(name)
                require(not part.is_absolute() and '..' not in part.parts and '\\' not in name, 'Unsafe source member')
                candidates.append(binding(root.joinpath(*part.parts), profile['allowed_roots']))
    require(len(candidates) == 1, 'Cannot uniquely locate original configuration; add its config path to the run selection')
    return candidates[0]


def capture_segment(spec, profile, known):
    study = study_path(spec['run'])
    original = known.get(str(study))
    hint = deepcopy(original[1]) if original else {}
    hint.update({k: v for k, v in spec.items() if k != 'run'})
    summary_binding = binding(study / 'summary.json', profile['allowed_roots'])
    # A byte-identical authenticated seed avoids reparsing a 300MB summary.
    if original and summary_binding == original[1]['summary']:
        config_binding = binding(hint['config']['path'] if isinstance(hint['config'], dict) else hint['config'], profile['allowed_roots'])
        summary = {'status': original[1]['scientific_status'], 'source_manifest_sha256': original[1]['source_manifest_sha256'],
                   'config_sha256': original[1]['config']['sha256'], 'job_id': original[1]['job_id'], 'config': load(config_binding['path'])}
    else:
        summary = load(study / 'summary.json')
    require(summary.get('status') in ('passed', 'failed'), 'A terminal passed/failed summary is required; original status is never invented')
    job = str(summary.get('job_id', ''))
    require(job.isdigit(), 'Original job identity missing')
    require(re.fullmatch('[a-f0-9]{64}', summary.get('source_manifest_sha256', '')) is not None, 'Original source pin missing')
    if hint.get('source_manifest_sha256'):
        require(summary['source_manifest_sha256'] == hint['source_manifest_sha256'], 'Declared original source identity changed')
    config_binding = source_and_config(summary, hint, profile, study)
    config = load(config_binding['path'])
    kind = 'noisy' if 'learning_rate' in config and 'initial_step_size' not in config else 'fixed'
    case = study / ('trajectory' if kind == 'noisy' else 'case-0')
    if hint.get('case_directory'):
        case = Path(hint['case_directory'])
    with (case / 'states.jsonl').open('rb') as stream:
        first = json.loads(stream.readline())
    execution = next((summary[k] for k in ('execution', 'serial_execution', 'extended_execution')
                      if isinstance(summary.get(k), dict) and {'start', 'updates', 'target'} <= set(summary[k])), {})
    offset = spec.get('global_offset', first.get('global_step', execution.get('start', config.get('extension', {}).get('start_global_update', 0))))
    requested = spec.get('requested_updates', execution.get('updates', hint.get('requested_updates', config.get('extension', {}).get('segment_updates', config.get('steps')))))
    if execution:
        require(offset == execution['start'] and requested == execution['updates'], 'Declared segment scope differs from original execution')
    require(type(offset) is int and offset >= 0 and type(requested) is int and requested > 0, 'Original segment range is missing')
    row = {'id': hint.get('id', study.parent.parent.name + '-' + study.parent.name), 'case_directory': str(case),
           'anchor_directory': hint.get('anchor_directory', str(study if kind == 'noisy' else case)),
           'global_offset': offset, 'requested_updates': requested, 'source_manifest_sha256': summary['source_manifest_sha256'],
           'job_id': job, 'scientific_status': summary['status'], 'config': config_binding, 'summary': summary_binding}
    for name, path in [('states', case / 'states.jsonl'), ('updates', case / 'updates.jsonl'),
                       ('case_report', case / 'report.json'), ('retained_index', case / 'retained-frames.json')]:
        if path.exists() or name in ('states', 'updates', 'retained_index'):
            row[name] = binding(path, profile['allowed_roots'])
    for name in ('stdout', 'stderr'):
        explicit = hint.get(name)
        path = Path(explicit['path'] if isinstance(explicit, dict) else explicit) if explicit else None
        if path is None:
            matches = [p / (name + '.' + job + '.log') for p in (study, study.parent, study.parent.parent, study.parent.parent.parent)]
            matches = [p for p in matches if p.is_file()]
            require(len(matches) == 1, 'Cannot uniquely locate original ' + name + '; add its path to the selection')
            path = matches[0]
        row[name] = binding(path, profile['allowed_roots'])
    check_lsf(row['stdout']['path'], job, summary['status'])
    row['expected_counts'] = counts(row['states']['path'], row['updates']['path'], offset,
                                    row['anchor_directory'], profile['allowed_roots'])
    return row, config, kind


def capture(selection, profile, additional_seeds=()):
    require(selection.get('version') == VERSION, 'Wrong run-selection version')
    items = selection.get('runs')
    require(isinstance(items, list) and 1 <= len(items) <= MAX_RUNS, 'Select between1 and24 saved runs')
    _, known = seed_catalog(profile)
    for seed in reversed(additional_seeds):
        for arm in seed['plan']['arms']:
            for segment in arm['segments']:
                known[str(Path(segment['summary']['path']).parent)] = (arm, segment)
    arms = []
    for item in items:
        spec = {'segments': [{'run': item}]} if isinstance(item, str) else deepcopy(item)
        if 'segments' not in spec:
            spec['segments'] = [{'run': spec.pop('run')}]
        require(spec['segments'] and len(spec['segments']) <= 100, 'Missing or excessive ordered segments')
        found = [capture_segment(s, profile, known) for s in spec['segments']]
        segments = [r[0] for r in found]
        config, kind = found[0][1:]
        rate = config['learning_rate' if kind == 'noisy' else 'initial_step_size']
        require(all(k == kind and c['learning_rate' if k == 'noisy' else 'initial_step_size'] == rate for _, c, k in found),
                'Grouped segments changed trajectory schema or learning rate')
        start = segments[0]['global_offset']
        target = spec.get('global_target', config.get('extension', {}).get('target_global_update', config['steps']))
        applied = sum(s['expected_counts']['committed_updates'] for s in segments)
        end = start
        for s in segments:
            require(s['global_offset'] == end, 'Segment group has a gap, overlap or incorrect order')
            end += s['expected_counts']['committed_updates']
        require(end <= target, 'Saved prefix exceeds the declared target')
        identity = spec.get('id', 'run-' + hashlib.sha256(str(study_path(spec['segments'][0]['run'])).encode()).hexdigest()[:12])
        require(re.fullmatch('[a-z][a-z0-9_-]*', identity) is not None, 'Unsafe report run ID')
        metric = spec.get('primary_metric', 'dbmol_total_loss' if 'dbmol' in str(config.get('version', '')).lower()
                          or 'dbmol' in str(config.get('arm', '')).lower() else
                          'affinity_loss' if str(config.get('arm', '')).startswith('boltz') else 'contact_loss')
        require(metric in METRICS, 'Unsupported primary loss; use contact_loss, affinity_loss or dbmol_total_loss')
        status = 'failed' if any(s['scientific_status'] == 'failed' for s in segments) else 'passed' if end == target else 'partial'
        frames = sum(s['expected_counts']['retained_frames'] for s in segments) - len(segments) + 1
        arm = {'id': identity, 'label': spec.get('label', config.get('label', identity)), 'kind': kind, 'rate': rate,
               'global_start': start, 'global_target': target, 'target_updates': target, 'segments': segments,
               'scientific_status': status, 'primary_metric': metric,
               'status_detail': spec.get('status_detail', 'Original scientific status is preserved; saved progress is separate from validation.'),
               'continuation_status': spec.get('continuation_status', 'not_recorded'), 'metric_labels': {},
               'expected_counts': {'committed_updates': applied, 'scalar_states': applied + 1,
                                   'retained_frames': frames, 'latest_global_step': end},
               'expected_committed_updates': applied, 'expected_retained_frames': frames}
        arms.append(arm)
    require(len({a['id'] for a in arms}) == len(arms), 'Duplicate report run IDs')
    return {'version': 'jq1-live-saved-report-v1', 'edition': selection['edition'],
            'captured_at': datetime.now(timezone.utc).isoformat(), 'arms': arms,
            'monitor_remains_paused': True, 'automatic_optimizer_submission': False}
