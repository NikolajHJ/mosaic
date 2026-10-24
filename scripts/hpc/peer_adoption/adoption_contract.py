"""Standard-library proof and ownership contracts; no numerical operations.

This module is prospective metadata tooling. A metadata CPU pass is independent
of the future native results; production adoption requires those real results.
Scientific sources, CPU/native proofs and saved trajectories remain immutable.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import xml.etree.ElementTree as ET

BASE = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
GIB = 1024 ** 3
VERSION = 'protenix02-peer-adoption-v1'
REVISION = 'c74c659344123937608077e3e4078f75f09cd743'
NOISY_PARENTS = (Path('/work3/s234818/mosaic/runs'), Path('/work3/s234818/mosaic/releases'))
PROFILES = {
    'boltz30k': {
        'source': BASE / 'boltz-lr0025-release-03/source',
        'manifest': 'c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6',
        'root': BASE / 'boltz-lr0025-03',
        'control': BASE / 'boltz-pro02-adoption-04',
        'launcher': BASE / 'boltz-pro02-adoption-release-04/source',
        'cpu_cases': 827,
        'cpu_sha256': '06efbb027f1008a0921494af0e226c3fc7a78c0a8f4ab01b93e8b50f2c042836',
        'arms': {'boltz2': (30000, 3), 'boltz2_chemistry': (30000, 3)},
        'cpu_parent': 'cpu',
        'native_ids': {'boltz2': ('29805950', '29805951'),
                       'boltz2_chemistry': ('29805958', '29805959')},
        'job_prefix': 'mosaic-bol30k-',
        'original_wrapper': 'scripts/hpc/boltz_serial_stage.lsf',
        'config_prefix': 'boltz_serial_',
    },
    'dbmol10k': {
        'source': BASE / 'boltz-dbmol-pilot-release-02/source',
        'manifest': '600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed',
        'root': BASE / 'boltz-dbmol-pilot-02',
        'control': BASE / 'dbmol-pro02-adoption-06',
        'launcher': BASE / 'dbmol-pro02-adoption-release-06/source',
        'cpu_cases': 1163,
        'cpu_sha256': '387070e8fe56b17ddbd7c8e0b82d2d6a44d9aa99e7015d3fbe4757ce82c0819b',
        'arms': {'boltz2_dbmol': (10000, 4)},
        'cpu_parent': 'cpu-05',
        'native_ids': {'boltz2_dbmol': ('29805980', '29805981')},
        'job_prefix': 'mosaic-dbmol-',
        'original_wrapper': 'scripts/hpc/dbmol_storage_stage.lsf',
        'original_launcher': BASE / 'boltz-dbmol-storage-release-05/source',
        'original_launcher_manifest': '865303de06fc0a0bdf9297857c5bbc78c6ccd9ee05177e372a51869adc599bc3',
    },
    'protenix50k40k': {
        'source': BASE / 'protenix-extended-release-03/source',
        'manifest': 'da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562',
        'root': BASE / 'protenix-extended-03',
        'cpu_cases': 866,
        'arms': {'random': (50000, 5), 'continuation': (40000, 4)},
        'cpu_parent': 'cpu',
        'job_prefix': 'mosaic-pro50k-',
        'original_wrapper': 'scripts/hpc/extended_stage.lsf',
        'config_prefix': 'extended_',
    },
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def file_record(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Proof is not a regular original file')
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def exclusive(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    exclusive(temporary, value)
    temporary.replace(path)


def verify_seal(root, expected, *, historical_missing_revision=False, metadata_only=False):
    """Require complete regular-file, read-only source, before any imports."""
    root = Path(root)
    require(not root.is_symlink() and root.is_dir(), 'Sealed source root missing or linked')
    require(not (root.stat().st_mode & 0o222), 'Sealed source root is writable')
    manifest = root / 'source_manifest.json'
    require(sha(manifest) == expected, 'Source manifest identity differs')
    data = read(manifest)
    files = data['files']
    require(isinstance(files, dict) and files, 'Source lacks its authenticated file map')
    if historical_missing_revision:
        require(expected == 'a30f980fd8fe95001fdfc6780d720bdde6486d2421d7e4942b119ace61777f00'
                and 'git_revision_reference' not in data,
                'Historical missing-field evidence scope differs')
    elif metadata_only and 'git_revision_reference' not in data:
        require(expected == '865303de06fc0a0bdf9297857c5bbc78c6ccd9ee05177e372a51869adc599bc3'
                and data['scientific_source_manifest_sha256'] ==
                '600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed',
                'Original metadata-only launcher scope differs')
    else:
        require(data['git_revision_reference'] == REVISION,
                'Source lacks the authenticated revision')
    observed = set()
    records = []
    for directory, dirs, names in os.walk(root, followlinks=False):
        require(stat.S_ISDIR(Path(directory).stat().st_mode)
                and not (Path(directory).stat().st_mode & 0o222), 'Sealed source directory is writable')
        require(not any((Path(directory) / name).is_symlink() for name in dirs), 'Source directory link')
        for name in names:
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            require(not path.is_symlink() and stat.S_ISREG(path.stat().st_mode), 'Special source file')
            require(not (path.stat().st_mode & 0o222), 'Sealed source is writable')
            observed.add(rel)
    require(observed == set(files) | {'source_manifest.json'}, 'Sealed source file inventory differs')
    for rel, digest in files.items():
        key = Path(rel)
        require(not key.is_absolute() and '..' not in key.parts and key.as_posix() == rel,
                'Unsafe source member')
        path = root / key
        require(sha(path) == digest, 'Sealed source member changed')
        records.append(file_record(path))
    records.append(file_record(manifest))
    return records


def verify_proofs(certificate):
    require(certificate['status'] == 'passed', 'Proof certificate did not pass')
    proofs = certificate['proofs']
    require(isinstance(proofs, list) and proofs, 'Empty proof chain')
    paths = set()
    for item in proofs:
        path = Path(item['path'])
        require(str(path) not in paths and path.is_absolute() and path.is_file() and not path.is_symlink(),
                'Duplicate, missing or linked certificate proof')
        require(sha(path) == item['sha256'], 'Certificate proof bytes changed')
        if 'bytes' in item:
            require(path.stat().st_size == item['bytes'], 'Certificate proof length changed')
        paths.add(str(path))
    return proofs


def verify_certificate(path, source_manifest, cases=None):
    certificate = read(path)
    require(certificate['source_manifest_sha256'] == source_manifest, 'Certificate scientific source differs')
    if cases is not None:
        require(type(certificate['cases']) is int and certificate['cases'] == cases,
                'Allocated CPU case scope differs')
    verify_proofs(certificate)
    return certificate


def profile_config(profile, arm):
    p = PROFILES[profile]
    require(arm in p['arms'], 'Unknown scientific arm')
    name = 'dbmol_pilot' if profile == 'dbmol10k' else p['config_prefix'] + arm
    return p['source'] / 'examples/configs' / (name + '.json')


def native_name(profile, arm, index=0):
    p = PROFILES[profile]
    require(arm in p['arms'] and type(index) is int and 0 <= index < p['arms'][arm][1],
            'Undeclared native boundary')
    return ('pilot' if profile == 'dbmol10k' else arm) + '-native-' + format(index, '04d')


def callback_name(profile, arm, index=0):
    return native_name(profile, arm, index).replace('-native-', '-after-native-')


def resource_vector(phase):
    if phase == 'native':
        return ['-q', 'gpul40s', '-n', '4', '-R', 'span[hosts=1]', '-R',
                'rusage[mem=8GB]', '-gpu', 'num=1:mode=exclusive_process', '-W', '02:00']
    require(phase == 'after-native', 'Unknown original resource phase')
    return ['-q', 'hpc', '-n', '2', '-R', 'span[hosts=1]', '-R', 'rusage[mem=4GB]', '-W', '00:30']


def original_command(profile, label, phase, dependency):
    p = PROFILES[profile]
    root, source = p['root'], p['source']
    args = ['bsub', *resource_vector(phase), '-J', p['job_prefix'] + label,
            '-oo', str(root / label / 'stdout.%J.log'), '-eo', str(root / label / 'stderr.%J.log')]
    if dependency is not None:
        args += ['-w', dependency]
    if profile == 'dbmol10k':
        launcher = p['original_launcher']
        args += ['bash', '-l', str(launcher / p['original_wrapper']), str(launcher),
                 p['original_launcher_manifest'], str(source), str(root), p['manifest']]
    else:
        args += ['bash', '-l', str(source / p['original_wrapper']), str(source), str(root), p['manifest']]
    return args + [phase, str(root / label)]


def original_submissions(profile, arm):
    """Read actual native0 IDs; never guess IDs or reinterpret queued as passed."""
    p = PROFILES[profile]
    jobs = read(p['root'] / 'jobs.json')
    native, callback = native_name(profile, arm), callback_name(profile, arm)
    result = {}
    for label, phase in [(native, 'native'), (callback, 'after-native')]:
        path = p['root'] / label / 'submission.json'
        value = read(path)
        require(value['outcome'] == 'submitted' and re.fullmatch(r'\d+', value['job_id'])
                and jobs[label] == value['job_id'] and value['phase'] == phase,
                'Actual original native job mapping differs')
        if 'native_ids' in p:
            require(value['job_id'] == p['native_ids'][arm][0 if phase == 'native' else 1],
                    'Original actual native/callback ID differs from adopted campaign')
        dependency = None if phase == 'native' else 'ended(' + result['native']['job_id'] + ')'
        require(value['dependency'] == dependency and value['args'] == original_command(profile, label, phase, dependency),
                'Actual original source/resources/dependency vector differs')
        result[phase] = value
    return result


def lsf_terminal(parent, job, *, expected=None):
    """Authenticate real same-job persisted LSF; no fabricated scheduler state."""
    path = Path(parent) / ('stdout.' + job + '.log')
    text = path.read_text()
    rows = re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$', text, re.M)
    require(len(rows) == 1 and rows[0][0] == job, 'Missing unique original same-job LSF footer')
    status = rows[0][1]
    if status == 'Done':
        require('Successfully completed.' in text and 'Exited with exit' not in text, 'Original LSF success missing')
    else:
        require(re.search(r'Exited with exit code [1-9]\d*', text) is not None, 'Unclassified original LSF failure')
    require(expected is None or status == expected, 'Original job did not finish in required state')
    return {'job_id': job, 'status': status, 'proof': file_record(path), 'scheduler_state_invented': False}


def verify_native_units(profile):
    """Authenticate actual full native/projection chain; no model evaluation."""
    p = PROFILES[profile]
    verify_seal(p['source'], p['manifest'])
    cpu = p['root'] / 'cpu-certificate.json'
    certificate = verify_certificate(cpu, p['manifest'], p['cpu_cases'])
    if 'cpu_sha256' in p:
        require(sha(cpu) == p['cpu_sha256'], 'Original scientific CPU certificate changed')
    if profile == 'dbmol10k':
        verify_seal(p['original_launcher'], p['original_launcher_manifest'], metadata_only=True)
        require(certificate['launcher_manifest_sha256'] == p['original_launcher_manifest'],
                'Original DBMol launcher certificate differs')
    projection = p['root'] / 'projection.json'
    values = read(projection)
    units, records = {}, [file_record(cpu)]
    for arm in p['arms']:
        actual = original_submissions(profile, arm)
        initial = p['root'] / callback_name(profile, arm)
        lsf_terminal(initial, actual['after-native']['job_id'], expected='Done')
        observed = []
        paths = sorted(p['root'].glob(native_name(profile, arm)[:-4] + '*-certificate.json'))
        require(paths, 'No authenticated native producer certificate')
        for cp in paths:
            cert = verify_certificate(cp, p['manifest'])
            label = cp.name.removesuffix('-certificate.json')
            ready = p['root'] / label / 'payload/study/assessment.json'
            value = read(ready)
            require(any(x['path'] == str(ready) and x['sha256'] == sha(ready) for x in cert['proofs']),
                    'Native assessment missing from certificate proofs')
            require(value['status'] == 'passed' and value['source_manifest_sha256'] == p['manifest']
                    and value['config_sha256'] == sha(profile_config(profile, arm))
                    and value['arm'] == ('protenix' if profile == 'protenix50k40k' else arm),
                    'Native readiness source/config/arm differs')
            producer = read(p['root'] / label / 'submission.json')
            require(producer['outcome'] == 'submitted' and read(p['root'] / 'jobs.json')[label] == producer['job_id'],
                    'Native producer actual ID differs')
            lsf = lsf_terminal(p['root'] / label, producer['job_id'], expected='Done')
            require(any(x['path'] == lsf['proof']['path'] and x['sha256'] == lsf['proof']['sha256'] for x in cert['proofs']),
                    'Native certificate lacks actual same-job LSF')
            unit = value['storage_projection']['maximum_pilot_update_bytes']
            require(type(unit) is int and unit > 0, 'Native empirical projection invalid')
            observed.append(unit)
            records += [file_record(cp), file_record(ready)]
        unit = values['unit_bytes'] if profile == 'dbmol10k' else values[arm]['unit_bytes']
        require(type(unit) is int and unit == max(observed), 'Live capacity maximum differs from passed native evidence')
        units[arm] = unit
    if profile == 'dbmol10k':
        require(values['status'] == 'passed' and values['source_manifest_sha256'] == p['manifest']
                and values['full_projected_bytes'] == 2 * units['boltz2_dbmol'] * 10000 + 4 * GIB,
                'DBMol live full projection formula differs')
    else:
        require(set(values) == set(p['arms']), 'Unexpected peer projection arms')
    return {'status': 'passed', 'profile': profile, 'units': units,
            'full': {arm: 2 * unit * p['arms'][arm][0] + p['arms'][arm][1] * GIB for arm, unit in units.items()},
            'source_manifest_sha256': p['manifest'], 'cpu_certificate': file_record(cpu),
            'live_projection': file_record(projection), 'proofs': records}


def pending_dependencies(profiles):
    """Only actual own-ended callbacks; missing IDs yield waiting, not a guess."""
    jobs, missing, failed, running = [], [], [], []
    for profile in profiles:
        p = PROFILES[profile]
        for arm in p['arms']:
            label = callback_name(profile, arm)
            if not (p['root'] / label / 'submission.json').exists():
                missing.append({'profile': profile, 'arm': arm, 'reason': 'actual_callback_not_submitted'})
                continue
            actual = original_submissions(profile, arm)
            job = actual['after-native']['job_id']
            log = p['root'] / label / ('stdout.' + job + '.log')
            if log.exists() and re.search(r'^Subject:', log.read_text(), re.M):
                terminal = lsf_terminal(p['root'] / label, job)
                if terminal['status'] == 'Exited':
                    failed.append({'profile': profile, 'arm': arm, **terminal})
            else:
                running.append({'profile': profile, 'arm': arm, 'job_id': job,
                                'status': 'actual_callback_not_terminal',
                                'scheduler_state_invented': False})
            jobs.append(job)
    require(len(jobs) == len(set(jobs)), 'Duplicate peer callback IDs')
    return {'status': 'failed_peer' if failed else ('waiting_unknown_actual_ids' if missing else 'actual_callbacks_known'),
            'missing': missing, 'failed': failed, 'actual_callback_ids': jobs,
            'running': running, 'all_callbacks_terminal': not running and not missing,
            'dependency': None if failed or missing else '&&'.join('ended(' + job + ')' for job in jobs)}


def proof_inventory(profile):
    """Pin immutable proof bytes; intentionally exclude mutable jobs/projection."""
    p = PROFILES[profile]
    paths = {p['root'] / 'cpu-certificate.json'}
    for arm in p['arms']:
        native = p['root'] / native_name(profile, arm)
        callback = p['root'] / callback_name(profile, arm)
        for parent in (native, callback):
            paths.update(x for x in parent.rglob('*') if x.is_file())
        paths.add(p['root'] / (native.name + '-certificate.json'))
    require(all(x.is_file() and not x.is_symlink() for x in paths), 'Missing original proof inventory')
    return [file_record(path) for path in sorted(paths)]


def verify_inventory(records):
    for record in records:
        require(file_record(record['path']) == record, 'Immutable original proof changed')
    return True


def absence_of_production(profile):
    p = PROFILES[profile]
    root = p['root']
    require(not list(root.glob('*-segment-*')) and not list(root.glob('*-completion.json'))
            and not (root / 'completion.json').exists(), 'Optimization output or unknown segment marker already exists')
    require(not list(root.glob('dispatch-*')), 'Original dispatch/unknown outcome still owns production')
    jobs = read(root / 'jobs.json')
    require(not any('-segment-' in key for key in jobs), 'Original jobs already include optimization')
    for arm in p['arms']:
        for index in range(1, p['arms'][arm][1]):
            label = native_name(profile, arm, index)
            require(not (root / label).exists() and label not in jobs, 'Boundary native already admitted before adoption')
    return True


def claim_ownership(profile, metadata_cpu):
    """Caller holds original dispatch.lock; all original callbacks are terminal."""
    p = PROFILES[profile]
    control, root = p['control'], p['root']
    require(metadata_cpu['status'] == 'passed' and metadata_cpu['version'] == VERSION,
            'Metadata CPU certificate has not passed')
    for arm in p['arms']:
        actual = original_submissions(profile, arm)
        lsf_terminal(root / callback_name(profile, arm), actual['after-native']['job_id'], expected='Done')
    absence_of_production(profile)
    inventory = proof_inventory(profile)
    # Every path below is exclusive. An unknown partial outcome is never retried.
    target = control / 'adoption-snapshot'
    target.mkdir(exist_ok=False)
    for name in ('jobs.json', 'projection.json'):
        source = root / name
        destination = target / name
        with destination.open('xb') as stream:
            stream.write(source.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
    record = {'version': VERSION, 'status': 'ownership_claimed', 'profile': profile,
              'control_root': str(control), 'output_root': str(root),
              'source_manifest_sha256': p['manifest'], 'utc': datetime.now(timezone.utc).isoformat(),
              'metadata_cpu_certificate': file_record(control / 'metadata-cpu-certificate.json'),
              'snapshots': [file_record(target / x) for x in ('jobs.json', 'projection.json')],
              'immutable_original_proofs': inventory,
              'mutable_metadata_policy': 'actual added IDs only; nondecreasing native empirical projection maxima',
              'scientific_saved_results_changed': False}
    exclusive(root / 'protenix02-adoption-owner.json', record)
    exclusive(control / 'ownership.json', record)
    verify_inventory(inventory)
    return record


def verify_metadata_xml(path, count, modules):
    """Exact actual rows, raw log, attribution, zero failures/errors/skips."""
    path = Path(path)
    tree = ET.parse(path)
    rows = list(tree.iter('testcase'))
    require(type(count) is int and count > 0 and len(rows) == count, 'Metadata actual case count differs')
    require(not any(list(tree.iter(key)) for key in ('failure', 'error', 'skipped')), 'Metadata fixture failure/error/skip')
    observed = {part for case in rows for part in case.get('classname', '').split('.')}
    require(set(modules) <= observed, 'Missing metadata module attribution')
    require(re.search(r'\b' + str(count) + r'\s+passed\b', path.with_suffix('.log').read_text()) is not None,
            'Metadata raw log/XML scope differs')
    return [file_record(path), file_record(path.with_suffix('.log'))]


def verify_failed_protenix_history():
    """Preserve old641 startup failures, never synthesize its missing projection."""
    source = BASE / 'protenix-extended-release-01/source'
    root = BASE / 'protenix-extended-01'
    manifest = 'a30f980fd8fe95001fdfc6780d720bdde6486d2421d7e4942b119ace61777f00'
    records = verify_seal(source, manifest, historical_missing_revision=True)
    cpu = root / 'cpu-certificate.json'
    require(sha(cpu) == 'c0f784d523d358dfd09d7f57fcfe333fbb5c069d93074681ea35562a5f423171',
            'Old641 original CPU certificate changed')
    verify_certificate(cpu, manifest)
    require(not (root / 'projection.json').exists() and not list(root.glob('*-segment-*')),
            'Old641 failure acquired invented projections or production')
    jobs = read(root / 'jobs.json')
    for arm, native_job, callback_job in [('random', '29798159', '29798160'),
                                          ('continuation', '29798161', '29798162')]:
        for phase, job in [('native', native_job), ('after-native', callback_job)]:
            label = arm + '-' + phase + '-0000'
            parent = root / label
            submission = read(parent / 'submission.json')
            dependency = None if phase == 'native' else 'ended(' + native_job + ')'
            args = ['bsub', *resource_vector(phase), '-J', 'mosaic-pro50k-' + label,
                    '-oo', str(parent / 'stdout.%J.log'), '-eo', str(parent / 'stderr.%J.log')]
            if dependency:
                args += ['-w', dependency]
            args += ['bash', '-l', str(source / 'scripts/hpc/extended_stage.lsf'),
                     str(source), str(root), manifest, phase, str(parent)]
            require(submission['outcome'] == 'submitted' and submission['job_id'] == jobs[label] == job
                    and submission['phase'] == phase and submission['dependency'] == dependency
                    and submission['args'] == args, 'Old641 actual failed vector differs')
            terminal = lsf_terminal(parent, job, expected='Exited')
            records += [file_record(parent / 'submission.json'), terminal['proof']]
            if phase == 'native':
                error = parent / ('stderr.' + job + '.log')
                text = error.read_text()
                require("KeyError: 'git_revision_reference'" in text and 'source_identity' in text,
                        'Old641 premodel failure cause differs')
                study = parent / 'payload/study'
                require(not list(study.rglob('*')), 'Old641 premodel failure has scientific outputs')
                records.append(file_record(error))
    return {'status': 'failed_originals_preserved', 'old_projection_invented': False,
            'scientific_results_changed': False, 'proofs': records + [file_record(cpu)]}


def unknown_noisy_capsules():
    """Fail closed on additional external capsules lacking declared hard bounds.

    The known01 capsule has its existing authenticated1GiB bound. Neither local
    dirty plans nor an absent job listing can establish remote capacity scope.
    This reader performs no deletions and never imports another plan's code.
    """
    known = {'20261023-protenix-noisy-01', '20261023-protenix-noisy-01.tar.gz'}
    observed, additional = [], []
    for parent in NOISY_PARENTS:
        require(parent.is_dir() and not parent.is_symlink(), 'Noisy capsule inventory parent unavailable')
        for path in sorted(parent.glob('*protenix-noisy*')):
            require(not path.is_symlink(), 'Unknown noisy capsule symlink')
            item = {'path': str(path), 'kind': 'directory' if path.is_dir() else 'file'}
            observed.append(item)
            if path.name not in known:
                additional.append(item)
    return {'status': 'unknown_capsules_blocked' if additional else 'declared_capsules_only',
            'observed': observed, 'unknown_capsules': additional,
            'unsealed_other_plan_imported': False, 'additional_future_capacity_established': False}
