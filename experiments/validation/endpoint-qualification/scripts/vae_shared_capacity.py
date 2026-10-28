"""Read-only authentication/forecast for the exact concurrent VAE campaign.

No VAE module is imported, tensor loaded, setting changed or job controlled.
The empirical whole-campaign forecast is held until Mosaic work ends. It is
checked at every entry/periodic admission and is NOT a new VAE hard output cap.
"""
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import stat
import time

TAG = 'source140-epoch40-to45-continuation-attempt002'
ROOT = PurePosixPath('/work3/s234818/latentmol/runs') / TAG
SOURCE = PurePosixPath('/work3/s234818/latentmol/releases/source-140-epoch40-to45-continuation-attempt002')
PLAN = PurePosixPath('/work3/s234818/latentmol/submissions') / TAG / (TAG + '.json')
PLAN_SHA = 'a794d3f7a9b3bcbcdc357aadb6feade06806a4692b08a65e7c3a0e8e0a9420f6'
MANIFEST_SHA = 'ac6b06edeff9ce6d35413d84890ee3738a68f517230ac06bd6345f6f86ca4d65'
ARCHIVE_SHA = '375272634d1836625e3d9a2bb8548e93f4cf4ac982ad1853fd0c621af8a344ef'
VERSION = 'mosaic-authenticated-vae-shared-quota-v1'
GIB = 1024**3


def require(value, message):
    if not value: raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def binding(path):
    path = Path(path); info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not path.is_symlink()
            and all(not p.is_symlink() for p in path.parents), 'VAE metadata linked/special')
    with path.open('rb') as f: digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': info.st_size, 'sha256': digest}


def validate_profile(profile):
    require(type(profile) is dict and set(profile) == {'version','plan','archive','manifest','quota_before',
            'baseline_bytes','baseline_files','growth_bytes','growth_files','job_templates'}, 'Shared VAE profile fields differ')
    require(profile['version'] == VERSION and profile['plan']['path'] == str(PLAN)
            and profile['plan']['sha256'] == PLAN_SHA
            and profile['archive']['path'] == str(SOURCE) + '.tar.gz'
            and profile['archive']['sha256'] == ARCHIVE_SHA
            and profile['manifest']['path'] == str(SOURCE / 'release_manifest.json')
            and profile['quota_before']['path'] == str(ROOT / 'quota-before-001.json'), 'Shared VAE identities differ')
    for key in ('baseline_bytes','baseline_files','growth_bytes','growth_files'):
        require(type(profile[key]) is int and profile[key] >= 0, 'Shared VAE forecast must be finite integer')
    require(profile['growth_bytes'] == 10 * GIB and profile['growth_files'] == 30000,
            'Original VAE10GiB headroom/declared metadata allowance changed')
    templates = profile['job_templates']
    require(type(templates) is dict and set(templates) == {'gpu','callback'}
            and all(type(v) is str and v for v in templates.values()), 'Exact VAE command templates missing')
    require('--plan-sha256 ' + PLAN_SHA in templates['gpu'] and '--plan-sha256 ' + PLAN_SHA in templates['callback']
            and ARCHIVE_SHA in templates['gpu'] and MANIFEST_SHA in templates['gpu'], 'VAE template lacks release/plan pins')
    return profile


def authenticate(profile):
    validate_profile(profile)
    for key in ('plan','archive','manifest','quota_before'):
        require(binding(profile[key]['path']) == profile[key], 'Authenticated VAE metadata changed: ' + key)
    plan = read(PLAN); manifest = read(SOURCE / 'release_manifest.json')
    require(plan['campaign'] == str(ROOT) and plan['release'] == str(SOURCE)
            and plan['epoch45_extension']['start_epoch'] == 40 and plan['epoch45_extension']['end_epoch'] == 45
            and plan['epoch45_extension']['save_every_epoch'] is True, 'VAE authorized endpoint changed')
    require(manifest['sha256'] == MANIFEST_SHA
            and hashlib.sha256(json.dumps(manifest['files'],sort_keys=True).encode()).hexdigest() == MANIFEST_SHA,
            'VAE source manifest digest differs')
    for name, digest in manifest['files'].items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and '..' not in relative.parts and str(relative) == name,
                'VAE source member path unsafe')
        require(binding(SOURCE / name)['sha256'] == digest, 'VAE executable source member changed: ' + name)
    quota = read(ROOT / 'quota-before-001.json')
    require(quota['minimum_gib'] == 10, 'Original VAE required quota headroom differs')
    source = ast.parse(Path(SOURCE / 'scripts/hpc/chemistry_campaign.py').read_bytes())
    wanted = ast.dump(ast.parse('max(10., 2. + (100 - epoch) / 16 + (100 - self.plan["parent_epoch"]) / 16)',mode='eval').body)
    require(sum(isinstance(n,ast.Call) and ast.dump(n) == wanted for n in ast.walk(source)) == 1,
            'Original VAE quota computation changed')
    return {'source_files':len(manifest['files']), 'source_sha256':MANIFEST_SHA,
            'original_minimum_free_gib':10,'VAE_jobs_or_settings_changed':False}


def matches(command, template, kind):
    """Exact argv family, permitting only serial index and authenticated resume."""
    actual, expected = shlex.split(command), shlex.split(template)
    if len(actual) != len(expected): return False
    indices = [i + 1 for i, token in enumerate(expected[:-1]) if token == '--segment']
    number = actual[indices[0]] if indices else None
    if kind == 'callback':
        i = expected.index('--after-stage') + 1
        stage = actual[i]
        if not stage.startswith('segment-') or len(stage) != len('segment-001'): return False
        number = stage.split('-')[-1]
    if number is None or not number.isdigit() or not 1 <= int(number) <= 80: return False
    for i, token in enumerate(expected):
        if i in indices:
            if actual[i] != str(int(number)): return False
        elif token == 'source104-chemistry-segment-001':
            if actual[i] != 'source104-chemistry-segment-' + str(int(number)).zfill(3): return False
        elif kind == 'callback' and token == 'segment-001':
            if actual[i] != 'segment-' + str(int(number)).zfill(3): return False
        elif i and expected[i-1] == '--resume':
            if int(number) == 1:
                if actual[i] != token: return False
            elif (not actual[i].startswith(str(ROOT / 'segments') + '/')
                  or '..' in PurePosixPath(actual[i]).parts
                  or str(PurePosixPath(actual[i])) != actual[i]):
                return False
        elif i and expected[i-1] == '--resume-sha256':
            import re
            if not re.fullmatch('[0-9a-f]{64}', actual[i]): return False
            if int(number) == 1 and actual[i] != token: return False
        elif actual[i] != token: return False
    return True


def inventory(root, timeout=180):
    root = Path(root); deadline = time.monotonic() + timeout
    pending, rows = [root], []
    while pending:
        require(time.monotonic() < deadline, 'Complete VAE inventory timeout')
        directory = pending.pop()
        require(directory.is_dir() and not directory.is_symlink()
                and all(not p.is_symlink() for p in directory.parents), 'VAE output root linked/special')
        with os.scandir(directory) as entries:
            for entry in entries:
                p = Path(entry.path); info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode): pending.append(p)
                elif stat.S_ISREG(info.st_mode): rows.append({'path':str(p.relative_to(root)), 'bytes':info.st_size})
                else: raise ValueError('VAE inventory linked/special: ' + str(p))
    return {'root':str(root),'complete':True,'nofollow':True,'actual_bytes':sum(r['bytes'] for r in rows),
            'actual_files':len(rows),'rows':sorted(rows,key=lambda r:r['path'])}


def observe(profile, rows, on_inventory_failure=None):
    auth = authenticate(profile)
    accounted = []
    for row in rows:
        if any(matches(row['COMMAND'], template, kind) for kind,template in profile['job_templates'].items()):
            require(row['QUEUE'] == ('gpua100' if matches(row['COMMAND'],profile['job_templates']['gpu'],'gpu') else 'hpc'),
                    'VAE actual queue differs')
            accounted.append(row['JOBID'])
    deadline = time.monotonic() + 180
    for number in range(1,4):
        try:
            current = inventory(ROOT, timeout=max(0,deadline-time.monotonic()))
            break
        except FileNotFoundError as error:
            missing = Path(error.filename) if error.filename else None
            allowed = (missing is not None and missing.is_absolute() and ROOT in missing.parents
                       and missing.name.endswith(('.tmp','.partial')))
            record = {'status':'failed_incomplete_inventory_not_admission','attempt':number,'path':error.filename,
                      'recognized_owned_atomic_temporary':allowed,'partial_totals_used':False,'files_skipped':False}
            if on_inventory_failure is not None: on_inventory_failure(number, record)
            if not allowed or number == 3: raise
    cap = profile['baseline_bytes'] + profile['growth_bytes']
    cap_files = profile['baseline_files'] + profile['growth_files']
    require(current['actual_bytes'] <= cap and current['actual_files'] <= cap_files,
            'VAE output exceeded admitted empirical forecast; stop own Mosaic work')
    return {'version':VERSION,'authentication':auth,'accounted_job_ids':accounted,'inventory':current,
            'cap_bytes':cap,'cap_files':cap_files,
            'remaining_bytes':cap-current['actual_bytes'],'remaining_files':cap_files-current['actual_files'],
            'forecast_held_until_Mosaic_work_ends':True,'VAE_hard_output_cap_added':False,
            'whole_account_reservation':False,'VAE_jobs_or_settings_changed':False}


def augment(result, observation):
    """Current stale charge + remaining allowance equals entire forecast once."""
    value = dict(result)
    current = observation['inventory']
    value['available_bytes'] -= current['actual_bytes']
    value['available_files'] -= current['actual_files']
    value['required_bytes'] += observation['remaining_bytes']
    value['required_files'] += observation['remaining_files']
    value['byte_margin'] = value['available_bytes'] - value['required_bytes']
    value['file_margin'] = value['available_files'] - value['required_files']
    value['status'] = 'passed' if value['byte_margin'] >= 0 and value['file_margin'] >= 0 else 'failed'
    value['shared_vae_accounting'] = observation
    return value
