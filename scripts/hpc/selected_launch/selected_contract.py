"""Standard-library selected-scope proofs; no scientific arithmetic or imports."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import xml.etree.ElementTree as ET

import legacy_contract as legacy

BASE = legacy.BASE
GIB = legacy.GIB
VERSION = 'independent-nonchemistry-stride20-admission-v2'
CONTROL = BASE / 'nonchem-stride20-02/startup-fix-03'
LAUNCHER = BASE / 'nonchem-stride20-startup-release-03/source'
PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
PROFILES = ('boltz30k', 'dbmol10k', 'protenix50k40k')
SELECTED = {'boltz30k': 'boltz2', 'dbmol10k': 'boltz2_dbmol', 'protenix50k40k': 'continuation'}
COUNTS = {'boltz30k': 3, 'dbmol10k': 4, 'protenix50k40k': 4}
CPU_PINS = {'boltz30k': '06efbb027f1008a0921494af0e226c3fc7a78c0a8f4ab01b93e8b50f2c042836',
            'dbmol10k': '387070e8fe56b17ddbd7c8e0b82d2d6a44d9aa99e7015d3fbe4757ce82c0819b',
            'protenix50k40k': 'ab3148ea6dd8d25361f85e9d8283578e0cf4429844d820a4dab6b03673ef577b'}
NOISE_SOURCE = Path('/work3/s234818/mosaic/releases/20261023-protenix-noisy-01/source')
NOISE_MANIFEST = '7bb970f48f6f907b7dc05e74a9bc2645a313addf814f08a0eeca11582f3b84ef'
NOISE_ROOT = Path('/work3/s234818/mosaic/runs/20261025-protenix-noisy-05')
NOISE_STUDY = 'experiments/brd4-jq1/protenix-noisy-1000'
require, read, sha, exclusive, atomic, file_record = (
    legacy.require, legacy.read, legacy.sha, legacy.exclusive, legacy.atomic, legacy.file_record)


def profile(name):
    require(name in PROFILES, 'Only the three declared non-chemistry chains may run')
    return legacy.PROFILES[name]


def production_entries(root):
    """Separate real phase markers from authenticated regular startup JSON files."""
    rows=[]
    phase=r'(?:boltz2|boltz2_chemistry|pilot|continuation|random)-(?:after-)?segment-\d{4}(?:-certificate\.json)?'
    metadata=r'(?:startup-(?:command-)?(?:(?:boltz2|boltz2_chemistry|pilot|continuation|random)-)?segment-\d{4}(?:-certificate)?(?:-started)?|(?:boltz2|pilot|continuation)-stride20-v3-(?:command-)?segment-\d{4}(?:-certificate|-startup)?(?:-started)?)\.json'
    for path in Path(root).iterdir():
        if re.fullmatch(phase,path.name):rows.append(path)
        elif '-segment-' in path.name:
            require(re.fullmatch(metadata,path.name) is not None and path.is_file() and not path.is_symlink(),
                    'Unknown or nonregular segment-related marker')
    return rows


def arm_label(name, phase, index):
    require(name in SELECTED and phase in ('native', 'after-native', 'segment', 'after-segment')
            and type(index) is int and 0 <= index < COUNTS[name], 'Undeclared selected phase/boundary')
    require(index > 0 or phase not in ('native', 'after-native'), 'Never repeat original native0')
    arm = 'pilot' if name == 'dbmol10k' else SELECTED[name]
    return arm + '-' + phase + '-' + format(index, '04d')


def decode(name, label):
    match = re.fullmatch(r'([a-z0-9_]+)-(native|after-native|segment|after-segment)-(\d{4})', label)
    require(match is not None, 'Unknown scientific stage label')
    phase, index = match.group(2), int(match.group(3))
    require(label == arm_label(name, phase, index), 'Deferred or wrong arm cannot execute')
    return phase, index


def verify_own(manifest):
    return legacy.verify_seal(LAUNCHER, manifest)


def verify_source(name):
    p = profile(name)
    proofs = legacy.verify_seal(p['source'], p['manifest'])
    if name == 'dbmol10k':
        proofs += legacy.verify_seal(p['original_launcher'], p['original_launcher_manifest'], metadata_only=True)
    cpu = p['root'] / 'cpu-certificate.json'
    require(sha(cpu) == CPU_PINS[name], 'Original scientific CPU certificate changed')
    legacy.verify_certificate(cpu, p['manifest'], p['cpu_cases'])
    cfg = read(legacy.profile_config(name, SELECTED[name]))
    if name == 'dbmol10k':
        require(cfg['steps'] == 10000, 'DBMol requested budget differs')
    else:
        require(cfg['chemistry_enabled'] is False and cfg['element_js_weight'] == 0,
                'Selected chain must retain its original native-only objective')
    return proofs + [file_record(cpu), file_record(legacy.profile_config(name, SELECTED[name]))]


def native_units(name):
    """All authentic maxima for the selected arm; never certify deferred science."""
    p = profile(name)
    arm = SELECTED[name]
    proofs = verify_source(name)
    actual = legacy.original_submissions(name, arm)
    legacy.lsf_terminal(p['root'] / legacy.callback_name(name, arm), actual['after-native']['job_id'], expected='Done')
    paths = sorted(p['root'].glob(legacy.native_name(name, arm)[:-4] + '*-certificate.json'))
    require(paths, 'Selected native certificate absent')
    observed = []
    for cp in paths:
        cert = legacy.verify_certificate(cp, p['manifest'])
        label = cp.name.removesuffix('-certificate.json')
        parent = p['root'] / label
        ready_path = parent / 'payload/study/assessment.json'
        ready = read(ready_path)
        require(any(v['path'] == str(ready_path) and v['sha256'] == sha(ready_path) for v in cert['proofs']),
                'Native assessment is not certificate-bound')
        require(ready['status'] == 'passed' and ready['source_manifest_sha256'] == p['manifest']
                and ready['config_sha256'] == sha(legacy.profile_config(name, arm))
                and ready['arm'] == ('protenix' if name == 'protenix50k40k' else arm),
                'Selected native source/config/arm differs')
        producer = read(parent / 'submission.json')
        require(producer['outcome'] == 'submitted' and read(p['root'] / 'jobs.json')[label] == producer['job_id'],
                'Actual selected producer ID differs')
        lsf = legacy.lsf_terminal(parent, producer['job_id'], expected='Done')
        require(any(v['path'] == lsf['proof']['path'] and v['sha256'] == lsf['proof']['sha256'] for v in cert['proofs']),
                'Original native same-job LSF proof missing')
        unit = ready['storage_projection']['maximum_pilot_update_bytes']
        require(type(unit) is int and unit > 0 and ready['runtime_gate']['status'] == 'passed'
                and ready['runtime_gate']['conservative_projected_seconds'] < 82800,
                'Selected native runtime/projection failed')
        observed.append(unit)
        proofs += [file_record(cp), file_record(ready_path)]
    live = read(p['root'] / 'projection.json')
    unit = live['unit_bytes'] if name == 'dbmol10k' else live[arm]['unit_bytes']
    require(unit == max(observed), 'Selected empirical maximum was reduced or invented')
    total = 10000 if name == 'dbmol10k' else (30000 if name == 'boltz30k' else 40000)
    dense = 2 * unit * total + COUNTS[name] * GIB
    if name == 'dbmol10k':
        require(live['full_projected_bytes'] == dense and live['status'] == 'passed', 'DBMol original capacity differs')
    from retention import full_projection,policy
    full=full_projection(unit,total,COUNTS[name])
    return {'status': 'passed', 'arm': arm, 'unit_bytes': unit, 'full': full,
            'original_dense_full':dense,'artifact_retention':policy(),'proofs': proofs}


def previous_admission():
    """Authenticate the failed launcher without rewriting its role/status records."""
    old_manifest='e63f830f4f01f0740151fdd0b2baeb125d0faf9a927b5cd2e57b7400fea9d4c4'
    legacy.verify_seal(BASE/'nonchem-selected-release-01/source',old_manifest)
    old_cpu=read(BASE/'nonchem-selected-01/cpu-certificate.json')
    require(old_cpu['status']=='passed' and old_cpu['cases']==42 and old_cpu['launcher_manifest_sha256']==old_manifest,'Original metadata42 outcome differs')
    legacy.verify_proofs(old_cpu)
    failed=BASE/'nonchem-stride20-01'
    legacy.verify_seal(BASE/'nonchem-stride20-release-01/source','af73bea37be4620d1515052a805b43db0a892aa74cd2e783708cc2d66ce07502')
    legacy.lsf_terminal(failed/'cpu','29809167',expected='Exited')
    legacy.lsf_terminal(failed/'after-cpu','29809168',expected='Exited')
    tests=list(ET.parse(failed/'cpu/payload/selected.xml').iter('testcase'))
    failures=[case for case in tests if case.find('failure') is not None]
    require(len(tests)==66 and len(failures)==3 and all('test_actual_sealed_trajectory_dense_sparse_exact_parity' in x.get('name','') for x in failures)
            and not any(case.find(k) is not None for case in tests for k in ['error','skipped']),'Prior66-case fixture failure differs')
    require(not (failed/'cpu-certificate.json').exists(),'Failed fixture study cannot be relabeled passed')
    passing=BASE/'nonchem-stride20-02'
    legacy.verify_seal(BASE/'nonchem-stride20-release-02/source','f3be7c2f85d495ae042ee8092eecb5274c6298cc8f12907a7ba9ba6d8fb740db')
    cert=read(passing/'cpu-certificate.json')
    require(cert['status']=='passed' and cert['cases']==67 and cert['launcher_manifest_sha256']=='f3be7c2f85d495ae042ee8092eecb5274c6298cc8f12907a7ba9ba6d8fb740db','Prior real67 pass differs')
    legacy.verify_proofs(cert)
    for name,job in [('boltz30k','29809221'),('dbmol10k','29809222'),('protenix50k40k','29809223')]:
        p=profile(name)['root']/'dispatch-0003';s=read(p/'submission.json')
        require(s['job_id']==job and s['launcher_manifest_sha256']=='f3be7c2f85d495ae042ee8092eecb5274c6298cc8f12907a7ba9ba6d8fb740db','Prior startup owner differs')
        legacy.lsf_terminal(p,job,expected='Exited')
        require('retention.py' in (p/('stderr.'+job+'.log')).read_text(),'Prior startup wrapper failure proof missing')
    noise=read(passing/'noise-gpu/submission.json')
    require(noise['job_id']=='29809228' and noise['outcome']=='submitted' and noise['launcher_manifest_sha256']=='f3be7c2f85d495ae042ee8092eecb5274c6298cc8f12907a7ba9ba6d8fb740db','Existing independent noisy job differs')
    for name,job in [('boltz30k','29809135'),('dbmol10k','29809136'),('protenix50k40k','29809137')]:
        root=profile(name)['root'];p=root/'dispatch-0002';sub=read(p/'submission.json')
        require(sub['job_id']==job and sub['outcome']=='submitted' and sub['phase']=='dispatch' and sub['launcher_manifest_sha256']==old_manifest,'Failed prior dispatch provenance differs')
        legacy.lsf_terminal(p,job,expected='Exited')
        require('ValueError: Proof certificate did not pass' in (p/('stderr.'+job+'.log')).read_text(),'Prior failure cause differs')
        owner=read(root/'nonchem-selected-owner.json')
        require(owner['status']=='selected_owner' and owner['launcher_manifest_sha256']==old_manifest and owner['profile']==name,'Prior role metadata differs')
        for row in owner['proofs']:require(sha(row['path'])==row['sha256'],'Prior ownership proof changed')
        require(not production_entries(root),'Prior scientific/unknown production exists')
    path=BASE/'nonchem-stride20-retirement-01/assessment.json'
    expected='c75c65397c2b36378c8127be7da3d2dde803fa8d38ab51a594e2b1d3839896ab'
    require(sha(path)==expected,'Exact unstarted noisy-job retirement differs')
    value=read(path)
    require(value['status']=='intentionally_retired_before_scientific_start' and value['job_id']=='29809146' and value['scientific_updates']==0,'Noisy predecessor was not retired before science')
    for row in value['proofs']:require(sha(row['path'])==row['sha256'],'Retirement proof changed')
    return file_record(path)


def deferred_absence():
    """No production/unknown marker is allowed in the two explicitly deferred arms."""
    rows = []
    for name, arm in [('boltz30k', 'boltz2_chemistry'), ('protenix50k40k', 'random')]:
        root = profile(name)['root']
        jobs = read(root / 'jobs.json')
        require(not list(root.glob(arm + '-segment-*')) and not list(root.glob(arm + '-after-segment-*'))
                and not (root / (arm + '-completion.json')).exists()
                and not any(key.startswith(arm + '-segment-') for key in jobs),
                'Deferred production already exists; no silent budget exclusion')
        callback = root / (arm + '-after-native-0000')
        sub = read(callback / 'submission.json')
        expected = 'Done' if arm == 'boltz2_chemistry' else 'Exited'
        lsf = legacy.lsf_terminal(callback, sub['job_id'], expected=expected)
        rows.append({'profile': name, 'arm': arm, 'future_admission': False,
                     'production_absent': True, 'terminal_callback': lsf})
    # Earlier observers must be terminal failures and must never be replayed.
    for root, job in [(BASE / 'boltz-pro02-adoption-04', '29806694'),
                      (BASE / 'dbmol-pro02-adoption-06', '29806701')]:
        parent = root / 'observe-0001'
        sub = read(parent / 'submission.json')
        require(sub['job_id'] == job and read(parent / 'failure.json')['status'] == 'failed_peer',
                'Previous failed observer provenance differs')
        legacy.lsf_terminal(parent, job, expected='Exited')
    return rows


def verify_noise_source():
    require(sha(NOISE_SOURCE / 'source_manifest.json') == NOISE_MANIFEST, 'Original noise source changed')
    value = read(NOISE_SOURCE / 'source_manifest.json')
    files = value['files']
    actual = {p.relative_to(NOISE_SOURCE).as_posix() for p in NOISE_SOURCE.rglob('*') if p.is_file() or p.is_symlink()}
    require(actual == set(files) | {'source_manifest.json'}, 'Noise source member scope differs')
    for path in [NOISE_SOURCE, *NOISE_SOURCE.rglob('*')]:
        require(not path.is_symlink() and (path.is_dir() or path.is_file()) and not path.stat().st_mode & 0o222,
                'Noise source is not immutable regular content')
    for rel, digest in files.items():
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts
                and sha(NOISE_SOURCE / rel) == digest, 'Noise source member changed')
    config = NOISE_SOURCE / NOISE_STUDY / 'runs/20261023-protenix-noisy-01/config.json'
    cfg = read(config)
    require(cfg['steps'] == 1000 and cfg['learning_rate'] == .3 and cfg['objective'] == 'native_only_pocket_contact',
            'Original non-chemistry noise settings changed')
    return file_record(config)


def verify_xml(path, module_names=None):
    rows = list(ET.parse(path).iter('testcase'))
    require(rows and not any(case.find(k) is not None for case in rows for k in ('failure', 'error', 'skipped')),
            'Allocated CPU fixtures failed/errored/skipped')
    modules = {part for case in rows for part in case.get('classname', '').split('.')}
    require(module_names is None or set(module_names) <= modules, 'Allocated CPU module attribution missing')
    require(re.search(r'\b' + str(len(rows)) + r'\s+passed\b', Path(path).with_suffix('.log').read_text()),
            'Raw CPU XML/log count differs')
    return len(rows)
