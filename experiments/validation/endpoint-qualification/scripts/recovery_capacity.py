"""Complete inherited futures plus new Protenix/DBMol recovery allowances.

Metadata only. No deletion, array evaluation, scheduler mutation or reduced
forecast/reserve. The authenticated qualification04 guard retains all prior
campaigns and the two already queued chemistry validation futures.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

BASE = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
SOURCE = BASE / 'chemistry-native-qualification-release-04/source'
PIN = '5d8f4ef0f8bb820e76d2e92442492f604393e79b58725fd01f9bb042b3114863'
EXTRA = (272 + 512) * 1024**2
CHARGED = ('protenix-endpoint-fix-07', 'protenix-endpoint-fix-release-07',
           'protenix-endpoint-fix-08', 'protenix-endpoint-fix-release-08',
           'dbmol-endpoint-recovery-10', 'dbmol-endpoint-recovery-release-10')


def require(value, message):
    if not value:
        raise ValueError(message)


def validate(prior, charged, excess=0):
    require(prior.get('status') == 'passed' and prior.get('original_gate_changed') is False
            and prior.get('reserve_bytes') == 4 * 1024**3 and prior.get('file_reserve') == 100000,
            'Passed whole inherited futures and unchanged reserves required')
    require(all(type(v) is int and v >= 0 for v in (charged, excess))
            and all(type(prior.get(k)) is int for k in ('required_bytes', 'available_bytes')),
            'Finite complete byte accounting required')
    result = {'status': 'failed', 'inherited_gate': prior,
              'required_bytes': prior['required_bytes'] + EXTRA + excess,
              'available_bytes': prior['available_bytes'] - charged,
              'additional_full_recovery_allowance_bytes': EXTRA,
              'entire_recovery_stale_charge_bytes': charged,
              'reserve_bytes': 4 * 1024**3, 'file_reserve': 100000,
              'original_gate_changed': False, 'forecasts_removed': False}
    require(result['available_bytes'] >= result['required_bytes'], 'Full recovery futures do not fit')
    result['status'] = 'passed'
    return result


def gate(parent, phase, excess=0):
    raw = (SOURCE / 'source_manifest.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PIN, 'Recovery storage parent pin differs')
    manifest = json.loads(raw)
    require(len(manifest['files']) == 12, 'Recovery storage parent scope differs')
    for name, sha in manifest['files'].items():
        p = SOURCE / name
        require(not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest() == sha,
                'Recovery storage parent changed')
    spec = importlib.util.spec_from_file_location('_recovery_storage_parent', SOURCE / 'scripts/chemistry_qualification_ops.py')
    op = importlib.util.module_from_spec(spec); sys.modules[spec.name] = op; spec.loader.exec_module(op)
    inherited = op.inherited()
    prior = inherited.storage(Path(parent), phase)
    charged = 0
    for name in CHARGED:
        root = BASE / name
        if not root.exists():
            continue
        require(root.is_dir() and not root.is_symlink(), 'Recovery accounting root linked/special')
        for directory, dirs, files in os.walk(root, followlinks=False):
            for name2 in dirs:
                p = Path(directory) / name2
                require(not p.is_symlink() and p.is_dir(), 'Recovery accounting directory linked/special')
            for name2 in files:
                p = Path(directory) / name2; info = p.lstat()
                require(stat.S_ISREG(info.st_mode), 'Recovery accounting file linked/special')
                charged += info.st_size
    result = validate(prior, charged, excess)
    with (Path(parent) / ('recovery-capacity-' + phase + '.json')).open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    return result
