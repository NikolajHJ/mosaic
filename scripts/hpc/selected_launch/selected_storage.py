"""Explicit selected future-growth accounting; all existing bytes stay charged."""
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

from selected_contract import (BASE, CONTROL, GIB, NOISE_ROOT, NOISE_SOURCE,
                               PROFILES, SELECTED, deferred_absence, exclusive,
                               native_units, profile, require, verify_noise_source)

OLD = BASE / 'restart-lrreset-04'
OLD_FULL = {'protenix': 1493521824, 'boltz2': 1748021824, 'boltz2_chemistry': 5144021824}
OLD_CONTROLS = (BASE / 'boltz-pro02-adoption-04', BASE / 'dbmol-pro02-adoption-06', BASE/'nonchem-selected-01',BASE/'nonchem-stride20-01',BASE/'nonchem-stride20-02')
OLD_NOISE = Path('/work3/s234818/mosaic/runs/20261023-protenix-noisy-01')
RETIRED_NOISE = Path('/work3/s234818/mosaic/runs/20261024-protenix-noisy-04')


def helper():
    source = profile('protenix50k40k')['source']
    sys.path.insert(0, str(source / 'scripts'))
    return importlib.import_module('fixed25_lrreset_storage')


def budget_remaining(full, current):
    require(type(full) is int and type(current) is int and 0 <= current <= full,
            'Current selected output exceeds its full authenticated projection')
    return full - current


def external_inventory(tree):
    allowed = {str(OLD_NOISE),str(RETIRED_NOISE),str(NOISE_ROOT),str(NOISE_SOURCE.parent),str(NOISE_SOURCE.parent)+'.tar.gz'}
    rows = []
    for parent in (Path('/work3/s234818/mosaic/runs'), Path('/work3/s234818/mosaic/releases'),
                   Path('/zhome/95/e/206121/mosaic-runs')):
        require(parent.is_dir() and not parent.is_symlink(), 'External inventory parent missing/linked')
        for path in sorted(parent.glob('*protenix-noisy*')):
            require(str(path) in allowed and not path.is_symlink(), 'An unadopted noisy capsule needs separate admission')
            rows.append({'path': str(path), 'bytes': tree(path)})
    verify_noise_source()
    return rows


def storage_gate(parent, phase, *unused, **ignored):
    """New declared scope, never a retrofit pass of an older failed gate."""
    path = Path(parent) / ('selected-storage-' + phase)
    path.mkdir(exist_ok=False)
    record = {'status': 'failed', 'phase': phase, 'version': 'selected-nonchem-full-storage-v1',
              'scope': 'All selected full remaining projections; deferred chemistry gets no execution/future admission; every existing byte retained',
              'reserve_bytes': 4 * GIB, 'file_reserve': 100000,
              'full_selected_production_admission': True, 'matrix_stride':20, 'loss_stride':1,
              'historical_failed_gates_changed': False}
    try:
        tool = helper()
        tree = tool.tree_bytes
        record['deferred_arms'] = deferred_absence()
        selected = {}
        future = 0
        for key in PROFILES:
            evidence = native_units(key)
            prefix = 'pilot' if key == 'dbmol10k' else SELECTED[key]
            current = sum(tree(p) for p in profile(key)['root'].glob(prefix + '-*'))
            remaining = budget_remaining(evidence['full'], current)
            selected[key] = {'full': evidence['full'], 'current': current, 'remaining': remaining,
                             'unit_bytes': evidence['unit_bytes']}
            future += remaining
        record['selected'] = selected
        old = {key: tree(OLD / ('run-' + key)) for key in OLD_FULL}
        future += sum(budget_remaining(OLD_FULL[key], old[key]) for key in OLD_FULL)
        record['historical_remaining'] = {key: OLD_FULL[key] - old[key] for key in OLD_FULL}
        # The new control lives inside the existing1GiB control cap. Count each
        # physical tree once; its entire nested output remains bounded/charged.
        control_roots=tuple(p for p in (*OLD_CONTROLS,CONTROL) if not any(p!=q and p.is_relative_to(q) for q in (*OLD_CONTROLS,CONTROL)))
        controls = {str(p): tree(p) for p in control_roots}
        future += sum(budget_remaining(GIB, current) for current in controls.values())
        record['metadata_controls'] = controls
        # Retain the old prospective noise allowance and independently fund the new pilot.
        noise_current = {str(p): tree(p) for p in (OLD_NOISE, NOISE_ROOT)}
        future += sum(budget_remaining(GIB, current) for current in noise_current.values())
        record['noise_caps'] = noise_current
        record['retired_noise_current_bytes']=tree(RETIRED_NOISE)
        record['retired_noise_future_admission']=False
        future += 2 * GIB + 512 * 1024**2
        # The original noisy-pilot admission separately retained this existing
        # prospective scope. Keep its full allowance, independent of metadata.
        future += GIB
        record['retained_learned_prior_allowance_bytes'] = GIB
        # Separate conservative allowance for the new immutable metadata release/upload.
        # The containing analysis root is charged in full again for stale quota.
        future += GIB
        record['new_metadata_release_allowance_bytes'] = GIB
        inventory = external_inventory(tree)
        record['external_capsules'] = inventory
        raw = subprocess.run(['getquota_work3.sh'], capture_output=True, text=True, check=True, timeout=60).stdout
        stale = tool.stale_work_bytes() + sum(row['bytes'] for row in inventory)
        gate = tool.gate_record('work3', raw, future, stale, shutil.disk_usage(BASE).free,
                                tool.inode_observation(os.statvfs(BASE)))
        record.update(status=gate['status'], projected_bytes=future, work3=gate,
                      limitation='Fresh conservative observation, not a reservation; all-zero filesystem inode counters remain unreported supplemental evidence')
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        exclusive(path / 'gate.json', record)
    require(record['status'] == 'passed', 'Full selected byte/file/reserve admission failed')
    return record
