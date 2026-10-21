"""Finite work3 quota and inode gates for the independent LR0.3 experiment."""

import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import json
from pathlib import Path
import re
import shutil
import subprocess


RUN_ROOT = Path('/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-protenix-lr03-01')
RESERVE = 4 * 1024**3
FILE_RESERVE = 100000


def parse_work3(text):
    finite = []
    for line in text.splitlines():
        fields = [x.strip() for x in re.split(r'\|+', line) if x.strip()]
        if not fields or fields[0] != 's234818':
            continue
        if len(fields) != 6 or fields[1] != '306121':
            raise ValueError('Unexpected work3 account/quota schema')
        if fields[3] == 'unlimited':
            continue
        values = []
        for raw, rounding in ((fields[2], ROUND_CEILING), (fields[3], ROUND_FLOOR)):
            match = re.fullmatch(r'(\d+(?:\.\d+)?)\s+(Byte|KiB|MiB|GiB|TiB)', raw)
            if not match:
                raise ValueError('Unrecognized finite work3 quota units')
            factor = 1024 ** ('Byte', 'KiB', 'MiB', 'GiB', 'TiB').index(match[2])
            value = Decimal(match[1]) * factor / 1024**3
            values.append(int(value.to_integral_value(rounding=rounding)) * 1024**3)
        used, hard = values
        files, file_hard = int(fields[4]), int(fields[5])
        if not 0 <= used < hard or not 0 <= files < file_hard:
            raise ValueError('Invalid/exhausted finite work3 quota')
        finite.append({'used_bytes_upper': used, 'hard_bytes_lower': hard,
                       'remaining_bytes': hard - used, 'remaining_files': file_hard - files})
    if not finite:
        raise ValueError('No authenticated finite work3 quota; unlimited rows alone are insufficient')
    return {'finite_pools': finite, 'available_bytes_lower': min(x['remaining_bytes'] for x in finite),
            'available_files_lower': min(x['remaining_files'] for x in finite)}


def check_storage(run_dir, projected_bytes):
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_relative_to(RUN_ROOT) or type(projected_bytes) is not int or projected_bytes < 0:
        raise ValueError('Wrong independent experiment storage scope/projection')
    run_dir.mkdir(parents=True, exist_ok=True)
    command = subprocess.run(['getquota_work3.sh'], capture_output=True, text=True, check=True, timeout=60)
    quota = parse_work3(command.stdout)
    # Subtract the complete experiment again to cover stale quota reporting.
    current = sum(p.stat().st_size for p in RUN_ROOT.rglob('*') if p.is_file())
    available = min(quota['available_bytes_lower'], shutil.disk_usage(run_dir).free) - current
    required = projected_bytes + RESERVE
    record = {'status': 'passed' if available >= required and quota['available_files_lower'] >= FILE_RESERVE else 'failed',
              'utc': datetime.now(timezone.utc).isoformat(), 'run_dir': str(run_dir),
              'projected_bytes': projected_bytes, 'reserve_bytes': RESERVE,
              'required_bytes': required, 'available_bytes_conservative': available,
              'experiment_bytes_subtracted': current, 'file_reserve': FILE_RESERVE,
              'quota': quota, 'raw_quota': command.stdout,
              'scope': 'Separate work3 artifacts; no new outputs consume the live Boltz home budget',
              'limitation': 'Conservative rounded finite quota, all finite pools; fresh checks are not reservations'}
    (run_dir / 'work3-storage-gate.json').write_text(json.dumps(record, indent=2) + '\n')
    if record['status'] != 'passed':
        raise ValueError('Independent run exceeds fresh work3 byte/inode budget')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--projected-bytes', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(check_storage(args.run_dir, args.projected_bytes)))
