"""Syntax, sealed target files and config consistency only; no model arithmetic."""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--targets', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    args = p.parse_args()
    paths = [*Path('src').rglob('*.py'), *Path('tests').rglob('*.py'), *Path('scripts').rglob('*.py')]
    for path in paths:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    protocol = yaml.safe_load(args.protocol.read_text())
    for digest in (protocol['confidence_sha256'], protocol['targets']['sha256'],
                   protocol['base_config']['sha256']):
        assert re.fullmatch(r'[0-9a-f]{64}', digest), 'Invalid pinned SHA256 format'
    assert hashlib.sha256(args.targets.read_bytes()).hexdigest() == protocol['targets']['sha256']
    targets = json.loads(args.targets.read_text())['targets']
    assert len(targets) == 10 and len({t['sequence_sha256'] for t in targets}) == 10
    for target in targets:
        assert hashlib.sha256(target['sequence'].encode()).hexdigest() == target['sequence_sha256']
        for key in ('raw_record', 'msa'):
            asset = target[key]
            assert hashlib.sha256((args.targets.parent / asset['path']).read_bytes()).hexdigest() == asset['sha256']
    assert protocol['counts'] == list(range(6, 16))
    assert len(targets) * len(protocol['counts']) == 100
    print(json.dumps({'status': 'static_checks_passed', 'python_files_parsed': len(paths),
                      'targets': 10, 'planned_trajectories': 100,
                      'numerical_or_native_tests': 'not_run_locally'}))


if __name__ == '__main__':
    main()
