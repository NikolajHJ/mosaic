"""Local stdlib-only package preparation; never stages or submits anything."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import tarfile


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def declaration(path, name):
    tree = ast.parse(Path(path).read_bytes())
    matches = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            matches.append(ast.literal_eval(node.value))
    if len(matches) != 1:
        raise ValueError('Missing unique static declaration ' + name)
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--profile', choices=('boltz30k', 'dbmol10k'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    shared = repository / 'scripts/hpc/peer_adoption'
    consumer = repository / ('experiments/brd4-jq1/boltz-lr0025/runs/20261024-boltz-pro02-adoption-04'
                             if args.profile == 'boltz30k' else
                             'experiments/brd4-jq1/boltz-dbmol/runs/20261024-dbmol-pro02-adoption-06')
    fixture = repository / 'experiments/brd4-jq1/boltz-lr0025/runs/20261024-boltz-pro02-adoption-04/fixture_adoption.py'
    count = declaration(shared / 'adoption_ops.py', 'METADATA_CASES')
    if type(count) is not int or count <= 0 or declaration(fixture, 'METADATA_CASES') != count:
        raise ValueError('Exact reviewed positive allocated fixture count not installed')
    contract = (shared / 'adoption_contract.py').read_text()
    if "'manifest': None" in contract:
        raise ValueError('A future peer manifest pin is not established')
    if args.output.exists():
        raise FileExistsError('Unknown/existing package output; inspect rather than overwrite')
    args.output.mkdir(parents=True, exist_ok=False)
    aliases = {
        'README.md': consumer / 'README.md',
        'adoption-plan.json': consumer / 'adoption-plan.json',
        'scripts/adoption_contract.py': shared / 'adoption_contract.py',
        'scripts/adoption_ops.py': shared / 'adoption_ops.py',
        'scripts/hpc/peer_adoption_cpu.lsf': shared / 'cpu.lsf',
        'scripts/hpc/peer_adoption_stage.lsf': shared / 'stage.lsf',
        'tests/test_peer_adoption.py': fixture,
    }
    source = args.output / 'source'
    source.mkdir()
    files, provenance = {}, {}
    for name, original in aliases.items():
        raw = original.read_bytes()
        if original.suffix == '.py':
            ast.parse(raw)
            compile(raw, str(original), 'exec')
        elif original.suffix == '.json':
            json.loads(raw)
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw)
        files[name] = digest(target)
        provenance[name] = {'path': str(original), 'bytes': len(raw), 'sha256': files[name]}
    revision = 'c74c659344123937608077e3e4078f75f09cd743'
    manifest = {'version': 'mosaic-provenance-peer-adoption-launcher-v1',
                'profile': args.profile, 'git_revision_reference': revision,
                'metadata_cpu_cases': count, 'files': files,
                'scientific_source_unchanged': True,
                'metadata_cpu_is_scientific_readiness': False}
    path = source / 'source_manifest.json'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    archive = args.output / 'source.tar.gz'
    with tarfile.open(archive, 'x:gz') as stream:
        for name in sorted([*files, 'source_manifest.json']):
            stream.add(source / name, arcname=name, recursive=False)
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        if len(members) != len(files) + 1 or not all(x.isfile() and not x.issym() and not x.islnk() for x in members):
            raise ValueError('Prepared archive scope differs')
        for member in members:
            raw = stream.extractfile(member).read()
            expected = digest(path) if member.name == 'source_manifest.json' else files[member.name]
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError('Prepared archive member changed')
    receipt = {'status': 'prepared_local_only', 'profile': args.profile,
               'source_file_count': len(files), 'manifest_sha256': digest(path),
               'archive': {'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest(archive)},
               'canonical_members': provenance, 'metadata_cpu_cases_planned': count,
               'numerical_execution': False, 'allocated_cpu_passed': False,
               'staged': False, 'submitted': False}
    with (args.output / 'preparation.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    # Local package artifacts are immutable; canonical working sources stay editable.
    for entry in source.rglob('*'):
        entry.chmod(0o555 if entry.is_dir() else 0o444)
    source.chmod(0o555)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
