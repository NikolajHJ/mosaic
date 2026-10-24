"""Exact641 fork with authenticated625 provenance; local stdlib packaging only."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
PARENT_ARCHIVE = REPO / 'results/hpc/20261023-protenix-extended-01/source02/source.tar.gz'
PARENT_ARCHIVE_SHA = '8ba1c3370fee1f09b96af139dbf1403bb3c187c52d447ec4b0b23ebe4a65f63d'
PARENT_MANIFEST = 'a30f980fd8fe95001fdfc6780d720bdde6486d2421d7e4942b119ace61777f00'
ANCESTOR_ARCHIVE = REPO / 'results/hpc/20261022-brd4-jq1-25-fixed-lrreset-04/source02/source.tar.gz'
ANCESTOR_ARCHIVE_SHA = '4d6d297451d9820543abc3298622e6599014c4b96bc999e104dd732825f56e4e'
ANCESTOR_MANIFEST = 'c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057'
REVISION = 'c74c659344123937608077e3e4078f75f09cd743'
ALIASES = {
    'extended_ops.py': 'scripts/extended_ops.py',
    'extended_protocol.py': 'scripts/extended_protocol.py',
    'extended_driver.py': 'examples/brd4_25_atoms.py',
    'extended_contract.py': 'scripts/campaign25_contract.py',
    'extended_cpu.lsf': 'scripts/hpc/extended_cpu.lsf',
    'extended_stage.lsf': 'scripts/hpc/extended_stage.lsf',
    'fixture_extended.py': 'tests/test_extended_protenix.py',
    'random.json': 'examples/configs/extended_random.json',
    'continuation.json': 'examples/configs/extended_continuation.json',
    'parent.json': 'docs/extended_parent.json',
    'predecessor-retirement.json': 'docs/extended_retirement.json',
    'README.md': 'docs/protenix_extended.md',
    'extended_startup.py': 'scripts/extended_startup.py',
    'fixture_extended_startup.py': 'tests/test_extended_startup.py',
}
MUTABLE = {'scripts/extended_ops.py', 'scripts/hpc/extended_cpu.lsf',
           'docs/protenix_extended.md', 'scripts/extended_startup.py',
           'tests/test_extended_startup.py'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unpack(path, archive_sha, manifest_sha, count):
    if sha(path.read_bytes()) != archive_sha:
        raise ValueError('Historical archive changed')
    files = {}
    with tarfile.open(path) as archive:
        for member in archive:
            name = member.name.removeprefix('source/')
            safe = PurePosixPath(name)
            if (not member.isfile() or not member.name.startswith('source/') or name in files
                    or safe.is_absolute() or '..' in safe.parts or '\\' in name or ':' in name):
                raise ValueError('Unsafe historical member')
            files[name] = archive.extractfile(member).read()
    raw = files.pop('source_manifest.json')
    manifest = json.loads(raw)
    if sha(raw) != manifest_sha or len(files) != count or set(files) != set(manifest['files']):
        raise ValueError('Historical full source scope changed')
    if any(sha(data) != manifest['files'][name] for name, data in files.items()):
        raise ValueError('Historical execution bytes changed')
    return files, manifest


def original():
    files, manifest = unpack(PARENT_ARCHIVE, PARENT_ARCHIVE_SHA, PARENT_MANIFEST, 641)
    _, ancestor = unpack(ANCESTOR_ARCHIVE, ANCESTOR_ARCHIVE_SHA, ANCESTOR_MANIFEST, 625)
    if ancestor.get('git_revision_reference') != REVISION:
        raise ValueError('Revision is not the exact authenticated ancestor field')
    return files


def initialize(files):
    for local, member in ALIASES.items():
        if local in ('README.md', 'extended_startup.py', 'fixture_extended_startup.py'):
            continue
        path = HERE / local
        if path.exists():
            raise FileExistsError('Refusing canonical overwrite: ' + str(path))
        data = files[member]
        if local == 'extended_ops.py':
            data = data.replace(b"ROOT=BASE/'protenix-extended-01'", b"ROOT=BASE/'protenix-extended-03'")
            data = data.replace(b"SOURCE=BASE/'protenix-extended-release-01/source'", b"SOURCE=BASE/'protenix-extended-release-03/source'")
        path.write_bytes(data)


def seal(previous):
    files = dict(previous)
    for local, member in ALIASES.items():
        files[member] = (HERE / local).read_bytes().replace(b'\r\n', b'\n')
    for name, data in files.items():
        if name.endswith('.py'):
            ast.parse(data, filename=name)
    changed = {name: sha(data) for name, data in files.items() if previous.get(name) != data}
    if set(changed) != MUTABLE:
        raise ValueError('Undeclared metadata/controller changes: ' + repr(set(changed)))
    scientific = [name for name in previous if name.startswith('src/')]
    if len(scientific) != 145 or any(files[name] != previous[name] for name in scientific):
        raise ValueError('Scientific library bytes changed')
    if any(files[name] != data for name, data in previous.items() if name not in MUTABLE):
        raise ValueError('Protected641 driver/protocol/config/check/parent bytes changed')
    inheritance = {'version': 'authenticated-ancestor-git-revision-v1',
                   'ancestor_manifest_sha256': ANCESTOR_MANIFEST,
                   'ancestor_archive_sha256': ANCESTOR_ARCHIVE_SHA,
                   'field': 'git_revision_reference', 'value': REVISION}
    manifest = {'files': {name: sha(data) for name, data in sorted(files.items())},
                'parent_manifest_sha256': PARENT_MANIFEST,
                'git_revision_reference': REVISION, 'git_revision_inheritance': inheritance,
                'local_engineering_predecessor': {'source_manifest_sha256':'f1c15dc99052a8c4764e4ea0d0e9d8ca068e1c496e12c8c76c55251233b1220e',
                    'archive_sha256':'e04c57d588ef3bc531902f8ccc8fbb2d634ceea78dc8cbe6640cfbb3607d6b36',
                    'status':'unsubmitted_local_candidate_preserved',
                    'reason':'Genuine GPU startup moved to short-lived same-allocation child before production'},
                'scope': 'Protenix metadata/startup recovery with GPU child isolation; complete scientific641 source unchanged'}
    raw = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    output = REPO / 'results/hpc' / HERE.name / 'source01'
    output.mkdir(parents=True, exist_ok=False)
    archive_path = output / 'source.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for name, data in {**files, 'source_manifest.json': raw}.items():
            member = tarfile.TarInfo('source/' + name); member.mode = 0o444; member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    (output / 'source_manifest.json').write_bytes(raw)
    package = {'tag': HERE.name, 'manifest_sha256': sha(raw), 'source_files': len(files),
               'archive_bytes': archive_path.stat().st_size, 'archive_sha256': sha(archive_path.read_bytes()),
               'parent_manifest_sha256': PARENT_MANIFEST, 'git_revision_reference': REVISION,
               'git_revision_inheritance': inheritance, 'changed_members': changed,
               'unchanged_scientific_src_members': len(scientific),
               'unchanged_original641_members': len(previous) - len(set(previous) & MUTABLE),
               'execution': 'Local stdlib byte/AST packaging only; allocated CPU/GPU checks pending'}
    with (HERE / 'package.json').open('x', encoding='utf8') as handle:
        json.dump(package, handle, indent=2); handle.write('\n')
    print(json.dumps(package, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--initialize', action='store_true'); parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    if args.initialize == args.seal:
        raise ValueError('Choose one explicit preparation action')
    files = original()
    initialize(files) if args.initialize else seal(files)


if __name__ == '__main__':
    main()
