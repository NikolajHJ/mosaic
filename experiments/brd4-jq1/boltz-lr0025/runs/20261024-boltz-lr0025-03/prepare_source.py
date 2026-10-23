"""New immutable provenance correction; local stdlib bytes/AST packaging only.

The scientific633 members are inherited exactly. A trusted625 manifest supplies
its real revision field. Only controller, wrappers, docs and startup fixtures
may differ. Initialize once; seal explicitly after the shared helper is ready.
"""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ARCHIVE = REPO / 'results/hpc/20261023-boltz-lr0025-02/source01/source.tar.gz'
ARCHIVE_SHA = '40b4a087c2735de93ed1896f7534618f32d8713dae4d83fa80d36b49b1924dfa'
MANIFEST = '3689071a4e7dd135071ff5b420c026543c4d10887c951077dff63d498d9f812c'
ANCESTOR = REPO / 'results/hpc/20261022-brd4-jq1-25-fixed-lrreset-04/source02/source.tar.gz'
ANCESTOR_ARCHIVE = '4d6d297451d9820543abc3298622e6599014c4b96bc999e104dd732825f56e4e'
ANCESTOR_MANIFEST = 'c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057'
REVISION = 'c74c659344123937608077e3e4078f75f09cd743'
ALIASES = {
    'boltz_serial_protocol.py': 'scripts/boltz_serial_protocol.py',
    'boltz_serial_ops.py': 'scripts/boltz_serial_ops.py',
    'boltz_serial_driver.py': 'examples/brd4_25_atoms.py',
    'boltz_serial_contract.py': 'scripts/campaign25_contract.py',
    'boltz_serial_cpu.lsf': 'scripts/hpc/boltz_serial_cpu.lsf',
    'boltz_serial_stage.lsf': 'scripts/hpc/boltz_serial_stage.lsf',
    'fixture_boltz_serial.py': 'tests/test_boltz_serial.py',
    'fixture_protenix_initialization.py': 'tests/test_protenix_initialization.py',
    'boltz2.json': 'examples/configs/boltz_serial_boltz2.json',
    'boltz2_chemistry.json': 'examples/configs/boltz_serial_boltz2_chemistry.json',
    'README.md': 'docs/boltz_serial30k.md',
}
SHARED = {
    'scripts/hpc/startup/premodel_startup.py': 'scripts/premodel_startup.py',
    'tests/core/fixture_premodel_startup.py': 'tests/test_premodel_startup.py',
}
PROTECTED = ('examples/brd4_25_atoms.py', 'scripts/campaign25_contract.py',
             'scripts/boltz_serial_protocol.py', 'tests/test_boltz_serial.py',
             'tests/test_protenix_initialization.py',
             'examples/configs/boltz_serial_boltz2.json',
             'examples/configs/boltz_serial_boltz2_chemistry.json')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unpack(path, archive_sha, manifest_sha, count):
    if sha(path.read_bytes()) != archive_sha:
        raise ValueError('Historical archive bytes changed')
    files = {}
    with tarfile.open(path) as archive:
        for member in archive:
            name = member.name.removeprefix('source/')
            safe = PurePosixPath(name)
            if (not member.isfile() or not member.name.startswith('source/') or name in files
                    or safe.is_absolute() or '..' in safe.parts or '\\' in name or ':' in name):
                raise ValueError('Unsafe/duplicate historical source member')
            files[name] = archive.extractfile(member).read()
    raw = files.pop('source_manifest.json');manifest = json.loads(raw)
    if sha(raw) != manifest_sha or len(files) != count or set(files) != set(manifest['files']):
        raise ValueError('Historical manifest/full member scope changed')
    for name, data in files.items():
        if sha(data) != manifest['files'][name]:
            raise ValueError('Historical source member changed: ' + name)
    return files, manifest


def source():
    files, _ = unpack(ARCHIVE, ARCHIVE_SHA, MANIFEST, 633)
    _, ancestor = unpack(ANCESTOR, ANCESTOR_ARCHIVE, ANCESTOR_MANIFEST, 625)
    if ancestor.get('git_revision_reference') != REVISION:
        raise ValueError('Revision is not the exact authenticated ancestor625 field')
    return files


def initialize(files):
    for local, target in ALIASES.items():
        if local == 'README.md': continue
        path = HERE / local
        if path.exists(): raise FileExistsError('Canonical initialization refuses overwrite: ' + str(path))
        data = files[target]
        if local == 'boltz_serial_ops.py':
            data = data.replace(b"ROOT = BASE / 'boltz-lr0025-02'", b"ROOT = BASE / 'boltz-lr0025-03'")
            data = data.replace(b"SOURCE = BASE / 'boltz-lr0025-release-02/source'", b"SOURCE = BASE / 'boltz-lr0025-release-03/source'")
            data = data.replace(b"DBMOL_ROOT = BASE / 'boltz-dbmol-pilot-01'", b"DBMOL_ROOT = BASE / 'boltz-dbmol-pilot-02'")
            data = data.replace(b"BASE / 'boltz-dbmol-pilot-release-01/source'", b"BASE / 'boltz-dbmol-pilot-release-02/source'")
        path.write_bytes(data)


def seal(original):
    files = dict(original)
    for local, target in ALIASES.items():
        files[target] = (HERE / local).read_bytes().replace(b'\r\n', b'\n')
    for local, target in SHARED.items():
        data = (REPO / local).read_bytes().replace(b'\r\n', b'\n')
        if not data: raise ValueError('Startup helper/fixtures must be ready before sealing')
        files[target] = data
    for name, data in files.items():
        if name.endswith('.py'): ast.parse(data, filename=name)
    scientific = [name for name in original if name.startswith('src/')]
    if len(scientific) != 145 or any(files[name] != original[name] for name in scientific):
        raise ValueError('Provenance correction changed scientific library bytes')
    if any(files[name] != original[name] for name in PROTECTED):
        raise ValueError('Driver/contract/protocol/config/original fixture bytes changed')
    changed = {name: sha(data) for name, data in files.items() if original.get(name) != data}
    allowed = {'scripts/boltz_serial_ops.py', 'scripts/hpc/boltz_serial_cpu.lsf',
               'scripts/hpc/boltz_serial_stage.lsf', 'docs/boltz_serial30k.md', *SHARED.values()}
    if not set(changed) <= allowed or not set(SHARED.values()) <= set(changed):
        raise ValueError('Undeclared source/controller changes')
    inheritance = {'version': 'authenticated-ancestor-git-revision-v1',
                   'ancestor_manifest_sha256': ANCESTOR_MANIFEST,
                   'ancestor_archive_sha256': ANCESTOR_ARCHIVE,
                   'field': 'git_revision_reference', 'value': REVISION,
                   'scope': 'Exact original625 metadata field; no inferred revision'}
    manifest = {'files': {name: sha(data) for name, data in sorted(files.items())},
                'parent_manifest_sha256': MANIFEST,
                'git_revision_reference': REVISION,
                'git_revision_inheritance': inheritance,
                'scope': 'Metadata provenance correction with actual pre-model startup regressions; scientific633 library/driver/protocol/configs unchanged'}
    raw = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    output = REPO / 'results/hpc' / HERE.name / 'source01'
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(output / 'source.tar.gz', 'w:gz') as archive:
        for name, data in {**files, 'source_manifest.json': raw}.items():
            member = tarfile.TarInfo('source/' + name);member.mode = 0o444;member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    (output / 'source_manifest.json').write_bytes(raw)
    package = {'tag': HERE.name, 'manifest_sha256': sha(raw), 'source_files': len(files),
               'archive_bytes': (output / 'source.tar.gz').stat().st_size,
               'archive_sha256': sha((output / 'source.tar.gz').read_bytes()),
               'parent_manifest_sha256': MANIFEST, 'changed_members': changed,
               'git_revision_reference': REVISION, 'git_revision_inheritance': inheritance,
               'scientific_src_changes': [], 'unchanged_scientific_library_members': 145,
               'execution': 'Local stdlib AST/JSON/byte packaging only; allocated checks pending'}
    with (HERE / 'package.json').open('x', encoding='utf8') as stream:
        json.dump(package, stream, indent=2);stream.write('\n')
    print(json.dumps(package, indent=2))


def main():
    p = argparse.ArgumentParser();p.add_argument('--initialize', action='store_true');p.add_argument('--seal', action='store_true');a = p.parse_args()
    if a.initialize == a.seal: raise ValueError('Choose one explicit preparation phase')
    files = source()
    initialize(files) if a.initialize else seal(files)


if __name__ == '__main__':
    main()
