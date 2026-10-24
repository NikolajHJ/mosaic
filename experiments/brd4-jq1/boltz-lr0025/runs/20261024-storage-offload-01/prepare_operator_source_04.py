"""Seal distinct04 performance/container tooling; stdlib byte work only.

No backup, deletion, submission or allocated pass occurs here. Future use of04
requires its own CPU44 and explicit operation/adoption bindings; source03 and
its intentionally retired partial backup remain immutable historical evidence.
"""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
PARENT_SHA = '9af242392c14b3fc09804f0868b49f0d650245a1f075d43c3b132d994545230d'
PARENT_ARCHIVE_SHA = '7b2be2caadf109d2e94945d606511004bf54a781341c006344d9c93ce586f02b'
OVERLAYS = {'offload_protocol.py': 'offload_protocol_04.py',
            'fixture_backup_contract.py': 'fixture_backup_contract_04.py',
            'cleanup_cpu.py': 'cleanup_cpu_04.py', 'cleanup_cpu.lsf': 'cleanup_cpu_04.lsf'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def count_cases(raw):
    count = 0
    for node in ast.parse(raw).body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith('test_'):
            continue
        rows = 1
        for item in node.decorator_list:
            if (isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                    and item.func.attr == 'parametrize'):
                rows *= len(ast.literal_eval(item.args[1]))
        count += rows
    return count


def main():
    original = REPO / 'results/hpc/20261024-storage-offload-01/operator-source-03/source.tar.gz'
    assert original.stat().st_size == 26508 and sha(original) == PARENT_ARCHIVE_SHA
    files = {}
    with tarfile.open(original) as tar:
        for member in tar:
            assert member.isfile() and '/' not in member.name and member.name not in files
            files[member.name] = tar.extractfile(member).read()
    raw_parent = files.pop('source-manifest.json')
    assert hashlib.sha256(raw_parent).hexdigest() == PARENT_SHA
    old = json.loads(raw_parent)
    assert len(files) == 11 and set(files) == set(old['files'])
    for name, digest in old['files'].items():
        assert hashlib.sha256(files[name]).hexdigest() == digest
    audit_path = HERE / 'operator-verification-03.json'
    audit = json.loads(audit_path.read_bytes())
    assert audit['status'] == 'passed' and audit['cases'] == 32 and audit['job_id'] == '29806228'
    assert audit['source_manifest_sha256'] == PARENT_SHA
    for name, local in OVERLAYS.items():
        files[name] = (HERE / local).read_bytes()
    for name, raw in files.items():
        if name.endswith('.py'):
            ast.parse(raw, filename=name)
    assert count_cases(files['fixture_backup_contract.py']) == 44
    # Every original payload opener, hash, race, active-job and unlink guard
    # remains unchanged; this proof is AST/data-level, not runtime validation.
    parent_protocol = ast.parse(tar_protocol(original))
    candidate_protocol = ast.parse(files['offload_protocol.py'])
    parent_nodes = {n.name: n for n in parent_protocol.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    candidate_nodes = {n.name: n for n in candidate_protocol.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for name in parent_nodes.keys() - {'validate_plan', 'pack_stream'}:
        assert ast.dump(parent_nodes[name], include_attributes=False) == ast.dump(candidate_nodes[name], include_attributes=False), name
    restored = files['offload_protocol.py'].decode()
    restored = restored.replace('    membership = compile_plan_membership(roots, protections)\n', '')
    restored = restored.replace('        inside, blocked = membership(path)\n', '')
    restored = restored.replace("require(inside, 'File outside approved exact candidate roots')",
        "require(any(below(path, root) for root in roots), 'File outside approved exact candidate roots')")
    restored = restored.replace("require(str(path) not in paths and not blocked, 'Duplicate or protected current dependency')",
        "require(str(path) not in paths and not is_protected(path, protections), 'Duplicate or protected current dependency')")
    restored_node = next(n for n in ast.parse(restored).body if isinstance(n, ast.FunctionDef) and n.name == 'validate_plan')
    assert ast.dump(restored_node, include_attributes=False) == ast.dump(parent_nodes['validate_plan'], include_attributes=False)
    old_pack = ast.dump(parent_nodes['pack_stream'], include_attributes=False)
    copy_pack = ast.parse(ast.unparse(candidate_nodes['pack_stream'])).body[0]
    for node in ast.walk(copy_pack):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'GzipFile':
            assert [(k.arg, ast.literal_eval(k.value)) for k in node.keywords if k.arg == 'compresslevel'] == [('compresslevel', 1)]
            node.keywords = [k for k in node.keywords if k.arg != 'compresslevel']
    assert ast.dump(copy_pack, include_attributes=False) == old_pack
    manifest = {**old, 'allocated_fixture_rows_expected': 44,
        'parent_manifest_sha256': PARENT_SHA,
        'correction': 'Canonical root/protection lookup prepared once per plan; gzip compression explicitly level1. Original guards/member bytes unchanged; compressed container bytes have fresh authenticated hashes.',
        'gzip_compresslevel': 1,
        'scope': 'Unvalidated successor tooling; no backup/deletion authority, scientific admission or original scientific change. Explicit future CPU44 and operation/adoption required.',
        'parent_adoption': {'source_manifest_sha256': PARENT_SHA, 'cpu_job_id': '29806228', 'cases': 32,
            'local_independent_verification_sha256': sha(audit_path), 'proofs': audit['proofs']},
        'files': {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}}
    raw_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    folder = REPO / 'results/hpc/20261024-storage-offload-01/operator-source-04'
    folder.mkdir(exist_ok=False)
    archive = folder / 'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name, raw in {**files, 'source-manifest.json': raw_manifest}.items():
            member = tarfile.TarInfo(name); member.size, member.mode, member.mtime = len(raw), 0o444, 0
            tar.addfile(member, io.BytesIO(raw))
    result = {'status': 'sealed_unvalidated', 'source_files': len(files), 'allocated_cases_expected': 44,
        'manifest_sha256': hashlib.sha256(raw_manifest).hexdigest(),
        'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size, 'archive': str(archive),
        'parent_source_manifest_sha256': PARENT_SHA, 'parent_cpu_job_id': '29806228', 'parent_cases': 32,
        'gzip_compresslevel': 1, 'backup_or_deletion_executed': False, 'allocated_validation_passed': False}
    for path in (folder / 'source-manifest.json', HERE / 'operator-source-manifest-04.json'):
        with path.open('xb') as stream: stream.write(raw_manifest)
    with (HERE / 'operator-package-04.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))


def tar_protocol(path):
    with tarfile.open(path) as archive:
        return archive.extractfile('offload_protocol.py').read()


if __name__ == '__main__':
    main()
