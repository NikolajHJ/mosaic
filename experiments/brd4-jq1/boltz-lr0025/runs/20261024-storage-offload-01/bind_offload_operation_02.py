"""Unexecuted local data-only binder for the distinct legacy-offload-02.

No bound helper is emitted until genuine allocated operator04/44-case evidence
and its independent audit exist. The four reviewed operation01 helper files are
never edited. Only whitelisted literal identities change in new helper copies;
the separately parsed embedded remote programs receive the same AST check.
This program performs opaque-byte/XML/source metadata work only. It has no SSH,
backup, deletion, submission, scientific execution or automatic follow-up.
"""
from __future__ import annotations

import argparse
import ast
import base64
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import tokenize
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
RESULTS = REPO / 'results/hpc/20261024-storage-offload-01'
ROOT = '/work3/s234818/mosaic/runs/20261024-storage-offload-01'
OLD_MANIFEST = '9af242392c14b3fc09804f0868b49f0d650245a1f075d43c3b132d994545230d'
NEW_MANIFEST = '8a6420047f7260eedd1219c4313c835dae4bb2a13cd39d4914f9e87abcd5bafe'
SPEC_SHA = 'cb8e794d5f67731ff7389c99fd3827a7ad0874f2055879166d932760ad12a9f4'
AUDIT_CODE_SHA = '441ec4d6f05a9bf8357edc5fe3b421cbc7b5893ace1f959924cf05b4540ee6c2'
SOURCE_PINS = {
    'run_approved_backup.py': '182bd5f47a5a1173831eb7f8ebd857149a15dbb7b24ff8574b2c6d2377f28a26',
    'collect_offload_receipts.py': 'fc9d3b591d8f00e584ddd5369e033a8e94031d817300e80fb31773f3a29ce72c',
    'execute_verified_removal.py': '0bee28202e8d7498a712d8a7ffc324352c9f8f1cd42a602dabcbd3cf068b91e5',
    'run_post_deletion_audit.py': '04a6d2b7979c3ea0bc6ce378070c80bc665af8e974ad773d6f6f44cb2aa9d49d',
}
MAX_BYTES = 128 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate JSON field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def raw_file(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/linked local metadata proof')
    with path.open('rb') as stream:
        raw = stream.read(MAX_BYTES + 1)
    require(len(raw) <= MAX_BYTES, 'Metadata proof exceeds cap')
    return raw


def prove_actual_cpu(evidence):
    """Recheck actual audit/source/archive/CPU rows; no fixture is executed."""
    audit_raw = raw_file(HERE / 'operator-verification-04.json'); audit = strict_json(audit_raw)
    job = audit['job_id']
    require(isinstance(job, str) and re.fullmatch(r'\d+', job) and job != '29806228'
            and audit['version'] == 'mosaic-offload-allocated-operator-verification-v1'
            and audit['status'] == 'passed' and audit['cases'] == 44
            and audit['failures'] == audit['errors'] == audit['skips'] == 0
            and audit['source_manifest_sha256'] == NEW_MANIFEST
            and audit['scientific_admission_or_result_claimed'] is False
            and audit['original_remote_files_deleted'] == 0,
            'No genuine independent operator04/44-case pass')
    evidence = Path(evidence)
    require(evidence.resolve().is_relative_to(RESULTS.resolve()), 'CPU evidence outside dedicated result scope')
    bundle_raw = raw_file(evidence / 'stdout.bin'); bundle = strict_json(bundle_raw)
    require(sha(bundle_raw) == audit['bundle_sha256'], 'Actual CPU bundle differs from independent audit')
    files = {}
    for path, row in bundle['files'].items():
        data = base64.b64decode(row['base64'], validate=True)
        require(len(data) == row['bytes'] and sha(data) == row['sha256'], 'Actual proof byte hash differs')
        files[path] = data
    require(audit['proofs'] == {path: dict(bytes=len(data), sha256=sha(data)) for path, data in files.items()},
            'Independent complete proof inventory differs')
    source, cpu = ROOT + '/operator-source-04', ROOT + '/cpu-04'
    manifest_raw = raw_file(HERE / 'operator-source-manifest-04.json'); manifest = strict_json(manifest_raw)
    require(sha(manifest_raw) == NEW_MANIFEST and files[source + '/source-manifest.json'] == manifest_raw
            and len(manifest['files']) == 11 and manifest['allocated_fixture_rows_expected'] == 44
            and manifest['gzip_compresslevel'] == 1 and manifest['parent_manifest_sha256'] == OLD_MANIFEST,
            'Actual immutable04 source differs')
    for name, digest in manifest['files'].items():
        require(Path(name).name == name and sha(files[source + '/' + name]) == digest, 'Actual source04 member differs')
    package = strict_json(raw_file(HERE / 'operator-package-04.json'))
    archive = Path(package['archive']); archive_raw = raw_file(archive)
    require(package['manifest_sha256'] == NEW_MANIFEST and package['archive_bytes'] == len(archive_raw) == 31403
            and package['archive_sha256'] == sha(archive_raw) == 'ce1fb5252bbd4ba57669198187efec8d77c041b33b958ee352c10248e5d3fd7b',
            'Local source04 archive differs')
    with tarfile.open(fileobj=io.BytesIO(archive_raw)) as tar:
        members = list(tar)
        require(len(members) == 12 and len({x.name for x in members}) == 12
                and {x.name for x in members} == set(manifest['files']) | {'source-manifest.json'}, 'Unsafe source archive scope')
        for item in members:
            require(item.isfile() and not item.linkname and Path(item.name).name == item.name
                    and tar.extractfile(item).read() == files[source + '/' + item.name], 'Source archive member differs')
    # Source04 explicitly adopts actual source03 and its successful32 cases.
    parent = manifest['parent_adoption']
    require(parent['source_manifest_sha256'] == OLD_MANIFEST and parent['cpu_job_id'] == '29806228'
            and parent['cases'] == 32 and parent['local_independent_verification_sha256']
                == sha(raw_file(HERE / 'operator-verification-03.json')), 'Original03 history adoption differs')
    for path, pin in parent['proofs'].items():
        require(path in files and len(files[path]) == pin['bytes'] and sha(files[path]) == pin['sha256'],
                'Actual04 bundle does not preserve adopted original03 proof bytes')
    required = {cpu + '/' + name for name in ('submission-started.json', 'submission.json',
        'stdout.' + job + '.log', 'stderr.' + job + '.log', 'payload/assessment.json', 'payload/invocation.json',
        'payload/operator-contract.xml', 'payload/operator-contract.log', 'payload/storage-gate.json')}
    require({p for p in files if p.startswith(cpu + '/')} == required, 'Actual CPU04 proof scope differs')
    submission = strict_json(files[cpu + '/submission.json'])
    expected = ['bsub', '-q', 'hpc', '-n', '2', '-R', 'span[hosts=1]', '-R', 'rusage[mem=4GB]', '-W', '00:30',
        '-J', 'mosaic-storage-offload-cpu-04', '-oo', cpu + '/stdout.%J.log', '-eo', cpu + '/stderr.%J.log',
        'bash', '-l', source + '/cleanup_cpu.lsf', NEW_MANIFEST]
    require(submission['job_id'] == job and submission['outcome'] == 'submitted'
            and submission['args'] == expected and submission['source_manifest_sha256'] == NEW_MANIFEST,
            'Actual CPU04 submission/resources differ')
    scheduler = bundle['scheduler'][0]; text = ' '.join(scheduler['stdout'].split())
    require(scheduler['returncode'] == 0 and 'Job <' + job + '>' in text and 'Status <DONE>' in text
            and 'Queue <hpc>' in text and 'Done successfully.' in text, 'Actual same-job CPU04 scheduler success absent')
    log = files[cpu + '/stdout.' + job + '.log'].decode()
    require(re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$', log, re.M) == [(job, 'Done')]
            and 'Successfully completed.' in log, 'Actual CPU04 LSF success absent')
    cases = list(ET.fromstring(files[cpu + '/payload/operator-contract.xml']).iter('testcase'))
    require(len(cases) == len({x.attrib['name'] for x in cases}) == 44
            and all(not list(row) and row.attrib['classname'].split('.')[-1] == 'fixture_backup_contract' for row in cases),
            'Raw CPU04 44-case attribution/pass differs')
    require(re.search(r'\b44\s+passed\b', files[cpu + '/payload/operator-contract.log'].decode()), 'Raw CPU04 log count differs')
    assessment = strict_json(files[cpu + '/payload/assessment.json'])
    require(assessment['status'] == 'passed' and assessment['cases'] == 44 and assessment['job_id'] == job
            and assessment['queue'] == 'hpc' and assessment['source_manifest_sha256'] == NEW_MANIFEST,
            'Actual CPU04 assessment differs')
    for pin in assessment['proofs']:
        require(pin['path'] in files and len(files[pin['path']]) == pin['bytes'] and sha(files[pin['path']]) == pin['sha256'],
                'Assessment original proof hash differs')
    gate = strict_json(files[cpu + '/payload/storage-gate.json'])
    require(gate['gate']['status'] == 'passed' and gate['maximum_persistent_bytes'] == 128 * 1024**2
            and gate['gate']['reserve_bytes'] == 4 * 1024**3 and gate['gate']['file_reserve'] == 100000
            and gate['scientific_admission'] is False and gate['scientific_projection_changed'] is False,
            'Actual bounded metadata entry admission differs')
    return job, audit_raw, bundle_raw


def substitutions(job):
    return (
        (OLD_MANIFEST, NEW_MANIFEST),
        ('29806228', job),
        ('operator-source-manifest-03.json', 'operator-source-manifest-04.json'),
        ('operator-verification-03.json', 'operator-verification-04.json'),
        ('operator-source-03', 'operator-source-04'),
        ('cpu-03', 'cpu-04'),
        ('legacy-offload-01', 'legacy-offload-02'),
        ('backup-01', 'backup-02'),
        ('approved-selection-01.json', 'approved-selection-02.json'),
        ('post-deletion-audit-01', 'post-deletion-audit-02'),
        ('post-audit-01', 'post-audit-02'),
        ('CPU32', 'CPU44'),
        ('operator32', 'operator44'),
        (r'\b32\s+passed\b', r'\b44\s+passed\b'),
    )


def bind_string(value, job):
    for old, new in substitutions(job):
        value = value.replace(old, new)
    return value


def bind_program(source, job, *, embedded=False):
    """Replace token literals only, then independently prove AST equivalence."""
    original = ast.parse(source)
    remote_values = {node.value.value for node in original.body if isinstance(node, ast.Assign)
        and any(isinstance(x, ast.Name) and x.id == 'REMOTE' for x in node.targets)
        and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)}
    remote_changes = {value: bind_program(value, job, embedded=True) for value in remote_values}

    def changed(value):
        if isinstance(value, str):
            return remote_changes.get(value, bind_string(value, job))
        if type(value) is int and value == 32:
            return 44
        return value

    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.STRING:
            value = ast.literal_eval(token.string)
            if isinstance(value, str):
                candidate = changed(value)
                if candidate != value:
                    token = token._replace(string=repr(candidate))
        elif token.type == tokenize.NUMBER and token.string == '32':
            token = token._replace(string='44')
        tokens.append(token)
    bound = tokenize.untokenize(tokens)
    candidate = ast.parse(bound)

    class OnlyLiteralData(ast.NodeTransformer):
        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant(value=changed(node.value)), node)

    expected = OnlyLiteralData().visit(copy.deepcopy(original))
    require(ast.dump(expected, include_attributes=False) == ast.dump(candidate, include_attributes=False),
            'Binder changed executable syntax instead of whitelisted literal data')
    compile(candidate, '<embedded-bound-helper>' if embedded else '<bound-helper>', 'exec')
    # No obsolete operation/CPU/source identity can silently remain in code.
    values = [x.value for x in ast.walk(candidate) if isinstance(x, ast.Constant) and isinstance(x.value, str)]
    for value in values:
        require(all(old not in value for old in (OLD_MANIFEST, '29806228', 'operator-source-03', 'cpu-03',
            'legacy-offload-01', 'backup-01', 'approved-selection-01.json', 'post-audit-01', 'post-deletion-audit-01')),
            'An obsolete literal identity remains in a new helper')
    return bound


def save(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-evidence', type=Path, required=True,
                   help='Existing independently audited actual CPU04 evidence directory under ignored results')
    args = p.parse_args()
    job, audit_raw, bundle_raw = prove_actual_cpu(args.cpu_evidence)
    require(sha(raw_file(HERE / 'post_deletion_audit.py')) == AUDIT_CODE_SHA
            and sha(raw_file(HERE / 'post-audit-pin-spec-candidate-01.json')) == SPEC_SHA,
            'Generic approved post-audit code/pins changed')
    prepared = {}
    for name, digest in SOURCE_PINS.items():
        raw = raw_file(HERE / name); require(sha(raw) == digest, 'Reviewed original01 helper changed: ' + name)
        bound = bind_program(raw.decode('utf8'), job)
        target = HERE / (Path(name).stem + '_02.py')
        require(not target.exists(), 'A bound helper already exists; inspect rather than replay')
        prepared[target] = bound.encode('utf8')
    start = HERE / 'operation-02-binding-started.json'; result = HERE / 'operation-02-binding.json'
    require(not start.exists() and not result.exists(), 'Existing/unknown operation02 binding outcome')
    save(start, dict(status='unknown_until_four_bound_helpers_complete', utc=datetime.now(timezone.utc).isoformat(),
        binder_sha256=sha(raw_file(__file__)), source_manifest_sha256=NEW_MANIFEST,
        operator_cpu_job_id=job, actual_cases=44, allocated_verification_sha256=sha(audit_raw),
        actual_cpu_bundle_sha256=sha(bundle_raw), source_helpers=SOURCE_PINS,
        intended_outputs=[x.name for x in prepared], remote_execution=False))
    outputs = {}
    for target, raw in prepared.items():
        with target.open('xb') as stream:
            stream.write(raw)
        outputs[target.name] = dict(bytes=len(raw), sha256=sha(raw),
            original_helper=target.name.removesuffix('_02.py') + '.py', syntax_changes='whitelisted literal data only')
        remotes = [node.value.value for node in ast.parse(raw).body if isinstance(node, ast.Assign)
            and any(isinstance(x, ast.Name) and x.id == 'REMOTE' for x in node.targets)
            and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)]
        outputs[target.name]['embedded_remote_programs'] = [dict(bytes=len(value.encode()), sha256=sha(value.encode()),
            syntax_changes='independently parsed whitelisted literal data only') for value in remotes]
    # Rehash original files after all writes: never mutate reviewed01 helpers.
    require(all(sha(raw_file(HERE / name)) == digest for name, digest in SOURCE_PINS.items()),
            'Historical original01 helper was changed')
    save(result, dict(status='four_new_operation02_helpers_bound_unexecuted', utc=datetime.now(timezone.utc).isoformat(),
        operation_id='legacy-offload-02', local_backup_directory='backup-02', approved_selection='approved-selection-02.json',
        remote_operator_source='operator-source-04', actual_cpu_parent='cpu-04', actual_cpu_job_id=job,
        actual_allocated_cases=44, source_manifest_sha256=NEW_MANIFEST, allocated_verification_sha256=sha(audit_raw),
        actual_cpu_bundle_sha256=sha(bundle_raw), sources=SOURCE_PINS, outputs=outputs,
        literal_substitution_whitelist=substitutions(job), generic_post_audit_code_sha256=AUDIT_CODE_SHA,
        approved_pin_spec_sha256=SPEC_SHA, original03_history_adopted_by_actual04_cpu=True,
        original01_files_unchanged=True, generated_helpers_executed=False, remote_execution=False,
        backup_or_deletion_authority_created=False, scientific_or_storage_admission_pass_claimed=False))
    print(json.dumps(dict(status='four_new_operation02_helpers_bound_unexecuted', actual_cpu_job_id=job,
                          cases=44, outputs=outputs, executed=False)))


if __name__ == '__main__':
    main()
