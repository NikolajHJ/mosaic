"""Unexecuted local literal-only binding for the distinct single-link offload03.

Keeps every original01/02 helper and partial transfer unchanged. Requires actual
operator04 CPU44 evidence, intentional retirement02 and the newly approved
plan/protection03. It emits only three new collect/remove/read-only-audit
helpers. The unchanged generic run_approved_backup_02.py serves plan03 too.
There is no network, backup, deletion, job or scientific action in this binder.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tokenize


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ROOT = '/work3/s234818/mosaic/runs/20261024-storage-offload-01'
RESULTS = REPO / 'results/hpc/20261024-storage-offload-01'
BINDER02_SHA = '368142296f3c95e15e5b6e853e51272794fa3851a1bcb8899b2c97a37d845444'
BINDING02_SHA = 'f909f1b66614d3006bea92229b3598ccb1ae4fa5312b56602ffeaef1186822b8'
MANIFEST = '8a6420047f7260eedd1219c4313c835dae4bb2a13cd39d4914f9e87abcd5bafe'
AUDIT04_SHA = 'd712220f553d0988b0e1e3ae9655d78369999665f71641e5a670c52e163d3ce1'
BUNDLE04_SHA = '9bb9692472172b2c673734879fa6d103143b5106a5e8b1c179cf0d95a32f3add'
AUDIT_CODE_SHA = '441ec4d6f05a9bf8357edc5fe3b421cbc7b5893ace1f959924cf05b4540ee6c2'
PIN_SPEC_SHA = 'cb8e794d5f67731ff7389c99fd3827a7ad0874f2055879166d932760ad12a9f4'
SOURCE_PINS = {
    'collect_offload_receipts_02.py': 'b785aaeab24af70c7378ac7f90ba2ef5041ea42f7ec731409f06b5412cb6da17',
    'execute_verified_removal_02.py': 'e936e8e24d6eaf120b188305a937d072d6bf2690ecd7fdaa57185c7929856fae',
    'run_post_deletion_audit_02.py': 'e29174c0e51b792d94a27d9260168dc63fef2b6da8a7bec9da5119be358d0b4d',
}
LITERAL_SUBSTITUTIONS = (
    ('legacy-offload-02', 'legacy-offload-03'),
    ('backup-02', 'backup-03'),
    ('approved-selection-02.json', 'approved-selection-03.json'),
    ('flat-protection-approved-01.json', 'flat-protection-approved-03.json'),
    ('protection-01.json', 'protection-03.json'),
    ('post-deletion-audit-02', 'post-deletion-audit-03'),
    ('post-audit-02', 'post-audit-03'),
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def raw_file(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/linked local metadata proof')
    with path.open('rb') as stream:
        raw = stream.read(128 * 1024**2 + 1)
    require(len(raw) <= 128 * 1024**2, 'Metadata proof exceeds cap')
    return raw


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate JSON field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def verify_inputs(evidence):
    # Reuse the exact reviewed metadata verifier only; importing its definitions
    # does not execute its main or emit operation02 files.
    verifier = HERE / 'bind_offload_operation_02.py'
    require(sha(raw_file(verifier)) == BINDER02_SHA, 'Reviewed CPU metadata verifier changed')
    spec = importlib.util.spec_from_file_location('exact_reviewed_offload02_metadata', verifier)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    job, audit_raw, bundle_raw = helper.prove_actual_cpu(evidence)
    require(job == '29806439' and sha(audit_raw) == AUDIT04_SHA and sha(bundle_raw) == BUNDLE04_SHA,
            'Actual source04 CPU44 authority differs')
    binding_raw = raw_file(HERE / 'operation-02-binding.json'); binding = strict_json(binding_raw)
    require(sha(binding_raw) == BINDING02_SHA and binding['status'] == 'four_new_operation02_helpers_bound_unexecuted'
            and binding['source_manifest_sha256'] == MANIFEST and binding['actual_allocated_cases'] == 44
            and binding['actual_cpu_job_id'] == job and binding['allocated_verification_sha256'] == AUDIT04_SHA,
            'Reviewed bound02 helper provenance differs')
    require(sha(raw_file(HERE / 'run_approved_backup_02.py')) == 'a3296c53fb63eb14fdeb0fedcc77ec45e83d2a88ae10f8078a11fa20663c1543'
            and sha(raw_file(HERE / 'post_deletion_audit.py')) == AUDIT_CODE_SHA
            and sha(raw_file(HERE / 'post-audit-pin-spec-candidate-01.json')) == PIN_SPEC_SHA,
            'Generic backup, audit code or approved pins changed')
    approval_raw = raw_file(HERE / 'approved-selection-03.json'); approval = strict_json(approval_raw)
    protection_raw = raw_file(HERE / 'flat-protection-approved-03.json'); protection = strict_json(protection_raw)
    plan_path = RESULTS / 'backup-03/approved-plan.json'; plan_raw = raw_file(plan_path); plan = strict_json(plan_raw)
    require(approval['status'] == 'approved_for_backup_then_conditional_identical_copy_removal'
            and approval['operation_id'] == plan['operation_id'] == 'legacy-offload-03'
            and approval['plan_sha256'] == sha(plan_raw) and approval['plan_bytes'] == len(plan_raw)
            and Path(approval['plan_path']).resolve() == plan_path.resolve()
            and plan['cleanup_root'] == ROOT and plan['status'] == 'approved_archive_then_remove'
            and plan['operator_source_manifest_sha256'] == approval['operator_source_manifest_sha256'] == MANIFEST
            and plan['allocated_operator_verification_sha256'] == AUDIT04_SHA
            and approval['allocated_operator_job_id'] == job and approval['allocated_operator_cases'] == 44,
            'New exact approved plan/source/CPU identity differs')
    require(approval['selection_policy'] == 'complete_single_link_inode_groups_only'
            and approval['files'] == len(plan['files']) == 25413
            and approval['logical_bytes'] == sum(row['bytes'] for row in plan['files']) == 15725430139
            and approval['excluded_linked_alias_files'] == 324 and approval['excluded_linked_inode_groups'] == 123,
            'Root-approved single-link-only selection differs')
    require(protection['version'] == 'mosaic-storage-offload-protection-v1'
            and sha(protection_raw) == approval['protection_sha256']
            and plan['protection_manifest'] == dict(path=ROOT + '/protection-03.json', sha256=sha(protection_raw)),
            'New exact expanded protection03 differs')
    # Selection authoring owns its exact inactive/single-link classification.
    # This binder requires the old protection to remain included verbatim.
    old_protection = strict_json(raw_file(HERE / 'flat-protection-approved-01.json'))
    require(all(set(old_protection[k]) <= set(protection[k]) for k in ('exact', 'subtrees', 'negative')),
            'New protection dropped an original current-experiment guard')
    retirement_raw = raw_file(HERE / 'stream-retirement-02.json'); retirement = strict_json(retirement_raw)
    require(retirement['version'] == 'mosaic-intentional-backup-retirement-proof-v1'
            and retirement['status'] == 'intentionally_stopped_own_incomplete_backup'
            and retirement['old_exact_producer_running'] is False and retirement['all_remote_originals_retained'] is True
            and retirement['path'] == ROOT + '/stream-retirement-complete-02.json'
            and retirement['sha256'] == '3324ff5a10fe9e461ab1946c4072859a8a09d4ffce72eca9d09cebcf2a9a2da2'
            and retirement['bytes'] == 1132, 'No genuine intentional incomplete02 retirement')
    require(set(retirement['files']) == {ROOT + '/stream-retirement-intent-02.json', retirement['path']},
            'Retirement proof scope differs')
    for path, pin in retirement['files'].items():
        local = Path(pin['local_path']); data = raw_file(local)
        require(local.resolve().is_relative_to((RESULTS / 'backup-02/retirement-evidence-01').resolve())
                and pin['path'] == path and len(data) == pin['bytes'] and sha(data) == pin['sha256'],
                'Actual retirement02 raw proof changed')
    completed = strict_json(raw_file(retirement['local_path']))
    require(completed['pid'] == 177888 and completed['status'] == 'intentionally_stopped_own_incomplete_backup'
            and completed['producer_no_longer_running'] is True and completed['remote_originals_retained'] is True
            and completed['backup_verified'] is False and completed['original_files_deleted'] == 0,
            'Retirement02 does not preserve the original files')
    return approval_raw, protection_raw, plan_raw, retirement_raw


def bind_text(source, *, embedded=False):
    original = ast.parse(source)
    remotes = {node.value.value for node in original.body if isinstance(node, ast.Assign)
        and any(isinstance(x, ast.Name) and x.id == 'REMOTE' for x in node.targets)
        and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)}
    transformed_remotes = {value: bind_text(value, embedded=True) for value in remotes}

    def literal(value):
        if value in transformed_remotes:
            return transformed_remotes[value]
        for old, new in LITERAL_SUBSTITUTIONS:
            value = value.replace(old, new)
        return value

    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.STRING:
            value = ast.literal_eval(token.string)
            if isinstance(value, str) and literal(value) != value:
                token = token._replace(string=repr(literal(value)))
        tokens.append(token)
    output = tokenize.untokenize(tokens); candidate = ast.parse(output)

    class StringDataOnly(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(literal(node.value)), node)
            return node

    expected = StringDataOnly().visit(copy.deepcopy(original))
    require(ast.dump(expected, include_attributes=False) == ast.dump(candidate, include_attributes=False),
            'New03 helper changed executable syntax or nonstring data')
    compile(candidate, '<embedded03>' if embedded else '<bound03>', 'exec')
    # Never globally replace _02: the unchanged allocated verifier alias is02.
    for node in ast.walk(candidate):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            require(all(old not in node.value for old, _ in LITERAL_SUBSTITUTIONS), 'Old operation/protection literal remains')
    return output


def save(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-evidence', type=Path, required=True)
    args = parser.parse_args()
    approval_raw, protection_raw, plan_raw, retirement_raw = verify_inputs(args.cpu_evidence)
    prepared = {}
    for name, expected in SOURCE_PINS.items():
        raw = raw_file(HERE / name)
        require(sha(raw) == expected, 'Reviewed bound02 source changed: ' + name)
        target = HERE / (name.removesuffix('_02.py') + '_03.py')
        require(not target.exists(), 'Existing bound03 helper: inspect rather than replay')
        prepared[target] = bind_text(raw.decode('utf8')).encode('utf8')
    start = HERE / 'operation-03-binding-started.json'; result = HERE / 'operation-03-binding.json'
    require(not start.exists() and not result.exists(), 'Existing/unknown operation03 binding outcome')
    save(start, dict(status='unknown_until_three_bound_helpers_complete', utc=datetime.now(timezone.utc).isoformat(),
        binder_sha256=sha(raw_file(__file__)), source_manifest_sha256=MANIFEST, actual_cpu_job_id='29806439', actual_cases=44,
        allocated_verification_sha256=AUDIT04_SHA, actual_cpu_bundle_sha256=BUNDLE04_SHA,
        approved_selection_sha256=sha(approval_raw), plan_sha256=sha(plan_raw), protection_sha256=sha(protection_raw),
        retirement02_sha256=sha(retirement_raw), intended_outputs=[x.name for x in prepared], remote_execution=False))
    outputs = {}
    for path, raw in prepared.items():
        with path.open('xb') as stream:
            stream.write(raw)
        remotes = [node.value.value for node in ast.parse(raw).body if isinstance(node, ast.Assign)
            and any(isinstance(x, ast.Name) and x.id == 'REMOTE' for x in node.targets)
            and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)]
        outputs[path.name] = dict(bytes=len(raw), sha256=sha(raw), syntax_changes='whitelisted string data only',
            embedded_remote_programs=[dict(bytes=len(x.encode()), sha256=sha(x.encode())) for x in remotes])
    require(all(sha(raw_file(HERE / name)) == digest for name, digest in SOURCE_PINS.items()),
            'Original bound02 helper changed')
    save(result, dict(status='three_new_operation03_helpers_bound_unexecuted', utc=datetime.now(timezone.utc).isoformat(),
        operation_id='legacy-offload-03', local_backup_directory='backup-03', approved_selection='approved-selection-03.json',
        source_manifest_sha256=MANIFEST, actual_cpu_job_id='29806439', actual_allocated_cases=44,
        allocated_verification_sha256=AUDIT04_SHA, actual_cpu_bundle_sha256=BUNDLE04_SHA,
        approved_selection_sha256=sha(approval_raw), plan_sha256=sha(plan_raw), protection_sha256=sha(protection_raw),
        retirement02_sha256=sha(retirement_raw), sources=SOURCE_PINS, outputs=outputs,
        literal_substitution_whitelist=LITERAL_SUBSTITUTIONS,
        unchanged_generic_backup_helper=dict(path='run_approved_backup_02.py', sha256='a3296c53fb63eb14fdeb0fedcc77ec45e83d2a88ae10f8078a11fa20663c1543'),
        unchanged_standalone_verifier='verify_stream_backup_02.py', generic_post_audit_code_sha256=AUDIT_CODE_SHA,
        approved_pin_spec_sha256=PIN_SPEC_SHA, old_helpers_and_evidence_unchanged=True,
        generated_helpers_executed=False, remote_execution=False, backup_or_deletion_authority_created=False,
        scientific_or_storage_admission_pass_claimed=False))
    print(json.dumps(dict(status='three_new_operation03_helpers_bound_unexecuted', outputs=outputs,
        unchanged_generic_backup_helper='run_approved_backup_02.py', executed=False)))


if __name__ == '__main__':
    main()
