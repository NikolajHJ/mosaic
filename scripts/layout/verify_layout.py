"""Verify source layout with stdlib AST, byte and path checks; never import models."""

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import contextlib
import io
import importlib.util
import importlib.machinery
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, baseline=None, live_receipts=()):
    registry = json.loads((root / 'experiments/migration-map.json').read_text(encoding='utf-8'))
    records = dict(registry['files'])
    nested_path = root / 'experiments/folder-layout.json'
    nested = json.loads(nested_path.read_text(encoding='utf-8')) if nested_path.is_file() else {}
    records.update(nested.get('shared_tests', {}))
    for old, adapter in nested.get('relocated_compatibility', {}).items():
        records[old] = dict(records[old], compatibility_path=adapter)
    errors = []
    counts = Counter()
    concurrent_receipt_changes = []
    metadata_only_source_changes = []
    changed_since_migration = []
    for old, item in records.items():
        original, canonical = root / item.get('compatibility_path', old), root / item['path']
        for path in (original, canonical):
            if not path.is_file() or not path.resolve().is_relative_to(root):
                errors.append('Missing/escaping migration path: ' + str(path))
        if not canonical.is_file():
            continue
        if digest(canonical) != item['after_sha256']:
            changed_since_migration.append(item['path'])
            if baseline:
                errors.append('Canonical source differs from migration record: ' + item['path'])
        if canonical.suffix == '.py':
            ast.parse(canonical.read_text(encoding='utf-8-sig'), filename=str(canonical))
            if old != item['path']:
                ast.parse(original.read_text(encoding='utf-8-sig'), filename=str(original))
                if '_execute_legacy(globals(),' not in original.read_text(encoding='utf-8'):
                    errors.append('Legacy file is not a forwarding entry point: ' + old)
            counts['canonical_python_asts'] += 1
        elif canonical.suffix in {'.json', '.html'}:
            if original.read_bytes() != canonical.read_bytes():
                errors.append('Declarative compatibility copy differs: ' + old)
            counts['declarative_byte_pairs'] += 1
    if len({r['path'] for r in records.values()}) != len(records):
        errors.append('Multiple owners of one canonical file')
    if nested:
        hook = root / 'tests/conftest.py'
        if not hook.is_file():
            errors.append('Missing legacy-alias pytest collection hook')
        else:
            ast.parse(hook.read_text(encoding='utf-8'))
        for folder in ('scripts', 'tests'):
            unexpected = [p.name for p in (root / folder).iterdir() if p.is_file() and p.name not in {'README.md', 'conftest.py', '__init__.py'}]
            if unexpected:
                errors.append(f'Unclassified flat files under {folder}: {unexpected}')
        tests = [Path(item['path']).stem for old, item in records.items() if old.startswith('tests/')]
        if len(tests) != len(set(tests)):
            errors.append('Test helper module names are ambiguous')
        counts['unique_test_modules'] = len(tests)
        pytest = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))['tool']['pytest']['ini_options']
        if pytest['testpaths'] != nested['pytest_discovery_roots'] or pytest['pythonpath'] != nested['test_module_directories']:
            errors.append('Pytest canonical discovery/helper paths differ from nested layout')
        discovered = [path.resolve() for directory in pytest['testpaths'] for path in (root / directory).rglob('test_*.py') if '__pycache__' not in path.parts]
        expected = {(root / item['path']).resolve() for old, item in records.items() if old.startswith('tests/')}
        if set(discovered) != expected or len(discovered) != len(expected):
            errors.append('Default discovery does not reach each canonical test module exactly once')
        counts['canonical_test_discovery_paths'] = len(discovered)
        for directory in set(nested['test_module_directories']) | {str(Path(p).parent) for p in nested['relocated_compatibility'].values()}:
            if not (root / directory / 'README.md').is_file():
                errors.append('Nested source directory lacks README: ' + directory)

    docs = list((root / 'experiments').rglob('*.md'))
    docs += [root / name for name in ['AGENTS.md', 'README.md', 'scripts/README.md']]
    docs += list((root / 'scripts').rglob('README.md'))
    docs += list((root / 'tests').rglob('README.md'))
    for directory in (root / 'experiments').rglob('*'):
        if directory.is_dir() and '__pycache__' not in directory.parts and not (directory / 'README.md').is_file():
            errors.append('Experiment folder lacks README: ' + str(directory))
    for document in docs:
        content = document.read_text(encoding='utf-8-sig')
        if sum(line.startswith('```') for line in content.splitlines()) % 2:
            errors.append('Unbalanced code fence: ' + str(document))
        for match in re.finditer(r'\[[^\]\n]+\]\(([^)]+)\)', content):
            target = match[1]
            if target.startswith(('http:', 'https:', 'mailto:', '#')) or '<' in target:
                continue
            file = target.split('#', 1)[0]
            if file and not (document.parent / file).exists():
                errors.append(f'Broken link: {document.relative_to(root)} -> {target}')
            counts['local_document_links'] += 1
        counts['documents'] += 1

    spec = importlib.util.spec_from_file_location('mosaic_layout_verification_runtime', root / 'scripts/layout/experiment_layout.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)  # stdlib-only infrastructure, no experiment imports
    runtime.activate(include_tests=True)
    if nested:
        helper_names = set()
        for old, item in records.items():
            if not old.startswith('tests/'):
                continue
            tree = ast.parse((root / item['path']).read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('test_'):
                    helper_names.add(node.module)
                elif isinstance(node, ast.Import):
                    helper_names.update(alias.name for alias in node.names if alias.name.startswith('test_'))
        for name in sorted(helper_names):
            if importlib.machinery.PathFinder.find_spec(name) is None:
                errors.append('Nested test helper cannot be resolved without import: ' + name)
        counts['test_helper_modules_resolved_without_import'] = len(helper_names)
    names = [Path(item['path']).stem for old, item in records.items() if old.endswith('.py') and not old.startswith('tests/')]
    if len(names) != len(set(names)):
        errors.append('Canonical script module names are ambiguous')
    for name in names:
        if importlib.machinery.PathFinder.find_spec(name) is None:
            errors.append('Canonical script module cannot be resolved: ' + name)
    counts['module_names_resolved_without_import'] = len(names)
    for old, item in records.items():
        if runtime.resolve(root / old) != root / item['path']:
            errors.append('Path resolver mismatch: ' + old)
    counts['working_path_resolutions'] = len(records)
    if nested:
        aliases = runtime.release_aliases(root)
        for old, data in aliases.items():
            if (root / old).exists():
                errors.append('Release-only flat alias leaked into working checkout: ' + old)
            if old.endswith('.py'):
                ast.parse(data, filename=old)
        counts['release_only_aliases_checked'] = len(aliases)
    with tempfile.TemporaryDirectory(prefix='mosaic-layout-verification-') as temporary:
        temp = Path(temporary)
        # An old sealed source must retain its legacy path, not acquire a new one.
        old = 'examples/configs/brd4_jq1_25_fixed_lrreset_protenix_10000.json'
        source = temp / 'old-source' / old
        source.parent.mkdir(parents=True)
        source.write_text('{}', encoding='utf-8')
        if runtime.resolve(source) != source:
            errors.append('Old sealed-source fallback changed')
        counts['sealed_source_fallback_checks'] = 1
        # Exercise only synthetic stdlib code, including dataclasses and patched globals.
        module_file = temp / 'synthetic.py'
        module_file.write_text('from dataclasses import dataclass\nVALUE = 3\n@dataclass\nclass Item:\n    value: int\ndef value():\n    return VALUE\n', encoding='utf-8')
        from types import ModuleType
        module = ModuleType('_mosaic_layout_synthetic')
        sys.modules[module.__name__] = module
        try:
            runtime.execute_legacy(module.__dict__, module_file)
            module.VALUE = 7
            if module.value() != 7 or module.Item(1).value != 1:
                errors.append('Legacy module globals/dataclass behavior changed')
            if module.__file__ != str(module_file):
                errors.append('Legacy source identity does not identify canonical file')
        finally:
            del sys.modules[module.__name__]
        counts['synthetic_compatibility_checks'] = 3
        # Extract only the stdlib source-identity function; never import Mosaic/io/JAX.
        provenance = ast.parse((root / 'src/mosaic/small_molecule/provenance.py').read_text(encoding='utf-8'))
        function = next(n for n in provenance.body if isinstance(n, ast.FunctionDef) and n.name == 'source_identity')
        namespace = {'Path': Path, 'json': json, 'subprocess': subprocess, 'file_sha256': digest, 'identity_digest': lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<stdlib source identity extraction>', 'exec'), namespace)
        source_root = temp / 'identity-source'
        source_root.mkdir()
        (source_root / 'pyproject.toml').write_text('[project]\nname="synthetic"\n', encoding='utf-8')
        (source_root / 'source_manifest.json').write_text('{"git_revision_reference":"synthetic"}', encoding='utf-8')
        experiment_file = source_root / 'experiments/family/study/scripts/driver.py'
        experiment_file.parent.mkdir(parents=True)
        experiment_file.write_text('VERSION = "before"\n', encoding='utf-8')
        before = namespace['source_identity'](source_root)
        experiment_file.write_text('VERSION = "after"\n', encoding='utf-8')
        after = namespace['source_identity'](source_root)
        if 'experiments/family/study/scripts/driver.py' not in before['files'] or before['execution_content_sha256'] == after['execution_content_sha256']:
            errors.append('Canonical experiment code is not covered by execution identity')
        counts['synthetic_execution_identity_checks'] = 2
        verifier_path = root / records['scripts/verify_stage0_release.py']['path']
        verifier_spec = importlib.util.spec_from_file_location('_mosaic_stdlib_source_verifier', verifier_path)
        verifier = importlib.util.module_from_spec(verifier_spec)
        verifier_spec.loader.exec_module(verifier)
        manifest_path = source_root / 'source_manifest.json'
        manifest_path.write_text(json.dumps({'files': {'pyproject.toml': digest(source_root / 'pyproject.toml'), 'experiments/family/study/scripts/driver.py': digest(experiment_file)}}), encoding='utf-8')
        prior_argv = sys.argv
        sys.argv = [str(verifier_path), str(source_root), digest(manifest_path)]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                verifier.main()
            extra = experiment_file.with_name('unmanifested.py')
            extra.write_text('# synthetic unmanifested input\n', encoding='utf-8')
            try:
                verifier.main()
            except ValueError as error:
                if 'Unmanifested execution input' not in str(error):
                    raise
            else:
                errors.append('Source authentication accepted unmanifested experiment code')
            extra.unlink()
            experiment_file.write_text('# synthetic changed input\n', encoding='utf-8')
            try:
                verifier.main()
            except ValueError as error:
                if 'Source content mismatch' not in str(error):
                    raise
            else:
                errors.append('Source authentication accepted changed experiment code')
        finally:
            sys.argv = prior_argv
        counts['synthetic_source_authentication_checks'] = 3

    git = shutil.which('git')
    bash = Path(git).resolve().parents[1] / 'bin/bash.exe' if git and sys.platform == 'win32' else Path(shutil.which('bash') or '')
    if not bash.is_file():
        errors.append('Bash syntax checker unavailable')
    else:
        shells = {root / old for old in records if old.endswith(('.lsf', '.sh'))}
        shells |= {root / item['path'] for old, item in records.items() if old.endswith(('.lsf', '.sh'))}
        for script in sorted(shells):
            check = subprocess.run([str(bash), '-n', script.as_posix()], capture_output=True, text=True)
            if check.returncode:
                errors.append(f'Bash syntax: {script.relative_to(root)}: {check.stderr.strip()}')
            counts['shell_syntax_checks'] += 1

    if baseline:
        state = json.loads((baseline / 'snapshot.json').read_text(encoding='utf-8'))
        for group in ['scientific_library', 'receipts']:
            for name, expected in state[group].items():
                if not (root / name).is_file() or digest(root / name) != expected:
                    if name == 'src/mosaic/small_molecule/provenance.py' and (root / name).is_file() and hashlib.sha256((root / name).read_bytes().replace(b'("src", "tests", "examples", "scripts", "experiments")', b'("src", "tests", "examples", "scripts")')).hexdigest() == expected:
                        metadata_only_source_changes.append({'path': name, 'change': 'Execution-source identity additionally includes experiments/; restoring only the enumeration tuple exactly reproduces the original file hash.'})
                    elif group == 'receipts' and name in live_receipts and (root / name).is_file():
                        concurrent_receipt_changes.append({'path': name, 'before_sha256': expected, 'observed_sha256': digest(root / name), 'scope': 'Explicitly named live operator log; retained in place. Its external writer continued during migration. No unchanged-byte claim.'})
                    else:
                        errors.append(f'Protected {group} file differs: {name}')
                counts['protected_' + group + '_files'] += 1
        # Ensure each scientific function retains its signature, non-path numeric
        # literals and guard messages. This is static evidence, not model parity.
        for old, item in records.items():
            if not old.endswith('.py'):
                continue
            before = ast.parse((baseline / old).read_text(encoding='utf-8-sig'))
            after = ast.parse((root / item['path']).read_text(encoding='utf-8-sig'))
            def numeric(tree):
                parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
                result = Counter()
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Constant) or not isinstance(node.value, (float, int)):
                        continue
                    ancestors = []
                    current = node
                    while current in parents:
                        current = parents[current]
                        ancestors.append(current)
                    if any(isinstance(p, ast.Subscript) and isinstance(p.value, ast.Attribute) and p.value.attr == 'parents' for p in ancestors):
                        continue
                    if any(isinstance(p, ast.Call) and ast.unparse(p.func).startswith('_layout_sys.') for p in ancestors):
                        continue
                    result[(type(node.value).__name__, repr(node.value))] += 1
                return result
            if numeric(before) != numeric(after):
                errors.append('Non-path numeric literals changed: ' + old)
            def signatures(tree):
                return [(n.name, ast.dump(n.args)) for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if signatures(before) != signatures(after):
                # Defaults containing a repository path legitimately change.
                def simple(tree):
                    return [(n.name, [a.arg for a in n.args.posonlyargs + n.args.args + n.args.kwonlyargs]) for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                if simple(before) != simple(after):
                    errors.append('Function interface changed: ' + old)
            def guards(tree):
                return [n.exc.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call) and n.exc.args and isinstance(n.exc.args[0], ast.Constant) and isinstance(n.exc.args[0].value, str)]
            if guards(before) != guards(after):
                errors.append('Existing guard messages changed: ' + old)
            counts['static_scientific_source_comparisons'] += 1
    return {'scope': 'stdlib layout/AST/byte/syntax checks only; no numerical/model/native checks or submissions', 'created_utc': datetime.now(timezone.utc).isoformat(), 'counts': dict(counts), 'canonical_content_changed_since_migration': changed_since_migration, 'metadata_only_source_changes': metadata_only_source_changes, 'concurrent_receipt_changes': concurrent_receipt_changes, 'errors': errors, 'status': 'passed' if not errors else 'failed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--live-receipt', action='append', default=[], help='Exact currently written operator log to report separately, never as unchanged.')
    args = parser.parse_args()
    result = verify(args.root.resolve(), args.baseline, args.live_receipt)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
