"""Audit documentation files, local references and account assumptions without models.

Reads all repository documentation text and supported document attachments.
It does not connect to HPC, check remote ACLs, execute documented commands,
or certify scientific outcomes or imported organizational policies.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote
import xml.etree.ElementTree as ET
import zipfile

TEXT = {'.md', '.tex', '.rst', '.txt'}
BINARY = {'.pdf', '.docx', '.pptx'}
CURRENT_GUIDES = {
    'docs/HPC_GUIDE.md', 'docs/codex_dtu_hpc_integration.md',
    'docs/codex_HPC_Katinka.md', 'docs/README.md',
}
ACCOUNT = re.compile(r'/work3/(?!(?:YOUR|USER)\b)[A-Za-z_][\w.-]*(?:/[^\s`"<>]*)?|/zhome/\d+/[a-z]/\d+(?:/[^\s`"<>]*)?|C:[/\\]Users[/\\][^\s`"<>]+|\bUser\s+s\d{6}\b')
CODE_PATH = re.compile(r'(?<![\w/])(?:scripts|examples|experiments|src|tests|docs)/[\w./+-]+\.(?:py|lsf|sh|json|html|toml|pdf|md)(?![\w/])')
LINK = re.compile(r'(?<!!)\[[^\]\n]+\]\((<[^>]+>|[^)\n]+)\)|!\[[^\]\n]*\]\((<[^>]+>|[^)\n]+)\)')
ILLUSTRATIVE_PATHS = {'docs/HPC_HELP.tex': {'scripts/run_pipeline.sh'}}


def category(name):
    if name.startswith('docs/onenote_export/'):
        return 'imported_reference_snapshot'
    if name.startswith('scripts/hpc/receipts/'):
        return 'historical_submission_evidence'
    if '/runs/' in name or name == 'docs/experiment_ledger.md' or name.startswith('docs/history/'):
        return 'historical_run_or_workstation_record'
    if name in CURRENT_GUIDES:
        return 'current_reusable_guide'
    if name == 'docs/HPC_HELP.tex':
        return 'general_reference_with_project_caveats'
    if name == 'docs/stage_0_usage.md':
        return 'mixed_usage_and_dated_evidence'
    if name.startswith('docs/') and name not in {'docs/DOCUMENTATION_AUDIT.md'}:
        return 'dated_protocol_or_project_reference'
    return 'repository_navigation_or_policy'


def extract(path):
    if path.suffix.lower() in TEXT:
        return path.read_text(encoding='utf-8-sig'), 'text_read'
    if path.suffix.lower() in {'.docx', '.pptx'}:
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name == 'word/document.xml' or re.fullmatch(r'ppt/(?:slides/slide|notesSlides/notesSlide)\d+\.xml', name)]
            chunks = []
            for member in members:
                node = ET.fromstring(archive.read(member))
                chunks.extend(n.text or '' for n in node.iter() if n.tag.rsplit('}', 1)[-1] == 't')
        return '\n'.join(chunks), 'ooxml_text_extracted_not_visual_or_policy_validation'
    tool = shutil.which('pdftotext')
    if not tool:
        return '', 'pdf_text_extractor_unavailable'
    run = subprocess.run([tool, '-layout', str(path), '-'], capture_output=True, timeout=60)
    if run.returncode:
        return '', 'pdf_text_extraction_failed'
    text = run.stdout.decode('utf-8', errors='replace')
    return text, 'pdf_text_extracted_not_visual_validation' if text.strip() else 'pdf_has_no_extractable_text'


def audit(root):
    files = set(root.glob('*.md'))
    for folder in ['docs', 'experiments', 'scripts', 'examples', 'tests', 'src']:
        files.update(p for p in (root / folder).rglob('*') if p.is_file() and p.suffix.lower() in TEXT | BINARY and '__pycache__' not in p.parts)
    migration = json.loads((root / 'experiments/migration-map.json').read_text(encoding='utf-8'))['files']
    nested = root / 'experiments/folder-layout.json'
    if nested.is_file():
        migration.update(json.loads(nested.read_text(encoding='utf-8'))['shared_tests'])
    rows, failures, counts = [], [], Counter()
    for path in sorted(files):
        name = path.relative_to(root).as_posix()
        kind = category(name)
        try:
            raw = path.read_bytes()
            if path.suffix.lower() in TEXT:
                content, scope = raw.decode('utf-8-sig'), 'text_read'
            else:
                content, scope = extract(path)
        except (OSError, UnicodeError, zipfile.BadZipFile, ET.ParseError, subprocess.TimeoutExpired) as error:
            raw, content, scope = b'', '', 'unreadable: ' + type(error).__name__
        account_refs = [{'line': content.count('\n', 0, match.start()) + 1, 'reference': match[0]} for match in ACCOUNT.finditer(content)]
        broken, legacy, missing_code, illustrative = [], [], [], []
        # Markdown links within illustrative fenced examples are not actual links.
        link_content = re.sub(r'(?ms)^```[^\n]*\n.*?^```\s*$', '', content) if path.suffix == '.md' else ''
        for match in LINK.finditer(link_content):
            target = match[1] or match[2]
            target = target[1:-1] if target.startswith('<') and target.endswith('>') else target
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith(('#', '//')):
                continue
            target = target.split(' "', 1)[0]
            file = unquote(target.split('#', 1)[0])
            if not file or any(token in file for token in ['<', '>', '{', '}', '*', '$']):
                continue
            resolved = root / file.lstrip('/') if file.startswith('/') else path.parent / file
            if not resolved.exists():
                broken.append({'target': target, 'line': link_content.count('\n', 0, match.start()) + 1})
        for match in CODE_PATH.finditer(content):
            target = match[0]
            if target in migration and migration[target]['path'] != target:
                legacy.append(target)
            elif target in ILLUSTRATIVE_PATHS.get(name, set()):
                illustrative.append(target)
            elif not (root / target).exists() and not (path.parent / target).exists():
                missing_code.append(target)
        row = {'path': name, 'category': kind, 'sha256': hashlib.sha256(raw).hexdigest(), 'read_scope': scope, 'characters_read': len(content), 'account_references': account_refs, 'broken_markdown_links': broken, 'legacy_compatibility_references': sorted(set(legacy)), 'unresolved_literal_repository_paths': sorted(set(missing_code)), 'illustrative_non_repository_paths': sorted(set(illustrative))}
        if name in CURRENT_GUIDES and account_refs:
            failures.append('Current reusable guide contains concrete account defaults: ' + name)
        if name in CURRENT_GUIDES and broken:
            failures.append('Current reusable guide contains broken links: ' + name)
        rows.append(row)
        counts[kind] += 1
    bound_sources = []
    for old, item in migration.items():
        path = root / item['path']
        if path.suffix not in {'.py', '.lsf', '.sh'}:
            continue
        matches = list(ACCOUNT.finditer(path.read_text(encoding='utf-8-sig')))
        if matches:
            bound_sources.append({'path': item['path'], 'owner': item['owner'], 'account_reference_count': len(matches)})
    return {
        'version': 'documentation-portability-audit-v1',
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'All discovered documentation text and supported attachments read; local links/source references and account assumptions scanned. No HPC access, remote permission check, command execution, scientific validation or imported organizational-policy validation.',
        'counts': {'documents': len(rows), 'by_category': dict(counts), 'account_bound_canonical_sources': len(bound_sources), 'documents_with_account_references': sum(bool(r['account_references']) for r in rows), 'broken_markdown_links': sum(len(r['broken_markdown_links']) for r in rows), 'unresolved_literal_repository_paths': sum(len(r['unresolved_literal_repository_paths']) for r in rows)},
        'account_bound_canonical_sources': bound_sources,
        'documents': rows,
        'current_guide_failures': failures,
        'status': 'passed_with_documented_historical_or_imported_findings' if not failures else 'failed_current_guides',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'counts': result['counts'], 'current_guide_failures': result['current_guide_failures'], 'status': result['status']}, indent=2))
    raise SystemExit(1 if result['current_guide_failures'] else 0)


if __name__ == '__main__':
    main()
