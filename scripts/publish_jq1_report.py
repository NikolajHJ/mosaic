"""Publish byte-verified saved HTML assets; local stdlib file I/O only.

The input must already have passed verify_hpc_evidence.py. This step also checks
every copied report file against that evidence manifest, then changes the small
latest pointer atomically. It never reads numerical arrays or renders molecules.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('extracted', type=Path)
    parser.add_argument('viewer', type=Path)
    parser.add_argument('edition')
    args = parser.parse_args()
    extracted = args.extracted.resolve(strict=True)
    verified = extracted.with_name(extracted.name + '-download.json')
    if not verified.is_file() or not args.edition.startswith('edition-') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.edition):
        raise ValueError('Require completed byte verification and a safe new edition name')
    manifest = json.loads((extracted / 'evidence_manifest.json').read_text(encoding='utf-8'))['files']
    candidates = list(extracted.glob('**/payload/report/report_receipt.json'))
    if len(candidates) != 1:
        raise ValueError('Require one report in this verified archive')
    report = candidates[0].parent
    receipt = json.loads(candidates[0].read_text(encoding='utf-8'))
    if receipt['input_jobs'] != ['29770179', '29770093', '29770124'] or receipt['n_atoms'] != 25:
        raise ValueError('Wrong native studies or atom scope')
    if digest(report / 'report.html') != receipt['html_sha256'] or (report / 'report.html').stat().st_size != receipt['html_bytes']:
        raise ValueError('HTML receipt bytes differ')
    sources = sorted(p for p in report.rglob('*') if p.is_file())
    for source in sources:
        entry = manifest[source.relative_to(extracted).as_posix()]
        if source.is_symlink() or source.stat().st_size != entry['bytes'] or digest(source) != entry['sha256']:
            raise ValueError('Verified report bytes changed: ' + str(source))
    args.viewer.mkdir(parents=True, exist_ok=True)
    destination = args.viewer / args.edition
    destination.mkdir(exist_ok=False)
    # All external assets precede the HTML, preserving already published editions.
    for source in sorted(sources, key=lambda p: p.name == 'report.html'):
        target = destination / source.relative_to(report)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(target) != digest(source):
            raise ValueError('Published copy differs')
    latest = {'path': args.edition + '/report.html', 'complete': receipt['complete'],
              'included_updates': receipt['included_updates'], 'html_sha256': receipt['html_sha256'],
              'published_utc': datetime.now(timezone.utc).isoformat()}
    temporary = args.viewer / 'latest.json.tmp'
    temporary.write_text(json.dumps(latest, indent=2) + '\n', encoding='utf-8')
    temporary.replace(args.viewer / 'latest.json')
    wrapper = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BRD4 JQ1 — latest verified progress report</title><style>body{margin:0;font:14px system-ui;background:#eef3f6;color:#17313b}header{padding:12px 20px}iframe{border:0;width:100%;height:calc(100vh - 60px)}a{color:#246bb2}</style>
<header><strong>Latest verified 25-atom JQ1 report</strong> · <span id="status">Loading…</span> · <a id="direct" target="_blank">Open edition</a></header>
<iframe title="Verified saved molecular trajectories" id="report"></iframe><script>
let current='';async function refresh(){try{const response=await fetch('latest.json',{cache:'no-store'});if(!response.ok)throw Error('Latest pointer unavailable');const saved=await response.json();if(!/^edition-[a-z0-9-]+\\/report\\.html$/.test(saved.path))throw Error('Invalid edition path');if(saved.path!==current){current=saved.path;document.getElementById('report').src=current;document.getElementById('direct').href=current;}document.getElementById('status').textContent=(saved.complete?'COMPLETE':'IN PROGRESS')+' · included updates '+saved.included_updates.map(x=>x??'queued').join(' / ');}catch(error){document.getElementById('status').textContent=error.message;}}refresh();setInterval(refresh,60000);
</script></html>'''
    temporary = args.viewer / 'report.html.tmp'
    temporary.write_text(wrapper, encoding='utf-8')
    temporary.replace(args.viewer / 'report.html')
    (destination / 'publication.json').write_text(json.dumps(latest | {'verified_input': str(extracted), 'verified_files_copied': len(sources)}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(latest))


if __name__ == '__main__':
    main()
