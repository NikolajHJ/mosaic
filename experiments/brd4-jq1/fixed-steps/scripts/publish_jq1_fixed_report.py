"""Publish independently byte-verified fixed-step JQ1 report assets.

Only standard-library byte, XML and JSON checks run here. Scientific frame
validation and rendering belong to the authenticated allocated report job.
Published editions are immutable; only the new fixed-report pointer changes.
"""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import xml.etree.ElementTree as ET

from jq1_fixed_report_contract import ARMS

VERSION = 'jq1-25-fixed-saved-report-v1'
VIEWER_KIND = 'jq1-fixed-step-viewer-v1'
EXPECTED_JOBS = [pin['job'] for pin in ARMS.values()]
HEX = re.compile(r'[0-9a-f]{64}')
EDITION = re.compile(r'edition-(?:[0-9]{4}|final(?:-[0-9]{4})?)')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def safe_name(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name and '\x00' not in name, 'Unsafe evidence/asset path')
    pure = PurePosixPath(name)
    require(not pure.is_absolute() and all(part not in ('', '.', '..') for part in pure.parts) and str(pure) == name, 'Unsafe evidence/asset path')
    require(all(part.rstrip(' .') == part and part.split('.')[0].upper() not in {'CON', 'PRN', 'AUX', 'NUL', *('COM'+str(i) for i in range(1, 10)), *('LPT'+str(i) for i in range(1, 10))} for part in pure.parts), 'Unsafe platform path')
    return pure


def safe_path(root, name):
    pure = safe_name(name)
    path = root.joinpath(*pure.parts)
    require(path.resolve().is_relative_to(root.resolve()), 'Evidence path escaped its root')
    for item in (path, *path.parents):
        if item == root.parent:
            break
        require(not item.is_symlink(), 'Evidence symlinks are forbidden')
    return path


def verify_tree(extracted):
    require(extracted.is_dir() and not extracted.is_symlink(), 'Missing verified extraction')
    download_path = extracted.with_name(extracted.name + '-download.json')
    require(download_path.is_file() and not download_path.is_symlink(), 'Require completed archive byte verification')
    downloaded = read(download_path)
    archive = Path(downloaded.get('archive', '')).resolve(strict=True)
    require(archive.is_file() and not archive.is_symlink() and HEX.fullmatch(downloaded.get('sha256', '')), 'Invalid verified archive receipt')
    require(digest(archive) == downloaded['sha256'], 'Verified archive bytes changed')
    manifest_path = extracted / 'evidence_manifest.json'
    require(manifest_path.is_file(), 'Missing evidence manifest')
    # Bind the extracted manifest to the exact downloaded archive, rather than
    # trusting an editable extraction/download marker by itself.
    with tarfile.open(archive, 'r:*') as handle:
        names = set()
        for member in handle.getmembers():
            name = member.name.rstrip('/') if member.isdir() else member.name
            safe_name(name)
            require(name not in names and (member.isdir() or member.isfile()), 'Unsafe archive member')
            names.add(name)
        member = handle.getmember('evidence_manifest.json')
        require(member.isfile(), 'Archive evidence manifest is not a regular file')
        require(handle.extractfile(member).read() == manifest_path.read_bytes(), 'Extracted evidence manifest differs from archive')
    manifest = read(manifest_path)['files']
    require(isinstance(manifest, dict) and manifest, 'Empty evidence manifest')
    require(downloaded.get('verified_files') == len(manifest), 'Download verified-file count differs')
    expected = set(manifest) | {'evidence_manifest.json'}
    actual = set()
    for path in extracted.rglob('*'):
        require(not path.is_symlink(), 'Extracted symlink is forbidden')
        if path.is_file():
            actual.add(path.relative_to(extracted).as_posix())
    require(actual == expected, 'Extracted file set differs from the verified evidence manifest')
    for name, entry in manifest.items():
        path = safe_path(extracted, name)
        require(type(entry.get('bytes')) is int and entry['bytes'] >= 0 and HEX.fullmatch(entry.get('sha256', '')), 'Malformed evidence byte receipt')
        require(path.is_file() and path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'], 'Evidence bytes changed: ' + name)
    return manifest, downloaded


def validate_report(report, extracted):
    receipt = read(report / 'report_receipt.json')
    require(receipt.get('version', receipt.get('report_version')) == VERSION and receipt.get('report_version', VERSION) == VERSION, 'Wrong fixed report version')
    require(receipt.get('input_jobs') == EXPECTED_JOBS and receipt.get('n_atoms') == 25, 'Wrong fixed scientific jobs or atom scope')
    require(type(receipt.get('complete')) is bool and receipt.get('chunk_size') == 100 and receipt.get('frame_stride_floats') == 1650, 'Wrong report frame contract')
    html = (report / 'report.html').read_text(encoding='utf-8')
    require((report / 'report.html').stat().st_size == receipt.get('html_bytes') and digest(report / 'report.html') == receipt.get('html_sha256'), 'HTML receipt bytes differ')
    match = re.search(r'<script\s+id="manifest"\s+type="application/json">(.*?)</script>', html, re.S)
    require(match is not None, 'Missing embedded saved report manifest')
    data = json.loads(match.group(1))
    require(data.get('version') == VERSION and data.get('selection_policy') == 'fixed_step' and data.get('n_atoms') == 25 and data.get('frame_stride_floats') == 1650 and data.get('chunk_size') == 100, 'Wrong embedded fixed frame contract')
    require(data.get('complete') is receipt['complete'] and isinstance(data.get('studies'), list) and len(data['studies']) == 3, 'Embedded report/receipt scope differs')
    require(receipt.get('states') == [s.get('states') for s in data['studies']] and receipt.get('included_updates') == [s.get('latest_update') for s in data['studies']], 'Embedded captured prefixes differ')
    require(isinstance(receipt.get('arm_bindings'), list) and len(receipt['arm_bindings']) == 3, 'Missing per-arm proof bindings')
    require(receipt.get('milestone_progress') == {s['arm']: (0 if s.get('latest_update') is None or s['latest_update'] < 1500 else 10000 if s['latest_update'] == 10000 else 1500 + ((s['latest_update'] - 1500) // 1000) * 1000) for s in data['studies']}, 'Published per-arm milestones differ')
    assets = {'report.html', 'report_receipt.json', 'loss-overlays.svg'}
    for study, binding, (arm, pin) in zip(data['studies'], receipt['arm_bindings'], ARMS.items()):
        fixed = {'arm': arm, 'job_id': pin['job'], 'learning_rate': pin['rate'], 'selection_policy': 'fixed_step', 'source_manifest_sha256': pin['manifest'], 'config_sha256': pin['config'], 'readiness_sha256': pin['ready']}
        require(all(study.get(k) == v and binding.get(k) == v for k, v in fixed.items()), 'Fixed scientific arm binding changed: ' + arm)
        require(study.get('native_status') == binding.get('native_status') and study.get('native_status') in ('PEND', 'RUN', 'DONE', 'EXIT', 'UNKNOWN'), 'Native scheduler status differs')
        states, latest, chunks = study.get('states'), study.get('latest_update'), study.get('chunks')
        require(type(states) is int and 0 <= states <= 10001 and isinstance(chunks, list), 'Invalid saved frame count')
        if not states:
            require(latest is None and chunks == [] and not receipt['complete'], 'An empty arm cannot contain invented frames or completion')
            require(not study.get('snapshot_steps'), 'Empty arm has snapshot steps')
            continue
        require(type(latest) is int and latest == states - 1 and study.get('snapshot_steps') in (None, list(range(states))), 'Sparse/noncontiguous saved updates are forbidden')
        require(study.get('counts', {}).get('applied') == latest and study.get('counts', {}).get('forced_fallback') == 0, 'Fixed-step counts differ')
        require(not receipt['complete'] or (states == 10001 and study['native_status'] == 'DONE'), 'A final report needs every full native completion')
        cursor = 0
        for chunk in chunks:
            require(type(chunk.get('count')) is int and 1 <= chunk['count'] <= 100 and chunk.get('start') == cursor, 'Unbounded/noncontiguous frame chunks')
            cursor += chunk['count']
            for kind in ('binary', 'metadata'):
                name = chunk.get(kind)
                require(isinstance(name, str) and len(PurePosixPath(name).parts) == 1, 'Chunk asset must be a safe basename')
                relative = arm + '/' + str(safe_name(name))
                require(relative not in assets, 'Duplicate frame asset')
                assets.add(relative)
                path = safe_path(report, relative)
                require(path.is_file() and digest(path) == chunk.get(kind + '_sha256'), 'Saved chunk checksum differs')
                if kind == 'binary':
                    require(path.stat().st_size == chunk['count'] * 1650 * 4, 'Saved probability byte length differs')
                else:
                    rows = read(path)
                    require(isinstance(rows, list) and [row.get('step') for row in rows] == list(range(chunk['start'], cursor)), 'Saved metadata steps differ')
            require(cursor <= states, 'Chunk extends beyond its captured prefix')
        require(cursor == states, 'Saved frames are missing')
        curve_name = arm + '/curves.json'
        assets.add(curve_name)
        curves = read(safe_path(report, curve_name))
        require(isinstance(curves, list) and [row.get('step') for row in curves] == list(range(states)), 'Saved scalar points are sparse/noncontiguous')
    actual = {p.relative_to(report).as_posix() for p in report.rglob('*') if p.is_file()}
    require(actual == assets, 'Unexpected or missing served report assets')
    payload = report.parent
    assessment = read(payload / 'assessment.json')
    require(assessment.get('status') == 'passed' and str(assessment.get('job_id', '')).isdigit(), 'Missing allocated report assessment')
    require(assessment.get('report_receipt_sha256') == digest(report / 'report_receipt.json'), 'Assessment does not bind this report receipt')
    require(assessment.get('complete') is receipt['complete'] and assessment.get('n_atoms') == 25 and assessment.get('states') == receipt['states'] and assessment.get('included_updates') == receipt['included_updates'], 'Assessment captured scope differs')
    require(assessment.get('input_jobs', assessment.get('native_jobs')) == EXPECTED_JOBS, 'Assessment scientific jobs differ')
    source_manifest = payload / 'report-source-manifest.json'
    require(source_manifest.is_file() and digest(source_manifest) == assessment.get('report_source_manifest_sha256'), 'Allocated report source manifest differs')
    cpu_dir = payload / 'cpu-proof'
    cpu = read(cpu_dir / 'assessment.json')
    require(cpu.get('status') == 'passed' and cpu.get('source_manifest_sha256') == assessment['report_source_manifest_sha256'], 'Report CPU certificate differs')
    require(cpu.get('suite') == 'jq1-fixed-saved-report-contract-v1', 'Wrong allocated report fixture suite')
    proofs = cpu.get('proofs')
    require(isinstance(proofs, list) and proofs, 'Missing report CPU proof bindings')
    require(assessment.get('cpu_certificate_sha256') == digest(cpu_dir / 'assessment.json'), 'Allocated report CPU certificate checksum differs')
    bundled = read(cpu_dir / 'bundled-proofs.json')
    require(isinstance(bundled, list) and len(bundled) == len(proofs), 'Missing bundled CPU proof paths')
    original = {p.get('path'): p.get('sha256') for p in proofs}
    require(len(original) == len(proofs) and all(isinstance(name, str) and HEX.fullmatch(value or '') for name, value in original.items()), 'Malformed original CPU proof bindings')
    seen, bound_paths = set(), []
    for proof in bundled:
        name, local, checksum = proof.get('original_path'), proof.get('local_path'), proof.get('sha256')
        require(name in original and name not in seen and original[name] == checksum, 'Bundled CPU proof identity differs')
        path = safe_path(payload, local)
        require(path.is_relative_to(cpu_dir) and path.is_file() and digest(path) == checksum, 'Bundled CPU proof bytes missing or changed')
        seen.add(name)
        bound_paths.append(path)
    cpu_job = str(cpu.get('producer_job_id', cpu.get('job_id', '')))
    require(cpu_job.isdigit(), 'Missing allocated report CPU job')
    texts = [p.read_text(encoding='utf-8', errors='replace') for p in bound_paths if p.suffix != '.xml']
    schedulers = [path for path in bound_paths if re.fullmatch(r'scheduler-final-\d+\.txt', path.name)]
    require(len(schedulers) == 1, 'Missing unique actual report CPU scheduler proof')
    # LSF wraps field names and values across lines; authenticate the compact
    # identity while retaining the original bound file bytes and hashes.
    compact = re.sub(r'\s+', '', schedulers[0].read_text(encoding='utf-8'))
    require(re.findall(r'Job<([^>]+)>', compact) == [cpu_job]
            and re.findall(r'Status<([^>]+)>', compact) == ['DONE']
            and re.findall(r'Queue<([^>]+)>', compact) == ['hpc'],
            'Missing actual DONE report CPU scheduler proof')
    require(any(re.search(r'Job\s*<?' + re.escape(cpu_job) + r'>?\b', text) and 'Successfully completed.' in text for text in texts), 'Missing same-job successful CPU LSF proof')
    xml_files = sorted({path for path in bound_paths if path.suffix == '.xml'})
    require(xml_files, 'Missing raw allocated report XML')
    cases = 0
    for xml in xml_files:
        tree = ET.parse(xml)
        rows = tree.findall('.//testcase')
        require(rows and not tree.findall('.//failure') and not tree.findall('.//error') and not tree.findall('.//skipped'), 'Raw report fixtures did not all pass')
        cases += len(rows)
    require(type(cpu.get('tests')) is int and cases == cpu['tests'] and cases > 0, 'Raw report fixture count differs')
    return receipt, data


def validate_viewer(viewer, extracted):
    require(not viewer.is_relative_to(extracted) and not extracted.is_relative_to(viewer), 'Viewer and verified input must be separate')
    require('20261021-brd4-jq1-25-report' not in viewer.parts, 'Historical report viewer must remain unchanged')
    for path in (viewer, *viewer.parents):
        require(not path.is_symlink(), 'Viewer symlinks are forbidden')
    marker = viewer / 'viewer-kind.json'
    if viewer.exists() and any(viewer.iterdir()):
        require(marker.is_file() and read(marker) == {'version': VIEWER_KIND, 'report_version': VERSION}, 'Require a separate fixed-step viewer')
    return marker


WRAPPER = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BRD4 JQ1 — latest fixed learning-rate progress</title><style>body{margin:0;font:14px system-ui;background:#eef3f6;color:#17313b}header{padding:12px 20px}iframe{border:0;width:100%;height:calc(100vh - 64px)}a{color:#246bb2}</style>
<header><strong>Latest verified fixed learning-rate JQ1 report</strong> · <span id="status">Loading…</span> · <a id="direct" target="_blank" rel="noopener">Open edition</a></header>
<iframe title="Verified saved fixed-step molecular trajectories" id="report"></iframe><script>
let current='';async function refresh(){try{const response=await fetch('latest.json',{cache:'no-store'});if(!response.ok)throw Error('Latest pointer unavailable');const saved=await response.json();if(saved.version!=='jq1-25-fixed-saved-report-v1'||!/^edition-(?:[0-9]{4}|final(?:-[0-9]{4})?)\\/report\\.html$/.test(saved.path))throw Error('Invalid fixed edition pointer');if(saved.path!==current){current=saved.path;document.getElementById('report').src=current;document.getElementById('direct').href=current;}document.getElementById('status').textContent=(saved.complete?'COMPLETE':'IN PROGRESS')+' · captured updates '+saved.included_updates.map(x=>x??'no saved state').join(' / ');}catch(error){document.getElementById('status').textContent=error.message;}}refresh();setInterval(refresh,60000);
</script></html>'''


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def publish(extracted, viewer, edition):
    require(isinstance(edition, str) and EDITION.fullmatch(edition), 'Require a safe immutable edition name')
    require(not Path(extracted).is_symlink() and not Path(viewer).is_symlink(), 'Root symlinks are forbidden')
    extracted = Path(extracted).resolve(strict=True)
    viewer = Path(viewer).absolute()
    validate_viewer(viewer, extracted)
    viewer = viewer.resolve()
    marker = validate_viewer(viewer, extracted)
    manifest, downloaded = verify_tree(extracted)
    candidates = list(extracted.glob('**/payload/report/report_receipt.json'))
    require(len(candidates) == 1, 'Require exactly one fixed report in this archive')
    report = candidates[0].parent
    receipt, data = validate_report(report, extracted)
    sources = sorted(p for p in report.rglob('*') if p.is_file())
    # Validate the entire source before creating a new immutable publication.
    for source in sources:
        require(source.relative_to(extracted).as_posix() in manifest, 'Unverified served report asset')
    destination = viewer / edition
    require(not destination.exists(), 'Immutable edition already exists')
    viewer.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        atomic_json(marker, {'version': VIEWER_KIND, 'report_version': VERSION})
    destination.mkdir(exist_ok=False)
    for source in sorted(sources, key=lambda p: p.name == 'report.html'):
        target = destination / source.relative_to(report)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        require(target.stat().st_size == source.stat().st_size and digest(target) == digest(source), 'Published asset copy differs')
    latest = {'version': VERSION, 'path': edition + '/report.html', 'complete': receipt['complete'],
              'input_jobs': EXPECTED_JOBS, 'included_updates': receipt['included_updates'], 'states': receipt['states'],
              'milestone_progress': receipt['milestone_progress'], 'html_sha256': receipt['html_sha256'],
              'report_receipt_sha256': digest(report / 'report_receipt.json'), 'published_utc': datetime.now(timezone.utc).isoformat()}
    atomic_json(destination / 'publication.json', latest | {'verified_input': str(extracted), 'verified_archive_sha256': downloaded['sha256'], 'verified_files_copied': len(sources)})
    wrapper_temp = viewer / 'report.html.tmp'
    with wrapper_temp.open('x', encoding='utf-8') as stream:
        stream.write(WRAPPER)
    wrapper_temp.replace(viewer / 'report.html')
    # The pointer changes only after every asset and the new wrapper are ready.
    atomic_json(viewer / 'latest.json', latest)
    return latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('extracted', type=Path)
    parser.add_argument('viewer', type=Path)
    parser.add_argument('edition')
    args = parser.parse_args()
    print(json.dumps(publish(args.extracted, args.viewer, args.edition)))


if __name__ == '__main__':
    main()
