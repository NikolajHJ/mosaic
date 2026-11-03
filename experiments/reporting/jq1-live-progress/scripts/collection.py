"""Assemble a focused navigation page over already-rendered saved reports.

No scientific rendering or data conversion: Mosaic assets stay in their canonical
immutable edition; external self-contained reports are copied byte-for-byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


OWNER = Path(__file__).resolve().parents[1]
REPO = OWNER.parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def local(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def checked(path, expected):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or sha(path) != expected:
        raise ValueError('Missing/changed report input: ' + str(path))
    return path


def assemble(config_path):
    config_path = Path(config_path).resolve()
    config = load(config_path)
    viewer = local(config['viewer']).resolve()
    name = config['collection']
    if not re.fullmatch(r'[a-z0-9-]+', name):
        raise ValueError('Unsafe collection name')
    destination = viewer / 'collections' / name
    if destination.exists():
        raise FileExistsError('Immutable collection exists; use a new collection name')
    pointer = load(viewer / 'latest.json')
    if not re.fullmatch(r'edition-[0-9]{4}', pointer['edition']):
        raise ValueError('Unsupported publication pointer')
    edition = viewer / 'editions' / pointer['edition']
    manifest_path = checked(edition / 'manifest.json', pointer['manifest_sha256'])
    manifest = load(manifest_path)
    receipt = load(edition / 'report-receipt.json')
    for filename in ('report.html', 'manifest.json', 'assessment.json', 'cpu-certificate.json'):
        row = receipt['files'][filename]
        path = checked(edition / filename, row['sha256'])
        if path.stat().st_size != row['bytes']:
            raise ValueError('Report byte length differs')
    arms = {arm['id']: arm for arm in manifest['arms']}
    ids = [row['id'] for row in config['runs']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate selected run')
    rows, attachments = [], []
    for row in config['runs']:
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', row['id']):
            raise ValueError('Unsafe run ID')
        value = dict(row)
        if 'attachment' in row:
            item = row['attachment']
            artifact = checked(local(item['path']), item['sha256'])
            proof = checked(local(item['manifest']), item['manifest_sha256'])
            if load(proof)['outputs']['index.html'] != item['sha256']:
                raise ValueError('External manifest does not bind the HTML')
            evidence = checked(local(item['evidence']), item['evidence_sha256'])
            value['url'] = 'attachments/' + row['id'] + '.html'
            value['renderer'] = 'external'
            attachments.append((artifact, value['url'], item['sha256']))
            attachments.append((proof, 'attachments/' + row['id'] + '-manifest.json', item['manifest_sha256']))
            attachments.append((evidence, 'attachments/' + row['id'] + '-evidence.json', item['evidence_sha256']))
            del value['attachment']
        else:
            arm = arms[row['id']]
            if arm['global_start'] != 0 or arm['latest_global_step'] != row['target']:
                raise ValueError('Selected run is not the requested complete history: ' + row['id'])
            value['url'] = '../../editions/' + pointer['edition'] + '/report.html?arm=' + row['id']
            value['renderer'] = 'mosaic'
            value['scientific_status'] = arm['scientific_status']
        rows.append(value)
    data = {k: config[k] for k in ('title', 'subtitle', 'date', 'highlights', 'comparison_note')}
    data.update(runs=rows, edition=pointer['edition'], full_report='../../report.html')
    template = (OWNER / 'templates/collection.html').read_text(encoding='utf-8')
    encoded = json.dumps(data, ensure_ascii=True).replace('<', '\\u003c')
    html = template.replace('/*COLLECTION_DATA*/null', encoded)
    # Validate everything before creating a new immutable output directory.
    destination.mkdir(parents=True, exist_ok=False)
    output_files = {}
    for source, relative, digest in attachments:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
        checked(target, digest)
        output_files[relative] = {'bytes': target.stat().st_size, 'sha256': digest}
    for filename, value in (('report.html', html), ('selection.json', json.dumps(data, indent=2))):
        target = destination / filename
        with target.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(value)
        output_files[filename] = {'bytes': target.stat().st_size, 'sha256': sha(target)}
    proof = {
        'version': 'mosaic-report-collection-v1',
        'execution': 'metadata navigation and byte-preserving report packaging only',
        'config_sha256': sha(config_path),
        'canonical_edition': pointer['edition'],
        'canonical_manifest_sha256': sha(manifest_path),
        'canonical_receipt_sha256': sha(edition / 'report-receipt.json'),
        'canonical_render_job': receipt['job_id'],
        'included_runs': ids,
        'omitted_mosaic_runs': [key for key in arms if key not in ids],
        'files': output_files,
        'saved_results_modified': False,
        'scientific_certification_claim': False,
    }
    with (destination / 'collection-receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(proof, stream, indent=2)
    return {'output': str(destination), 'runs': len(rows), 'edition': pointer['edition'],
            'url': config['base_url'].rstrip('/') + '/collections/' + name + '/report.html'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.config), indent=2))


if __name__ == '__main__':
    main()
