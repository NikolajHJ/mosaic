"""Independent byte/XML/actual scheduler+LSF audit; no local fixtures or science."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import tarfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ROOT = '/work3/s234818/mosaic/runs/20261024-storage-offload-01'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--cases', type=int, required=True)
    args = parser.parse_args()
    raw_bundle = (REPO / 'results/hpc/20261024-storage-offload-01' / args.evidence / 'stdout.bin').read_bytes()
    bundle = json.loads(raw_bundle)
    files = {}
    for path, record in bundle['files'].items():
        raw = base64.b64decode(record['base64'], validate=True)
        assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256']
        files[path] = raw
    suffix = format(args.attempt, '02d')
    source, parent = ROOT + '/operator-source-' + suffix, ROOT + '/cpu-' + suffix
    package = json.loads((HERE / ('operator-package-' + suffix + '.json')).read_bytes())
    assert hashlib.sha256(files[source + '/source-manifest.json']).hexdigest() == package['manifest_sha256']
    manifest = json.loads(files[source + '/source-manifest.json'])
    assert len(manifest['files']) == 11
    archive = Path(package['archive'])
    assert archive.stat().st_size == package['archive_bytes']
    assert hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest() == package['archive_sha256']
    with tarfile.open(archive) as tar:
        entries = list(tar)
        assert len(entries) == 12 and len({x.name for x in entries}) == 12
        assert {x.name for x in entries} == set(manifest['files']) | {'source-manifest.json'}
        for item in entries:
            assert item.isfile() and not item.linkname
            assert tar.extractfile(item).read() == files[source + '/' + item.name]
    for name, digest in manifest['files'].items():
        assert hashlib.sha256(files[source + '/' + name]).hexdigest() == digest
    submission = json.loads(files[parent + '/submission.json'])
    assert submission['job_id'] == args.job and submission['outcome'] == 'submitted'
    expected = ['bsub', '-q', 'hpc', '-n', '2', '-R', 'span[hosts=1]', '-R', 'rusage[mem=4GB]',
        '-W', '00:30', '-J', 'mosaic-storage-offload-cpu-' + suffix,
        '-oo', parent + '/stdout.%J.log', '-eo', parent + '/stderr.%J.log',
        'bash', '-l', source + '/cleanup_cpu.lsf', package['manifest_sha256']]
    assert submission['args'] == expected and submission['source_manifest_sha256'] == package['manifest_sha256']
    raw_scheduler = bundle['scheduler'][0]
    scheduler = ' '.join(raw_scheduler['stdout'].split())
    assert raw_scheduler['returncode'] == 0 and 'Job <' + args.job + '>' in scheduler
    assert 'Status <DONE>' in scheduler and 'Queue <hpc>' in scheduler and 'Done successfully.' in scheduler
    log = files[parent + '/stdout.' + args.job + '.log'].decode()
    assert re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$', log, re.M) == [(args.job, 'Done')]
    assert 'Successfully completed.' in log
    rows = list(ET.fromstring(files[parent + '/payload/operator-contract.xml']).iter('testcase'))
    assert len(rows) == args.cases and len({x.attrib['name'] for x in rows}) == args.cases
    assert all(not list(x) and x.attrib['classname'].split('.')[-1] == 'fixture_backup_contract' for x in rows)
    assessment = json.loads(files[parent + '/payload/assessment.json'])
    assert assessment['status'] == 'passed' and assessment['job_id'] == args.job
    assert assessment['cases'] == args.cases and assessment['queue'] == 'hpc'
    assert assessment['source_manifest_sha256'] == package['manifest_sha256']
    for proof in assessment['proofs']:
        assert len(files[proof['path']]) == proof['bytes']
        assert hashlib.sha256(files[proof['path']]).hexdigest() == proof['sha256']
    gate = json.loads(files[parent + '/payload/storage-gate.json'])
    assert gate['gate']['status'] == 'passed' and gate['maximum_persistent_bytes'] == 128 * 1024**2
    assert gate['gate']['reserve_bytes'] == 4 * 1024**3 and gate['gate']['file_reserve'] == 100000
    assert gate['scientific_admission'] is False and gate['scientific_projection_changed'] is False
    result = {'version': 'mosaic-offload-allocated-operator-verification-v1', 'status': 'passed',
        'job_id': args.job, 'cases': args.cases, 'failures': 0, 'errors': 0, 'skips': 0,
        'source_manifest_sha256': package['manifest_sha256'],
        'bundle_sha256': hashlib.sha256(raw_bundle).hexdigest(),
        'proofs': {p: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} for p, raw in files.items()},
        'scientific_admission_or_result_claimed': False,
        'original_remote_files_deleted': 0,
        'scope': 'Allocated byte/metadata operator fixtures, actual job/source/resource/LSF/XML bindings. Scientific guards unchanged.'}
    with (HERE / ('operator-verification-' + suffix + '.json')).open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print('verified actual DONE/LSF; raw passing cases', args.cases, 'source_files11; original deletion0')


if __name__ == '__main__':
    main()
