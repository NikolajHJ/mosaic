"""Stdlib-only audit of a failed startup; never a scientific gradient pass."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
ROOT = BASE + '/boltz-dbmol-pilot-01'
SCIENCE = BASE + '/boltz-dbmol-pilot-release-01/source'
LAUNCHER = BASE + '/boltz-dbmol-storage-release-04/source'
MANIFEST = '24c207da5510fa579576a04ad32bf1afc6e1f23f51dcb5251d82a4ebdfa8f436'
LAUNCHER_MANIFEST = 'df89def61047469e0db8ee6f4b235c75344dd4ad9f35a0833b0839f23859e9fa'
ARCHIVE_SHA = 'cfe7d8e0b80c455b84474769c46b7551b66588eeeee9ad327500fe6ab8ce918b'
SCIENCE_ARCHIVE_SHA = 'ec333fd3e0b33d9e5b82e0cd5e4e12c14ffb4f91782c18cd820d4bfab53de1f8'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    folder = REPO / 'results/hpc' / HERE.name / 'cpu-evidence-02'
    receipt = read(folder / 'delivery-receipt.json')
    archive_path = folder.with_suffix('.tar.gz')
    require(receipt['sha256'] == ARCHIVE_SHA and archive_path.stat().st_size == receipt['bytes']
            and digest(archive_path.read_bytes()) == ARCHIVE_SHA, 'Delivered archive identity differs')
    metadata = read(folder / 'metadata.json')
    files = metadata['files']
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        require(len(members) == len(files) + 1, 'Delivered inventory count differs')
        seen = set()
        for member in members:
            path = PurePosixPath(member.name)
            require(member.isfile() and not path.is_absolute() and '..' not in path.parts
                    and '\\' not in member.name and member.name not in seen, 'Unsafe delivered member')
            seen.add(member.name)
            value = archive.extractfile(member).read()
            require((folder / member.name).read_bytes() == value, 'Extracted file differs')
            if member.name != 'metadata.json':
                item = files[member.name]
                require(len(value) == item['bytes'] and digest(value) == item['sha256'], 'Proof changed')
        require(seen == set(files) | {'metadata.json'}, 'Delivered member scope differs')
    cpu_path = HERE / 'cpu-verification-02.json'
    cpu = read(cpu_path)
    require(cpu['status'] == 'passed' and cpu['cpu_cases'] == 1135
            and cpu['cpu_certificate_sha256'] == '14199ca0444d614385c5bda4863fd449e063a34540085513c3985080e02b5880'
            and cpu['evidence_archive_sha256'] == ARCHIVE_SHA, 'Actual CPU pass not authenticated')
    source_archive = REPO / 'results/hpc/20261023-boltz-dbmol-pilot-01/source01/source.tar.gz'
    require(digest(source_archive.read_bytes()) == SCIENCE_ARCHIVE_SHA, 'Scientific archive changed')
    with tarfile.open(source_archive) as archive:
        manifest_raw = archive.extractfile('source/source_manifest.json').read()
        manifest = json.loads(manifest_raw)
        require(digest(manifest_raw) == MANIFEST and len(manifest['files']) == 657
                and 'git_revision_reference' not in manifest, 'Missing manifest-field diagnosis differs')
        raw_driver = archive.extractfile('source/scripts/dbmol_driver.py').read()
        raw_provenance = archive.extractfile('source/src/mosaic/small_molecule/provenance.py').read()
        require(digest(raw_driver) == manifest['files']['scripts/dbmol_driver.py']
                and digest(raw_provenance) == manifest['files']['src/mosaic/small_molecule/provenance.py'],
                'Driver/provenance bytes changed')
    driver_tree = ast.parse(raw_driver)
    calls = [(node.func.id, node.lineno) for node in ast.walk(driver_tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    source_line = [line for name, line in calls if name == 'source_identity']
    worker_line = [line for name, line in calls if name == 'DBMolWorker']
    require(len(source_line) == len(worker_line) == 1 and source_line[0] < worker_line[0],
            'Failure no longer precedes worker construction')
    require(b'"git_revision_reference"' in raw_provenance, 'Source metadata lookup differs')
    jobs = read(folder / 'run/jobs.json')
    require(jobs['pilot-native-0000'] == '29802648' and jobs['pilot-after-native-0000'] == '29802649'
            and not any(name.startswith('pilot-segment-') for name in jobs), 'Unexpected production job')
    proofs = {}
    times = {}
    for label, job, phase, prefix in [('pilot-native-0000', '29802648', 'native', 'mosaic-dbmol-'),
                                      ('pilot-after-native-0000', '29802649', 'after-native', 'mosaic-dbmol-')]:
        parent = folder / 'run' / label
        submission = read(parent / 'submission.json')
        require(submission['job_id'] == job and submission['outcome'] == 'submitted'
                and submission['phase'] == phase
                and submission['args'][-3:] == [MANIFEST, phase, ROOT + '/' + label],
                'Actual source/phase/submission differs')
        args = submission['args']
        require(args[:5] == (['bsub', '-q', 'gpul40s', '-n', '4'] if phase == 'native'
                            else ['bsub', '-q', 'hpc', '-n', '2'])
                and args[args.index('-J') + 1] == prefix + label
                and LAUNCHER_MANIFEST in args and SCIENCE in args and LAUNCHER in args,
                'Actual resource/source vector differs')
        require(submission['dependency'] == (None if phase == 'native' else 'ended(29802648)'),
                'Own-ended dependency differs')
        stdout = (parent / ('stdout.' + job + '.log')).read_text(encoding='utf-8')
        stderr = (parent / ('stderr.' + job + '.log')).read_text(encoding='utf-8')
        require(re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$', stdout, re.M)
                == [(job, 'Exited')] and 'Exited with exit code 1.' in stdout,
                'Same-job actual LSF failure missing')
        query = metadata['scheduler'][label]
        require(query['job_id'] == job and re.fullmatch(r'Job\s+<' + job + r'>\s+is\s+not\s+found\s*', query['raw']),
                'Aged-out live history is not authenticated')
        if phase == 'native':
            require("KeyError: 'git_revision_reference'" in stderr and 'in source_identity' in stderr
                    and 'dbmol_driver.py", line ' + str(source_line[0]) in stderr,
                    'Actual pre-worker metadata failure differs')
            require("'--mode', 'validate'" in stderr and "'--offset', '0'" in stderr
                    and "'--segment-steps', '2500'" in stderr, 'Actual driver scope differs')
            gate = read(parent / 'storage-entry/gate.json')
            require(gate['status'] == 'passed', 'Producer did not pass entry storage')
        else:
            require('ValueError: Producer failed' in stderr, 'Callback failure differs')
        times[label] = {name: re.findall(r'^' + name + r' at (.+)$', stdout, re.M)
                        for name in ['Started', 'Terminated']}
        for tail in ['submission.json', 'stdout.' + job + '.log', 'stderr.' + job + '.log']:
            name = 'run/' + label + '/' + tail
            proofs[name] = files[name]
    require('run/projection.json' not in files
            and not any(name.endswith('/assessment.json') and name.startswith('run/pilot-native-0000/') for name in files),
            'An unexpected native projection/readiness needs inspection')
    value = {'status': 'failed_native_startup_evidence_verified', 'verified_utc': datetime.now(timezone.utc).isoformat(),
             'observed_utc': metadata['observed_utc'], 'source_manifest_sha256': MANIFEST,
             'launcher_manifest_sha256': LAUNCHER_MANIFEST, 'cpu_recheck_passed': True, 'cpu_cases': 1135,
             'cpu_verification_sha256': digest(cpu_path.read_bytes()), 'files_verified': len(files),
             'evidence_archive_bytes': receipt['bytes'], 'evidence_archive_sha256': ARCHIVE_SHA,
             'native_job': '29802648', 'callback_job': '29802649', 'persisted_lsf_status': 'Exited',
             'persisted_lsf_exit_code': 1, 'live_scheduler_status': 'unavailable_live_history',
             'failure': "KeyError: 'git_revision_reference'", 'failure_kind': 'source_manifest_startup_metadata',
             'source_metadata_line': source_line[0], 'worker_constructor_line': worker_line[0],
             'native_model_worker_reached': False, 'native_gradient_checks_run': False,
             'native_gradient_passed': False, 'optimization_started': False, 'production_submissions': [],
             'original_saved_results_changed': False, 'new_jobs_submitted_by_this_audit': False,
             'native_failure_times_lsf_dtulocal': times, 'proofs': proofs,
             'scope': 'Independent stdlib byte/XML/source-AST/metadata/submission/LSF audit only. No numerical evaluation, native restart or gate waiver.'}
    with (HERE / 'native-failure-verification-01.json').open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2);stream.write('\n')
    print(json.dumps({key: value[key] for key in ['status', 'cpu_cases', 'native_job', 'failure', 'native_gradient_checks_run', 'optimization_started']}))


if __name__ == '__main__':
    main()
