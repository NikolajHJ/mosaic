"""Independent stdlib source/raw-proof/schema audit of the metadata correction."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile

import round2_verify as original

PIN = '00e223b9b453aa0e390f315951eee3406916ac8c8499284945967a72fbf38698'
ARCHIVE = 'ae2031e920ec7d40058a04215fd4c3bb767e16832672538f12e11b9a5564348b'
BASE = original.BASE
ROOT = BASE + 'chemistry-search-speed-gpu-adoption-03'
SOURCE = BASE + 'chemistry-search-speed-gpu-adoption-release-03/source'


def read(p):
    return json.loads(Path(p).read_bytes())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compact(text):
    return ''.join(text.split())


def archive_check(path):
    raw = Path(path).read_bytes()
    assert len(raw) == 9050 and hashlib.sha256(raw).hexdigest() == ARCHIVE
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        members = t.getmembers()
        assert len(members) == len({m.name for m in members}) == 6
        for m in members:
            assert m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts
        raw_manifest = t.extractfile('source_manifest.json').read()
        assert hashlib.sha256(raw_manifest).hexdigest() == PIN
        manifest = json.loads(raw_manifest)
        assert set(manifest['files']) == {m.name for m in members} - {'source_manifest.json'}
        for name, digest in manifest['files'].items():
            assert hashlib.sha256(t.extractfile(name).read()).hexdigest() == digest
        assert manifest['metadata_cases'] == 19 and manifest['scientific_source_changed'] is False
        assert manifest['parent_source_manifest_sha256'] == original.PIN
    return manifest


def command(phase, dependency=None):
    args = ['bsub', '-q', 'hpc', '-n', '2' if phase == 'cpu' else '1', '-R', 'span[hosts=1]',
            '-R', 'rusage[mem=2GB]', '-W', '00:30' if phase == 'cpu' else '00:20',
            '-J', 'mosaic-chem-speed-gpu-adoption03-' + phase,
            '-oo', ROOT + '/' + phase + '/stdout.%J.log', '-eo', ROOT + '/' + phase + '/stderr.%J.log']
    if dependency:
        assert phase == 'after-cpu' and re.fullmatch(r'ended\(\d+\)', dependency)
        args += ['-w', dependency]
    return args + ['bash', '-l', SOURCE + '/scripts/gpu_adoption_stage.lsf', SOURCE, PIN, phase]


def actual_terminal(data, files, phase):
    job = data['jobs'][phase]; q = data['scheduler'][phase]
    assert q['job_id'] == job and q['returncode'] == 0
    text = compact(q['stdout'])
    assert re.findall(r'Job<(\d+)>', text) == [job]
    assert re.findall(r'Status<([^>]+)>', text) == ['DONE']
    assert re.findall(r'Queue<([^>]+)>', text) == ['hpc']
    assert re.findall(r'JobName<([^>]+)>', text) == ['mosaic-chem-speed-gpu-adoption03-' + phase]
    log = files / phase / ('stdout.' + job + '.log'); value = compact(log.read_text())
    assert value.count(f'Subject:Job{job}:<mosaic-chem-speed-gpu-adoption03-{phase}>incluster<dcc>Done') == 1
    assert value.count('Successfullycompleted.') == 1 and 'inqueue<hpc>' in value
    return sha(log)


def main():
    observation = Path(sys.argv[1]).resolve()
    data = read(observation / 'adoption/raw.json'); files = observation / 'adoption/files'
    parent = read(observation / 'parent/raw.json'); parent_files = observation / 'parent/files'
    for block, directory in [(data, files), (parent, parent_files)]:
        for name, row in block['files'].items():
            rel = PurePosixPath(name)
            assert not rel.is_absolute() and '..' not in rel.parts and ':' not in name and '\\' not in name
            p = directory / name
            assert p.is_file() and not p.is_symlink() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256']
    manifest = archive_check(files / 'source.tar.gz')
    old_root = BASE + 'chemistry-search-speed-02/'
    for proof in manifest['preserved_callback_failure']:
        assert proof['path'].startswith(old_root)
        assert sha(parent_files / proof['path'][len(old_root):]) == proof['sha256']
    original.archive_check(parent_files / 'source.tar.gz')
    initial = read(files / 'initial-submissions.json')
    assert initial['manifest_sha256'] == PIN and initial['archive_sha256'] == ARCHIVE and initial['archive_bytes'] == 9050
    original.storage_check(initial['storage'])
    assert data['jobs'] == initial['jobs'] == {'cpu': '29809552', 'after-cpu': '29809553'}
    for phase, job in data['jobs'].items():
        sub = read(files / phase / 'submission.json')
        assert sub['job_id'] == job and sub['status'] == 'submitted' and sub['returncode'] == 0
        assert sub['manifest_sha256'] == PIN and sub['argv'] == command(phase, sub['dependency'])
        assert sub == initial['submissions'][phase]
    assert initial['submissions']['after-cpu']['dependency'] == 'ended(29809552)'
    cpu = read(files / 'cpu/payload/assessment.json')
    assert cpu['status'] == 'passed_metadata_checks_original_CPU_adopted' and cpu['job_id'] == '29809552'
    assert cpu['metadata_cases'] == 19 and cpu['manifest_sha256'] == PIN
    assert cpu['scientific_source_changed'] is False and cpu['original_callback_status'] == 'failed_preserved'
    assert cpu['gpu_submitted'] is False
    assert sha(files / 'cpu/payload/tests.xml') == cpu['raw_xml_sha256']
    original.xml_check(files / 'cpu/payload/tests.xml', {'tests.test_gpu_adoption_terminal': 19})
    original_lsf = actual_terminal(data, files, 'cpu')
    callback = read(files / 'after-cpu/payload/assessment.json')
    assert callback['status'] == 'metadata_checks_passed_GPU_submitted_not_passed'
    assert callback['job_id'] == '29809553' and callback['manifest_sha256'] == PIN
    assert callback['adoption_cpu_lsf_sha256'] == original_lsf
    assert callback['adoption_cpu_assessment_sha256'] == sha(files / 'cpu/payload/assessment.json')
    assert callback['raw_xml_sha256'] == cpu['raw_xml_sha256'] and callback['metadata_cases'] == 19
    assert callback['scientific_source_changed'] is False and callback['gpu_submitted'] is True
    actual_terminal(data, files, 'after-cpu')
    original.storage_check(callback['storage'])
    assert callback['plan_sha256'] == cpu['plan_sha256'] == sha(parent_files / 'plan.json')
    for key, rel in [('original_cpu_assessment_sha256', 'cpu/payload/assessment.json'),
                     ('original_cpu_lsf_sha256', 'cpu/stdout.29809477.log'),
                     ('original_cpu_xml_sha256', 'cpu/payload/tests.xml'),
                     ('original_benchmark_sha256', 'cpu/payload/benchmark/assessment.json')]:
        assert callback[key] == cpu[key] == sha(parent_files / rel)
    assert callback['gpu'] == read(parent_files / 'gpu/submission.json')
    gpu = callback['gpu']; assert gpu['manifest_sha256'] == original.PIN
    assert gpu['argv'] == original.command('gpu') and gpu['status'] == 'submitted' and gpu['returncode'] == 0
    assert parent['jobs']['gpu'] == gpu['job_id']
    assert re.findall(r'Job <(\d+)> is submitted', gpu['stdout']) == [gpu['job_id']]
    assert callback['combined_current_bytes'] < 1024**3
    report = {'status': 'passed_independent_metadata_source_byte_XML_LSF_dispatch_audit',
              'captured_utc': data['captured_utc'], 'source_manifest_sha256': PIN, 'source_files': 5,
              'raw_metadata_cases': 19, 'original_numerical_cases_adopted': 143,
              'actual_metadata_CPU_job': data['jobs']['cpu'], 'actual_callback_job': data['jobs']['after-cpu'],
              'gpu_job': gpu['job_id'], 'gpu_status_claim': 'actually_submitted_not_a_GPU_execution_pass',
              'actual_GPU_runtime_source_manifest_sha256': original.PIN,
              'cpu_assessment_sha256': sha(files / 'cpu/payload/assessment.json'),
              'callback_assessment_sha256': sha(files / 'after-cpu/payload/assessment.json'),
              'raw_observation_sha256': sha(observation / 'raw.json'), 'verifier_sha256': sha(__file__),
              'storage_required_bytes': callback['storage']['required_bytes'],
              'storage_available_bytes': callback['storage']['available_bytes'],
              'original_failure_preserved': True, 'original_scientific_runs_changed': False,
              'scope': 'Stdlib artifact/schema audit; no local scientific evaluation'}
    with (observation / 'verification.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
