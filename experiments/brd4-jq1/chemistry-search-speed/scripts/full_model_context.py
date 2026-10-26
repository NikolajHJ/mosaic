"""Allocated metadata adoption of the exact isolated replay producer.

No JAX/model/array is imported. This can propose a compiler environment only
for a NEW native pilot, never certify or restart any failed saved study.
"""
import json
import re
import time
from pathlib import Path

BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
PRODUCER = {'root': BASE+'/chemistry-replay-02', 'source': BASE+'/chemistry-replay-release-02/source',
            'manifest_sha256': '89b290831e8ebf5633dba07736189a18848cd835d1d6cdf8a3650a5f056a928b',
            'cpu_certificate_sha256': '0241b63c7ebc62dff3a50ae2b20ac0b767ca3e883bc7b2bb7aca46a44ef722d7',
            'gpu_job_id': '29818513'}


def producer(ops, manifest):
    ops.require(manifest.get('replay_context_producer') == PRODUCER,
                'Only the actual declared replay02 producer may qualify a context')
    source = Path(PRODUCER['source']); root = Path(PRODUCER['root'])
    own = ops.verify(source, PRODUCER['manifest_sha256'])
    ops.require(own['version'] == 'chemistry-softmax-replay-source-v2' and len(own['files']) == 11,
                'Replay producer source scope differs')
    certificate = root/'cpu-certificate.json'
    ops.require(ops.sha(certificate) == PRODUCER['cpu_certificate_sha256'],
                'Actual replay CPU certificate differs')
    replay = ops._module(source/'scripts/replay_ops.py', '_native_context_replay_metadata')
    cert = replay.verify_cpu(PRODUCER['manifest_sha256'], own)
    ops.require(cert['cases'] == 67 and cert['job_id'] == '29818483', 'Replay CPU67 identity differs')
    sub = json.loads((root/'gpu/submission.json').read_bytes())
    ops.require(sub['status'] == 'submitted' and sub['job_id'] == PRODUCER['gpu_job_id']
                and sub['source_manifest_sha256'] == PRODUCER['manifest_sha256']
                and sub['argv'] == replay.argv('gpu', PRODUCER['manifest_sha256']),
                'Replay actual GPU submission/resources differ')
    return root, source, own, replay


def qualify_predecessor(ops, manifest):
    root, source, own, replay = producer(ops, manifest)
    log = root/'gpu/stdout.29818513.log'
    # An ended dependency is not success. Wait only for the actual same-job LSF
    # footer, not for an expected scientific outcome; an EXIT footer fails.
    for index in range(11):
        raw = log.read_text() if log.exists() else ''
        if 'Subject:Job' in ''.join(raw.split()):
            break
        ops.require(index < 10, 'Actual replay terminal LSF proof unavailable')
        time.sleep(2)
    ops._lsf_success(raw, '29818513', 'mosaic-chem-replay02-gpu', 'gpul40s')
    top_path = root/'gpu/payload/assessment.json'; top = json.loads(top_path.read_bytes())
    admission = json.loads((source/'admission.json').read_bytes())
    expected = {'status': 'original_failure_and_independent_Boltz_pass_preserved', 'points': 8,
                'input_files': len(admission['preserved_proofs']), 'can_certify_or_restart_original_study': False}
    ops.require(top['status'] == 'passed' and top['job_id'] == '29818513'
                and top['source_manifest_sha256'] == PRODUCER['manifest_sha256']
                and top['model_evaluations'] == 0 and top['optimizer_updates'] == 0
                and top['production_authorized'] is False and top['original_results_changed'] is False
                and top['preserved_inputs'] == top['preserved_inputs_after'] == expected,
                'Replay wrapper did not pass with preserved original scope')
    result_path = root/'gpu/payload/diagnostic/assessment.json'
    ops.require(top['diagnostic_assessment']['path'] == str(result_path), 'Replay result redirected')
    ops._binding(top['diagnostic_assessment'])
    result = json.loads(result_path.read_bytes()); replay.completed_diagnostic(result)
    ops.require(result['input_before'] == result['input_after'] and result['matching_control_selected_as_fix'] is False,
                'Original replay inputs changed or failed study reclassified')
    proofs = [ops._file_binding(path) for path in
              (root/'gpu/submission.json', root/'cpu-certificate.json', log, top_path, result_path)]
    records = []
    for row in result['controls']:
        path = root/'gpu/payload/diagnostic'/(row['control']+'-'+str(row['repeat']))/'control.json'
        ops.require(row['assessment'] == str(path) and ops.sha(path) == row['assessment_sha256'],
                    'Exact completed control path/hash differs')
        records.append(json.loads(path.read_bytes())); proofs.append(ops._file_binding(path))
    qualifier = ops._module(ops.SOURCE/'scripts/replay_context_qualification.py', '_new_native_context_qualification')
    qualification = qualifier.qualify(records)
    ops.require(qualification['selected_policy'] is not None,
                'No all-eight-point isolated compiler context qualified; preserve negative evidence')
    return {'version': 'new-full-model-context-adoption-v1', 'producer': PRODUCER,
            'qualification': qualification, 'proofs': proofs,
            'original_failure_reclassified': False, 'native_gradient_passed': False}


def adopt_cpu(ops, manifest):
    assessment = json.loads((ops.ROOT/'cpu/payload/assessment.json').read_bytes())
    binding = assessment.get('qualified_context')
    ops.require(assessment.get('status') == 'passed' and type(binding) is dict
                and binding.get('path') == str(ops.ROOT/'cpu/payload/qualified-context.json'),
                'CPU-qualified context redirected')
    ops._binding(binding)
    saved = json.loads(Path(binding['path']).read_bytes())
    ops.require(saved == qualify_predecessor(ops, manifest),
                'CPU context/actual diagnostic proofs changed before native dispatch')
    return saved, binding


def apply_environment(inherited, qualified):
    """Only fixed compiler controls; no optimizer/model/loss settings exist here."""
    choice = qualified['qualification']
    policy = choice['selected_policy']
    if policy not in ('hash0', 'autotune0', 'hash0_autotune0'):
        raise ValueError('No declared compiler context')
    expected = {}
    if 'hash0' in policy:
        expected['PYTHONHASHSEED'] = '0'
    if 'autotune0' in policy:
        expected['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
    if choice['selected_environment'] != expected or choice['native_gradient_passed'] is not False:
        raise ValueError('Compiler context claims unsupported settings/native acceptance')
    result = dict(inherited)
    if 'XLA_FLAGS' in expected:
        existing = result.get('XLA_FLAGS', '')
        if re.search(r'--xla_gpu_autotune_level(?:=|\s)', existing):
            if existing.strip() != expected['XLA_FLAGS']:
                raise ValueError('Ambiguous inherited compiler override')
        else:
            expected = {**expected, 'XLA_FLAGS': (existing+' '+expected['XLA_FLAGS']).strip()}
    result.update(expected)
    return result
