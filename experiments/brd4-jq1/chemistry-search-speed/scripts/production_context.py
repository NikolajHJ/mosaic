"""Exact metadata transport of a passed NEW Protenix compiler context.

Boltz gets no new override. Neither branch evaluates an array or changes a
model/loss/optimizer. Actual native acceptance remains in pilot_pass.
"""
from pathlib import Path


def read_context(op, arm, plan):
    if arm == 'boltz':
        return {'version': 'chemistry-production-context-v1', 'arm': arm, 'environment': {},
                'scientific_settings_changed': False}
    op.require(arm == 'protenix', 'Undeclared chemistry context arm')
    row = plan['pilots'][arm]; source = Path(row['source']); root = Path(row['root'])
    pilot = op.pilot_metadata(arm, plan)
    manifest = pilot.verify(source, row['manifest_sha256'])
    pilot.verify_cpu_certificate(row['manifest_sha256'], manifest, row['cpu_certificate_sha256'])
    tool = op.module(source/'scripts/full_model_context.py', '_production_pro_context_metadata')
    qualified, binding = tool.adopt_cpu(pilot.context_api(), manifest)
    top_path = root/'protenix/payload/assessment.json'; top = op.read(top_path)
    sub = op.read(root/'protenix/submission.json')
    op.require(top['status'] == 'passed' and top['job_id'] == row['job_id'] == sub['job_id']
               and top['source_manifest_sha256'] == row['manifest_sha256']
               and top['qualified_context'] == binding
               and top['execution_context_overrides'] == qualified['qualification']['selected_environment'],
               'Passed Protenix native pilot used a different compiler context')
    proof = op.lsf_success(root/'protenix', sub)
    return {'version': 'chemistry-production-context-v1', 'arm': arm,
            'environment': qualified['qualification']['selected_environment'],
            'qualified_context': binding, 'native_assessment': op.binding(top_path), 'native_LSF': proof,
            'scientific_settings_changed': False}


def apply(environment, record):
    if record.get('scientific_settings_changed') is not False:
        raise ValueError('Compiler transport cannot change scientific settings')
    overrides = record['environment']
    allowed = ({}, {'PYTHONHASHSEED': '0'}, {'XLA_FLAGS': '--xla_gpu_autotune_level=0'},
               {'PYTHONHASHSEED': '0', 'XLA_FLAGS': '--xla_gpu_autotune_level=0'})
    if overrides not in allowed or (record['arm'] == 'boltz' and overrides) or (record['arm'] == 'protenix' and not overrides):
        raise ValueError('Undeclared or cross-arm compiler override')
    if record['arm'] not in ('protenix', 'boltz'):
        raise ValueError('Unknown compiler-context arm')
    result = dict(environment)
    if 'XLA_FLAGS' in overrides:
        existing = result.get('XLA_FLAGS', '').strip()
        if existing not in ('', overrides['XLA_FLAGS']):
            raise ValueError('Unvalidated ambient compiler flags')
    result.update(overrides)
    return result
