"""Authenticate an existing replay-context proof; change no scientific formula.

This is an explicit read-only dependency on chemistry-search-speed's sealed
full-model05 metadata consumer, not an import of its mutable working tree.
Fresh exact replay and precision gates at this different endpoint still apply.
"""
import importlib.util
import os
from pathlib import Path
import re
import sys

BASE=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
SOURCE=BASE/'chemistry-full-model-release-05/source'
PIN='ab875927a328c34650d2e43baf6dfbe323e8f9bcf5600f39bac78f3ccdfb5d8c'
CPU='c4a88a77f1ae5b4d20aa8c6f3b7fa59152cb295b4791b70c2bc737e1e0857980'


def require(value,message):
    if not value:raise ValueError(message)


def validate(value):
    qualification=value.get('qualification',{})
    require(qualification.get('version')=='isolated-replay-context-qualification-v1'
            and qualification.get('selected_policy')=='autotune0'
            and qualification.get('selected_environment')=={'XLA_FLAGS':'--xla_gpu_autotune_level=0'}
            and qualification.get('native_gradient_passed') is False
            and qualification.get('original_scientific_gate_reclassified') is False
            and qualification.get('saved_probability_substitution') is False
            and qualification.get('can_certify_or_restart_original_study') is False
            and qualification.get('fresh_full_model_validation_required') is True,
            'Exact predeclared replay candidate/scope differs')
    require(value.get('original_failure_reclassified') is False
            and value.get('native_gradient_passed') is False
            and type(value.get('proofs')) is list and value['proofs'],
            'Compiler evidence is not endpoint/native acceptance')
    return value


def qualify(verifier):
    require(str(os.environ.get('LSB_JOBID','')).isdigit(),'Allocated context authentication required')
    manifest=verifier(SOURCE,PIN)
    require(len(manifest['files'])==34,'Exact sealed compiler-context consumer differs')
    spec=importlib.util.spec_from_file_location('_endpoint_original_context_ops',SOURCE/'scripts/full_model_ops.py')
    ops=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ops;spec.loader.exec_module(ops)
    require(ops.SOURCE==SOURCE,'Context consumer source differs')
    certificate=ops.verify_cpu_certificate(PIN,manifest,CPU)
    require(certificate['cpu_cases']==408 and certificate['job_id']=='29818536',
            'Original allocated context CPU identity differs')
    value,receipt=ops.context_tools().adopt_cpu(ops.context_api(),manifest)
    validate(value)
    proofs=list(value['proofs'])+[receipt,ops._file_binding(ops.ROOT/'cpu-certificate.json')]
    for row in proofs:ops._binding(row)
    return {**value,'proofs':proofs,'consumer_source_manifest_sha256':PIN,
            'consumer_CPU_certificate_sha256':CPU,'endpoint_qualified':False}


def environment(inherited,qualified):
    validate(qualified)
    result=dict(inherited)
    existing=result.get('XLA_FLAGS','').strip()
    # Ambiguous inherited flags must not be silently overridden or accumulated.
    flags=re.findall(r'--xla_gpu_autotune_level(?:=|\s+)(\S+)',existing)
    require(not flags or flags==['0'],'Conflicting inherited compiler context')
    require('--xla_gpu_autotune_level' not in existing or flags==['0'],
            'Malformed inherited compiler context')
    if not flags:result['XLA_FLAGS']=(existing+' --xla_gpu_autotune_level=0').strip()
    return result
