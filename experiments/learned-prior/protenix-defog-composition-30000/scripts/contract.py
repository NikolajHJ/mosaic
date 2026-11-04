"""Closed metadata for the declared composition branch; parent science retained."""
from copy import deepcopy
import math
from parent_contract import sha, canonical_sha, ELEMENT_NATIVE_INDICES, weight, teacher_time, query_identity, verify_source
from parent_contract import validate as parent_validate
VERSION = "protenix-defog-moving-teacher-composition-descent-30000-v2"

def validate(c):
    if c['version'] != VERSION or c.get('output_filesystem') != 'home' or c['max_output_bytes'] != 3489660928:
        raise ValueError('Changed composition-branch version/output contract')
    if c.get('native_execution_profile') != {'policy': 'unchanged_parent_fp32_default', 'XLA_FLAGS': ''}:
        raise ValueError('Changed parent native execution profile')
    if c.get('step_acceptance') != {'version': 'fixed_teacher_primal_descent_v1', 'strict_decrease': True, 'nonzero_displacement': True, 'teacher_refresh_inside_backtracking': False, 'maximum_halvings': 12} or c.get('foreign_endpoint_policy') != {'coarse_failure': 'preserved_diagnostic_not_fresh_trajectory_prerequisite', 'fine_reference_job': '29853002', 'fine_reference_result': '19af0b0cabc58a19fb62c284a8f01c9f0a6fd6f2f7f344488dc7cdc5b3f32513', 'qualification': 'Both FP64 one-sided limits and centralFD agreeAD near1e-6; no runtime/model correction adopted'}:
        raise ValueError('Changed explicit descent/admission policy')
    comp = c['composition']
    if comp != {'version':'fixed_guacamol_population_js_v1', 'weight':0.7276566386277723, 'reference':[0.7428855101028519,0.10721655769374808,0.11250068056895664,0.014712321621999118,0.014251088885263644,0.008433841127180616], 'smoothing':1e-8, 'source_result_sha256':'0e1c4d3cbc2e19c95eb8426d326cc738d5b4f624d4c9221985150d771d1da93c', 'source_job':'29852742', 'source_config_sha256':'fa2e5596f829413659449118317988afaeeb3623b7ea1c2c4379f5ad61980744', 'upstream_source_sha256':'cd88305793ea8d78204a4e482749068e9b35171f3314782c8371f57e34b0d944', 'reference_scope':'fixed upstream GuacaMol source frequencies conditioned on six-element support; not a validity condition'}:
        raise ValueError('Changed fixed empirical reference or calibrated coefficient')
    old=deepcopy(c);old['version']='protenix-defog-moving-teacher-balanced-30000-v1';old['max_output_bytes']=4294967296
    parent_validate(old)
    return c
