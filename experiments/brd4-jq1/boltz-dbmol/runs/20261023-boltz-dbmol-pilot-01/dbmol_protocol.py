"""Exact binding-only composite scope and saved-state replay; imports lazy."""
import hashlib, json, math
from pathlib import Path

VERSION='mosaic-brd4-jq1-dbmol-lr0025-10000-v1'
SUITE='mosaic-dbmol-pilot-native-components-v1'
SEGMENT=2500
TARGET=10000
POCKET='9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43'
CONTEXT='fd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4'
CONTEXT_FILE='a18a331269170aac3281b0a2ecad313aab81756c9b77e048d7e2cbaa4d8f1093'
AFFINITY_MODEL='395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16'
AFFINITY_CHECKPOINT='dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e'
STRUCTURAL_CHECKPOINT='090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1'
ARITHMETIC='contact_head_and_assembly_float64_v1'
COMPONENTS=('dbmol_affinity_loss','dbmol_contact_loss','dbmol_confidence_loss','dbmol_total_loss')

def require(ok,message):
    if not ok:raise ValueError(message)

def read(path):return json.loads(Path(path).read_bytes())

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def config25(c):
    original=Path(__file__).resolve().parents[1]/'examples/configs/brd4_jq1_25_fixed_lrreset_boltz2_10000.json'
    expected=read(original)
    expected.update(version=VERSION,initial_step_size=.0025,
        label='JQ125 / DBMol binding-only / fixedLR0.0025 / fresh10000',
        dbmol={'version':SUITE,'arithmetic':ARITHMETIC,'contact_weight':2.5,
               'confidence_weight':2.5,'affinity':'mean(value)-mean(raw_binary_logit)',
               'segment_updates':SEGMENT,'chemistry_enabled':False})
    require(c==expected,'Undeclared DBMol pilot setting/objective')
    return c

def update_parameters(c):
    config25(c);return {'step_size':.0025,'max_backtracks':1,'acceptance_policy':'fixed_step'}

def verify_attempt_rates(case,c):
    policy=update_parameters(c)
    require(case.get('acceptance_policy')=='fixed_step','Unexpected search/acceptance policy')
    for u in case['updates']:
        attempts=u['attempts'];require(len(attempts)==1,'Fixed step must have one candidate')
        a=attempts[0]
        require(a['trial']==0 and a['rate']==.0025 and u['selected_trial']==0
            and u['accepted_step_size']==.0025 and u['fallback_used'] is False
            and u['selection_policy']=='fixed_step','Executed rate or policy changed')
        numeric=[u[k] for k in ['fixed_before','fixed_after','repair_weight']]
        numeric += [a[k] for k in ['autodiff_surrogate','independent_primal_surrogate','repair_weight']]
        numeric += list(a['components'].values())
        require(all(type(x) in (int,float) and math.isfinite(x) for x in numeric)
            and u['fixed_after']==a['independent_primal_surrogate']
            and u['repair_weight']==a['repair_weight']==0
            and type(u['no_op']) is bool and type(u['fixed_increased']) is bool
            and u['fixed_increased'] is (u['fixed_after']>u['fixed_before']),
            'Missing finite candidate or honest increase/no-op bookkeeping')
        require(set(COMPONENTS)<=set(a['components']),'Missing live composite metrics')
    return policy

def verify_cpu_certificate(path,digest,manifest):
    require(sha(path)==digest,'CPU certificate bytes differ');c=read(path)
    require(c['status']=='passed' and c['suite']==SUITE and c['source_manifest_sha256']==manifest,'Wrong CPU certificate')
    for p in c['proofs']:require(sha(p['path'])==p['sha256'],'CPU proof changed')
    return c

def verify_identity(identity):
    require(identity['protocol']=='mosaic-boltz-dbmol-v1'
        and identity['objective_arithmetic']==ARITHMETIC
        and identity['checkpoint_sha256']==AFFINITY_CHECKPOINT
        and identity['structural_checkpoint_sha256']==STRUCTURAL_CHECKPOINT
        and identity['affinity_model_digest']==AFFINITY_MODEL
        and identity['context_digest']==CONTEXT and identity['context_file_sha256']==CONTEXT_FILE
        and identity['pocket_definition_sha256']==POCKET and identity['diffusion_gradient'] is False
        and identity['recycling_steps']==0 and identity['affinity_mw_correction'] is False
        and identity['n_atoms']==25 and identity['optimized_output']=='dbmol_total_loss'
        and identity['scalar_dtype']=='float64' and identity['graph_dtype']=='float32'
        and identity['gradient_dtype']=='float32','Changed checkpoint/context/pocket/precision/objective')
    for k in ['model_digest','structural_model_digest','full_context_digest']:
        require(isinstance(identity[k],str) and len(identity[k])==64,'Malformed model/context digest')
    return identity

def frozen(after,before):
    keys=['model_digest','affinity_model_digest','structural_model_digest','context_digest','full_context_digest',
          'context_file_sha256','checkpoint_sha256','structural_checkpoint_sha256']
    require(after.get('frozen') is True and all(after[k]==before[k] for k in keys),'Frozen identity changed')

def scope(offset,steps,snapshot,digest):
    require(type(offset) is int and offset in range(0,TARGET,SEGMENT) and steps==SEGMENT,'Undeclared segment')
    require((offset>0)==bool(snapshot and digest) and (snapshot is None)==(digest is None),'Missing exact predecessor')
    return {'version':'dbmol-exact-fp32-logits-v1','start':offset,'end':offset+steps,'updates':steps,
            'target':TARGET,'input_snapshot':None if snapshot is None else str(snapshot),'input_snapshot_sha256':digest}

def restore(path,digest,key):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.chemistry import GraphLogits
    require(sha(path)==digest,'Saved predecessor bytes changed')
    with np.load(path,allow_pickle=False) as z:
        for k,shape in [('X',(25,6)),('E_upper',(300,5)),('atom_logits',(25,6)),('bond_logits',(300,5))]:
            require(z[k].shape==shape and z[k].dtype==np.float32 and np.isfinite(z[k]).all(),'Wrong predecessor FP32 shape/finite')
        require(z['key_data'].dtype==np.uint32,'Wrong predecessor key dtype')
        np.testing.assert_array_equal(z['key_data'],np.asarray(jax.random.key_data(key)))
        value=GraphLogits(jnp.asarray(z['atom_logits']),jnp.asarray(z['bond_logits']))
        for k,v in [('X',value.graph().X),('E_upper',value.graph().E_upper)]:np.testing.assert_array_equal(z[k],np.asarray(v))
    return value

def verify_readiness(path,digest,config,manifest,execution=None):
    require(sha(path)==digest,'Readiness assessment changed');a=read(path)
    require(a['status']=='passed' and a['config_sha256']==sha(config) and a['source_manifest_sha256']==manifest
        and a['arm']=='boltz2_dbmol' and a['n_atoms']==25 and a['pocket_definition_sha256']==POCKET
        and a['diffusion_gradient'] is False,'Readiness/config/source differs')
    verify_identity(a['native_identity'])
    if execution is not None:require(a['execution']==execution,'Readiness boundary differs')
    for p in a['proofs']:require(sha(p['path'])==p['sha256'],'Native proof changed')
    s=read(Path(path).with_name('summary.json'));p=read(Path(path).parent/'early_pilot/report.json')
    require(s['status']=='passed' and s['native_gradient_checks']['status']=='passed'
        and s['native_gradient_checks']['all_components_and_total_raw_fd_passed'] is True
        and s['runtime_gate']['status']=='passed' and len(p['updates'])==10 and p['status']=='passed','Missing component FD/pilot/runtime acceptance')
    verify_native_report(s['native_gradient_checks'])
    frozen(s['frozen_identity_after'],a['native_identity']);verify_attempt_rates(p,read(config))
    return a

def verify_native_report(r):
    require(r['status']=='passed' and r['policy']=='dbmol-native-raw-fd-v1'
        and set(r['components'])==set(COMPONENTS)
        and r['all_components_and_total_raw_fd_passed'] is True
        and r['hard_native_parity']['status']=='passed'
        and r['frozen_unchanged'] is True
        and r['key_and_logit_row_shift_nulls']=='passed'
        and r['live_primitive_recomposition']=='passed_exact_scalar_and_raw_gradients',
        'Full component raw-gradient policy not passed')
    for c in r['components'].values():
        require(c['status']=='passed' and c['epsilons']==[.01,.003,.001]
            and c['relative_tolerance']==.15 and c['required_adjacent_passes']==2
            and c['independent_primal'] is True and c['raw_unclipped_gradients'] is True
            and c['typed_bond_contrast_nonzero'] is True and c['production_jax_transport']=='passed'
            and [v['field'] for v in c['rows']]==['X','E_upper'],'Wrong raw FD scope')
        for row in c['rows']:
            require([v['epsilon'] for v in row['probes']]==[.01,.003,.001],'Changed FD scales')
            agrees=[]
            for v in row['probes']:
                analytic,fd=v['analytic'],v['finite_difference']
                calculated=all(math.isfinite(x) for x in [analytic,fd]) and abs(analytic)>1e-7 and analytic*fd>0 and abs(fd-analytic)<=.15*max(abs(fd),abs(analytic))
                require(v['agrees'] is calculated,'Raw derivative agreement misreported');agrees.append(calculated)
            require(any(x and y for x,y in zip(agrees,agrees[1:])),'Two adjacent original scales did not pass')
    return r
