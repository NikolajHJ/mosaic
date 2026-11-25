"""Read-only V100 re-score of exact saved hard graphs before a migration."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

VERSION = 'embedding-v100-cache-audit-v1'
WORK = Path('/work3/s234818/latentmol')
CONFIG = 'configs/embedding_v100_handoff_audit.json'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024*1024),b''): h.update(data)
    return h.hexdigest()

def verify(asset):
    path=Path(asset['path'])
    if sha(path)!=asset['sha256']: raise ValueError('Input changed: '+str(path))
    return path

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')

def operator(path):
    spec=importlib.util.spec_from_file_location('_handoff_operator',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def validate(config):
    if config['version']!=VERSION or config['gpu_profile']!='v10032' or config['step_candidates']!=1 or config['line_search'] is not False:
        raise ValueError('Invalid isolated audit policy')
    parent=config['parent'];root=WORK/'runs/source142-epoch36-guarded-embedding-long-attempt001'
    if parent['campaign']!=str(root) or parent['source_sha256']!='d2ac85ad644a94f3b5967537f1c777bb7318faf2a8507038888bc32728a0d805':raise ValueError('Original campaign mismatch')
    for asset in parent['artifacts'].values():
        if not Path(asset['path']).is_relative_to(root):raise ValueError('Audit input escaped parent campaign')
    qualification=verify(config['qualification']);result=json.loads(qualification.read_text())
    if (result['status']!='passed_device_qualification_not_continuation_adoption' or result['frozen_unchanged'] is not True
        or tuple(x['case'] for x in result['cases'])!=('case-001','case-002','case-003')):raise ValueError('Original V100 qualification not passed')
    for i,row in enumerate(result['cases'],1):
        if row['resume_sha256']!=parent['artifacts']['case%03d_state'%i]['sha256']:raise ValueError('Qualification does not cover pinned resume')
    for asset in parent['artifacts'].values():verify(asset)
    release=Path(parent['release']);operator(Path(__file__).with_name('embedding_operator.py')).authenticate(release,parent['source_sha256'],parent['plan_sha256'],'configs/epoch36_embedding_long.json')
    if sha(release/'embedding-release-manifest.json')!=parent['manifest_sha256']:raise ValueError('Parent manifest mismatch')
    return release

def audit(config,output):
    release=validate(config);sys.path.insert(0,str(release/'src'))
    import torch
    from latentmol.boltz import embedding_comparison as comparison
    from latentmol.boltz import embedding_long_optimization as long
    from latentmol.boltz import embedding_long_experiment as experiment
    from latentmol.boltz.embedding_experiment import load_native,load_vae
    from latentmol.train import state_digest
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    props=torch.cuda.get_device_properties(0)
    if 'V100' not in props.name or props.total_memory<30*1024**3:raise ValueError('Requires 32GB V100')
    native=json.loads((release/'configs/epoch36_embedding_long.json').read_text()); p=config['parent']; fitted=json.loads(verify(p['artifacts']['fit_result']).read_text())
    vae,_=load_vae(native,'cuda');scorer,_,bank=load_native(native,'cuda');bridge,caps=experiment.load_long_bridge(native,vae,bank,fitted,p['source_sha256'],p['plan_sha256'],'cuda')
    starts=torch.load(verify(p['artifacts']['starts']),map_location='cpu',weights_only=True)
    modules=dict(vae=vae,scorer=scorer,bank=bank,bridge=bridge); before={k:state_digest(x.state_dict()) for k,x in modules.items()};out=[]
    for i,item in enumerate(starts['cases'],1):
        name='case-%03d'%i;key='case%03d_'%i
        if item['record']['case_id']!=name:raise ValueError('Case order changed')
        prepared,z0=experiment.load_case(item,'cuda');base=comparison.FrozenComparisonObjective(scorer,prepared,bridge,vae,bank=bank,comparison_graph_policy=native['comparison_graph_policy']);guard=long.GuardedEmbeddingObjective(base,caps)
        state=torch.load(verify(p['artifacts'][key+'state']),map_location='cpu',weights_only=True);manifest=json.loads(verify(p['artifacts'][key+'manifest']).read_text());binding=json.loads(verify(p['artifacts'][key+'binding']).read_text());cache=json.loads(verify(p['artifacts'][key+'cache']).read_text())
        if (state['case_id']!=name or state['update']!=config['state_updates'][name] or state['signature_sha256']!=manifest['sha256'] or manifest['signature']['binding']!=binding or not torch.equal(state['z0'],z0.cpu()) or not state['gate_binding_verified']):raise ValueError('Saved state authentication failed')
        assessment=guard.assess(state['z'].to(z0),include_tensors=True);record=assessment['record']
        if not record['eligible'] or record['hard_graph_sha256']!=cache['hard_graph_sha256'] or state['best_valid']['hard_graph_sha256']!=cache['hard_graph_sha256'] or state['hard_cache_entries']!=1:raise ValueError('Cached hard graph is not the pinned current and best graph')
        long._validate_binding(binding,guard,z0);scores=guard.hard_score(assessment);old=cache['scores'];limits=native['native_tolerances'];rows={}
        if set(scores)!=set(old):raise ValueError('Hard score keys changed')
        for k in scores:
            if not math.isfinite(scores[k]) or not math.isfinite(old[k]):raise ValueError('Nonfinite hard score')
            error=abs(scores[k]-old[k]);limit=limits['scalar_atol']+limits['scalar_rtol']*abs(old[k]);rows[k]=dict(a100=old[k],v100=scores[k],absolute_error=error,original_scalar_tolerance=limit,within_tolerance=error<=limit)
        row=dict(case=name,update=state['update'],hard_graph_sha256=cache['hard_graph_sha256'],a100_cache_sha256=p['artifacts'][key+'cache']['sha256'],scores=rows,all_within_native_scalar_tolerance=all(x['within_tolerance'] for x in rows.values()),frozen_after=guard.assert_unchanged(full=True))
        write(output/(name+'.json'),row);out.append(row)
        if not row['all_within_native_scalar_tolerance']:raise ValueError('Cross-device hard score exceeds original scalar tolerance')
        del guard,base,prepared,state
    after={k:state_digest(x.state_dict()) for k,x in modules.items()}
    if after!=before:raise ValueError('Frozen model state changed')
    validate(config)
    return dict(status='passed_cross_device_cache_audit_not_migrated',cases=out,production_mutation=False,original_states_and_caches_unchanged=True,device=props.name)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--source-sha256',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    release=Path(__file__).resolve().parents[2]
    if not os.environ.get('LSB_JOBID') or os.environ.get('LATENTMOL_SOURCE_SHA256')!=a.source_sha256 or sha(a.config)!=a.plan_sha256:raise ValueError('Require allocated authenticated release')
    operator(Path(__file__).with_name('embedding_operator.py')).authenticate(release,a.source_sha256,a.plan_sha256,CONFIG)
    if release!=WORK/'releases/source-147-embedding-v100-cache-audit-attempt001' or a.output.resolve()!=WORK/'runs/source147-embedding-v100-cache-audit-attempt001/gpu':raise ValueError('Require isolated audit output')
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();header=dict(job_id=os.environ['LSB_JOBID'],source_sha256=a.source_sha256,plan_sha256=a.plan_sha256)
    try:result=audit(json.loads(a.config.read_text()),a.output)
    except BaseException as error:
        write(a.output/'result.json',dict(**header,status='failed_not_passed',error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started));raise
    write(a.output/'result.json',dict(**header,**result,elapsed_seconds=time.monotonic()-started))
if __name__=='__main__':main()
