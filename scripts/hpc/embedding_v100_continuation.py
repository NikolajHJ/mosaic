"""First V100 continuation segment from the verified, rebased checkpoint."""
import hashlib,importlib.util,json,os,shutil,sys,time,traceback
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def verify(asset):
 p=Path(asset['path'])
 if sha(p)!=asset['sha256']:raise ValueError('Pinned input changed: '+str(p))
 return p
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def write(path,value):
 with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def run(config,out,deadline):
 parent=config['parent'];release=Path(parent['release']);sys.path.insert(0,str(release/'src'))
 import torch
 from latentmol.boltz import embedding_long_experiment as exp,embedding_long_optimization as long,embedding_comparison as comparison
 from latentmol.boltz.embedding_experiment import load_vae,load_native
 from latentmol.optimization import StepControl
 if not all(Path(x.__file__).resolve().is_relative_to(release/'src') for x in (exp,long,comparison)):raise ValueError('Wrong numerical source')
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
 props=torch.cuda.get_device_properties(0)
 if 'V100' not in props.name or props.total_memory<30*1024**3:raise ValueError('Requires 32GB V100')
 original=json.loads((release/'configs/epoch36_embedding_long.json').read_text());pilot=json.loads(verify(config['passed_pilot']).read_text());
 if pilot['status']!='passed_one_isolated_resume_update_not_adopted' or pilot['result']['completed_update']!=21 or pilot['production_mutation'] is not False:raise ValueError('Verified isolated V100 resume pilot missing')
 clones=json.loads(verify(config['real_clones']).read_text())
 if clones['status']!='passed_real_state_clones_not_adopted' or clones['original_campaign_modified'] is not False or len(clones['cases'])!=3:raise ValueError('Real state clones did not pass')
 clone=clones['cases'][1];evidence=clone['evidence']
 if evidence['case']!='case-002' or evidence['update']!=20 or evidence['source_state']['sha256']!=parent['artifacts']['case002_state']['sha256']:raise ValueError('Incorrect resume state')
 old=verify(clone);src=old.parent.parent
 if src!=Path('/work3/s234818/latentmol/runs/source149-v100-real-rebase-attempt001/cpu/clones/case-002'):raise ValueError('Clone escaped source149')
 shutil.copytree(src,out/'trajectory');destination=out/'trajectory';resumed=destination/'states'/old.name
 if sha(resumed)!=clone['sha256']:raise ValueError('Copied checkpoint differs')
 fitted=json.loads(verify(parent['artifacts']['fit_result']).read_text());vae,_=load_vae(original,'cuda');scorer,_,bank=load_native(original,'cuda');bridge,caps=exp.load_long_bridge(original,vae,bank,fitted,parent['source_sha256'],parent['plan_sha256'],'cuda')
 starts=torch.load(verify(parent['artifacts']['starts']),map_location='cpu',weights_only=True);prepared,z0=exp.load_case(starts['cases'][1],'cuda')
 objective=long.GuardedEmbeddingObjective(comparison.FrozenComparisonObjective(scorer,prepared,bridge,vae,bank=bank,deadline=deadline,comparison_graph_policy=original['comparison_graph_policy']),caps)
 state=torch.load(resumed,map_location='cpu',weights_only=True);binding=json.loads(verify(parent['artifacts']['case002_binding']).read_text());assessment=objective.assess(state['z'].to(z0),include_tensors=True)
 if not assessment['record']['eligible'] or assessment['record']['hard_graph_sha256']!=state['last_hard_graph_sha256']:raise ValueError('Resume chemistry differs')
 scores=objective.hard_score(assessment);expected=evidence['v100_hard_score'];limits=original['native_tolerances']
 if set(scores)!=set(expected) or any(abs(scores[k]-expected[k])>limits['scalar_atol']+limits['scalar_rtol']*abs(expected[k]) for k in scores):raise ValueError('Fresh V100 hard scores differ from rebased cache tolerance')
 result=long.run_guarded_optimization(z0,objective,destination,case_id='case-002',binding=binding,target_updates=10000,segment_updates=config['segment_updates'],observation_every=10,resume=dict(path=str(resumed),sha256=sha(resumed)),learning_rate=original['optimization']['learning_rate'],max_gradient_norm=1.,trust_radius=3.,control=StepControl.from_mapping(original['step_control']),deadline=deadline,reserve_seconds=1800.,max_consecutive_noop=25)
 if result['starting_update']!=20 or result['completed_update']<=20 or result['scientific_stall']:raise ValueError('V100 continuation stalled or made no progress')
 if sha(old)!=clone['sha256']:raise ValueError('Source149 clone was modified')
 return dict(status='completed_first_v100_segment',result=result,source_clone_sha256=clone['sha256'],production_mutation=True,original_campaign_mutation=False,device=props.name,scope='Production continuation on V100 from the rebased case002 checkpoint')
def main():
 p=__import__('argparse').ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--source-sha256',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();work=Path('/work3/s234818/latentmol');release=Path(__file__).resolve().parents[2]
 if not os.environ.get('LSB_JOBID') or os.environ.get('LATENTMOL_SOURCE_SHA256')!=a.source_sha256 or sha(a.config)!=a.plan_sha256 or release!=work/'releases/source-151-v100-continuation-attempt001' or a.output.resolve()!=work/'runs/source151-v100-continuation-attempt001/gpu':raise ValueError('Require isolated allocated exact release')
 op=load(release/'scripts/hpc/embedding_operator.py','_operator');op.authenticate(release,a.source_sha256,a.plan_sha256,'configs/embedding_v100_continuation.json');config=json.loads(a.config.read_text());parent=config['parent'];op.authenticate(Path(parent['release']),parent['source_sha256'],parent['plan_sha256'],'configs/epoch36_embedding_long.json')
 if sha(Path(parent['release'])/'embedding-release-manifest.json')!=parent['manifest_sha256']:raise ValueError('Original manifest mismatch')
 a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();head=dict(job_id=os.environ['LSB_JOBID'],source_sha256=a.source_sha256,plan_sha256=a.plan_sha256)
 try:result=run(config,a.output,started+82800)
 except BaseException as e:write(a.output/'result.json',dict(**head,status='failed_not_passed',error=str(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started));raise
 write(a.output/'result.json',dict(**head,**result,elapsed_seconds=time.monotonic()-started))
if __name__=='__main__':main()
