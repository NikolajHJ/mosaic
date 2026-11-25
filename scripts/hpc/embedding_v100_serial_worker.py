"""Single V100 segment from an authenticated trajectory snapshot."""
import argparse,hashlib,importlib.util,json,os,shutil,sys,time,traceback
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def verify(a):
 p=Path(a['path']).resolve()
 if sha(p)!=a['sha256']:raise ValueError('Input hash changed: '+str(p))
 return p
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def write(path,value):
 with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def copy_file(src,dst):
 # The index is the only inherited file that future segments can append to.
 if Path(src).name=='index.jsonl':return shutil.copy2(src,dst)
 os.link(src,dst);return dst

def run(config,request,out,deadline):
 parent=config['parent'];release=Path(parent['release']);sys.path.insert(0,str(release/'src'))
 import torch
 from latentmol.boltz import embedding_long_experiment as exp,embedding_long_optimization as long,embedding_comparison as comparison
 from latentmol.boltz.embedding_experiment import load_vae,load_native
 from latentmol.optimization import StepControl
 if not all(Path(x.__file__).resolve().is_relative_to(release/'src') for x in (exp,long,comparison)):raise ValueError('Wrong numerical source')
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
 props=torch.cuda.get_device_properties(0)
 if 'V100' not in props.name or props.total_memory<30*1024**3:raise ValueError('Requires 32GB V100')
 name=request['case'];index={'case-001':0,'case-002':1,'case-003':2}[name];key='case%03d_'%(index+1)
 src=verify(request['state']);source_root=src.parent.parent
 allowed=(Path('/work3/s234818/latentmol/runs/source149-v100-real-rebase-attempt001/cpu/clones'),Path('/work3/s234818/latentmol/runs/source151-v100-continuation-attempt001/gpu'),Path('/work3/s234818/latentmol/runs/source152-v100-serial-continuation-attempt001'))
 if not any(source_root.is_relative_to(x) for x in allowed):raise ValueError('Resume escaped allowed trajectory roots')
 destination=out/'trajectory';shutil.copytree(source_root,destination,copy_function=copy_file);resumed=destination/'states'/src.name
 if sha(resumed)!=request['state']['sha256']:raise ValueError('Copied checkpoint differs')
 original=json.loads((release/'configs/epoch36_embedding_long.json').read_text());fitted=json.loads(verify(parent['artifacts']['fit_result']).read_text());vae,_=load_vae(original,'cuda');scorer,_,bank=load_native(original,'cuda');bridge,caps=exp.load_long_bridge(original,vae,bank,fitted,parent['source_sha256'],parent['plan_sha256'],'cuda')
 starts=torch.load(verify(parent['artifacts']['starts']),map_location='cpu',weights_only=True);prepared,z0=exp.load_case(starts['cases'][index],'cuda')
 objective=long.GuardedEmbeddingObjective(comparison.FrozenComparisonObjective(scorer,prepared,bridge,vae,bank=bank,deadline=deadline,comparison_graph_policy=original['comparison_graph_policy']),caps)
 state=torch.load(resumed,map_location='cpu',weights_only=True);binding=json.loads(verify(parent['artifacts'][key+'binding']).read_text())
 if state['case_id']!=name or state['update']!=request['start_update']:raise ValueError('Resume update/case mismatch')
 assessment=objective.assess(state['z'].to(z0),include_tensors=True)
 if not assessment['record']['eligible'] or assessment['record']['hard_graph_sha256']!=state['last_hard_graph_sha256']:raise ValueError('Resume chemistry differs')
 scores=objective.hard_score(assessment);cached=json.loads((destination/'hard-cache'/(state['last_hard_graph_sha256']+'.json')).read_text())['scores'];limits=original['native_tolerances']
 if set(scores)!=set(cached) or any(abs(scores[k]-cached[k])>limits['scalar_atol']+limits['scalar_rtol']*abs(cached[k]) for k in scores):raise ValueError('Fresh V100 hard scores differ from saved cache')
 result=long.run_guarded_optimization(z0,objective,destination,case_id=name,binding=binding,target_updates=10000,segment_updates=request['max_updates'],observation_every=10,resume=dict(path=str(resumed),sha256=sha(resumed)),learning_rate=original['optimization']['learning_rate'],max_gradient_norm=1.,trust_radius=3.,control=StepControl.from_mapping(original['step_control']),deadline=deadline,reserve_seconds=1800.,max_consecutive_noop=25)
 if result['starting_update']!=request['start_update'] or result['completed_update']<=request['start_update'] or result['scientific_stall']:raise ValueError('V100 segment stalled or made no progress')
 if sha(src)!=request['state']['sha256']:raise ValueError('Source checkpoint was modified')
 return dict(status='completed_v100_segment',case=name,result=result,source_state=request['state'],device=props.name,production=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--request',type=Path,required=True);p.add_argument('--request-sha256',required=True);p.add_argument('--source-sha256',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();work=Path('/work3/s234818/latentmol');release=Path(__file__).resolve().parents[2]
 if not os.environ.get('LSB_JOBID') or os.environ.get('LATENTMOL_SOURCE_SHA256')!=a.source_sha256 or sha(a.config)!=a.plan_sha256 or sha(a.request)!=a.request_sha256 or release!=work/'releases/source-152-v100-serial-continuation-attempt001' or not a.output.resolve().is_relative_to(work/'runs/source152-v100-serial-continuation-attempt001/jobs'):raise ValueError('Require isolated allocated exact release')
 op=load(release/'scripts/hpc/embedding_operator.py','_operator');op.authenticate(release,a.source_sha256,a.plan_sha256,'configs/embedding_v100_serial_continuation.json');config=json.loads(a.config.read_text());parent=config['parent'];op.authenticate(Path(parent['release']),parent['source_sha256'],parent['plan_sha256'],'configs/epoch36_embedding_long.json')
 if sha(Path(parent['release'])/'embedding-release-manifest.json')!=parent['manifest_sha256']:raise ValueError('Original manifest mismatch')
 request=json.loads(a.request.read_text());a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();head=dict(job_id=os.environ['LSB_JOBID'],source_sha256=a.source_sha256,plan_sha256=a.plan_sha256,request_sha256=a.request_sha256)
 try:result=run(config,request,a.output,started+82800)
 except BaseException as e:write(a.output/'result.json',dict(**head,status='failed_not_passed',error=str(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started));raise
 write(a.output/'result.json',dict(**head,**result,elapsed_seconds=time.monotonic()-started))
if __name__=='__main__':main()
