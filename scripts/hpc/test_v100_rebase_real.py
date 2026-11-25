"""Exercise the tested rebasing helper on isolated copies of actual committed states."""
import hashlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()

def verify(asset):
 p=Path(asset['path'])
 if sha(p)!=asset['sha256']:raise ValueError('Bound input changed: '+str(p))
 return p

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 release=Path(__file__).resolve().parents[2];work=Path('/work3/s234818/latentmol');out=work/'runs/source149-v100-real-rebase-attempt001/cpu'; config_path=release/'configs/embedding_v100_rebase_real.json'
 if not os.environ.get('LSB_JOBID') or release!=work/'releases/source-149-v100-real-rebase-attempt001':raise RuntimeError('Require isolated allocated source149')
 op=load(release/'scripts/hpc/embedding_operator.py','_operator');source=os.environ['LATENTMOL_SOURCE_SHA256'];plan=sha(config_path);op.authenticate(release,source,plan,'configs/embedding_v100_rebase_real.json')
 config=json.loads(config_path.read_text());corepath=release/'scripts/hpc/embedding_v100_rebase_core.py'
 if sha(corepath)!=config['tested_core_sha256']:raise ValueError('Rebase core differs from CPU-tested release')
 core=load(corepath,'_rebase');audit=json.loads(verify(config['audit']).read_text());test=json.loads(verify(config['cpu_test']).read_text())
 if (test['status']!='passed_cpu_rebase_fixtures_only' or test['source_sha256']!='a711f15bb2b995a3e49b4aa76ddf4dca20f2d8d718ebdc4dc9fc80a6b5443582' or test['junit']!={'tests':9,'failures':0,'errors':0,'skipped':0}):raise ValueError('Exact prior CPU qualification missing')
 if (audit['status']!='passed_cross_device_cache_audit_not_migrated' or audit['source_sha256']!='f9bff6a14b41f27128a5d5970de7e663bb0451e0c8b7d530b7e6f28ce44c1e8c' or str(audit['job_id'])!='29853377'):raise ValueError('Exact completed V100 audit missing')
 parent=config['parent'];source_release=Path(parent['release']);op.authenticate(source_release,parent['source_sha256'],parent['plan_sha256'],'configs/epoch36_embedding_long.json')
 if sha(source_release/'embedding-release-manifest.json')!=parent['manifest_sha256']:raise ValueError('Original parent manifest mismatch')
 out.mkdir(parents=True,exist_ok=False);started=time.monotonic();header=dict(job_id=os.environ['LSB_JOBID'],source_sha256=source,plan_sha256=plan)
 try:
  results=[]
  for i in range(1,4):
   case='case-%03d'%i;key='case%03d_'%i;src=Path(parent['campaign'])/'trajectories'/case
   r=core.rebase_case(case=case,source_root=src,source_state=parent['artifacts'][key+'state'],source_manifest=parent['artifacts'][key+'manifest'],source_cache=parent['artifacts'][key+'cache'],audit=audit,output_root=out/'clones'/case,expected_update=config['state_updates'][case]);results.append(r)
  result=dict(**header,status='passed_real_state_clones_not_adopted',cases=results,original_campaign_modified=False,production_optimization_started=False,elapsed_seconds=time.monotonic()-started)
 except BaseException as error:
  result=dict(**header,status='failed_not_passed',error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');raise
 (out/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
