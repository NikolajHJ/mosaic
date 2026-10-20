import hashlib,json,subprocess,tarfile
from pathlib import Path
root=Path('/zhome/95/e/206121/mosaic-runs')
run=root/'20261019-brd4-boltz-mix50-independent-01'
study=run/'payload/study';case=study/'case-0'
source=Path('/work3/s234818/mosaic/releases/20261019-brd4-boltz-affinity-02/source')
out=root/'20261020-brd4-boltz-mix50-completion-02';out.mkdir()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
scheduler=subprocess.check_output(['bjobs','-l','29759212'],text=True)
assert 'Status <DONE>' in scheduler and 'Done successfully.' in scheduler
stdout=run/'stdout.29759212.log'
assert 'Subject: Job 29759212:' in stdout.read_text() and 'Successfully completed.' in stdout.read_text()
(out/'scheduler-final.txt').write_text(scheduler)
r=read(study/'summary.json');c=read(case/'report.json')
assert r['status']==c['status']=='passed' and r['mode']=='run' and r['job_id']=='29759212'
assert len(r['cases'])==1
assert {k:v for k,v in r['cases'][0].items() if k not in ['label','initialization']}==c
assert c['requested_updates']==10000 and c['acceptance_policy']=='best_primal'
assert [x['step'] for x in c['states']]==list(range(10001))
assert [x['step'] for x in c['updates']]==list(range(10000))
for name,key in [('states.jsonl','states'),('updates.jsonl','updates')]:assert [json.loads(x) for x in (case/name).read_text().splitlines()]==c[key]
assert c['counts']['applied']==10000 and c['counts']['strict_improving']+c['counts']['forced_fallback']==10000
assert c['updates'][-1]['repair_weight']==50.0 and c['states'][-1]['repair_weight']==50.0
config=source/'examples/configs/brd4_boltz_affinity_mix50_10000.json'
assert sha(config)==r['config_sha256']=='97faf3b192a7aca566ad86752c2f3efacad69e89b076c88e67e8d14b5435de0e'
assert read(config)==r['config'] and r['config']['n_atoms']==6
assert sha(source/'source_manifest.json')=='5eb6e17ea8d19e52c99a589243228710994dcf137f59a6b9a6659a4fe719a7f4'
m=read(source/'source_manifest.json')
expected={n:v for n,v in m['files'].items() if n=='pyproject.toml' or (n.split('/')[0] in ['src','tests','examples','scripts'] and Path(n).suffix in ['.py','.json','.lsf','.sh','.toml','.txt'] and '__pycache__' not in Path(n).parts)}
assert r['source']['files']==expected
assert r['source']['execution_content_sha256']==hashlib.sha256(json.dumps(expected,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert r['context_sha256']=='254d5b7f11605fb41db6f5abf7015f9cc856a19dfe1d4b506ccaea9007200770'
assert r['reference_sha256']=='37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb'
identity=r['native_identity'];f=r['frozen_identity_after']
assert identity['model_digest']==f['model_digest']=='395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16'
assert identity['context_digest']==f['context_digest']=='495002ffa2a52cc85b58ef34424622763dfc4ca556139a98297e8789ae287aa5' and f['status']=='passed'
assert identity['context_file_sha256']==r['context_sha256'] and identity['optimized_output']=='affinity_pred_value' and identity['recycling_steps']==0
assert r['native_hard_parity']['status']=='passed'
for key in ['native_gradient_checks','terminal_gradient_checks']:
 check=read(study/(key+'/report.json'))
 assert check==r[key] and check['status']=='passed' and check['typed_bond_contrast_nonzero'] is True
 assert [x['field'] for x in check['rows']]==['X','E_upper']
 for row in check['rows']:
  assert [x['epsilon'] for x in row['probes']]==[.01,.003,.001]
  assert any(a['agrees'] and b['agrees'] for a,b in zip(row['probes'],row['probes'][1:]))
combined=read(case/'combined_initial_checks/report.json')
assert combined['status']=='passed' and c['initial_checks']==combined['directions']
objective=identity|{'context_sha256':r['context_sha256'],'coefficient':1.0}
assert c['affinity_objective_identity']==objective and c['best_observed_affinity']['objective_identity']==objective
selected={}
for name in ['summary.json','initialization.npz','native_gradient_checks/report.json','terminal_gradient_checks/report.json','case-0/report.json','case-0/states.jsonl','case-0/updates.jsonl','case-0/combined_initial_checks/report.json']:
 selected['study/'+name]=study/name
assert sha(study/'initialization.npz')==r['initialization_sha256']
for step in [0,5000,10000]:
 state=c['states'][step];p=(case/state['snapshot']).resolve()
 assert p.is_relative_to(case.resolve()) and sha(p)==state['snapshot_sha256']
 selected['study/case-0/'+state['snapshot']]=p
for name in ['submission-intent.json','submission.json','scheduler-submitted.txt','stdout.29759212.log','stderr.29759212.log','payload/gpu.txt','payload/zhome_quota_preflight.json']:
 if (run/name).is_file():selected['run/'+name]=run/name
selected['source/source_manifest.json']=source/'source_manifest.json'
selected['source/mix50_config.json']=config
selected['scheduler-final.txt']=out/'scheduler-final.txt'
audit={'status':'passed','job_id':'29759212','scheduler_status':'DONE','persisted_status':'passed','updates':10000,'states':10001,'counts':c['counts'],'final_repair_weight':50.0,'summary_sha256':sha(study/'summary.json'),'case_report_sha256':sha(case/'report.json'),'source_manifest_sha256':sha(source/'source_manifest.json'),'config_sha256':sha(config),'model_digest':f['model_digest'],'context_digest':f['context_digest'],'selected_snapshot_steps':[0,5000,10000],'scope':'completion-record/schema/source identity verification and selected snapshot bytes; not every snapshot byte or a new numerical/model evaluation','packaging_predecessor':'completion01 helper stopped on missing hashlib.file_digest in systemPython; no scientific artifact changed'}
(out/'completion-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
selected['completion-audit.json']=out/'completion-audit.json'
manifest={'files':{n:{'bytes':p.stat().st_size,'sha256':sha(p)} for n,p in selected.items()},'scope':audit['scope']}
(out/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=out.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz',compresslevel=1) as tf:
 tf.add(out/'evidence_manifest.json',arcname='evidence_manifest.json')
 for name,p in selected.items():tf.add(p,arcname=name)
receipt={'archive_path':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'manifest_sha256':sha(out/'evidence_manifest.json'),'files':len(selected),'completion':audit}
out.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
