"""Record only actual bootstrap/submission receipts; no model/numerical execution."""
from pathlib import Path
import hashlib,json
h=Path(__file__).resolve().parent
rows=[json.loads(line) for line in (h/'bootstrap-stdout-01.txt').read_text().splitlines() if line.startswith('{')]
jobs=[x for x in rows if x.get('outcome')=='submitted']
assert len(jobs)==2 and [j['job_id'] for j in jobs]==['29781363','29781364']
gate=next(x for x in rows if x.get('phase')=='preflight')
source=json.loads((h/'source-comparison.json').read_text())
for filename, value in [('initial-submissions.json',{'cpu':jobs[0],'callback':jobs[1]}),('preflight-storage-gate.json',gate)]:
 with (h/filename).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
campaign={'tag':h.name,'status':'Source622 staged/authenticated read-only; CPU and own-ended callback actually submitted. New CPU/native/long passes not yet established.',
 'authorization':{'rates':{'protenix':.3,'boltz2':.005,'boltz2_chemistry':.005},'fresh_updates_each':10000,'repair_schedule':{'initial':.001,'final':100.,'hold_updates':5000,'total_updates':10000},'acceptance_policy':'fixed_step','max_backtracks':1,'finite_uphill_equal_noop_applied':True},
 'source':source,'initial_jobs':{'cpu':jobs[0]['job_id'],'after_cpu':jobs[1]['job_id']},'cpu_submitted':True,'cpu_passed':False,'native_submitted':False,'native_passed':False,'long_submitted':False,
 'preflight':{'status':gate['status'],'required_bytes':gate['filesystems']['work3']['required_bytes'],'available_bytes':gate['filesystems']['work3']['available_bytes_conservative'],'concurrent_analysis_allowance_bytes':gate['concurrent_analysis_allowance_bytes'],'limitation':'CPU allowance only; not validation/long gate or reservation'},
 'work3_root':'/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-02','home_root':'/zhome/95/e/206121/mosaic-runs/'+h.name,
 'finished_predecessors':{'protenix':{'job':'29779177','scientific_status':'failed'},'boltz2':{'job':'29779178','scientific_status':'failed'},'boltz2_chemistry':{'job':'29779179','scientific_status':'passed'}},
 'existing_diagnostics_preserved':['29780821','29780823'],'existing_diagnostic_captures_preserved':['29780822','29780824'],'existing_final_presentation_preserved':'29780830',
 'new_report':None,'milestone_progress':{'protenix':0,'boltz2':0,'boltz2_chemistry':0},'no_old_run_cancelled_resumed_or_modified':True,'no_additional_cleanup':True}
with (h/'campaign.json').open('x') as f:json.dump(campaign,f,indent=2);f.write('\n')
print(json.dumps({'jobs':campaign['initial_jobs'],'source':source['manifest_sha256'],'preflight':campaign['preflight']}))
