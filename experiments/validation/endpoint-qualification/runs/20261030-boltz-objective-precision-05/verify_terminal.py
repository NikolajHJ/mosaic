"""Validate terminal diagnostic evidence with stdlib only; no gradient admission."""
import argparse,base64,hashlib,importlib.util,json
from pathlib import Path
spec=importlib.util.spec_from_file_location('_own_source_and_CPU_audit',Path(__file__).with_name('verify_capture.py'))
cp=importlib.util.module_from_spec(spec);spec.loader.exec_module(cp)
def require(value,message):
 if not value:raise ValueError(message)
def verify(packet,arm):
 basic=cp.verify(packet);require(basic.get('cases')==82,'Complete independently audited CPU82 required')
 files={n:base64.b64decode(r['base64'],validate=True) for n,r in packet['files'].items()}
 def read(path):return json.loads(files[path])
 plan=read(cp.SOURCE+'/plan.json');require(arm in plan['arms'],'Undeclared arm')
 phase='gpu-'+arm;parent=cp.ROOT+'/'+phase;job=packet['submissions'][phase]['job_id']
 text=''.join(files[parent+'/stdout.'+job+'.log'].decode().split())
 require(text.count('Subject:Job'+job+':')==1 and 'inqueue<gpul40s>' in text and text.count('Successfullycompleted.')==1
   and 'Exitedwithexitcode' not in text,'Actual same GPUjob successful LSF required')
 payload=parent+'/payload';report=read(payload+'/assessment.json');preservation=read(payload+'/preservation.json')
 require(report['status']=='complete_objective_precision_observations_not_admission' and report['arm']==arm
   and report['optimizer_updates']==0 and report['can_admit_new_child'] is False
   and report['can_certify_or_restart_original_study'] is False and report['saved_results_changed'] is False
   and report['acceptance_tolerances_changed'] is False,'Diagnostic cannot be reinterpreted as admission')
 require(preservation['status']=='inputs_and_sources_preserved' and preservation['optimizer_updates']==0
   and preservation['can_admit_new_child'] is False,'Complete source/input preservation required')
 require(read(payload+'/execution.json')['returncode']==0,'Original owned child/descendant cleanup failed')
 predeclared=read(payload+'/predeclared-cohort.json');scales=[.01,.003,.001,.0003,.0001,.00003,.00001,.000003,.000001]
 require(predeclared['epsilons']==scales and predeclared['directions']==plan['arms'][arm]['directions']
   and predeclared['selection_before_evaluations'] is True and predeclared['after_result_pair_selection'] is False,'Original predeclaration differs')
 expected=[(d,e) for d in predeclared['directions'] for e in scales]
 require([(r['direction'],r['epsilon']) for r in report['observations']]==expected,'Missing/reordered/covertly selected cohorts')
 repeats=report['independent_primal_and_AD_baseline_repeats']
 require(set(repeats)=={'False','True'} and all(len(r)==3 for r in repeats.values()),'Independent baseline repeats missing')
 native=report['identity'];closed=report['close'];resident=read(payload+'/native/frozen-and-residency.json')
 require(native['status']=='ready_complete_objective_reference' and closed['status']=='closed'
   and native['optimizer_updates']==0 and resident['status']=='passed_exact_production_before_during_after_reference'
   and resident['optimizer_updates']==0,'Exact producer/frozen residency handshake missing')
 require(native['frozen']==closed['frozen']==resident['frozen'],'Frozen/model/context records differ')
 require(all(r['all_castbacks_exact'] for r in native['promotions']),'Full reference promotion/castback proof missing')
 responses=report['responses'];require([r['id'] for r in responses]==list(range(1,len(responses)+1))
   and closed['requests']==len(responses),'Native response sequence/close count differs')
 bindings={r['path']:r for r in packet['binary_bindings_remote_only']}
 references=[]
 def raw_binding(row):
  b=row['raw_point'];require(b['path'].startswith(payload+'/native/') and b['path'].endswith('.npz')
    and bindings.get(b['path'])==b,'Referenced raw NPZ not independently rehashed')
  references.append(b['path'])
 for row in [native['production_before'],native['production_during'],closed['production_after'],*responses]:raw_binding(row)
 require(len(set(references))==len(responses)+3,'Raw observation identity reused/replaced')
 reference_responses=[r for r in responses if 'operation_audit' in r]
 require(reference_responses and all(r['dtype']=='torch.float64' and r['operation_audit']['narrow_floating_operations']==0
   and r['operation_audit']['status']=='passed_full_operation_FP64_audit' for r in reference_responses),'Full reference FP64 audit missing')
 if arm=='boltz_chemistry':
  replay=read(payload+'/original-combined-replay.json')
  require(replay['status']=='passed_exact_original_fixed_teacher_scalar_gradient_replay'
    and replay['preparation_identity']==plan['arms'][arm]['preparation_identity']==report['fixed_preparation_identity']
    and replay['repair_weight']==.001 and replay['optimizer_updates']==0,'Original fixed teacher/combined replay differs')
  for r in report['observations']:
   require(len(r['cohorts'])==3 and r['cohorts'][0]['coefficient']==.001 and r['cohorts'][-1]['coefficient']==100.,'Low/actual/high cohorts missing')
 else:
  require(report['fixed_preparation_identity'] is None and all(len(r['cohorts'])==1 and r['cohorts'][0]['coefficient']==0.
    for r in report['observations']),'Binding-only objective gained chemistry')
 return {'status':'independent_terminal_Diagnostic_byte_schema_CPU82_LSF_frozen_raw_NPZ_preservation_verified',
   'arm':arm,'job_id':job,'raw_native_requests':len(responses),'raw_NPZ_bindings_verified':len(references),
   'all_predeclared_cohorts_retained':True,'optimizer_updates':0,'can_admit_new_child':False,
   'can_certify_or_restart_original_study':False,'original_failures_preserved':True,
   'scientific_gradient_agreement_not_classified_locally':True,'source_manifest_sha256':cp.PIN}
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('packet',type=Path);parser.add_argument('--arm',required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
 require(not a.output.exists(),'Immutable verification already exists');raw=a.packet.read_bytes();p=json.loads(raw);result=verify(p,a.arm)
 result['packet_sha256']=hashlib.sha256(raw).hexdigest();result['observed_utc']=p['observed_utc']
 with a.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
