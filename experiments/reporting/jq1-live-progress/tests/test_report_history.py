"""Allocated CPU saved-history fixtures; no models or optimizer evaluation."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pytest
SCRIPTS=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(SCRIPTS))
import report_history as history
import report_data as data
import quick_inputs as inputs
spec=importlib.util.spec_from_file_location('_history_saved_fixture',Path(__file__).with_name('test_report_data.py'))
saved=importlib.util.module_from_spec(spec);spec.loader.exec_module(saved)

def fixture(root,dense):
 arm,segment,states,updates=saved.make_segment(root,n=20 if dense else 7,requested=20,status='passed' if dense else 'failed')
 case=Path(segment['case_directory']);segment.pop('retained_index')
 with np.load(case/'snapshots/step-0000.npz',allow_pickle=False) as src:arrays={k:src[k] for k in src.files}
 for index,state in enumerate(states):
  if dense or index==len(states)-1:
   values={**arrays,'contact_primal':np.asarray(state['contact_loss'],np.float32),**{k:np.asarray(state[k],np.float64) for k in ['refreshed_surrogate','autodiff_surrogate','repair_weight']}}
   p=case/('snapshots/step-'+str(index).zfill(4)+'.npz' if dense else 'latest-state.npz');np.savez(p,**values)
   if dense:state.update(snapshot='snapshots/'+p.name,snapshot_sha256=data.sha(p));state.pop('snapshot_retained',None)
   else:
    latest={**state,'snapshot':'latest-state.npz','snapshot_sha256':data.sha(p)};latest.pop('global_step',None);segment['latest_metadata']=saved.save(case/'latest-state.json',latest);segment['latest_snapshot']=saved.bind(p)
 segment['states']=saved.save_rows(case/'states.jsonl',states)
 segment['case_report']=saved.save(case/'report.json',{'states':states,'updates':updates})
 raw=inputs.load(segment['summary']['path']);raw['cases']=[{'states':states,'updates':updates}];segment['summary']=saved.save(Path(segment['summary']['path']),raw)
 if not dense:segment['summary']=segment['case_report'];segment['summary_role']='original_case_export_not_terminal_summary'
 segment['producer_submission']=saved.save(case/'submission.json',{'job_id':'123','status':'submitted'})
 segment['science_manifest']=saved.save(case/'source_manifest.json',{'files':{'config.json':segment['config']['sha256']}})
 segment.update(input_schema=history.DENSE if dense else history.PARTIAL,scientific_status='passed' if dense else 'unqualified',producer_status='passed' if dense else 'failed_original_LSF_saved_prefix')
 arm['scientific_status']='passed' if dense else 'unqualified'
 (root/'out').mkdir()
 return arm,segment

def test_dense_original_validates_every_frame_and_preserves_source_rows(tmp_path):
 arm,segment=fixture(tmp_path,True);value=history.collect_arm(arm,tmp_path/'out')
 assert value['global_start']==0 and value['latest_global_step']==20 and value['frames']['count']==2
 assert value['segment_proofs'][0]['original_dense_frames_validated']==21
 rows=inputs.load(tmp_path/'out/protenix-scalars.json')['rows'];assert rows[1]['state']['snapshot']=='snapshots/step-0001.npz'

def test_undisplayed_dense_snapshot_corruption_is_rejected(tmp_path):
 arm,segment=fixture(tmp_path,True);p=Path(segment['anchor_directory'])/'snapshots/step-0001.npz';p.write_bytes(p.read_bytes()+b'changed')
 with pytest.raises(ValueError,match='SHA changed'):history.collect_arm(arm,tmp_path/'out')

def test_terminal_partial_uses_actual_latest_and_never_claims_completion(tmp_path):
 arm,segment=fixture(tmp_path,False);value=history.collect_arm(arm,tmp_path/'out')
 assert value['committed_updates']==7 and value['frames']['count']==2 and not value['optimization_complete']
 assert value['scientific_status']=='unqualified' and value['segment_proofs'][0]['summary_role']=='original_case_export_not_terminal_summary'
 assert inputs.load(segment['case_report']['path'])['states'][-1]['snapshot_retained'] is False

@pytest.mark.parametrize('field',['contact_loss','step','preparation_identity'])
def test_partial_latest_must_match_actual_committed_endpoint(tmp_path,field):
 arm,s=fixture(tmp_path,False);p=Path(s['latest_metadata']['path']);v=inputs.load(p);v[field]='changed';s['latest_metadata']=saved.save(p,v)
 with pytest.raises(ValueError):history.collect_arm(arm,tmp_path/'out')

def test_partial_original_case_disagreement_is_rejected(tmp_path):
 arm,s=fixture(tmp_path,False);p=Path(s['case_report']['path']);v=inputs.load(p);v['updates'][0]['fixed_after']=999;s['case_report']=saved.save(p,v);s['summary']=s['case_report']
 with pytest.raises(ValueError,match='case export'):history.collect_arm(arm,tmp_path/'out')


def test_partial_signal_termination_remains_failed_and_reportable(tmp_path):
 p=tmp_path/'lsf.log';p.write_text('Subject: Job 123: <original> in cluster <dcc> Exited\nTERM_RUNLIMIT: job killed after reaching LSF run time limit.\nExited with signal termination: 9.\n')
 history.check_partial_lsf(p,'123')

@pytest.mark.parametrize('text',[
 'Subject: Job 124: <original> in cluster <dcc> Exited\nExited with signal termination: 9.',
 'Worker failure; TERM_RUNLIMIT; no terminal LSF header',
 'Subject: Job 123: <original> in cluster <dcc> Done\nSuccessfully completed.'])
def test_other_or_nonterminal_lsf_cannot_authorize_history(tmp_path,text):
 p=tmp_path/'lsf.log';p.write_text(text)
 with pytest.raises(ValueError):history.check_partial_lsf(p,'123')
