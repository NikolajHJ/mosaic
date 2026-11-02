"""Saved-array fixtures: allocated report CPU only, no model or optimizer."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import report_data as data
import report_exploratory as adapter
import quick_inputs as inputs
import quick_runtime as runtime

spec=importlib.util.spec_from_file_location('_exploratory_saved_fixture',Path(__file__).with_name('test_report_data.py'))
saved=importlib.util.module_from_spec(spec);spec.loader.exec_module(saved)


def fixture(root):
    arm,segment,states,updates=saved.make_segment(root)
    case=Path(segment['case_directory']);cfg=inputs.load(segment['config']['path']);cfg['steps']=20
    segment['config']=saved.save(Path(segment['config']['path']),cfg)
    identity={'model_digest':'a'*64,'context_digest':'b'*64}
    row={'research_profile':'exploratory','name':'protenix','start':0,'steps':20,'target':20,
         'rate':arm['rate'],'config':segment['config'],'science_source':{'manifest_sha256':segment['source_manifest_sha256']},'frozen_identity':identity}
    summary={'version':'mosaic-exploratory-continuation-v2','status':'completed_exploratory_updates','configuration':row,
        'name':'protenix','job_id':'123','global_start':0,'global_end':20,'global_target':20,'applied_updates':20,
        'derivative_qualified':False,'frozen_before':identity,'frozen_after':identity,'primal_scope':'primal_from_backward_call'}
    segment['summary']=saved.save(Path(segment['summary']['path']),summary)
    segment['case_report']=saved.save(case/'report.json',{'states':states,'updates':updates,'status':'passed'})
    segment['producer_plan']=saved.save(case/'producer-plan.json',{'arms':{'protenix':row}})
    segment['producer_submission']=saved.save(case/'submission.json',{'job_id':'123','plan':segment['producer_plan']})
    for key in ('producer_driver','producer_checksums','science_manifest'):segment[key]=saved.save(case/(key+'.json'),{'scope':'fixture'})
    segment.update(input_schema=adapter.SCHEMA,scientific_status='unqualified')
    arm.update(scientific_status='unqualified',primary_metric='contact_loss',schema_renderer_sha256='c'*64,
        expected_counts={'committed_updates':20,'scalar_states':21,'retained_frames':2,'latest_global_step':20})
    return arm,segment,summary


def test_completed_exploration_stays_unqualified_and_reuses_renderer(tmp_path):
    arm,segment,_=fixture(tmp_path);original=data.collect_segment
    (tmp_path/'out').mkdir()
    value=adapter.collect_arm(arm,tmp_path/'out')
    assert value['committed_updates']==20 and value['latest_global_step']==20 and value['optimization_complete']
    assert value['scientific_status']=='unqualified' and data.collect_segment is original
    assert value['segment_proofs'][0]['producer_status']=='completed_exploratory_updates'
    assert inputs.load(segment['summary']['path'])['status']=='completed_exploratory_updates'


@pytest.mark.parametrize('change',[
    {'status':'completed_device_smoke'},{'status':'running'},{'applied_updates':19},
    {'derivative_qualified':True},{'frozen_after':{'model_digest':'changed'}}])
def test_bad_or_disposable_completion_not_relabelled(tmp_path,change):
    _,_,summary=fixture(tmp_path)
    with pytest.raises(ValueError):adapter.header({**summary,**change})


def test_changed_raw_snapshot_still_rejected(tmp_path):
    arm,segment,_=fixture(tmp_path);states=inputs.load(segment['case_report']['path'])['states']
    path=Path(segment['anchor_directory'])/states[-1]['snapshot'];path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='SHA changed'):adapter.collect_arm(arm,tmp_path/'out')


def test_new_schema_identity_participates_in_cache_key(tmp_path):
    arm,_,_=fixture(tmp_path);other=deepcopy(arm);other['schema_renderer_sha256']='d'*64
    assert inputs.science_key(arm,'f'*64)!=inputs.science_key(other,'f'*64)


def test_runtime_uses_versioned_adapter(tmp_path):
    arm,_,_=fixture(tmp_path);arm['schema_renderer_sha256']=inputs.sha(SCRIPTS/'report_exploratory.py')
    manifest=runtime.render_plan({'edition':'edition-0001','captured_at':'fixture','arms':[arm]},tmp_path/'out','f'*64,[])
    assert manifest['arms'][0]['scientific_status']=='unqualified'


def test_helper_transport_is_local_only_and_preserves_command(tmp_path):
    import report as cli
    helper=r'C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe'
    connection=cli.Connection({'connection_helper':helper},tmp_path)
    argv=connection.args('python -',90)
    assert argv[:4]==[helper,'exec','--timeout','80'] and argv[-1].startswith('bash -lc ')
    assert 'connection_helper' in cli.LOCAL_ONLY_PROFILE


def test_cached_exploratory_wrapper_keeps_versioned_dispatch(tmp_path, monkeypatch):
    study = tmp_path / 'study'; study.mkdir()
    raw = {'version': 'mosaic-exploratory-continuation-v2',
           'status': 'completed_exploratory_updates'}
    (study / 'summary.json').write_bytes(inputs.encoded(raw))
    config = tmp_path / 'config.json'; config.write_bytes(inputs.encoded({}))
    hint = {'summary': inputs.binding(study / 'summary.json'),
            'config': inputs.binding(config), 'input_schema': adapter.SCHEMA,
            'scientific_status': 'unqualified', 'source_manifest_sha256': 'a' * 64,
            'job_id': '123'}
    seen = []
    def capture(spec, profile, path, summary):
        seen.append(summary)
        return 'versioned adapter', {}, 'fixed'
    monkeypatch.setattr(adapter, 'capture_segment', capture)
    result = inputs.capture_segment({'run': str(study)},
                                   {'allowed_roots': [str(tmp_path)]},
                                   {str(study): ({}, hint)})
    assert result[0] == 'versioned adapter' and seen == [raw]
