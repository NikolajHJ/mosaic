"""Allocated CPU saved-DeFoG corruption, scope, capture and cache controls."""
from copy import deepcopy
from pathlib import Path
import json
import sys
from unittest.mock import patch

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import quick_inputs as inputs
import quick_runtime as runtime
import report_defog as adapter
import report_data as data


def fixture(root, steps=2):
    root.mkdir(exist_ok=True)
    (root/'frames').mkdir()
    config = {'version': 'protenix-defog-moving-teacher-test', 'n_atoms': 25, 'seed': 17, 'steps': steps, 'learning_rate': .3,
              'max_step_halvings': 12, 'max_graph_cell_tv': .02,
              'teacher': {'time_ramp': {'initial': .2, 'final': .95, 'last_update': steps-1}},
              'weight_ramp': {'initial': .001, 'final': .1, 'last_update': steps-1}}
    data.write(root/'config.json', config)
    x = np.full((25, 6), 1/6, dtype=np.float32)
    e = np.tile(np.asarray([.05, .8, .05, .05, .05], np.float32), (300, 1))
    full = np.zeros((25, 12), np.float32)
    full[:, [0, 1, 2, 9, 3, 6]] = x
    atoms, bonds = x.argmax(-1).tolist(), e.argmax(-1).tolist()
    raw = {'sanitized': False, 'disconnected': False, 'canonical_smiles': None,
           'elements': ['C']*25, 'element_indices': atoms, 'bond_upper_indices': bonds,
           'pairs': [[i,j] for i in range(25) for j in range(i+1,25)], 'conversion_error': 'Fixture excessive valence'}
    states, updates = [], []
    for k in range(steps+1):
        time = adapter.schedule(config, k)
        arrays = {'X': x, 'E_upper': e, 'atom_logits': np.log(x), 'bond_logits': np.log(e),
                  'teacher_X': x, 'teacher_E': e, 'full_native_teacher_X': full,
                  'native_gradient_X': np.zeros_like(x), 'native_gradient_E_upper': np.zeros_like(e),
                  'native_gradient_atom_logits': np.zeros_like(x), 'native_gradient_bond_logits': np.zeros_like(e),
                  'prior_gradient_atom_logits': np.zeros_like(x), 'prior_gradient_bond_logits': np.zeros_like(e),
                  'combined_gradient_atoms': np.zeros_like(x), 'combined_gradient_bonds': np.zeros_like(e),
                  'key_data': np.asarray([0,17], np.uint32), 'teacher_time': np.asarray(time,np.float64),
                  'teacher_state_index': np.asarray(k,np.int64)}
        path = root/'frames'/('state-'+str(k)+'.npz')
        np.savez_compressed(path, **arrays)
        graph_sha = adapter.transport_digest({'X': x.tolist(), 'E': e.tolist()})
        state = {'state': k, 'teacher_state_index': k, 'teacher_time': time, 'weight': adapter.coefficient(config,k),
                 'native_loss': -.2, 'contact_score': .2, 'prior_kl': 0., 'refreshed_surrogate': -.2,
                 'teacher_drift_max_tv': None if k==0 else 0., 'snapshot': str(path.relative_to(root)),
                 'snapshot_sha256': data.sha(path), 'graph_sha256': graph_sha,
                 'teacher_sha256': graph_sha, 'raw_readout': deepcopy(raw),
                 'teacher_query_sha256': adapter.transport_digest({'graph_sha256':graph_sha,'state_index':k,'teacher_time':time})}
        states.append(state)
        if k:
            before = states[k-1]
            updates.append({'update':k,'teacher_state_index':k-1,'teacher_time':before['teacher_time'],
                'weight':before['weight'],'teacher_sha256':before['teacher_sha256'],
                'teacher_query_sha256':before['teacher_query_sha256'],'native_before':-.2,'native_after':-.2,
                'fixed_teacher_before':-.2,'fixed_teacher_after':-.2,'fixed_teacher_delta':0.,
                'rate':.3,'halvings':0,'max_graph_cell_tv':0.})
    for name, rows in (('states',states),('updates',updates)):
        (root/(name+'.jsonl')).write_bytes(b''.join(inputs.encoded(r) for r in rows))
    data.write(root/'summary.json', {'status':'failed','completed_updates':steps,'job':'123',
        'source_manifest_sha256':'a'*64,'config_sha256':data.sha(root/'config.json'),
        'error':'Original terminal combined atom gradient failed'})
    (root/'gpu.out').write_text('Subject: Job 123: fixture\nExited with exit code 1.\n')
    (root/'gpu.err').write_text('Original failure')
    segment = {'id':'fixture','case_directory':str(root),'anchor_directory':str(root),'global_offset':0,
        'requested_updates':steps,'source_manifest_sha256':'a'*64,'job_id':'123','scientific_status':'failed',
        **{n:inputs.binding(root/(n+'.jsonl' if n in ('states','updates') else n+'.json'))
           for n in ('states','updates','summary','config')}}
    arm = {'id':'protenix_defog','label':'DeFoG fixture','kind':'defog','rate':.3,'global_start':0,
        'global_target':steps,'target_updates':steps,'segments':[segment],'scientific_status':'failed',
        'primary_metric':'contact_loss','status_detail':'Original failure preserved','continuation_status':'not_planned',
        'schema_renderer_sha256':inputs.sha(SCRIPTS/'report_defog.py')}
    return arm, states, updates, config


def render(root, arm):
    out = root/'out'
    out.mkdir()
    return adapter.collect_arm(arm,out)


def test_saved_defog_all_matrices_validated_failed_verdict_and_no_scoring(tmp_path):
    arm, states, _, _ = fixture(tmp_path/'study')
    result = render(tmp_path, arm)
    assert result['scientific_status']=='failed' and result['committed_updates']==2
    assert result['saved_frames_count']==3 and result['frames']['steps']==[0,2]
    assert result['validity']=={'saved_states':3,'sanitized':0,'valid_connected':0,'distinct_valid_identities':0}
    rows = data.load(tmp_path/'out/protenix_defog-scalars.json')['rows']
    assert [r['state'] for r in rows]==states
    assert rows[-1]['losses']['teacher_time']==.95 and rows[-1]['losses']['contact_loss']==-.2
    assert len(result['segment_proofs'][0]['all_saved_snapshots'])==3


@pytest.mark.parametrize('field', ['teacher_time','teacher_state_index','X','teacher_X','combined_gradient_atoms'])
def test_corrupt_saved_defog_arrays_fail_even_with_rebound_file_hash(tmp_path, field):
    arm, states, _, config = fixture(tmp_path/'study')
    path = tmp_path/'study'/states[1]['snapshot']
    with np.load(path) as archive:
        arrays={name:archive[name] for name in archive.files}
    arrays[field] = arrays[field]+np.asarray(1,dtype=arrays[field].dtype)
    np.savez_compressed(path,**arrays)
    states[1]['snapshot_sha256']=data.sha(path)
    with pytest.raises(ValueError):
        adapter.validate_snapshot(path,states[1],config,False)


def test_undisplayed_defog_frame_corruption_is_not_hidden(tmp_path):
    arm, states, _, _ = fixture(tmp_path/'study')
    path=tmp_path/'study'/states[1]['snapshot']
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='SHA changed'):
        render(tmp_path,arm)


@pytest.mark.parametrize('field,value', [('teacher_time',.8),('weight',.7),('teacher_query_sha256','0'*64),('native_after',-.1)])
def test_defog_update_corruption_fails(tmp_path,field,value):
    arm, _, updates, _ = fixture(tmp_path/'study')
    updates[0][field]=value
    p=tmp_path/'study/updates.jsonl';p.write_bytes(b''.join(inputs.encoded(r) for r in updates))
    arm['segments'][0]['updates']=inputs.binding(p)
    with pytest.raises(ValueError):render(tmp_path,arm)


def test_defog_failed_outcome_cannot_be_relabelled(tmp_path):
    arm, _, _, _=fixture(tmp_path/'study');arm['scientific_status']='passed'
    with pytest.raises(ValueError,match='status differs'):render(tmp_path,arm)


def test_defog_discovery_accepts_original_job_and_state_schema(tmp_path):
    arm, _, _, _ = fixture(tmp_path/'study')
    spec={'run':str(tmp_path/'study'),'config':str(tmp_path/'study/config.json'),
          'stdout':str(tmp_path/'study/gpu.out'),'stderr':str(tmp_path/'study/gpu.err')}
    segment, config, kind=inputs.capture_segment(spec,{'allowed_roots':[str(tmp_path)]},{})
    assert kind=='defog' and segment['job_id']=='123' and segment['scientific_status']=='failed'
    assert segment['expected_counts']=={'committed_updates':2,'scalar_states':3,'retained_frames':2}
    assert 'retained_index' not in segment


def test_defog_adapter_changes_invalidate_only_defog_science_cache(tmp_path):
    arm, _, _, _=fixture(tmp_path/'study')
    before=inputs.science_key(arm,'f'*64)
    arm['schema_renderer_sha256']='b'*64
    assert inputs.science_key(arm,'f'*64)!=before


def test_defog_renders_through_canonical_engine_and_reuses_verified_cache(tmp_path):
    arm,_,_,_=fixture(tmp_path/'study')
    arm['expected_counts']={'committed_updates':2,'scalar_states':3,'retained_frames':2,'latest_global_step':2}
    plan={'edition':'edition-0001','captured_at':'2026-10-28T00:00:00Z','arms':[arm]}
    out=tmp_path/'first';value=runtime.render_plan(plan,out,'f'*64,[])
    seed={'root':out,'plan':plan,'manifest':value,'renderer_sha256':'f'*64,
          'receipt':{'files':{p.name:{'bytes':p.stat().st_size,'sha256':data.sha(p)} for p in out.iterdir()}}}
    with patch.object(adapter,'collect_arm',side_effect=AssertionError('Rebuilt unchanged DeFoG arm')):
        reused=runtime.render_plan(plan,tmp_path/'second','f'*64,[seed])
    assert reused['arms']==value['arms']
    path=tmp_path/'study/frames/state-1.npz';path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='length changed|SHA changed'):
        runtime.render_plan(plan,tmp_path/'third','f'*64,[seed])
