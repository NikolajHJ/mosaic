"""Failure denominators and paired sulfur accounting in saved-result summaries."""
import importlib.util
import json
from pathlib import Path

import pytest


def load_summary(monkeypatch):
    folder=Path(__file__).resolve().parents[2]/'experiments'/'chemistry-guidance'
    monkeypatch.syspath_prepend(str(folder))
    spec=importlib.util.spec_from_file_location('exchange_summary',folder/'summarize_exchange.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def cohort(folder,seed):
    folder.mkdir()
    def row(index,valid,sulfur=0):
        return dict(index=index,valid=valid,smiles=('C' if index==0 else 'CC') if valid else None,
            hard_sulfur=sulfur,expected_sulfur=float(sulfur),reference_hard_sulfur=0,
            substructure_changes=0,atom_exchanges=0,perturbation_norm=3.)
    baseline=[row(0,True),row(1,True)]
    guided=[row(0,False,1),row(1,True)]
    config=dict(seed=seed,count=2,updates=1,perturbation_norm=3.,arms=dict(bond_only=None,guided={}))
    results={arm:dict(endpoints=ends,valid=sum(r['valid'] for r in ends),checks=2,censored=0,
        invalid_attempts=1,target_edits=dict(atom=0,charge=0,bond=0)) for arm,ends in [('bond_only',baseline),('guided',guided)]}
    (folder/'result.json').write_text(json.dumps(dict(config=config,checkpoint_sha256='test',
        frozen_decoder_unchanged=True,elapsed_seconds=1.,arms=results)))
    trajectory=[]
    for arm,ends in [('bond_only',baseline),('guided',guided)]:
        for index in range(2):
            trajectory.append(dict(row(index,index==0),arm=arm,step=0))
            trajectory.append(dict(ends[index],arm=arm,step=1))
    (folder/'trajectory.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in trajectory))


def test_invalid_outputs_stay_in_validity_and_sulfur_denominators(tmp_path,monkeypatch):
    module=load_summary(monkeypatch)
    folders=[tmp_path/'a',tmp_path/'b']
    for seed,folder in enumerate(folders):cohort(folder,seed)
    result=module.summarize(folders);row=result['arms']['guided']
    assert result['starts']==4
    assert row['diversity']['validity']==.5
    assert row['hard_sulfur_drift']['mean']==.5  # Invalid molecules still counted.
    assert row['recovered']==2 and row['broke_initially_valid']==2
    assert row['common_valid_count']==2
    assert row['final_validity_difference_vs_bond']['mean']==-.5


def test_duplicate_cohort_is_rejected(tmp_path,monkeypatch):
    module=load_summary(monkeypatch);folder=tmp_path/'a';cohort(folder,0)
    with pytest.raises(ValueError,match='Duplicate'):module.summarize([folder,folder])


def test_native_missing_cohort_is_not_reported_as_success(tmp_path,monkeypatch):
    result=load_summary(monkeypatch).summarize_native([tmp_path/'missing'])
    assert result['status']=='incomplete'


def test_diversity_reports_unparseable_saved_valid_smiles(monkeypatch):
    module=load_summary(monkeypatch)
    bad='Cc1c2c(ccs1F)=CC(C(=O)NC1CCCCC1NC(C)N)=C2F'
    result=module.diversity([dict(valid=True,smiles='C'),dict(valid=True,smiles=bad),dict(valid=False,smiles=None)])
    assert result['valid']==2 and result['attempts']==3 and result['validity']==2/3
    assert result['fingerprint_evaluable']==1 and result['unparseable_valid_smiles']==[bad]
    assert result['unparseable_valid']==1 and result['mean_pairwise_morgan_distance'] is None


def test_native_invalid_endpoint_keeps_loss_unknown_and_counts_clipping(tmp_path,monkeypatch):
    folder=tmp_path/'native';records=folder/'current'/'arms'/'latent';records.mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value))
    def lines(path,values):path.write_text(''.join(json.dumps(v)+'\n' for v in values))
    write(folder/'comparison.json',dict(status='completed',seed=1,start=dict(smiles='C'),
        config=dict(arms=dict(current=None),design=dict(updates=50))))
    write(folder/'current'/'result.json',dict(status='completed',technical_checks={k:True for k in
        ('finite_required_binding_path','finite_chemistry_path','frozen_decoder_unchanged','frozen_native_close_completed')}))
    write(folder/'current'/'run.json',dict(native_identity=dict(model='toy')))
    lines(records/'candidates.jsonl',[dict(step=i,valid=i<50,smiles='C' if i<50 else None) for i in range(51)])
    lines(records/'metrics.jsonl',[dict(step=i,score=dict(binding_loss=-i/100)) for i in range(51)])
    lines(records/'updates.jsonl',[dict(step=i,bound_factor=.5) for i in range(50)])
    write(folder/'current'/'chemistry-observations.json',[dict(step=i,hard_sulfur=0,expected_sulfur=i/100,
        atom_exchanges=0,substructure_changes=0) for i in range(51)])
    write(folder/'current'/'hard-readouts.json',[dict(role='initial',step=0,loss=-.1),dict(role='endpoint',step=50,loss=None)])
    result=load_summary(monkeypatch).summarize_native([folder]);arm=result['stages']['50']['current']
    assert arm['endpoint_diversity']['valid']==0 and arm['valid_source_fraction']==1
    assert arm['per_seed'][0]['bound_hits']==50
    assert result['hard_readouts']['current'][1]['loss_delta_from_initial'] is None
    comparison=json.loads((folder/'comparison.json').read_text())
    comparison['config']['seeds']=[1,2,3]
    write(folder/'comparison.json',comparison)
    partial=load_summary(monkeypatch).summarize_native([folder])
    assert partial['status']=='partial' and partial['missing_seeds']==[2,3]
    assert partial['seeds']==[1] and '1 completed independent start(s) of 3 planned' in partial['limitations'][2]
