"""Allocated CPU tests for isolated cross-device trajectory rebasing."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

PATH=Path(__file__).resolve().parents[1]/'scripts/hpc/embedding_v100_rebase_core.py'
spec=importlib.util.spec_from_file_location('rebase_core',PATH);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def asset(path):return dict(path=str(path.resolve()),sha256=module.digest(path))


@pytest.fixture
def example(tmp_path):
    source=tmp_path/'original';(source/'states').mkdir(parents=True);(source/'hard-cache').mkdir();(source/'best').mkdir()
    signature=dict(case='case-001',source='fixed');signature_sha=module.json_digest(signature)
    manifest=source/'manifest.json';module.write_json(manifest,dict(signature=signature,sha256=signature_sha))
    key='a'*64;scores=dict(dbmol_total_loss=8.0,affinity_pred_value=2.0)
    cache=source/'hard-cache'/(key+'.json');module.write_json(cache,dict(signature_sha256=signature_sha,context_sha256='b'*64,hard_graph_sha256=key,scores=scores))
    entry=dict(key=key,sha256=module.digest(cache),previous=hashlib.sha256(b'').hexdigest())
    module.write_bytes(source/'hard-cache/index.jsonl',(json.dumps(entry,sort_keys=True)+'\n').encode())
    best=source/'best/best-000000.pt';torch.save(dict(graph=torch.tensor([1,2])),best)
    z0=torch.tensor([[0.25,0.5]]);z=torch.tensor([[0.3,0.55]])
    state=dict(case_id='case-001',update=1899,gate_binding_verified=True,signature_sha256=signature_sha,hard_cache_entries=1,last_hard_graph_sha256=key,hard_cache_chain=module.json_digest(entry),z0=z0,z=z,counts={'iterations':1899,'applied':1899},rng={'torch':torch.get_rng_state()},best_valid=dict(hard_graph_sha256=key,hard_loss=8.0,tensors=asset(best)))
    state_path=source/'states/state-001899.pt';torch.save(state,state_path)
    row=dict(case='case-001',update=1899,a100_cache_sha256=module.digest(cache),hard_graph_sha256=key,all_within_native_scalar_tolerance=True,scores={k:dict(a100=v,v100=v-1e-6,within_tolerance=True) for k,v in scores.items()})
    audit=dict(status='passed_cross_device_cache_audit_not_migrated',production_mutation=False,cases=[row])
    args=dict(case='case-001',source_root=source,source_state=asset(state_path),source_manifest=asset(manifest),source_cache=asset(cache),audit=audit,output_root=tmp_path/'v100-clone',expected_update=1899)
    return args,state,cache,source


def test_rebase_preserves_latent_rng_counts_and_original_files(example):
    args,state,cache,source=example;before={str(p):module.digest(p) for p in source.rglob('*') if p.is_file()}
    result=module.rebase_case(**args);saved=torch.load(result['path'],map_location='cpu',weights_only=True)
    assert torch.equal(saved['z'],state['z']) and torch.equal(saved['z0'],state['z0'])
    assert torch.equal(saved['rng']['torch'],state['rng']['torch']) and saved['counts']==state['counts']
    assert saved['best_valid']['hard_loss']==pytest.approx(7.999999)
    assert Path(saved['best_valid']['tensors']['path']).is_file()
    assert module.digest(saved['best_valid']['tensors']['path'])==saved['best_valid']['tensors']['sha256']
    new_cache=json.loads((args['output_root']/'hard-cache'/cache.name).read_text())
    assert new_cache['scores']['dbmol_total_loss']==pytest.approx(7.999999)
    entry=json.loads((args['output_root']/'hard-cache/index.jsonl').read_text())
    assert entry['sha256']==module.digest(args['output_root']/'hard-cache'/cache.name)
    assert saved['hard_cache_chain']==module.json_digest(entry)
    assert before=={str(p):module.digest(p) for p in source.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):module.rebase_case(**args)


@pytest.mark.parametrize('corruption',['audit_status','audit_mutated','audit_score','audit_graph','audit_failed','missing_score','source_hash','wrong_update'])
def test_untrusted_or_stale_evidence_cannot_create_clone(example,corruption):
    args,_,_,_=example;args=copy.deepcopy(args);row=args['audit']['cases'][0]
    if corruption=='audit_status':args['audit']['status']='failed'
    elif corruption=='audit_mutated':args['audit']['production_mutation']=True
    elif corruption=='audit_score':row['scores']['dbmol_total_loss']['a100']=9.0
    elif corruption=='audit_graph':row['hard_graph_sha256']='c'*64
    elif corruption=='audit_failed':row['all_within_native_scalar_tolerance']=False
    elif corruption=='missing_score':del row['scores']['dbmol_total_loss']
    elif corruption=='source_hash':args['source_state']['sha256']='0'*64
    else:args['expected_update']=1900
    with pytest.raises(ValueError):module.rebase_case(**args)
    assert not args['output_root'].exists()
