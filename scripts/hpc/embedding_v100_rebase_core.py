"""Create an isolated, provenance-bound V100 trajectory clone after a passed cache audit.

Never mutates the source trajectory. This helper does not dispatch jobs, cancel
jobs, waive original scientific checks, or adopt its output into production.
Call only inside an allocated HPC job after separately authenticating inputs.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def write_bytes(path, data):
    path=Path(path)
    with path.open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    return digest(path)


def write_json(path, value):
    return write_bytes(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())


def pinned(asset):
    path=Path(asset['path']).resolve()
    if digest(path)!=asset['sha256']: raise ValueError('Pinned source artifact changed: '+str(path))
    return path


def validate_audit(audit, case, state_sha, old_cache, cache_sha):
    if audit.get('status')!='passed_cross_device_cache_audit_not_migrated' or audit.get('production_mutation') is not False:
        raise ValueError('Require completed isolated V100 hard-score audit')
    rows=[row for row in audit['cases'] if row['case']==case]
    if len(rows)!=1:raise ValueError('Missing or duplicate audited case')
    row=rows[0]
    if (row['a100_cache_sha256']!=cache_sha or row['hard_graph_sha256']!=old_cache['hard_graph_sha256']
            or row['all_within_native_scalar_tolerance'] is not True or set(row['scores'])!=set(old_cache['scores'])):
        raise ValueError('Audit is not bound to this exact A100 cache')
    scores={}
    for key,record in row['scores'].items():
        if (record['a100']!=old_cache['scores'][key] or record['within_tolerance'] is not True
                or type(record['v100']) not in (int,float) or not math.isfinite(record['v100'])):
            raise ValueError('Score evidence does not match old cache')
        scores[key]=record['v100']
    if not state_sha or len(state_sha)!=64:raise ValueError('Missing immutable resume digest')
    return row,scores


def rebase_case(*,case,source_root,source_state,source_manifest,source_cache,audit,output_root,expected_update):
    # CPU Torch deserialization occurs only in an allocated HPC process.
    import torch
    source_root=Path(source_root).resolve();output_root=Path(output_root).resolve()
    if output_root.exists() or output_root.with_name(output_root.name+'.writing').exists():
        raise FileExistsError('Preserve previous migration attempt')
    state_path=pinned(source_state);manifest_path=pinned(source_manifest);cache_path=pinned(source_cache)
    if any(not p.is_relative_to(source_root) for p in (state_path,manifest_path,cache_path)):
        raise ValueError('Source artifact escaped its authenticated original trajectory')
    state=torch.load(state_path,map_location='cpu',weights_only=True)
    manifest=json.loads(manifest_path.read_text());old_cache=json.loads(cache_path.read_text())
    if (state['case_id']!=case or state['update']!=expected_update or not state['gate_binding_verified']
            or state['signature_sha256']!=manifest['sha256'] or json_digest(manifest['signature'])!=manifest['sha256']
            or old_cache['signature_sha256']!=manifest['sha256'] or state['hard_cache_entries']!=1
            or state['last_hard_graph_sha256']!=old_cache['hard_graph_sha256']
            or state['best_valid']['hard_graph_sha256']!=old_cache['hard_graph_sha256']):
        raise ValueError('Only the exact single-graph authenticated saved trajectory can be rebased')
    index_path=source_root/'hard-cache/index.jsonl';rows=[json.loads(line) for line in index_path.read_text().splitlines()]
    if (len(rows)!=1 or rows[0]['key']!=old_cache['hard_graph_sha256'] or rows[0]['sha256']!=source_cache['sha256']
            or rows[0]['previous']!=hashlib.sha256(b'').hexdigest() or json_digest(rows[0])!=state['hard_cache_chain']):
        raise ValueError('Original hard-cache chain did not authenticate')
    audit_row,scores=validate_audit(audit,case,source_state['sha256'],old_cache,source_cache['sha256'])
    if audit_row['update']!=expected_update:raise ValueError('Audited latent update changed')
    old_best=pinned(state['best_valid']['tensors'])
    if not old_best.is_relative_to(source_root/'best'):raise ValueError('Best artifact escaped source trajectory')
    temp=output_root.with_name(output_root.name+'.writing');temp.mkdir(parents=True,exist_ok=False)
    for name in ('states','segments','observations','hard-cache','best'): (temp/name).mkdir()
    write_bytes(temp/'manifest.json',manifest_path.read_bytes())
    new_cache=copy.deepcopy(old_cache);new_cache['scores']=scores
    cache_destination=temp/'hard-cache'/cache_path.name
    cache_sha=write_json(cache_destination,new_cache)
    new_entry=dict(key=old_cache['hard_graph_sha256'],sha256=cache_sha,previous=hashlib.sha256(b'').hexdigest())
    write_bytes(temp/'hard-cache/index.jsonl',(json.dumps(new_entry,sort_keys=True,allow_nan=False)+'\n').encode())
    best_path=temp/'best'/old_best.name
    write_bytes(best_path,old_best.read_bytes())
    rebased=copy.deepcopy(state)
    rebased['hard_cache_chain']=json_digest(new_entry)
    rebased['best_valid']['hard_loss']=scores['dbmol_total_loss']
    rebased['best_valid']['tensors']=dict(path=str(output_root/'best'/old_best.name),sha256=digest(best_path))
    filename='state-%06d.pt'%expected_update; state_target=temp/'states'/filename
    with state_target.open('xb') as stream:
        torch.save(rebased,stream);stream.flush();os.fsync(stream.fileno())
    verify=torch.load(state_target,map_location='cpu',weights_only=True)
    if (verify['update']!=state['update'] or verify['counts']!=state['counts'] or
            not torch.equal(verify['z'],state['z']) or not torch.equal(verify['z0'],state['z0'])):
        raise ValueError('Rebased latent and cumulative progress changed')
    evidence=dict(version='v100-rebased-trajectory-v1',case=case,update=expected_update,
        source_state=source_state,source_manifest=source_manifest,source_cache=source_cache,
        source_hard_score=old_cache['scores'],v100_hard_score=scores,
        hard_graph_sha256=old_cache['hard_graph_sha256'],rebased_cache_sha256=cache_sha,
        rebased_state_sha256=digest(state_target),prior_observation_paths_remain_source142=True,
        mutable_production_adopted=False,continuation_requires_separate_atomic_cutover=True)
    write_json(temp/'v100-handoff-evidence.json',evidence)
    temp.rename(output_root)
    return dict(path=str(output_root/'states'/filename),sha256=evidence['rebased_state_sha256'],evidence=evidence)
