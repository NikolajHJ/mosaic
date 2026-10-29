"""Read-only forecast for the exact source144 timing probe; no foreign imports.

Hold a separate512MiB/3000-file forecast above its complete baseline. The sealed
probe writes two points, fourteen19216-byte saved-state copies and small timing
JSON. This is conservative admission headroom, not a new external hard cap.
All actual selected bytes are stale-charged; no running external job is changed.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat

TAG = 'source144-embedding-runtime-probe-attempt002'
WORK = '/work3/s234818/latentmol'
SOURCE = WORK + '/releases/source-144-embedding-runtime-probe-attempt002'
ROOTS = (SOURCE, WORK+'/runs/'+TAG, WORK+'/submissions/'+TAG)
VERSION = 'mosaic-authenticated-runtime-probe-quota-v1'
MANIFEST = '1f7b5c9b19f05e1077f8b1f83ae8632f0981ccc827c2d35824a53b01b9a9beec'
PLAN = '3f89c017315e6537b4352c550fdac3e1ee2bfb9cf0549e681b10bade52facfbe'
CPU = 'e36bc8350d68c82da378032de2f16daeb3682f42c013399b6b54b46ea7588dcc'


def require(value, message):
    if not value: raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def binding(path):
    p=Path(path); i=p.lstat()
    require(stat.S_ISREG(i.st_mode) and all(not q.is_symlink() for q in (p,*p.parents)), 'Probe proof linked/special')
    with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p),'bytes':i.st_size,'sha256':digest}


def validate_profile(p):
    require(set(p)=={'version','proofs','baseline','growth_bytes','growth_files','job_id','command'}, 'Probe profile fields differ')
    require(p['version']==VERSION and p['job_id']=='29846982' and set(p['baseline'])==set(ROOTS), 'Probe exact identity differs')
    require(p['growth_bytes']==512*1024**2 and p['growth_files']==3000, 'Probe forecast differs')
    for row in p['baseline'].values():
        require(set(row)=={'bytes','files'} and all(type(v) is int and v>=0 for v in row.values()), 'Probe baseline malformed')
    expected={SOURCE+'/embedding-release-manifest.json':MANIFEST,
              SOURCE+'/configs/embedding_runtime_probe.json':PLAN,
              ROOTS[1]+'/cpu/result.json':CPU}
    require(type(p['proofs']) is list and len({r['path'] for r in p['proofs']})==len(p['proofs']), 'Probe proofs missing/duplicated')
    proofs={r['path']:r['sha256'] for r in p['proofs']}
    require(all(proofs.get(k)==v for k,v in expected.items()), 'Probe immutable source/config/CPU pins differ')
    require(p['command'].startswith('#!/bin/bash;set -euo pipefail;cd '+SOURCE+';')
            and '--output '+ROOTS[1]+'/gpu' in p['command']
            and '--plan-sha256 '+PLAN in p['command'] and p['command'].endswith('--cpu-sha256 '+CPU), 'Probe actual command differs')
    return p


def inventory(root):
    root=Path(root); total=files=0
    require(root.is_dir() and all(not q.is_symlink() for q in (root,*root.parents)), 'Probe inventory missing/linked')
    for d,ds,fs in os.walk(root,followlinks=False):
        for name in ds: require(stat.S_ISDIR(Path(d,name).lstat().st_mode), 'Probe directory linked/special')
        for name in fs:
            i=Path(d,name).lstat();require(stat.S_ISREG(i.st_mode), 'Probe file linked/special')
            total+=max(i.st_size,getattr(i,'st_blocks',0)*512);files+=1
    return {'root':str(root),'actual_bytes':total,'actual_files':files,'complete':True,'nofollow':True}


def observe(p,scheduler):
    validate_profile(p)
    for proof in p['proofs']:require(binding(proof['path'])==proof, 'Probe immutable evidence changed')
    manifest=read(SOURCE+'/embedding-release-manifest.json')
    members=manifest.get('files',manifest.get('members'))
    require(type(members) is dict and len(members)==629, 'Probe full sealed inventory differs')
    for name,pin in members.items():
        rel=PurePosixPath(name)
        require(not rel.is_absolute() and '..' not in rel.parts and str(rel)==name, 'Probe unsafe source member')
        require(binding(Path(SOURCE)/name)['sha256']==pin, 'Probe source member changed')
    intent=read(ROOTS[2]+'/gpu-intent.json');receipt=read(ROOTS[2]+'/gpu-receipt.json')
    require(receipt['job_id']==p['job_id'] and receipt['returncode']==0 and receipt['status']=='submitted_not_passed'
            and receipt['intent_sha256']==binding(ROOTS[2]+'/gpu-intent.json')['sha256']
            and intent['identity']['manifest_sha256']==MANIFEST and intent['identity']['plan_sha256']==PLAN,
            'Probe actual submission/CPU ancestry differs')
    rows=[inventory(root) for root in ROOTS]
    # These exact two scheduler logs are outside the three selected trees.
    for suffix in ('out','err'):
        f=Path(WORK+'/logs/'+TAG+'-gpu.'+p['job_id']+'.'+suffix)
        if f.exists():
            i=f.lstat();require(stat.S_ISREG(i.st_mode) and all(not q.is_symlink() for q in f.parents), 'Probe log linked/special')
            rows.append({'root':str(f),'actual_bytes':max(i.st_size,getattr(i,'st_blocks',0)*512),'actual_files':1,'complete':True,'nofollow':True})
    actual=sum(r['actual_bytes'] for r in rows); files=sum(r['actual_files'] for r in rows)
    cap=sum(r['bytes'] for r in p['baseline'].values())+p['growth_bytes']
    capfiles=sum(r['files'] for r in p['baseline'].values())+p['growth_files']
    require(actual<=cap and files<=capfiles, 'Probe exceeded held forecast; block own admission')
    ids=[]
    for row in scheduler:
        if row['JOBID']==p['job_id']:
            require(row['QUEUE']=='gpul40s' and row['COMMAND']==p['command'], 'Probe actual scheduler command differs')
            ids.append(row['JOBID'])
    return {'version':VERSION,'accounted_job_ids':ids,'source_files_authenticated':len(members),'inventories':rows,
            'actual_bytes':actual,'actual_files':files,'remaining_bytes':cap-actual,'remaining_files':capfiles-files,
            'forecast_held_until_Mosaic_work_ends':True,'external_jobs_settings_results_changed':False}


def augment(result,observation):
    value=dict(result)
    for resource in ('bytes','files'):
        value['available_'+resource]-=observation['actual_'+resource]
        value['required_'+resource]+=observation['remaining_'+resource]
    value['byte_margin']=value['available_bytes']-value['required_bytes']
    value['file_margin']=value['available_files']-value['required_files']
    value['status']='passed' if value['byte_margin']>=0 and value['file_margin']>=0 else 'failed'
    return value
