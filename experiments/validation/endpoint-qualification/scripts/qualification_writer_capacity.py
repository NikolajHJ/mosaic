"""Read-only source146 forecast; no foreign code import, job or file mutation."""
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import stat
import time

WORK='/work3/s234818/latentmol'
TAG='source146-embedding-v100-qualification-attempt001'
SOURCE=WORK+'/releases/source-146-embedding-v100-qualification-attempt001'
ROOTS=(SOURCE,WORK+'/runs/'+TAG,WORK+'/submissions/'+TAG)
MANIFEST='f288580e454e7d5fbc91fe76260e7f016f15908624f8bb15fe2a0ae63d92399a'
ARCHIVE='68b10f1cc0702ef8df6b34ce5956177a45d51bbcc0de78d570a379a7ce3eb64c'
PLAN='e1355e4b638c6f6bda2cdc6ba0e62cde9504dfeb5bf251bcaf10ec43b26a5afa'
CPU='b6f4f26ca1c7f1ccc89b52e0f9bf21e2752bf14353cca3273511a67e97f150b8'

def require(value,message):
    if not value:raise ValueError(message)

def read(p):return json.loads(Path(p).read_bytes())

def binding(path):
    p=Path(path);s=p.lstat()
    require(stat.S_ISREG(s.st_mode) and all(not q.is_symlink() for q in (p,*p.parents)),'Foreign proof linked/special')
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p),'bytes':s.st_size,'sha256':h}

def validate(profile):
    require(profile['job_id']=='29852732' and profile['source_files_authenticated']==638,'Exact source146 identity required')
    require(set(profile['baseline'])==set(ROOTS) and profile['growth_bytes']==512*1024**2 and profile['growth_files']==3000,
            'Independent full512MiB/3000file future forecast required')
    for row in profile['baseline'].values():
        require(set(row)=={'bytes','files'} and all(type(v) is int and v>=0 for v in row.values()),'Incomplete baseline')
    expected={SOURCE+'/embedding-release-manifest.json':MANIFEST,SOURCE+'/configs/embedding_v100_qualification.json':PLAN,
        ROOTS[1]+'/cpu/result.json':CPU,WORK+'/releases/source-146-embedding-v100-qualification-attempt001.tar.gz':ARCHIVE}
    proof={r['path']:r['sha256'] for r in profile['proofs']}
    require(all(proof.get(p)==h for p,h in expected.items()),'Foreign source/archive/CPU/plan pins differ')
    require(profile['command'].startswith('#!/bin/bash;set -euo pipefail;cd '+SOURCE+';')
        and '--output '+ROOTS[1]+'/gpu' in profile['command'] and '--plan-sha256 '+PLAN in profile['command']
        and profile['command'].endswith('--cpu-sha256 '+CPU),'Exact foreign accepted command differs')

def inventory(root):
    root=Path(root);require(root.is_dir() and all(not p.is_symlink() for p in (root,*root.parents)),'Foreign root linked/missing')
    total=count=0
    for d,ds,fs in os.walk(root,followlinks=False):
        for n in ds:require(stat.S_ISDIR(Path(d,n).lstat().st_mode),'Foreign directory linked/special')
        for n in fs:
            s=Path(d,n).lstat();require(stat.S_ISREG(s.st_mode),'Foreign output linked/special')
            total+=max(s.st_size,getattr(s,'st_blocks',0)*512);count+=1
    return {'root':str(root),'actual_bytes':total,'actual_files':count,'complete':True,'nofollow':True}

def observe(profile,rows):
    validate(profile)
    for p in profile['proofs']:require(binding(p['path'])==p,'Foreign immutable proof changed')
    members=read(SOURCE+'/embedding-release-manifest.json')['members'];require(len(members)==638,'Foreign manifest count differs')
    for name,h in members.items():
        rel=PurePosixPath(name);require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in name,'Unsafe source path')
        require(binding(Path(SOURCE)/name)['sha256']==h,'Foreign source member changed')
    intent=read(ROOTS[2]+'/gpu-intent.json');receipt=read(ROOTS[2]+'/gpu-receipt.json');cpu=read(ROOTS[1]+'/cpu/result.json')
    require(receipt['job_id']==profile['job_id'] and receipt['returncode']==0 and receipt['status']=='submitted_not_passed'
        and receipt['intent_sha256']==binding(ROOTS[2]+'/gpu-intent.json')['sha256'],'Foreign accepted submission differs')
    require(intent['identity']['archive_sha256']==ARCHIVE and intent['identity']['manifest_sha256']==MANIFEST
        and intent['identity']['plan_sha256']==PLAN and intent['dependency']['cpu_result_sha256']==CPU
        and cpu['status']=='passed' and cpu['junit']=={'tests':38,'failures':0,'errors':0,'skipped':0},'Foreign CPU ancestry differs')
    a=intent['bsub'];require(a[a.index('-q')+1]=='gpuv100' and a[a.index('-n')+1]=='4' and a[a.index('-W')+1]=='4:00',
                           'Foreign allocation identity differs')
    command=intent['command'];require(command[command.index('--source-sha256')+1]==ARCHIVE
        and command[command.index('--plan-sha256')+1]==PLAN and command[command.index('--cpu-sha256')+1]==CPU,
        'Foreign actual command pins differ')
    inventories=[inventory(root) for root in ROOTS]
    extras=[Path(WORK+'/releases/source-146-embedding-v100-qualification-attempt001.tar.gz'),
        *[Path(WORK+'/logs/'+TAG+'-gpu.'+profile['job_id']+'.'+suffix) for suffix in ('out','err')]]
    for p in extras:
        if p.exists():
            b=binding(p);s=p.stat();inventories.append({'root':str(p),'actual_bytes':max(s.st_size,getattr(s,'st_blocks',0)*512),
                'actual_files':1,'complete':True,'nofollow':True})
    actual=sum(r['actual_bytes'] for r in inventories);files=sum(r['actual_files'] for r in inventories)
    cap=sum(v['bytes'] for v in profile['baseline'].values())+profile['growth_bytes']
    filecap=sum(v['files'] for v in profile['baseline'].values())+profile['growth_files']
    require(actual<=cap and files<=filecap,'Foreign source146 exceeded held complete forecast')
    ids=[]
    for row in rows:
        if row['JOBID']==profile['job_id']:
            require(row['QUEUE']=='gpuv100' and row['COMMAND']==profile['command'],'Foreign actual scheduler identity differs')
            ids.append(row['JOBID'])
    return {'version':'source146-readonly-shared-forecast-v1','accounted_job_ids':ids,'inventories':inventories,
        'actual_bytes':actual,'actual_files':files,'remaining_bytes':cap-actual,'remaining_files':filecap-files,
        'forecast_not_foreign_hard_cap':True,'foreign_files_jobs_settings_changed':False}

def augment(result,observation):
    value=dict(result)
    for r in ('bytes','files'):
        value['available_'+r]-=observation['actual_'+r];value['required_'+r]+=observation['remaining_'+r]
    value['byte_margin']=value['available_bytes']-value['required_bytes'];value['file_margin']=value['available_files']-value['required_files']
    value['status']='passed' if value['byte_margin']>=0 and value['file_margin']>=0 else 'failed'
    return value

def install(rolling,profile,parent):
    """Compose complete external charging around the unchanged original assessor."""
    parse=rolling.parse_scheduler;classify=rolling.classify_jobs;assess=rolling.c.assess;latest={}
    def parsed(raw):
        rows=parse(raw);latest['observation']=observe(profile,rows)
        p=Path(parent)/('source146-observation-'+str(time.time_ns())+'.json')
        with p.open('x') as f:json.dump(latest['observation'],f,indent=2);f.write('\n')
        latest['proof']=binding(p);return rows
    def classified(rows,known,own,source,accounted_external=()):
        return classify(rows,known,own,source,set(accounted_external)|set(latest['observation']['accounted_job_ids']))
    def assessed(*args,**kwargs):
        result=augment(assess(*args,**kwargs),latest['observation']);result['source146_forecast_evidence']=latest['proof'];return result
    rolling.parse_scheduler=parsed;rolling.classify_jobs=classified;rolling.c.assess=assessed
    return rolling
