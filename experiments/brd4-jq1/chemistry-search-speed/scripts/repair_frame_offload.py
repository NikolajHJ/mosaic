"""Byte-exact retirement of three already delivered historical report frame copies.

Only frames-*.bin beneath the three explicit report parents are eligible.
Native states/losses, report proofs, extracted sources and localhost viewers
cannot be selected. Each remote phase has an exclusive outcome directory.
"""
import argparse,ast,base64,hashlib,json
from pathlib import Path,PurePosixPath
import shlex,subprocess,tarfile

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261027-repair-storage-01'
BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
PLAN_NAME='frame-plan-02.json'
GROUPS=[('20261022-brd4-jq1-lrboost-analysis-01','interim-0001','29780829'),
        ('20261022-brd4-jq1-lrboost-analysis-01','interim-0002','29781041'),
        ('20261023-brd4-jq1-lrreset-analysis-03','interim-0001','29796556')]

def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8',newline='\n') as stream:json.dump(v,stream,indent=2);stream.write('\n')

def prepare():
    rows=[];backups=[];parents=[]
    protected=set(json.loads((REPO/'results/hpc/20261026-chemistry-storage-recovery-01/plan-06/plan.json').read_bytes())['protected_paths'])
    for tag,name,job in GROUPS:
        folder=REPO/'results/hpc'/tag/name;archive=folder.with_suffix('.tar.gz')
        manifest_raw=(folder/'evidence_manifest.json').read_bytes();manifest=json.loads(manifest_raw)
        assert manifest['job_id']==job
        seen=set();selected=[]
        with tarfile.open(archive,'r:gz') as tar:
            for member in tar:
                part=PurePosixPath(member.name)
                assert not part.is_absolute() and '..' not in part.parts and '\\' not in member.name
                if member.isdir():continue
                assert member.isfile() and member.name not in seen;seen.add(member.name)
                with tar.extractfile(member) as stream:
                    if member.name=='evidence_manifest.json':assert stream.read()==manifest_raw;continue
                    bound=manifest['files'][member.name]
                    assert member.size==bound['bytes'] and hashlib.file_digest(stream,'sha256').hexdigest()==bound['sha256']
                if '/payload/report/' not in member.name or not part.name.startswith('frames-') or part.suffix!='.bin':continue
                assert sha(folder/member.name)==bound['sha256']
                remote='/work3/s234818/mosaic/runs/'+member.name
                assert remote.startswith(BASE+'/') and remote not in protected
                parent=remote.split('/payload/report/')[0]
                selected.append({'path':remote,**bound,'parent':parent,'producer_job':job})
        assert seen==set(manifest['files'])|{'evidence_manifest.json'} and selected
        parent_key=selected[0]['parent'].removeprefix('/work3/s234818/mosaic/runs/')
        parents.append({'path':selected[0]['parent'],'job_id':job,
            'submission':{'path':selected[0]['parent']+'/submission.json',**manifest['files'][parent_key+'/submission.json']},
            'stdout':{'path':selected[0]['parent']+'/stdout.'+job+'.log',**manifest['files'][parent_key+'/stdout.'+job+'.log']}})
        assert all(r['parent']==parents[-1]['path'] for r in selected)
        rows+=selected;backups.append({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),
            'manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),'all_files_verified':len(manifest['files'])})
    assert len({r['path'] for r in rows})==len(rows)
    plan={'version':'delivered-historical-report-frame-retirement-v1','files':rows,'backups':backups,'parents':parents,
          'native_states_and_losses_changed':False,'other_projects_changed':False,'settings_changed':False}
    write(LOCAL/PLAN_NAME,plan)
    print(json.dumps({'status':'all_archive_members_and_selected_laptop_frames_reverified','files':len(rows),
                      'logical_bytes':sum(r['bytes'] for r in rows),'archives':backups},indent=2))

COMMON=r'''
import base64,hashlib,json,os,re,stat,sys,time
from datetime import datetime,timezone
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01');root=base/'repair-frame-retirement-03'
raw=base64.b64decode(PLAN,validate=True);assert hashlib.sha256(raw).hexdigest()==PIN;plan=json.loads(raw)
assert plan['version']=='delivered-historical-report-frame-retirement-v1'
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(p,v):
 with Path(p).open('x',encoding='utf8') as stream:json.dump(v,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
def current():
 source=base/'chemistry-production-release-04/source';pin='01e2de2a3413bd903590dd331c5cedf6b756e856a83df2b73ad65e02b9f9fceb'
 manifest=json.loads((source/'source_manifest.json').read_bytes());assert sha(source/'source_manifest.json')==pin and len(manifest['files'])==17
 for name,digest in manifest['files'].items():assert sha(source/name)==digest
 sys.dont_write_bytecode=True;sys.path.insert(0,str(source/'scripts'));import production_ops as op
 _,declared=op.own_plan(pin);protected=set()
 def bindings(value):
  if isinstance(value,dict):
   if set(('path','bytes','sha256'))<=set(value):op.check_binding({k:value[k] for k in ('path','bytes','sha256')});protected.add(value['path'])
   for v in value.values():bindings(v)
  elif isinstance(value,list):
   for v in value:bindings(v)
 bindings(manifest);bindings(declared)
 for arm in ('protenix','boltz'):bindings(op.adoption(arm))
 terminal=[]
 for p in plan['parents']:
  op.check_binding(p['submission'])
  parent=Path(p['path']);sub=op.read(parent/'submission.json');assert sub['outcome']=='submitted' and sub['returncode']==0 and sub['job_id']==p['job_id']
  compact=''.join((parent/('stdout.'+p['job_id']+'.log')).read_text().split())
  assert compact.count('Subject:Job')==1 and ('Subject:Job'+p['job_id']+':<'+sub['args'][sub['args'].index('-J')+1]+'>incluster<dcc>Done') in compact and compact.count('Successfullycompleted.')==1
  assert compact.count('inqueue<hpc>')==1 and 'Exitedwithexit' not in compact
  terminal.append(op.binding(parent/('stdout.'+p['job_id']+'.log')))
 assert not any(row['path'] in protected for row in plan['files'])
 return {'status':'current_recovery_source_pilots_and_all_bound_inputs_verified','protected_bindings':len(protected),'source_manifest_sha256':pin,'actual_post_job_terminal_LSF':terminal,'archive_captured_stdout_is_not_final_LSF':True}
def check(row):
 p=Path(row['path']);parent=Path(row['parent'])
 assert parent in {base/'interim-0001',base/'interim-0002',base/'lrreset-analysis-03/interim-0001'}
 assert p.is_relative_to(parent/'payload/report') and re.fullmatch(r'frames-[0-9]{5}\.bin',p.name)
 for q in (p,*p.parents):assert not q.is_symlink()
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_uid==os.getuid() and s.st_size==row['bytes'] and sha(p)==row['sha256']
 return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_nlink]
'''

def remote(phase):
    if phase=='retire':
        raise RuntimeError('Abandoned: report-frame inspections found already-absent files; no retirement was selected')
    raw=(LOCAL/PLAN_NAME).read_bytes();plan=json.loads(raw)
    for row in plan['backups']:assert Path(row['archive']).stat().st_size==row['bytes'] and sha(row['archive'])==row['sha256']
    prefix='PLAN='+repr(base64.b64encode(raw).decode())+'\nPIN='+repr(hashlib.sha256(raw).hexdigest())+'\n'
    if phase=='inspect':
        suffix="""
assert not os.path.lexists(root),'Existing/unknown retirement; inspect instead of replay'
proof=current();identities={row['path']:check(row) for row in plan['files']}
print(json.dumps({'status':'exact_inactive_frame_copies_verified_no_deletion','plan_sha256':PIN,'identities':identities,'current':proof,'captured_utc':datetime.now(timezone.utc).isoformat()}))
"""
    else:
        before=json.loads((LOCAL/'frame-inspect03/stdout.json').read_bytes());receipt=json.loads((LOCAL/'frame-inspect03/receipt.json').read_bytes())
        assert receipt['returncode']==0 and before['status']=='exact_inactive_frame_copies_verified_no_deletion' and before['plan_sha256']==hashlib.sha256(raw).hexdigest()
        prefix+='BEFORE='+repr(before['identities'])+'\n'
        suffix="""
assert not os.path.lexists(root),'Existing/unknown retirement outcome; never replay'
proof=current();assert all(check(row)==BEFORE[row['path']] for row in plan['files'])
root.mkdir();(root/'plan.json').write_bytes(raw);write(root/'started.json',{'plan_sha256':PIN,'current':proof,'laptop_backups':plan['backups']})
removed=[]
with (root/'journal.jsonl').open('x',encoding='utf8') as journal:
 for row in plan['files']:
  assert check(row)==BEFORE[row['path']]
  journal.write(json.dumps({'event':'intent',**row})+'\n');journal.flush();os.fsync(journal.fileno())
  Path(row['path']).unlink();removed.append(row)
  journal.write(json.dumps({'event':'removed','path':row['path']})+'\n');journal.flush();os.fsync(journal.fileno())
after=current();assert all(not os.path.lexists(r['path']) for r in removed)
write(root/'completion.json',{'status':'exact_duplicate_frames_retired_after_verified_laptop_backup','plan_sha256':PIN,'removed_files':len(removed),'removed_logical_bytes':sum(r['bytes'] for r in removed),'after':after,'journal_sha256':sha(root/'journal.jsonl'),'original_native_states_or_losses_changed':False})
print((root/'completion.json').read_text())
"""
    program=prefix+COMMON+suffix;ast.parse(program)
    out=LOCAL/('frame-'+phase+'03');out.mkdir(exist_ok=False);(out/'program.py').write_text(program,encoding='utf8',newline='\n')
    cmd=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
         'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    receipt={'status':'unknown_inspect_exact_remote_journal_before_recovery','program_sha256':hashlib.sha256(program.encode()).hexdigest()}
    try:
        with (out/'stdout.json').open('xb') as output,(out/'stderr.txt').open('xb') as error:
            p=subprocess.run(cmd,input=program.encode(),stdout=output,stderr=error,timeout=300)
        receipt['returncode']=p.returncode
        if p.returncode:raise RuntimeError((out/'stderr.txt').read_text()[-3000:])
        value=json.loads((out/'stdout.json').read_bytes());receipt['status']=value['status'];print(json.dumps(value,indent=2) if phase=='retire' else json.dumps({'status':value['status'],'files':len(value['identities'])}))
    finally:
        receipt['stdout_sha256']=sha(out/'stdout.json');write(out/'receipt.json',receipt)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','inspect','retire'));a=parser.parse_args()
    if a.phase=='prepare':prepare()
    else:remote(a.phase)
