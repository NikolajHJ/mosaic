"""Read-only authenticated compact delivery collection; original big files are references."""
import base64,hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
REMOTE=r'''
import base64,hashlib,json,re,subprocess,sys
from pathlib import Path
root=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/lrreset-analysis-03')
source=root.with_name('lrreset-analysis-release-03')/'source'
manifest='8e3045ddfd44a701127deb7af7a163536dfbf699123ba883f98207d6d57b95d5'
name=sys.argv[1];assert re.fullmatch(r'interim-\d{4}|combined-report',name)
p=root/name;files={};references=[]
def digest(q):
 with q.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def add(q):
 q=Path(q);assert q.is_file() and not q.is_symlink()
 assert q.is_relative_to(root) or q.is_relative_to(source)
 n=('analysis/'+q.relative_to(root).as_posix()) if q.is_relative_to(root) else ('source/'+q.relative_to(source).as_posix())
 if n not in files:
  d=q.read_bytes();assert len(d)<=20000000
  files[n]={'path':str(q),'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest(),'base64':base64.b64encode(d).decode()}
def actual(parent):
 sub=json.loads((parent/'submission.json').read_bytes());job=sub['job_id']
 q=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30);raw=q.stdout+q.stderr;c=re.sub(r'\s+','',raw)
 assert q.returncode==0 and re.findall(r'Job<([^>]+)>',c)==[job] and re.findall(r'Status<([^>]+)>',c)==['DONE'] and re.findall(r'Queue<([^>]+)>',c)==['hpc']
 log=(parent/('stdout.'+job+'.log')).read_text()
 assert re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]* Done\s*$',log,re.M)==[job] and 'Successfully completed.' in log and 'Exited with exit' not in log
 add(parent/'submission.json');add(parent/('stdout.'+job+'.log'))
 return job,raw
job,scheduler=actual(p)
a=json.loads((p/'payload/assessment.json').read_bytes());receipt=json.loads((p/'payload/report/report_receipt.json').read_bytes())
assert a['status']=='passed' and a['job_id']==job and a['report_source_manifest_sha256']==manifest
assert receipt['input_jobs']==['29792670','29792671','29792672'] and receipt['states'][:2]==[10001,10001] and receipt['included_updates'][:2]==[10000,10000]
assert a['scientific_passed'] is False and a['campaign_status']=='failed'
assert a['report_receipt_sha256']==digest(p/'payload/report/report_receipt.json')
arc=json.loads(Path(str(p)+'.tar.receipt.json').read_bytes());q=Path(arc['path'])
assert q==Path(str(p)+'.tar.gz') and q.stat().st_size==arc['bytes'] and digest(q)==arc['sha256']
assert digest(source/'source_manifest.json')==manifest
meta=json.loads((source/'source_manifest.json').read_bytes());assert len(meta['files'])==17
for n,h in meta['files'].items():assert digest(source/n)==h
for q in [p/'submission-started.json',p/'scheduler-at-submission.txt',p/'payload/assessment.json',p/'payload/report/report_receipt.json',Path(str(p)+'.tar.receipt.json'),p/('storage-final/gate.json' if name=='combined-report' else 'storage-interim/gate.json'),root/'jobs.json',root/'cpu-certificate.json',root/'analysis-dispatch-started.json',root/'source-staging.json',root/'after-cpu/storage-dispatch/gate.json',source/'source_manifest.json',source/'analysis-input-pins.json']:
 add(q)
cpu=json.loads((root/'cpu-certificate.json').read_bytes())
assert cpu['status']=='passed' and cpu['tests']==284 and cpu['source_manifest_sha256']==manifest
for proof in cpu['proofs']:
 q=Path(proof['path']);assert digest(q)==proof['sha256'];add(q)
collectors={}
for arm in ['protenix','boltz2','boltz2_chemistry']:
 c=root/('complete-'+arm);assessment=c/'payload/assessment.json'
 if not assessment.exists():continue
 j,raw=actual(c);aa=json.loads(assessment.read_bytes());assert aa['status']=='passed' and aa['all_probability_frames_verified'] is True and aa['report_source_manifest_sha256']==manifest
 collectors[arm]={'job_id':j,'actual_scheduler':raw}
 for q in [assessment,c/'payload/snapshots.json',c/'payload/native-stdout.txt',*c.glob('payload/native-scheduler-*.txt')]:add(q)
 for proof in aa['proofs']:
  q=Path(proof['path']);assert digest(q)==proof['sha256'];references.append({'path':str(q),'bytes':q.stat().st_size,'sha256':proof['sha256'],'scope':'Remote byte rehash; original large scientific files not downloaded'})
print(json.dumps({'status':'passed_metadata_only','report_job':job,'actual_report_scheduler':scheduler,'report_archive':arc,'collectors':collectors,'remote_references':references,'files':files}))
'''

def main():
    name=sys.argv[1];out=REPO/'results/hpc'/HERE.name/('delivery-'+name);out.mkdir(exist_ok=False)
    import shlex
    command=' '.join(shlex.quote(x) for x in ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python','-c',REMOTE,name])
    q=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -c '+shlex.quote(command)],capture_output=True,timeout=180)
    (out/'ssh-stderr.bin').write_bytes(q.stderr);(out/'ssh-result.json').write_text(json.dumps({'returncode':q.returncode})+'\n')
    assert q.returncode==0,q.stderr.decode()[-3000:]
    j=json.loads(q.stdout);files=j.pop('files')
    for n,p in files.items():
        d=base64.b64decode(p.pop('base64'));assert len(d)==p['bytes'] and hashlib.sha256(d).hexdigest()==p['sha256']
        target=out/n;assert target.resolve().is_relative_to(out.resolve());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(d)
    (out/'evidence.json').write_text(json.dumps({**j,'files':files},indent=2)+'\n')
    (out/'archive-receipt.json').write_text(json.dumps(j['report_archive'],indent=2)+'\n')
    print(json.dumps({'folder':str(out),'files':len(files),'report_archive':j['report_archive']}))

if __name__=='__main__':main()
