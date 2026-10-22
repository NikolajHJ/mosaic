"""Stage the exact retry archive and submit its CPU gate/own callback once."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REPO = Path(__file__).resolve().parents[5]
TAG = '20261022-brd4-jq1-25-fixed-lrreset-04'

REMOTE = r'''
import ast,hashlib,json,os,re,subprocess,sys,tarfile
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=base/'restart-lrreset-04';release=base/'restart-lrreset-release-04';source=release/'source'
upload=base/'20261022-brd4-jq1-25-fixed-lrreset-04.tar.gz'
expected_sha,expected_bytes,manifest=sys.argv[1],int(sys.argv[2]),sys.argv[3]
assert root.is_dir() and not any(p.exists() for p in (release,upload,root/'cpu',root/'after-cpu',root/'source-staging.json',root/'native-dispatch-started.json',root/'long-dispatch-started.json')), 'Existing/unknown execution outcome prohibits duplicate bootstrap'
assert not Path('/zhome/95/e/206121/mosaic-runs/20261022-brd4-jq1-25-fixed-lrreset-04').exists()
probe=json.loads((root/'initialization-probe-jobs.json').read_bytes())
for phase in ('cpu','gpu'):
 job=probe[phase]['job_id']; parent=root/('initialization-probe-'+phase)
 result=subprocess.run(['bjobs','-l',job],text=True,capture_output=True,check=True,timeout=30)
 compact=re.sub(r'\s+','',result.stdout)
 assert re.findall(r'Job<([^>]+)>',compact)==[job] and re.findall(r'Status<([^>]+)>',compact)==['DONE']
 raw=(parent/('stdout.'+job+'.log')).read_text();assert 'Subject: Job '+job+':' in raw and 'Successfully completed.' in raw
 a=json.loads((parent/'payload/assessment.json').read_bytes());assert a['status']=='diagnostic_completed' and a['baseline_sha256']=='351789aa62d3ff111a4d2050a062cf844a1f29a87a7e352af465ff4299fd1bbf'
 (parent/'scheduler-verified-final.txt').write_text(result.stdout)
variants=json.loads((root/'initialization-probe-gpu/payload/assessment.json').read_bytes())['variants']
assert variants['cpu-saved_logits']['fields']['X']['equal'] and variants['gpu-saved_logits']['fields']['E_upper']['equal']
assert all(variants['gpu-regenerated']['fields'][k]['equal'] for k in ('atom_logits','bond_logits','key_data'))
with upload.open('xb') as out:
 for block in iter(lambda:sys.stdin.buffer.read(1048576),b''):out.write(block)
assert upload.stat().st_size==expected_bytes
assert hashlib.sha256(upload.read_bytes()).hexdigest()==expected_sha
release.mkdir();source.mkdir()
with tarfile.open(upload) as tar:
 seen=set()
 for member in tar:
  name=member.name.removeprefix('source/');p=Path(name)
  assert member.isfile() and member.name.startswith('source/') and not p.is_absolute() and '..' not in p.parts and name not in seen
  seen.add(name);dest=source/p;dest.parent.mkdir(parents=True,exist_ok=True)
  with dest.open('xb') as stream:stream.write(tar.extractfile(member).read())
raw=(source/'source_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==manifest
metadata=json.loads(raw);assert len(metadata['files'])==625 and set(metadata['files'])|{'source_manifest.json'}==seen
for name,expected in metadata['files'].items():
 data=(source/name).read_bytes();assert hashlib.sha256(data).hexdigest()==expected
 if name.endswith('.py'):ast.parse(data,filename=name)
parent=base/'restart-lrreset-release-03/source'
assert hashlib.sha256((parent/'source_manifest.json').read_bytes()).hexdigest()=='ff82857ecd4a985d0ba6cc16a87f1bf7b7bb2c7ffdbde18a183bbd63670ff567'
for name in metadata['files']:
 if (name.startswith('src/') and name!='src/mosaic/small_molecule/chemistry/trajectory.py') or name.startswith('examples/configs/brd4_jq1_25_fixed_lrreset_'):
  assert (source/name).read_bytes()==(parent/name).read_bytes(), 'Scientific library/configuration changed'
old=base/'restart-lrreset-03'
assert (source/'src/mosaic/small_molecule/chemistry/trajectory.py').read_bytes()==(parent/'src/mosaic/small_molecule/chemistry/trajectory.py').read_bytes().replace(b'                proposal = GraphLogits(',b'                proposal = type(evaluation.logits)(')
for arm in ('protenix','boltz2','boltz2_chemistry'):
 d=json.loads((old/('decision-'+arm+'.json')).read_text());assert d['status']==('failed' if arm=='protenix' else 'passed')
 for proof in d['proofs']:assert hashlib.sha256(Path(proof['path']).read_bytes()).hexdigest()==proof['sha256']
 summary=json.loads((old/('validate-'+arm)/'payload/study/summary.json').read_bytes())
 assert summary['status']==('failed' if arm=='protenix' else 'passed')
 if arm=='protenix':assert 'Mismatched elements: 150 / 150' in summary['error'] and 'native_gradient_checks' not in summary
 else:assert summary['native_gradient_checks']['status']=='passed'
assert not (old/'long-submissions.json').exists()
python='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
subprocess.run([python,str(source/'scripts/verify_stage0_release.py'),str(source),manifest],check=True)
for rel in ('scripts/hpc/fixed25_lrreset_cpu.lsf','scripts/hpc/fixed25_lrreset_stage.lsf'):subprocess.run(['bash','-n',str(source/rel)],check=True)
for p in source.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
source.chmod(0o555)
def write(path,value):
 with path.open('x') as out:json.dump(value,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
write(root/'source-staging.json',{'status':'passed','archive_sha256':expected_sha,'archive_bytes':expected_bytes,'manifest_sha256':manifest,'files':len(metadata['files']),'parent_144_library_files_and_all_configs_identical':True,'only_library_change_preserves_proposal_subclass':True,'old_failed_entry_jobs_preserved':True})
sys.path[:0]=[str(source/'scripts'),str(source/'src')]
import fixed25_lrreset_ops as ops
ops.validate_configs();ops.verify_finished_predecessors()
gate=ops.storage.check_work_storage(root/'preflight-quota',phase='preflight')
def submit(name,args):
 parent=root/name;parent.mkdir()
 command=['bsub',*args,'-oo',str(parent/'stdout.%J.log'),'-eo',str(parent/'stderr.%J.log')]
 if name=='cpu':command+=['bash','-l',str(source/'scripts/hpc/fixed25_lrreset_cpu.lsf'),str(source),str(root),manifest]
 else:command+=['bash','-l',str(source/'scripts/hpc/fixed25_lrreset_stage.lsf'),str(source),str(root),manifest,'after-cpu','none',str(parent),jobs['cpu']['job_id'],'0']
 write(parent/'submission-started.json',{'args':command,'outcome':'unknown'})
 record={'args':command,'outcome':'unknown'}
 try:
  r=subprocess.run(command,text=True,capture_output=True,timeout=60);record.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
  found=re.findall(r'Job <(\d+)>',r.stdout)
  if r.returncode==0 and len(found)==1:record.update(outcome='submitted',job_id=found[0])
 finally:write(parent/'submission.json',record)
 assert record['outcome']=='submitted','Unknown outcome; inspect receipts, never duplicate'
 raw=subprocess.run(['bjobs','-l',record['job_id']],capture_output=True,text=True,timeout=30)
 (parent/'scheduler-at-submission.txt').write_text(raw.stdout+raw.stderr)
 return record
jobs={}
jobs['cpu']=submit('cpu',['-q','hpc','-n','4','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','01:00','-J','mosaic-jq1-lrreset04-cpu'])
jobs['after_cpu']=submit('after-cpu',['-q','hpc','-n','1','-R','span[hosts=1]','-R','rusage[mem=2GB]','-W','00:30','-J','mosaic-jq1-lrreset04-after-cpu','-w','ended('+jobs['cpu']['job_id']+')'])
write(root/'initial-submissions-complete.json',jobs)
print(json.dumps({'status':'submitted_not_passed','root':str(root),'source':str(source),'manifest':manifest,'jobs':jobs,'preflight_gate':gate}))
'''


def main():
    folder = REPO / 'results/hpc' / TAG
    package = json.loads((folder / 'source02/package.json').read_text())
    archive = folder / 'source02/source.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == package['archive_sha256']
    marker = folder / 'submission-started.json'
    assert not marker.exists(), 'Inspect exact previous outcome before recovery'
    command = ' '.join(shlex.quote(x) for x in ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', '-c', REMOTE,
        package['archive_sha256'], str(package['archive_bytes']), package['manifest_sha256']])
    args = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
            'dtu-hpc', 'bash -l -c ' + shlex.quote(command)]
    marker.write_bytes(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'args': args}, indent=2).encode())
    result = subprocess.run(args, input=archive.read_bytes(), capture_output=True, timeout=180)
    (folder / 'submission-stdout.txt').write_bytes(result.stdout)
    (folder / 'submission-stderr.txt').write_bytes(result.stderr)
    (folder / 'submission-result.json').write_bytes(json.dumps({'returncode': result.returncode,
        'stdout_sha256': hashlib.sha256(result.stdout).hexdigest()}, indent=2).encode())
    print(result.stdout.decode()[-1400:]); print(result.stderr.decode()[-600:])
    assert result.returncode == 0, 'Preserved failed/unknown staging; do not replay'


if __name__ == '__main__':
    main()
