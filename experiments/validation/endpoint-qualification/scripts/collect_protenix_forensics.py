"""One-shot stdlib terminal proof/point-byte collection; never evaluates a model."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261025-protenix-endpoint-forensics-01'

PROGRAM=r"""
import hashlib,io,json,os,stat,sys,tarfile
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=base/'protenix-endpoint-forensics-01';source=base/'protenix-endpoint-forensics-release-01/source'
pin='d127d6abe87307a5d305edb6d04e3a32e3545c508e58b82ce45290e0f3fc03d8'
def sha(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
assert sha(source/'source_manifest.json')==pin
manifest=json.loads((source/'source_manifest.json').read_bytes())
for name,digest in manifest['files'].items():assert sha(source/name)==digest and not (source/name).is_symlink()
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
sys.path.insert(0,str(source/'scripts'));import protenix_forensics_ops as ops
ops.verify(source,pin);ops.external(manifest,ops.read(source/'plan.json'));ops.certificate(pin,manifest)
assessment=ops.read(root/'gpu/payload/assessment.json')
assert assessment['status']=='diagnostics_completed_original_failure_preserved' and assessment['job_id']=='29813996'
assert assessment['updates']==0 and assessment['protected_inputs_unchanged'] and assessment['frozen_unchanged']
assert assessment['source_trees_unchanged'] and not assessment['preservation_errors']
preserve=ops.read(root/'gpu/post-execution-preservation.json');assert preserve['returncode']==0 and preserve['preservation_status']=='passed'
ops.lsf_success((root/'gpu/stdout.29813996.log').read_text(),'29813996','mosaic-pro-forensics-gpu','gpul40s')
out=root/'delivery-01';assert not out.exists(),'Existing/unknown delivery outcome; inspect instead of replay'
out.mkdir();gate=ops.storage(out,'entry');assert gate['status']=='passed'
paths={}
for path in (root/'gpu').rglob('*'):
 assert not path.is_symlink() and (path.is_file() or path.is_dir())
 if path.is_file():paths[path.relative_to(root).as_posix()]=path
cert=ops.read(root/'cpu-certificate.json')
paths['cpu-certificate.json']=root/'cpu-certificate.json'
for proof in cert['proofs']:
 path=Path(proof['path']);ops.check_binding(proof);assert path.is_relative_to(root)
 paths[path.relative_to(root).as_posix()]=path
paths.update({'source_manifest.json':source/'source_manifest.json','plan.json':source/'plan.json',
 'original_admission.json':Path(ops.FAILED_ADMISSION['path']),
 'original_endpoint.npz':Path(ops.read(source/'plan.json')['arm']['parent_snapshot']['path'])})
files={};total=0
archive=out/'forensics-evidence.tar.gz'
with archive.open('xb') as target:
 with tarfile.open(fileobj=target,mode='w:gz') as tar:
  for name,path in sorted(paths.items()):
   assert not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode)
   size=path.stat().st_size;total+=size;assert total<=32*1024**2
   data=path.read_bytes();assert len(data)==size
   files[name]={'bytes':size,'sha256':hashlib.sha256(data).hexdigest()}
   member=tarfile.TarInfo(name);member.size=size;member.mode=0o444;member.mtime=0
   tar.addfile(member,io.BytesIO(data))
assert archive.stat().st_size<=16*1024**2
assert ops.owned_bytes(root,(root/'gpu/payload',))+ops.owned_bytes(source.parent)<=ops.METADATA_CAP
receipt={'archive':ops.binding(archive),'files':files,'original_results_changed':False,
 'scientific_array_analysis_performed':False,'same_job_LSF_success':'29813996',
 'live_scheduler_scope':'Records aged out; no GPU bjobs DONE record fabricated','storage_gate':gate}
ops.write(out/'receipt.json',receipt)
sys.stdout.buffer.write((json.dumps(receipt,separators=(',',':'))+'\n').encode());sys.stdout.buffer.flush()
with archive.open('rb') as stream:
 for block in iter(lambda:stream.read(1024**2),b''):sys.stdout.buffer.write(block)
sys.stdout.buffer.flush()
"""


def main():
    out=LOCAL/'delivery-01';out.mkdir(exist_ok=False)
    (out/'program.py').write_text(PROGRAM,encoding='utf8',newline='\n')
    argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
          'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    record={'status':'unknown','started_utc':datetime.now(timezone.utc).isoformat(),
            'argv':argv,'program_sha256':hashlib.sha256(PROGRAM.encode()).hexdigest()}
    try:
        with (out/'stderr.txt').open('xb') as err:
            process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
            process.stdin.write(PROGRAM.encode());process.stdin.close()
            header=process.stdout.readline(4*1024**2)
            (out/'header.json').write_bytes(header)
            receipt=json.loads(header);assert header.endswith(b'\n') and 'archive' in receipt
            count=0;digest=hashlib.sha256()
            with (out/'forensics-evidence.tar.gz').open('xb') as stream:
                while True:
                    block=process.stdout.read(1024**2)
                    if not block:break
                    count+=len(block);assert count<=receipt['archive']['bytes'];digest.update(block);stream.write(block)
            rc=process.wait(timeout=30)
            assert rc==0 and count==receipt['archive']['bytes'] and digest.hexdigest()==receipt['archive']['sha256']
            record.update(status='downloaded_byte_verified',returncode=rc,archive_bytes=count,archive_sha256=digest.hexdigest())
    finally:
        record['ended_utc']=datetime.now(timezone.utc).isoformat()
        (out/'transfer.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
