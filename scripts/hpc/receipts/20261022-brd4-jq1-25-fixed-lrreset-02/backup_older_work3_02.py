"""Fresh direct SSH byte backup only; no deletion or scientific evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys
h=Path(__file__).resolve().parent;repo=h.parents[3]
out=repo/'results/hpc'/h.name/'older-work3-backup-02.tar.gz'
script=h/'backup-older-work3-02.sh'
command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -s']
record={'started_utc':datetime.now(timezone.utc).isoformat(),'outcome':'unknown','command':command,'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'local_archive':str(out),'deletion_performed':False}
with (h/'older-work3-backup-02-started.json').open('x') as f:json.dump(record,f,indent=2)
try:
 with out.open('xb') as stream:
  r=subprocess.run(command,input=script.read_bytes(),stdout=stream,stderr=subprocess.PIPE,timeout=600)
 (h/'older-work3-backup-02-stderr.txt').write_bytes(r.stderr)
 record['returncode']=r.returncode
 assert r.returncode==0
 rows=[json.loads(line) for line in r.stderr.decode().splitlines() if line.startswith('{')]
 assert len(rows)==1
 receipt=rows[0]
 assert receipt['status']=='backup_stream_completed' and receipt['deletion_performed'] is False
 with out.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
 assert out.stat().st_size==receipt['bytes'] and actual==receipt['sha256']
 record.update(outcome='archive_bytes_verified',receipt=receipt)
 subprocess.run([sys.executable,str(repo/'scripts/verify_hpc_evidence.py'),str(out),actual],cwd=repo,check=True)
 record['outcome']='archive_and_every_original_file_verified'
except Exception as error:
 record['error']=repr(error)
 raise
finally:
 record['finished_utc']=datetime.now(timezone.utc).isoformat()
 record['observed_archive_bytes']=out.stat().st_size if out.exists() else 0
 with (h/'older-work3-backup-02-result.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record))
