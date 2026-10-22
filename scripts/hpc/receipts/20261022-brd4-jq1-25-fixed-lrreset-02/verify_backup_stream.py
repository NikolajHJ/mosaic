"""Verify every archive member directly, without thousands of Windows file opens."""
from pathlib import Path,PurePosixPath
import hashlib,json,tarfile
h=Path(__file__).resolve().parent;repo=h.parents[3]
r=json.loads((h/'older-work3-backup-02-result.json').read_text())
assert r['returncode']==0 and r['outcome']=='archive_bytes_verified'
archive=Path(r['local_archive']);receipt=r['receipt']
assert archive.stat().st_size==receipt['bytes'] and receipt['sha256']=='1f6b91446dba1866ff7c6e732a1ce2457e60c59661943fdbd2eb6cf870a305f4'
with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
observed={};metadata={}
with tarfile.open(archive,mode='r|gz') as t:
 for m in t:
  name=PurePosixPath(m.name)
  assert m.isfile() and not name.is_absolute() and '..' not in name.parts and m.name not in observed
  assert m.name.startswith('files/') or m.name in ('origin-map.json','evidence_manifest.json')
  stream=t.extractfile(m);hasher=hashlib.sha256();count=0;blocks=[]
  for b in iter(lambda:stream.read(1048576),b''):
   hasher.update(b);count+=len(b)
   if m.name in ('origin-map.json','evidence_manifest.json'):blocks.append(b)
  assert count==m.size
  observed[m.name]={'bytes':count,'sha256':hasher.hexdigest()}
  if blocks:metadata[m.name]=b''.join(blocks)
assert set(metadata)=={'origin-map.json','evidence_manifest.json'}
manifest=json.loads(metadata['evidence_manifest.json'])['files'];observed.pop('evidence_manifest.json')
assert observed==manifest
origins=json.loads(metadata['origin-map.json'])
assert set(origins)=={n for n in observed if n.startswith('files/')}
assert len(origins)==receipt['original_files']==34847
assert sum(observed[n]['bytes'] for n in origins)==receipt['original_logical_bytes']==1540133784
out=repo/'results/hpc'/h.name/'backup-stream-verification-02';out.mkdir(exist_ok=False)
for name,b in metadata.items():
 with (out/name).open('xb') as f:f.write(b)
record={'status':'passed','scope':'Independent local standard-library archive SHA/safe-member/type/path/size/every-file SHA verification by streaming; no scientific calculation or inference',
    'archive':str(archive),'archive_bytes':receipt['bytes'],'archive_sha256':receipt['sha256'],
    'verified_original_files':34847,'verified_manifest_files':len(observed),'original_logical_bytes':1540133784,
    'manifest_directory':str(out.resolve()),'interrupted_full_extraction_preserved':True,'original_result_scope':'Archive transfer bytes passed; slow full extraction intentionally stopped; independent complete stream verification supplies every-member proof','deletion_performed':False}
with (h/'older-work3-backup-02-stream-audit.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record))
