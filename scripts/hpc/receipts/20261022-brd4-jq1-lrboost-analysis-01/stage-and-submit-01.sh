set -euo pipefail
archive=/work3/s234818/mosaic/20261022-brd4-jq1-lrboost-analysis-01.tar.gz
base=/work3/s234818/mosaic/releases/20261022-brd4-jq1-lrboost-analysis-01
root=/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01
manifest=214c19387649a741a7ad42f1e3e027061631993b4ae33425071aba9005fdcbcc
[[ ! -e "$base" && ! -e "$root" ]]
module load python3/3.12.11
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
"$python" - "$archive" "$base" "$root" "$manifest" <<'EXTRACT'
from pathlib import Path,PurePosixPath
import sys,tarfile,hashlib,json
archive,base,root=map(Path,sys.argv[1:4]);manifest=sys.argv[4]
assert archive.stat().st_size==76750 and hashlib.sha256(archive.read_bytes()).hexdigest()=='9707bec755c9c116a03a56abceabbe1caeaf019922f4bd68cb2a99141546f3e1'
with tarfile.open(archive) as tf:
 members=tf.getmembers();assert len(members)==17 and len({m.name for m in members})==17
 assert all(m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts for m in members)
 blobs={m.name:tf.extractfile(m).read() for m in members}
 assert hashlib.sha256(blobs['source_manifest.json']).hexdigest()==manifest
 pins=json.loads(blobs['source_manifest.json'])['files']
 assert set(blobs)==set(pins)|{'source_manifest.json'}
 assert all(hashlib.sha256(blobs[n]).hexdigest()==s for n,s in pins.items())
 base.mkdir();source=base/'source';source.mkdir();root.mkdir()
 for name,data in blobs.items():
  target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as stream:stream.write(data)
 (root/'source-staging.json').write_text(json.dumps({'status':'authenticated','source_files':16,'manifest_sha256':manifest,'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},indent=2)+'\n')
EXTRACT
science=/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-lrboost-01/source
"$python" "$base/source/scripts/verify_stage0_release.py" "$base/source" "$manifest"
"$python" "$base/source/scripts/verify_stage0_release.py" "$science" 9605a5295b9fe8dfe29f15410b3a029038ea8a18feb16907f043abd5be13c323
bash -n "$base/source/scripts/hpc/jq1_lrboost_analysis_stage.lsf"
chmod -R a-w "$base/source"
export PYTHONPATH="$base/source/scripts:$base/source/tests:$science/scripts:$science/tests:$science/src" PYTHONDONTWRITEBYTECODE=1
"$python" "$base/source/scripts/jq1_lrboost_report_ops.py" --phase preflight --root "$root" --source "$base/source" --manifest "$manifest" --parent "$root"
"$python" - "$manifest" <<'SUBMIT'
import sys,json
import jq1_lrboost_report_ops as ops
manifest=sys.argv[1]
cpu=ops.submit('cpu','cpu',manifest)
ops.exclusive(ops.ROOT/'jobs.json',{'cpu':cpu['job_id'],'native_jobs':{a:p['job'] for a,p in ops.ARMS.items()},'report_manifest':manifest})
callback=ops.submit('after-cpu','certify-cpu',manifest,dependency='ended('+cpu['job_id']+')')
record={'cpu':cpu,'after_cpu':callback,'manifest_sha256':manifest,'scope':'Actual CPU and dependent certificate only; no diagnostic/collector/report ID inferred'}
ops.exclusive(ops.ROOT/'initial-submissions.json',record)
print('ACTUAL_SUBMISSIONS_JSON='+json.dumps(record))
SUBMIT
