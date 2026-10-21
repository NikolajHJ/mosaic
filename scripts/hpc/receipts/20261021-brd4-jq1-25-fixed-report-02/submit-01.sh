set -euo pipefail
archive=/work3/s234818/mosaic/20261021-brd4-jq1-25-fixed-report-02.tar.gz
base=/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-report-02
root=/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-report-02
manifest=3b78673801b3875fd040dff9d814a869e5f0172fa8dea164ffb85656e6b3ad5e
[[ ! -e "$base" && ! -e "$root" ]]
printf '%s  %s\n' 9ed7eb09c942a58f2cb8c9df5880ca7196215abcaf300551a74ee231d5e064e9 "$archive" | sha256sum -c -
[[ $(stat -c %s "$archive") == 53487 ]]
module load python3/3.12.11
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
"$python" - "$archive" "$base" "$root" <<'EXTRACT'
import sys,tarfile
from pathlib import Path
archive,base,root=map(Path,sys.argv[1:]);base.mkdir();(base/'source').mkdir();root.mkdir();(root/'cpu').mkdir();(root/'certify-cpu').mkdir()
with tarfile.open(archive,'r:gz') as tf:
    members=tf.getmembers()
    if len(members)!=14 or len({m.name for m in members})!=14 or any(not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in members):raise ValueError('Unsafe source archive')
    for m in members:
        q=base/'source'/m.name;q.parent.mkdir(parents=True,exist_ok=True)
        with q.open('xb') as stream:stream.write(tf.extractfile(m).read())
EXTRACT
science=/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-02/source
l03=/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-storage-03/source
l04=/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-storage-04/source
"$python" "$base/source/scripts/verify_stage0_release.py" "$base/source" "$manifest"
"$python" "$base/source/scripts/verify_stage0_release.py" "$science" 1e3a8a21e2fa68e24d1c9229238999c42a4a9b44bcc78344e73e188c87d1ee64
"$python" "$base/source/scripts/verify_stage0_release.py" "$l03" 9ee7bf1d6bf412acdc37fc83415476dbefdf45db89533571d4a646ee16f33772
"$python" "$base/source/scripts/verify_stage0_release.py" "$l04" 75a76dacf8de25f4ebd2d3b3fe114b1af6092064a01edf141650295300670ab3
bash -n "$base/source/scripts/hpc/jq1_fixed_report_stage.lsf"
chmod -R a-w "$base/source"
export PYTHONPATH="$base/source/scripts:$l04/scripts:$l03/scripts:$science/scripts:$science/src" PYTHONDONTWRITEBYTECODE=1
"$python" "$base/source/scripts/jq1_fixed_report_ops.py" --phase preflight --root "$root" --source "$base/source" --manifest "$manifest" --parent "$root"
"$python" - "$base/source" "$root" "$manifest" <<'SUBMIT'
import sys,json,re,subprocess,datetime
from pathlib import Path
source,root=map(Path,sys.argv[1:3]);manifest=sys.argv[3]
def submit(phase,producer=None):
    parent=root/phase
    command=['bsub','-q','hpc','-n','2' if phase=='cpu' else '1','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','00:30' if phase=='cpu' else '00:20','-J','mosaic-fixed-report02-'+phase,'-oo',str(parent/'stdout.%J.log'),'-eo',str(parent/'stderr.%J.log')]
    if producer:command+=['-w','ended('+producer+')']
    command+=['bash','-l',str(source/'scripts/hpc/jq1_fixed_report_stage.lsf'),str(source),str(root),manifest,phase,'none',str(parent)]
    r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,'args':command,'outcome':'unknown'}
    with (parent/'submission-started.json').open('x') as f:json.dump(r,f,indent=2)
    try:
        reply=subprocess.run(command,capture_output=True,text=True,timeout=60);r.update(returncode=reply.returncode,stdout=reply.stdout,stderr=reply.stderr)
        ids=re.findall(r'Job <(\d+)>',reply.stdout)
        if reply.returncode==0 and len(ids)==1:r.update(outcome='submitted',job_id=ids[0])
    except Exception as error:r['error']=repr(error)
    finally:
        with (parent/'submission.json').open('x') as f:json.dump(r,f,indent=2)
    if r['outcome']!='submitted':raise ValueError('Report submission failed/unknown; inspect receipts, no blind retry')
    q=subprocess.run(['bjobs','-l',r['job_id']],capture_output=True,text=True,timeout=30);(parent/'scheduler-at-submission.txt').write_text(q.stdout+q.stderr)
    return r
cpu=submit('cpu')
with (root/'jobs.json').open('x') as f:json.dump({'cpu':cpu['job_id'],'native_jobs':{'protenix':'29777709','boltz2':'29778203','boltz2_chemistry':'29778204'},'report_manifest':manifest},f,indent=2)
certificate=submit('certify-cpu',cpu['job_id'])
record={'cpu':cpu,'cpu_certificate':certificate,'report_manifest':manifest,'scope':'Actual report-only CPU submissions; collector/interim/final IDs not inferred'}
with (root/'initial-submissions.json').open('x') as f:json.dump(record,f,indent=2)
print('ACTUAL_SUBMISSIONS_JSON='+json.dumps(record))
SUBMIT
