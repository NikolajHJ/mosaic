set -euo pipefail
archive=/work3/s234818/mosaic/20261021-brd4-jq1-25-fixed-storage-04.tar.gz
base=/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-storage-04
root=/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-storage-04
manifest=75a76dacf8de25f4ebd2d3b3fe114b1af6092064a01edf141650295300670ab3
[[ ! -e "$base" && ! -e "$root" ]]
printf '%s  %s\n' 0ae3e8359f5ec7e6b25684746fca90aecc080c1fb2e86c59b71822981fd784a0 "$archive" | sha256sum -c -
[[ $(stat -c %s "$archive") == 25613 ]]
module load python3/3.12.11
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
"$python" - "$archive" "$base" "$root" <<'EXTRACT'
import json,sys,tarfile
from pathlib import Path
archive,base,root=map(Path,sys.argv[1:]);base.mkdir();(base/'source').mkdir();root.mkdir();(root/'cpu').mkdir();(root/'after-cpu').mkdir()
with tarfile.open(archive,'r:gz') as tf:
    members=tf.getmembers()
    if len(members)!=13 or any(not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in members):raise ValueError('Unsafe release members')
    if len({m.name for m in members})!=13:raise ValueError('Duplicate release member')
    for member in members:
        dest=base/'source'/member.name;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as stream:stream.write(tf.extractfile(member).read())
EXTRACT
"$python" "$base/source/scripts/verify_stage0_release.py" "$base/source" "$manifest"
"$python" "$base/source/scripts/verify_stage0_release.py" /work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-storage-03/source 9ee7bf1d6bf412acdc37fc83415476dbefdf45db89533571d4a646ee16f33772
"$python" "$base/source/scripts/verify_stage0_release.py" /work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-02/source 1e3a8a21e2fa68e24d1c9229238999c42a4a9b44bcc78344e73e188c87d1ee64
for wrapper in fixed25_recover_cpu.lsf fixed25_recover_after_cpu.lsf fixed25_recover_stage.lsf fixed25_recover_observe_pro.lsf; do bash -n "$base/source/scripts/hpc/$wrapper"; done
chmod -R a-w "$base/source"
export PYTHONPATH="$base/source/scripts:/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-storage-03/source/scripts:/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-02/source/scripts" PYTHONDONTWRITEBYTECODE=1
"$python" - "$root" <<'PREFLIGHT'
from pathlib import Path
import sys,json,subprocess
from fixed25_recover_storage import check_cpu_storage
r=Path(sys.argv[1]);ps=subprocess.run(['ps','-u','s234818','-o','pid=,etimes=,args='],capture_output=True,text=True,check=True,timeout=10)
active=[]
for row in ps.stdout.splitlines():
    fields=row.split(None,2)
    if len(fields)==3 and fields[2].split()[0].split('/')[-1]=='find' and '/zhome/95/e/206121/mosaic-runs' in fields[2] and '-mmin' in fields[2]:active.append(row)
with (r/'prior-inventory-process-inspection.json').open('x') as stream:json.dump({'scope':'Read-only exact Mosaic find inventory inspection','active_inventory_processes':active,'ps_returncode':ps.returncode},stream,indent=2)
if active:raise ValueError('Prior exact read-only inventory process remains active; do not duplicate')
print(json.dumps(check_cpu_storage(r,r/'launch-preflight')))
PREFLIGHT
"$python" - "$base/source" "$root" "$manifest" <<'SUBMIT'
import sys,json,re,subprocess,datetime
from pathlib import Path
source,root=map(Path,sys.argv[1:3]);manifest=sys.argv[3]
def submit(phase,producer=None):
    parent=root/phase
    command=['bsub','-q','hpc','-n','2' if phase=='cpu' else '1','-R','span[hosts=1]','-R','rusage[mem=4GB]' if phase=='cpu' else 'rusage[mem=2GB]','-W','00:20','-J','mosaic-fixed-storage04-'+phase,'-oo',str(parent/'stdout.%J.log'),'-eo',str(parent/'stderr.%J.log')]
    if producer:command+=['-w','ended('+producer+')']
    command+=['bash','-l',str(source/('scripts/hpc/fixed25_recover_cpu.lsf' if phase=='cpu' else 'scripts/hpc/fixed25_recover_after_cpu.lsf')),str(source),str(root),manifest]
    if producer:command+=[producer]
    receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,'args':command,'outcome':'unknown'}
    with (parent/'submission-started.json').open('x') as f:json.dump(receipt,f,indent=2)
    try:
        reply=subprocess.run(command,capture_output=True,text=True,timeout=60);receipt.update(returncode=reply.returncode,stdout=reply.stdout,stderr=reply.stderr)
        jobs=re.findall(r'Job <(\d+)>',reply.stdout)
        if reply.returncode==0 and len(jobs)==1:receipt.update(job_id=jobs[0],outcome='submitted')
    except Exception as error:receipt['error']=repr(error)
    finally:
        with (parent/'submission.json').open('x') as f:json.dump(receipt,f,indent=2)
    if receipt['outcome']!='submitted':raise ValueError('Unknown/failed submission; inspect exact receipts, never blind retry')
    actual=subprocess.run(['bjobs','-l',receipt['job_id']],capture_output=True,text=True,timeout=30)
    (parent/'scheduler-at-submission.txt').write_text(actual.stdout+actual.stderr)
    receipt['actual_scheduler_at_submission']=actual.stdout+actual.stderr
    return receipt
cpu=submit('cpu');callback=submit('after-cpu',cpu['job_id'])
record={'cpu':cpu,'callback':callback,'tool_manifest_sha256':manifest,'science_manifest_sha256':'1e3a8a21e2fa68e24d1c9229238999c42a4a9b44bcc78344e73e188c87d1ee64','scope':'Actual launcher-only CPU submissions; no new long-run ID is inferred'}
with (root/'initial-submissions.json').open('x') as f:json.dump(record,f,indent=2)
print('ACTUAL_SUBMISSIONS_JSON='+json.dumps(record))
SUBMIT
