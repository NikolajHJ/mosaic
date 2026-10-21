set -euo pipefail
archive=/work3/s234818/mosaic/20261021-brd4-jq1-25-fixed-storage-02.tar.gz
base=/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-storage-02
root=/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-storage-02
manifest=c8115e2686b94d78ff1889b9dc8161fdf00356639524853c46d4100fd8638fd9
printf '%s  %s\n' 8ed85ddae3e116640513b84e4786e8891315541768bb069d8cbfd0546124accc "$archive" | sha256sum -c -
[[ $(stat -c %s "$archive") == 18560 ]]
mkdir "$base" "$root"
mkdir "$base/source" "$root/cpu" "$root/after-cpu"
tar -xzf "$archive" -C "$base/source"
module load python3/3.12.11
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
"$python" "$base/source/scripts/verify_stage0_release.py" "$base/source" "$manifest"
bash -n "$base/source/scripts/hpc/fixed25_split_cpu.lsf"
bash -n "$base/source/scripts/hpc/fixed25_split_after_cpu.lsf"
bash -n "$base/source/scripts/hpc/fixed25_split_stage.lsf"
chmod -R a-w "$base/source"
export PYTHONPATH="$base/source/scripts:/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-02/source/scripts" PYTHONDONTWRITEBYTECODE=1
"$python" - "$root" <<'PREFLIGHT'
from pathlib import Path
import sys,json
from fixed25_split_storage import check_cpu_storage
r=Path(sys.argv[1]);print(json.dumps(check_cpu_storage(r,r/'launch-preflight')))
PREFLIGHT
"$python" - "$base/source" "$root" "$manifest" <<'SUBMIT'
import sys,json,re,subprocess,datetime
from pathlib import Path
source,root=map(Path,sys.argv[1:3]);manifest=sys.argv[3]
def submit(phase,producer=None):
    parent=root/phase
    command=['bsub','-q','hpc','-n','2' if phase=='cpu' else '1','-R','span[hosts=1]','-R','rusage[mem=4GB]' if phase=='cpu' else 'rusage[mem=2GB]','-W','00:20','-J','mosaic-fixed-storage02-'+phase,'-oo',str(parent/'stdout.%J.log'),'-eo',str(parent/'stderr.%J.log')]
    if producer:command+=['-w',f'ended({producer})']
    command+=['bash','-l',str(source/('scripts/hpc/fixed25_split_cpu.lsf' if phase=='cpu' else 'scripts/hpc/fixed25_split_after_cpu.lsf')),str(source),str(root),manifest]
    if producer:command+=[producer]
    receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,'args':command,'outcome':'unknown'}
    with (parent/'submission-started.json').open('x') as f:json.dump(receipt,f,indent=2)
    try:
        reply=subprocess.run(command,capture_output=True,text=True,timeout=60)
        receipt.update(returncode=reply.returncode,stdout=reply.stdout,stderr=reply.stderr)
        matches=re.findall(r'Job <(\d+)>',reply.stdout)
        if reply.returncode==0 and len(matches)==1:receipt.update(job_id=matches[0],outcome='submitted')
    except Exception as error:receipt['error']=repr(error)
    finally:
        with (parent/'submission.json').open('x') as f:json.dump(receipt,f,indent=2)
    if receipt['outcome']!='submitted':raise RuntimeError('Unknown/failed outcome; inspect receipt before recovery, never duplicate')
    actual=subprocess.run(['bjobs','-l',receipt['job_id']],capture_output=True,text=True,timeout=30)
    (parent/'scheduler-at-submission.txt').write_text(actual.stdout+actual.stderr)
    return receipt
cpu=submit('cpu');callback=submit('after-cpu',cpu['job_id'])
record={'cpu':cpu,'callback':callback,'science_manifest_sha256':'1e3a8a21e2fa68e24d1c9229238999c42a4a9b44bcc78344e73e188c87d1ee64','tool_manifest_sha256':manifest,'scope':'Actual launcher-only CPU submissions; long IDs require callback receipts'}
with (root/'initial-submissions.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
SUBMIT
