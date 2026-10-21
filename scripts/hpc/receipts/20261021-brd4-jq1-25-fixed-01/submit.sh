set -euo pipefail
archive=/work3/s234818/mosaic/20261021-brd4-jq1-25-fixed-01.tar.gz
base=/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-01
root=/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-01
manifest=dc4ac68d1818038ba177241311aecf62072a065ef147356186f1f49eba2d20f3
printf '%s  %s\n' ad7492baed5193abaa632810694c86e2ed29455cb5c6bd136b276cd59eb63e68 "$archive" | sha256sum -c -
mkdir -p /work3/s234818/mosaic/runs
mkdir "$base" "$root"
mkdir "$base/source" "$root/cpu" "$root/after-cpu"
tar -xzf "$archive" -C "$base/source"
module load python3/3.12.11
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
"$python" "$base/source/scripts/verify_stage0_release.py" "$base/source" "$manifest"
bash -n "$base/source/scripts/hpc/fixed25_cpu.lsf"
bash -n "$base/source/scripts/hpc/fixed25_stage.lsf"
chmod -R a-w "$base/source"
export PYTHONPATH="$base/source/scripts" PYTHONDONTWRITEBYTECODE=1
"$python" "$base/source/scripts/fixed25_storage.py" --run-dir "$root/launch-preflight" --projected-bytes 1073741824
"$python" - "$base/source" "$root" "$manifest" <<'PYREMOTE'
import sys,subprocess,json,re,datetime,pathlib
source,root,manifest=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3]
def submit(phase,producer=None):
 p=root/phase
 cmd=['bsub','-q','hpc','-n','4' if phase=='cpu' else '1','-R','span[hosts=1]','-R','rusage[mem=4GB]' if phase=='cpu' else 'rusage[mem=2GB]','-W','01:00' if phase=='cpu' else '00:20','-J','mosaic-jq1-fixed-'+phase,'-oo',str(p/'stdout.%J.log'),'-eo',str(p/'stderr.%J.log')]
 if producer:cmd+=['-w',f'ended({producer})']
 cmd+=['bash','-l',str(source/('scripts/hpc/fixed25_cpu.lsf' if phase=='cpu' else 'scripts/hpc/fixed25_stage.lsf')),str(source)]
 if phase=='cpu':cmd += [str(p),manifest]
 else:cmd += [str(root),manifest,'after-cpu','none',str(p),str(producer)]
 v=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
 receipt={'submitted_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'args':cmd,'returncode':v.returncode,'stdout':v.stdout,'stderr':v.stderr,'phase':phase}
 m=re.search(r'Job <(\d+)>',v.stdout)
 if m:receipt['job_id']=m[1]
 (p/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
 if v.returncode or not m:raise RuntimeError('Unknown/failed submission; inspect persisted receipt, never duplicate')
 a=subprocess.run(['bjobs','-l',m[1]],capture_output=True,text=True,timeout=30)
 (p/'scheduler-at-submission.txt').write_text(a.stdout+a.stderr)
 return receipt
cpu=submit('cpu');callback=submit('after-cpu',cpu['job_id'])
record={'cpu':cpu,'callback':callback,'long_run_submitted':False,'native_submitted':False}
(root/'initial-submissions.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
PYREMOTE
