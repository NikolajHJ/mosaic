set -euo pipefail
archive=/work3/s234818/mosaic/20261021-brd4-jq1-25-protenix-lr03-01.tar.gz
base=/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-protenix-lr03-01
root=/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-protenix-lr03-01
manifest=d5c1bf07868a6ef8f3c1a96a24afee11b76c8fb2affe334fb924d0a38b7849a0
printf '%s  %s\n' 8c50ef45cf1d60d90b380f21e39aa41cebd95fa6aefd01c0fe5e107fde68ba5c "$archive" | sha256sum -c -
mkdir -p /work3/s234818/mosaic/runs
mkdir "$base" "$root"
mkdir "$base/source" "$root/cpu" "$root/after-cpu"
tar -xzf "$archive" -C "$base/source"
module load python3/3.12.11
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
"$python" "$base/source/scripts/verify_stage0_release.py" "$base/source" "$manifest"
bash -n "$base/source/scripts/hpc/protenix_lr03_cpu.lsf"
bash -n "$base/source/scripts/hpc/protenix_lr03_stage.lsf"
chmod -R a-w "$base/source"
export PYTHONPATH="$base/source/scripts" PYTHONDONTWRITEBYTECODE=1
"$python" "$base/source/scripts/protenix_lr03_storage.py" --run-dir "$root/launch-preflight" --projected-bytes 1073741824
"$python" - "$base/source" "$root" "$manifest" <<'PYREMOTE'
import sys,subprocess,json,re,datetime,pathlib
source,root,manifest=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3]
def submit(phase,producer=None):
 p=root/phase
 cmd=['bsub','-q','hpc','-n','4' if phase=='cpu' else '1','-R','span[hosts=1]','-R','rusage[mem=4GB]' if phase=='cpu' else 'rusage[mem=2GB]','-W','01:00' if phase=='cpu' else '00:20','-J','mosaic-protenix-lr03-'+phase,'-oo',str(p/'stdout.%J.log'),'-eo',str(p/'stderr.%J.log')]
 if producer:cmd+=['-w',f'ended({producer})']
 cmd+=['bash','-l',str(source/('scripts/hpc/protenix_lr03_cpu.lsf' if phase=='cpu' else 'scripts/hpc/protenix_lr03_stage.lsf')),str(source)]
 if phase=='cpu':cmd += [str(p),manifest]
 else:cmd += [str(root),manifest,'after-cpu',str(p),str(producer)]
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
