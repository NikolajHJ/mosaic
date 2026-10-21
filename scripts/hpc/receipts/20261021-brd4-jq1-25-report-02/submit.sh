#!/bin/bash
set -euo pipefail
archive=/work3/s234818/mosaic/20261021-brd4-jq1-25-report-02.tar.gz
base=/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-report-02
root=/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-report-02
manifest=cff9a2fd0a1ee7b2b1e3fb53d68ee4a46bee9d1fa453506684d7cc7bbfeb75e0
printf '%s  %s\n' f6608dd46584f8f7c0cbac21d41677555e13f98b74a86b94a93df11afce39ceb "$archive" | sha256sum -c -
mkdir "$base" "$root"
mkdir "$base/source"
tar -xzf "$archive" -C "$base/source"
module load python3/3.12.11
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
"$python" "$base/source/scripts/verify_stage0_release.py" "$base/source" "$manifest"
bash -n "$base/source/scripts/hpc/jq1_25_report_stage.lsf"
chmod -R a-w "$base/source"
"$python" - "$base/source" "$root" "$manifest" <<'PY'
from datetime import datetime,timezone
import json,re,subprocess,sys
from pathlib import Path
source,root,manifest=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
jobs={}
def submit(name,phase,arm='none',dependency=None,walltime='00:30',slots='2',memory='4GB'):
    parent=root/name;parent.mkdir(exist_ok=False)
    command=['bsub','-q','hpc','-n',slots,'-R','span[hosts=1]','-R','rusage[mem='+memory+']','-W',walltime,
       '-J','mosaic-jq1-report-'+name,'-oo',str(parent/'stdout.%J.log'),'-eo',str(parent/'stderr.%J.log')]
    if dependency: command+=['-w',dependency]
    if phase=='interim': command+=['bash','-l',str(source/'scripts/hpc/jq1_25_interim_stage.lsf'),str(source),str(root),manifest,str(parent)]
    else: command+=['bash','-l',str(source/'scripts/hpc/jq1_25_report_stage.lsf'),str(source),str(root),manifest,phase,arm,str(parent)]
    reply=subprocess.run(command,capture_output=True,text=True,timeout=60)
    record={'submitted_utc':datetime.now(timezone.utc).isoformat(),'args':command,'returncode':reply.returncode,'stdout':reply.stdout,'stderr':reply.stderr}
    match=re.search(r'Job <(\d+)>',reply.stdout)
    if match: record['job_id']=match[1]
    (parent/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
    if reply.returncode or not match: raise RuntimeError('Submission failed or unknown; retain receipt and never duplicate blindly')
    jobs[name]=match[1];(root/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    live=subprocess.run(['bjobs','-l',match[1]],capture_output=True,text=True)
    (parent/'scheduler-at-submission.txt').write_text(live.stdout+live.stderr)
    print(name,match[1],flush=True)
    return match[1]
cpu=submit('cpu','cpu',walltime='01:00')
certificate=submit('cpu-certificate','certify-cpu',dependency='ended('+cpu+')',slots='1',memory='2GB',walltime='00:20')
collectors=[]
for arm,native in [('protenix','29770179'),('boltz2','29770093'),('boltz2_chemistry','29770124')]:
    collectors.append(submit('complete-'+arm,'capture',arm,dependency='ended('+native+')&&done('+certificate+')'))
submit('combined-report','report',dependency='&&'.join('done('+j+')' for j in [certificate,*collectors]),walltime='02:00')
submit('interim-0001','interim',dependency='done('+certificate+')',walltime='00:30')
print(json.dumps({'jobs':jobs,'scope':'Report/test/evidence jobs only; GPU experiments unchanged'}))
PY
