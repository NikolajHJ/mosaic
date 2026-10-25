"""Read-only compact failed-validation metadata; no scientific array loading."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REPO=Path(__file__).resolve().parents[4]
OUT=REPO/'results/hpc/20261025-continuation-admission-01/failure-evidence-01'


def main():
    program=r'''
from datetime import datetime,timezone
from pathlib import Path
import base64,hashlib,json
root=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-endpoint-continuation-01')
result={'captured_utc':datetime.now(timezone.utc).isoformat(),'mutated':False,'arms':{}}
for arm,offset in [('protenix',20000),('boltz',10000),('dbmol',5000)]:
 parent=root/(arm+'-validate-'+str(offset).zfill(5));study=parent/'payload/study'
 row={'files':{},'admission_findings':[],'study_exists':study.is_dir()}
 paths=[study/name for name in ['summary.json','assessment.json','worker.log','native_gradient_checks/report.json','early_pilot/report.json']]
 paths+=list(study.glob('*-prospective/admission.json'))+list(study.glob('*-prospective/*/admission.json'))
 paths+=list(study.glob('*-prospective/original-coarse.json'))+list(study.glob('*-prospective/*/original-coarse.json'))
 for path in paths:
  if not path.is_file():continue
  assert not path.is_symlink();data=path.read_bytes()
  item={'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
  assert len(data)<=32*1024**2
  item['base64']=base64.b64encode(data).decode();row['files'][str(path.relative_to(parent))]=item
  if path.name=='admission.json':
   document=json.loads(data)
   row['admission_findings'].append({k:document.get(k) for k in ['status','can_admit_new_child','failures','baseline_checks','field_outcomes','frozen_unchanged'] if k!='baseline_checks'})
 result['arms'][arm]=row
print(json.dumps(result))
'''
    OUT.mkdir(exist_ok=False);(OUT/'program.py').write_text(program,encoding='utf8',newline='\n')
    argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
          'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    with (OUT/'raw.json').open('xb') as output,(OUT/'stderr.txt').open('xb') as error:
        result=subprocess.run(argv,input=program.encode(),stdout=output,stderr=error,timeout=120)
    (OUT/'receipt.json').write_text(json.dumps({'returncode':result.returncode,'argv':argv,'mutated':False,
        'program_sha256':hashlib.sha256(program.encode()).hexdigest(),
        'raw_sha256':hashlib.sha256((OUT/'raw.json').read_bytes()).hexdigest()},indent=2)+'\n',encoding='utf8',newline='\n')
    assert result.returncode==0,(OUT/'stderr.txt').read_text()
    record=json.loads((OUT/'raw.json').read_bytes());verified=[]
    for arm,row in record['arms'].items():
        for relative,item in row['files'].items():
            assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
            raw=base64.b64decode(item['base64'],validate=True)
            assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
            path=OUT/'files'/arm/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            verified.append({k:v for k,v in item.items() if k!='base64'})
        print(arm,json.dumps(row['admission_findings']))
    (OUT/'verified-files.json').write_text(json.dumps(verified,indent=2)+'\n',encoding='utf8',newline='\n')
    print('Verified',len(verified),'metadata files; no NPZ arrays downloaded or evaluated')


if __name__=='__main__':main()
