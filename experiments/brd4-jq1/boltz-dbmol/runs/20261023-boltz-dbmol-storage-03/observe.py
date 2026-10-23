"""One compact, read-only query of actual current comparison/pilot jobs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
REMOTE = r'''
from datetime import datetime,timezone
import hashlib,json,re,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
result={'utc':datetime.now(timezone.utc).isoformat(),'campaigns':{},'scope':'Read-only actual scheduler and committed artifact metadata; no scientific evaluation'}
for key,root in [('baseline',base/'boltz-lr0025-02'),('dbmol',base/'boltz-dbmol-pilot-01')]:
 jobs=json.loads((root/'jobs.json').read_bytes());rows={}
 for label,job in jobs.items():
  parent=root/label;sub=json.loads((parent/'submission.json').read_bytes())
  assert sub['outcome']=='submitted' and sub['job_id']==job
  query=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=15)
  raw=query.stdout+query.stderr;compact=re.sub(r'\s+','',raw)
  row={'job':job,'phase':sub['phase'],'returncode':query.returncode,'raw_scheduler':raw,'status':re.findall(r'Status<([^>]+)>',compact),'artifacts':{},'raw_xml':{}}
  for path in [parent/'payload/driver-invocation.json',parent/'payload/study/assessment.json',parent/'storage-entry/gate.json',parent/'storage-cpu-entry/gate.json']:
   if path.is_file():row['artifacts'][str(path)]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
  if sub['phase']=='cpu':
   for path in (parent/'payload').glob('*.xml'):
    try:
     tree=ET.parse(path);row['raw_xml'][path.name]={name:len(list(tree.iter(name)))for name in ['testcase','failure','error','skipped']}
    except ET.ParseError as e:row['raw_xml'][path.name]={'incomplete':repr(e)}
  if sub['phase']=='segment':
   for field in ['states','updates']:
    path=parent/'payload/study/case-0'/(field+'.jsonl')
    if path.is_file():
     count=0
     with path.open('rb')as stream:
      for line in stream:
       if not line.endswith(b'\n'):break
       json.loads(line);count+=1
     row[field+'_newline_committed_rows']=count
  rows[label]=row
 result['campaigns'][key]={'root':str(root),'jobs':rows,'projection_present':(root/'projection.json').is_file(),'cpu_certificate_present':(root/'cpu-certificate.json').is_file()}
print(json.dumps(result))
'''


def main():
    folder = REPO / 'results/hpc' / HERE.name / 'observations'
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    command = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python -c ' + shlex.quote(REMOTE)
    argv = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'ConnectionAttempts=1', 'dtu-hpc', 'bash -l -c ' + shlex.quote(command)]
    result = subprocess.run(argv, capture_output=True, timeout=180)
    (folder / (stamp + '-stdout.bin')).write_bytes(result.stdout)
    (folder / (stamp + '-stderr.bin')).write_bytes(result.stderr)
    (folder / (stamp + '-receipt.json')).write_text(json.dumps({'returncode': result.returncode, 'args': argv}) + '\n')
    if result.returncode:
        raise RuntimeError('Read-only connection failed; no retry or job mutation')
    value = json.loads(result.stdout)
    print(json.dumps({'utc': value['utc'], 'campaigns': {
        key: {label: {'job': row['job'], 'status': row['status'], 'raw_xml': row['raw_xml']}
              for label, row in campaign['jobs'].items()}
        for key, campaign in value['campaigns'].items()}}, indent=2))


if __name__ == '__main__':
    main()
