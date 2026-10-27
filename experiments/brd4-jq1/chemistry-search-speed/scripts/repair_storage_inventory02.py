"""Bounded read-only Mosaic inventory plus exact current repair observation."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / 'results/hpc/20261027-repair-storage-02/inventory02'

def main():
    observer = REPO/'experiments/validation/endpoint-qualification/scripts/observe_endpoint_repair.py'
    spec = importlib.util.spec_from_file_location('_metadata_observer', observer)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    program = module.PROGRAM
    start=program.index('specs={');end=program.index('\nrecord=',start)
    specs={
      'precision05':('protenix-endpoint-fix-05','protenix-endpoint-fix-release-05/source','65b4bd350386a8550c3b588eb7de2b0f6084b32ad790a907851c632dba0f023e'),
      'chemistry04':('chemistry-production-04','chemistry-production-release-04/source','01e2de2a3413bd903590dd331c5cedf6b756e856a83df2b73ad65e02b9f9fceb'),
      'chemistry03':('chemistry-production-03','chemistry-production-release-03/source','c73796b041d8432ef1634e8c56920006170a094d3bdc74867c08f3b5415126d7')}
    program=program[:start]+'specs='+repr(specs)+program[end:]
    program=program.replace("assert re.fullmatch('[a-zA-Z0-9_-]+',name)", "assert not Path(name).is_absolute() and '..' not in Path(name).parts and re.fullmatch('[a-zA-Z0-9_/-]+',name)")
    program=program.replace("jobs=json.loads((root/'jobs.json').read_bytes());ids.update(jobs.values())", "jobs=json.loads((root/'jobs.json').read_bytes()) if (root/'jobs.json').is_file() else {};ids.update(jobs.values())")
    program=program.replace('parent=root/name', "parent=root/(name.removeprefix('control/') if label.startswith('chemistry') else name)")
    program=program.replace("for p in study.glob('*.jsonl'):","for p in study.rglob('*.jsonl'):")
    program=program.replace("details['jsonl'][p.name]", "details['jsonl'][p.relative_to(study).as_posix()]")
    program=program.replace("for p in root.glob('*.json'):collect(p)", "for p in root.glob('*.json'):collect(p)\n row['jobs_file_exists']=(root/'jobs.json').is_file();row['root_entries']=sorted(p.name for p in root.iterdir())\n for p in (root/'preflight').rglob('*.json'):collect(p)")
    extra=r'''
import heapq,os,time
start=time.monotonic();root=Path('/work3/s234818/mosaic');large=[];sums={};count=0;skipped=[]
for parent,dirs,files in os.walk(root,followlinks=False):
 assert time.monotonic()-start<75,'Incomplete inventory; no storage conclusion'
 dirs[:]=[n for n in dirs if n not in ('envs','models','.git','__pycache__') and not (Path(parent)/n).is_symlink()]
 for name in files:
  p=Path(parent)/name;s=p.lstat()
  if not stat.S_ISREG(s.st_mode) or p.is_symlink():skipped.append(str(p));continue
  rel=p.relative_to(root);key='/'.join(rel.parts[:2]);row=sums.setdefault(key,{'bytes':0,'files':0});row['bytes']+=s.st_size;row['files']+=1;count+=1
  if s.st_size>=8*1024**2:
   item={'path':str(p),'bytes':s.st_size,'identity':[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_nlink],'uid':s.st_uid}
   large.append(item)
record['inventory']={'status':'completed_metadata_only_no_deletion_authority','files':count,'elapsed_seconds':time.monotonic()-start,'groups':sums,'large':sorted(large,key=lambda r:r['bytes'],reverse=True),'skipped_special_or_linked':skipped}
try:
 q=subprocess.run(['getquota_work3.sh'],capture_output=True,text=True,timeout=30)
 record['quota']={'argv':['getquota_work3.sh'],'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr,'capacity_pass':False}
except OSError as error:record['quota']={'error':repr(error),'capacity_pass':False}
'''
    program=program.replace("record['compact_bytes']=total",extra+"\nrecord['compact_bytes']=total")
    ast.parse(program);OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'program.py').write_text(program,encoding='utf8',newline='\n')
    argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    receipt={'started_utc':datetime.now(timezone.utc).isoformat(),'status':'unknown_readonly_outcome','program_sha256':hashlib.sha256(program.encode()).hexdigest()}
    try:
        with (OUT/'raw.json').open('xb') as output,(OUT/'stderr.txt').open('xb') as error:
            p=subprocess.run(argv,input=program.encode(),stdout=output,stderr=error,timeout=170)
        receipt['returncode']=p.returncode
        if p.returncode:raise RuntimeError((OUT/'stderr.txt').read_text(errors='replace')[-3000:])
        value=json.loads((OUT/'raw.json').read_bytes())
        receipt.update(status='complete_readonly_inventory',files=value['inventory']['files'],captured_utc=value['captured_utc'])
        print(json.dumps({'captured_utc':value['captured_utc'],'jobs':{k:v['jobs'] for k,v in value['roots'].items()},'large_files':len(value['inventory']['large']),'top_groups':sorted(value['inventory']['groups'].items(),key=lambda r:r[1]['bytes'],reverse=True)[:12],'quota':value['quota']},indent=2))
    finally:
        receipt['raw_sha256']=hashlib.sha256((OUT/'raw.json').read_bytes()).hexdigest();receipt['ended_utc']=datetime.now(timezone.utc).isoformat()
        (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8',newline='\n')

if __name__=='__main__':main()
