"""One-shot immutable allocated scalar summary; no model or optimizer calls."""
import ast,base64,hashlib,json,shlex,subprocess
from pathlib import Path
REPO=Path(__file__).resolve().parents[4]
OWNER=REPO/'experiments/validation/endpoint-qualification'
RUN=OWNER/'runs/20261025-protenix-endpoint-forensics-01'
LOCAL=REPO/'results/hpc/20261025-protenix-endpoint-forensics-01'
RUNTIME=r"""
import hashlib,json,os,subprocess,sys
from pathlib import Path
out=Path(CFG['output_root']);source=out.parent.parent/'protenix-endpoint-forensics-release-01/source'
pin='d127d6abe87307a5d305edb6d04e3a32e3545c508e58b82ce45290e0f3fc03d8'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert os.environ.get('LSB_QUEUE')=='hpc' and os.environ.get('LSB_DJOB_NUMPROC')=='1'
assert sha(out/'summarize.py')==CFG['code_sha256'] and sha(source/'source_manifest.json')==pin
for name,digest in json.loads((source/'source_manifest.json').read_bytes())['files'].items():assert sha(source/name)==digest
sys.dont_write_bytecode=True;sys.path.insert(0,str(source/'scripts'));import protenix_forensics_ops as ops
m=ops.verify(source,pin);plan=ops.read(source/'plan.json');ops.external(m,plan);ops.certificate(pin,m)
p=out/'entry';p.mkdir();ops.storage(p,'entry')
python=ops.PYTHON
with (out/'self-test.json').open('x') as log:subprocess.run([python,str(out/'summarize.py'),'--self-test'],check=True,stdout=log)
command=[python,str(out/'summarize.py'),'--input',str(ops.ROOT/'gpu/payload/forensics/report.json'),'--input-sha256',CFG['diagnostic_report_sha256'],'--admission',str(ops.FAILED_ADMISSION['path']),'--admission-sha256',CFG['failed_admission_sha256'],'--output',str(out/'forensic-summary.json')]
ops.write(out/'invocation.json',{'argv':command,'model_calls':0,'updates':0});subprocess.run(command,check=True)
ops.external(m,plan)
ops.write(out/'completion.json',{'status':'allocated_scalar_analysis_completed','job_id':os.environ['LSB_JOBID'],'summary':ops.binding(out/'forensic-summary.json'),'code_sha256':CFG['code_sha256'],'original_results_changed':False,'updates':0,'can_admit_new_child':False})
"""
BOOTSTRAP=r"""
import base64,hashlib,json,os,re,shlex,subprocess,sys
from pathlib import Path
out=Path(CFG['output_root']);source=out.parent.parent/'protenix-endpoint-forensics-release-01/source'
pin='d127d6abe87307a5d305edb6d04e3a32e3545c508e58b82ce45290e0f3fc03d8'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(source/'source_manifest.json')==pin
for name,digest in json.loads((source/'source_manifest.json').read_bytes())['files'].items():assert sha(source/name)==digest
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True;sys.path.insert(0,str(source/'scripts'));import protenix_forensics_ops as ops
m=ops.verify(source,pin);ops.external(m,ops.read(source/'plan.json'));ops.certificate(pin,m)
assert not out.exists(),'Existing/unknown addendum outcome; never repeat'
out.mkdir();p=out/'preflight';p.mkdir();gate=ops.storage(p,'preflight')
code=base64.b64decode(CODE,validate=True);assert len(code)==CFG['code_bytes'] and hashlib.sha256(code).hexdigest()==CFG['code_sha256']
script=out/'summarize.py';script.write_bytes(code);script.chmod(0o444)
ops.write(out/'plan.json',CFG);(out/'plan.json').chmod(0o444)
runner=b'CFG='+repr(CFG).encode()+b'\n'+base64.b64decode(RUNNER,validate=True)
path=out/'run.py';path.write_bytes(runner);path.chmod(0o444);rpin=sha(path)
check="import hashlib; p="+repr(str(path))+"; assert hashlib.sha256(open(p,'rb').read()).hexdigest()=="+repr(rpin)+"; exec(compile(open(p,'rb').read(),p,'exec'))"
wrapper='#!/bin/bash\nset -euo pipefail\nmodule load python3/3.12.11\nexport PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""\n'+shlex.quote(ops.PYTHON)+' -c '+shlex.quote(check)+'\n'
launch=out/'analysis.lsf';launch.write_text(wrapper);launch.chmod(0o444);subprocess.run(['bash','-n',str(launch)],check=True)
argv=['bsub','-q','hpc','-n','1','-R','span[hosts=1]','-R','rusage[mem=2GB]','-W','00:15','-J','mosaic-pro-forensics-summary','-oo',str(out/'stdout.%J.log'),'-eo',str(out/'stderr.%J.log'),'bash','-l',str(launch)]
ops.write(out/'submission-started.json',{'status':'unknown','argv':argv})
r=subprocess.run(argv,capture_output=True,text=True,timeout=60);jobs=re.findall(r'Job <(\d+)> is submitted',r.stdout)
record={'status':'submitted' if r.returncode==0 and len(jobs)==1 else 'unknown','argv':argv,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'code_sha256':CFG['code_sha256'],'run_sha256':rpin,'wrapper_sha256':sha(launch),'storage_gate':gate}
if len(jobs)==1:record['job_id']=jobs[0]
ops.write(out/'submission.json',record);assert record['status']=='submitted','Unknown submission; inspect instead of repeat'
print(json.dumps(record))
"""
def main():
 code=(OWNER/'scripts/summarize_protenix_forensics.py').read_bytes().replace(b'\r\n',b'\n');ast.parse(code)
 cfg=json.loads((RUN/'analysis-plan.json').read_bytes());assert hashlib.sha256(code).hexdigest()==cfg['code_sha256']
 program='CFG='+repr(cfg)+'\nCODE='+repr(base64.b64encode(code).decode())+'\nRUNNER='+repr(base64.b64encode(RUNTIME.encode()).decode())+'\n'+BOOTSTRAP
 ast.parse(program)
 out=LOCAL/'analysis-submit-01';out.mkdir(exist_ok=False);(out/'program.py').write_text(program,encoding='utf8',newline='\n')
 argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
 with (out/'stdout.json').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:r=subprocess.run(argv,input=program.encode(),stdout=stdout,stderr=stderr,timeout=240)
 assert r.returncode==0,(out/'stderr.txt').read_text()
 record=json.loads((out/'stdout.json').read_bytes());(RUN/'analysis-submission.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8',newline='\n')
 print(json.dumps({k:record[k] for k in ('status','job_id','code_sha256')},indent=2))
if __name__=='__main__':main()
