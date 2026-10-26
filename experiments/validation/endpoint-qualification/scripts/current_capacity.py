"""Metadata-only admission including both already submitted chemistry futures.

Run the authenticated existing production03 guard in a fresh metadata process,
avoiding collisions with the current checker modules. No scientific import,
model execution, deletion or submission is performed here.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess

BASE=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
SOURCE=BASE/'chemistry-production-release-03/source'
PIN='c73796b041d8432ef1634e8c56920006170a094d3bdc74867c08f3b5415126d7'
PYTHON='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
GROUP_ALLOWANCE=512*1024**2+272*1024**2

def require(value,message):
    if not value:raise ValueError(message)

def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def validate(prior,excess):
    require(type(excess) is int and excess>=0,'Nonnegative complete forecast excess required')
    require(prior.get('status')=='passed' and prior.get('original_gate_changed') is False
            and prior.get('adopted_full_production') is True
            and prior.get('reserve_bytes')==4*1024**3 and prior.get('file_reserve')==100000,
            'Full current chemistry/old campaign admission and reserves required')
    require(set(prior['chemistry'])=={'protenix','boltz'}
            and all(type(v['full']) is int and v['full']>0 for v in prior['chemistry'].values()),
            'Both complete chemistry futures required')
    require(type(prior['required_bytes']) is int and type(prior['available_bytes']) is int,
            'Finite integer byte budget required')
    result={'status':'failed','original_gate_changed':False,'parent_gate':prior,
        'required_bytes':prior['required_bytes']+GROUP_ALLOWANCE+excess,
        'available_bytes':prior['available_bytes'],'reserve_bytes':4*1024**3,'file_reserve':100000,
        'new_group_allowance_bytes':GROUP_ALLOWANCE,'additional_projection_excess_bytes':excess,
        'scope':'Whole existing chemistry03/old budgets plus new continuation512MiB and precision272MiB; no forecast/reserve removed'}
    require(result['available_bytes']>=result['required_bytes'],'Current full futures plus new checker scopes do not fit')
    result['status']='passed'
    return result

def gate(parent,phase,excess=0):
    require(type(phase) is str and re.fullmatch('[a-z][a-z0-9_-]{0,40}',phase),'Invalid capacity phase')
    parent=Path(parent);require(parent.is_dir() and not parent.is_symlink(),'Owned capacity parent absent/linked')
    output=parent/('current-chemistry-capacity-'+phase);output.mkdir(exist_ok=False)
    program=r'''
import hashlib,json,os,sys
from pathlib import Path
source=Path(sys.argv[1]);pin=sys.argv[2];output=Path(sys.argv[3])
raw=(source/'source_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==pin
manifest=json.loads(raw);assert len(manifest['files'])==14
actual={p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file() or p.is_symlink()}
assert actual==set(manifest['files'])|{'source_manifest.json'}
for name,digest in manifest['files'].items():
 p=source/name;assert not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==digest
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
sys.path.insert(0,str(source/'scripts'))
import production_ops as op
assert op.SOURCE==source
op.storage(output,'entry',adopted=True)
'''
    command=[PYTHON,'-B','-c',program,str(SOURCE),PIN,str(output)]
    process=subprocess.run(command,capture_output=True,text=True,timeout=360)
    write(output/'invocation.json',{'argv':command,'returncode':process.returncode,
        'stdout':process.stdout,'stderr':process.stderr,'source_manifest_sha256':PIN,
        'numerical_evaluations':0,'jobs_submitted':False})
    require(process.returncode==0,'Authenticated full current-capacity process failed; raw evidence preserved')
    prior=json.loads((output/'production-storage-entry.json').read_bytes())
    try:result=validate(prior,excess)
    except BaseException as error:
        write(output/'assessment.json',{'status':'failed','error':repr(error),'parent_gate':prior});raise
    write(output/'assessment.json',result)
    return result
