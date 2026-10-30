"""Reuse full qualification and held protocol with the shared live storage owner."""
import argparse
import ast
import inspect
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

BASE=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
SCOPE=BASE/'nonchem-precision-continuation-26'
ROOT=SCOPE/'chemistry-continuation-05/combined-reference-qualification-03'
SOURCE=SCOPE/'combined-reference-qualification-release-03/source'
Q=SCOPE/'runtime-qualification-release-01/source'
SHARED=BASE/'runtime-production-release-05/source'
B=SCOPE/'boltz-precision-release-05/source'
sys.path.insert(1,str(SHARED/'scripts'))


def load(path,name):
    import importlib.util
    s=importlib.util.spec_from_file_location(name,path);v=importlib.util.module_from_spec(s);sys.modules[name]=v;s.loader.exec_module(v);return v


q=load(Q/'scripts/failure_diagnostic_ops.py','_new_combined_qualification_protocol')
q.ROOT=ROOT;q.SOURCE=SOURCE
read=q.read;write=q.write;sha=q.sha;module=q.module;helper=q.helper
PYTHON=q.PYTHON


def context(pin):
    op,plan=original_context(pin)
    q.require(plan['combined_reference_policy']=='protenix-fixed-chemistry-full-reference-v1','Explicit oracle policy required')
    return op,plan


def argv(phase,pin,dependency=None):
    q.require(phase in ('cpu','after-cpu','gpu-chemistry','after-chemistry'),'Chemistry qualification only')
    a=original_argv(phase,pin,dependency)
    a[a.index('-J')+1]='mosaic-Prochem-reference03-'+phase
    return a+['--shared-capacity-source',str(SHARED)]


def storage(op,parent,phase):
    shared=load(SHARED/'scripts/runtime_production_ops.py','_combined_current_live_storage_owner')
    current=read(SHARED/'plan.json');pin=sha(SHARED/'source_manifest.json')
    proxy=load(SHARED/'scripts/runtime_production_capacity.py','_combined_current_namespace_capacity')
    owner=proxy.proxy(op,current,SHARED,shared.ROOT,shared.ready_envelope(op,current,pin),shared.verify_qualified(op,current))
    own=q.owned_bytes(ROOT)
    q.require(own<512*1024**2 and owner.metadata_bytes()+512*1024**2-own<=2*1024**3,
              'Complete combined qualification exceeds funded shared control cap')
    rolling=load(shared.CAPACITY/'scripts/rolling_capacity.py','_combined_original_complete_capacity')
    proxy.configure(rolling,owner,current)
    import storage_policy_adapter
    plan=read(SOURCE/'plan.json');storage_policy_adapter.install(rolling,plan,parent,plan['shared_source146'])
    return rolling.storage(owner,parent,phase)


original_context=q.context;original_argv=q.argv
original_fixture_suite=q.fixture_suite


def fixture_suite(plan,payload):
    current=q.SOURCE
    try:
        q.SOURCE=Q
        count=original_fixture_suite(plan,payload)
    finally:q.SOURCE=current
    counts=read(SOURCE/'source_manifest.json')['cpu_counts']
    command=[PYTHON,'-m','pytest',*[str(SOURCE/(name.replace('.','/')+'.py')) for name in counts],
        '-q','--rootdir',str(SOURCE),'--import-mode=importlib','-o','addopts=','-p','no:cacheprovider',
        '--basetemp',str(Path(os.environ['TMPDIR'])/'combined-reference-fixtures'),
        '--junitxml='+str(payload/'combined-reference.xml')]
    write(payload/'combined-reference-command.json',{'argv':command,'counts':counts})
    environment={**os.environ,'PYTHONPATH':str(SOURCE/'scripts')+os.pathsep+str(q.CAPACITY/'scripts')}
    with (payload/'combined-reference.log').open('x') as stream:
        run=subprocess.run(command,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
    q.require(run.returncode==0,'Independent combined reference regressions failed')
    inherited=module(Path(plan['chemistry_source']['path'])/'scripts/chemistry_continuation_ops.py','_combined_raw_XML')
    added=inherited.xml(payload/'combined-reference.xml',counts)
    root=ET.parse(payload/'tests.xml').getroot();new=ET.parse(payload/'combined-reference.xml').getroot()
    root.extend([new] if new.tag=='testsuite' else list(new));ET.ElementTree(root).write(payload/'tests.xml',encoding='utf8',xml_declaration=True)
    inherited.xml(payload/'tests.xml',read(SOURCE/'source_manifest.json')['all_cpu_counts'])
    return count+added


q.context=context;q.argv=argv;q.storage=storage;q.fixture_suite=fixture_suite
# Complete original startup and fixture suite run for both inherited parsers;
# only the requested new chemistry GPU validator is dispatched.
tree=ast.parse(inspect.getsource(q.execute));changed=0
for n in ast.walk(tree):
    if isinstance(n,ast.For) and ast.unparse(n.iter)=="plan['qualification_arms']":
        n.iter=ast.Tuple([ast.Constant('chemistry')],ast.Load());changed+=1
q.require(changed==1,'Exactly one complete validator dispatch loop required')
exec(compile(ast.fix_missing_locations(tree),q.__file__,'exec'),q.__dict__)
submit=q.submit


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--pin',required=True)
    a=p.parse_args();q.execute(a.phase,a.pin)
