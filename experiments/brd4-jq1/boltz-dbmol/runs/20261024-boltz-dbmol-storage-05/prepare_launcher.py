"""Prepare launcher05; old executing members and all saved studies stay sealed."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
PREVIOUS='df89def61047469e0db8ee6f4b235c75344dd4ad9f35a0833b0839f23859e9fa'
OLD_SCIENCE='24c207da5510fa579576a04ad32bf1afc6e1f23f51dcb5251d82a4ebdfa8f436'
SCIENCE='600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed'
STARTUP_CASES=28

def sha(data):return hashlib.sha256(data).hexdigest()

def once(text,old,new):
    assert text.count(old)==1,repr(old)
    return text.replace(old,new)

def previous():
    archive=REPO/'results/hpc/20261023-boltz-dbmol-storage-04/source01/source.tar.gz'
    assert archive.stat().st_size==35608 and sha(archive.read_bytes())=='40ee3c1ea0e9181528c82c5d99e7d6347033a99051ba7e55b5ffdde1edb7cefc'
    files={}
    with tarfile.open(archive) as tar:
        for member in tar:
            assert member.isfile() and member.name.startswith('source/') and '..' not in Path(member.name).parts
            name=member.name.removeprefix('source/');assert name not in files
            files[name]=tar.extractfile(member).read()
    raw=files.pop('source_manifest.json');assert sha(raw)==PREVIOUS
    manifest=json.loads(raw);assert len(files)==8 and set(files)==set(manifest['files'])
    for name,data in files.items():assert sha(data)==manifest['files'][name]
    return files

def predecessor_pins():
    directory=REPO/'results/hpc/20261023-boltz-dbmol-storage-04/cpu-evidence-02'
    metadata=json.loads((directory/'metadata.json').read_bytes())['files']
    names=['cpu-certificate.json','cpu-04/submission.json','after-cpu-04/submission.json',
        'cpu-04/stdout.29802621.log','after-cpu-04/stdout.29802622.log',
        'pilot-native-0000/submission.json','pilot-after-native-0000/submission.json',
        'pilot-native-0000/stdout.29802648.log','pilot-after-native-0000/stdout.29802649.log',
        'pilot-native-0000/stderr.29802648.log','pilot-after-native-0000/stderr.29802649.log',
        'pilot-native-0000/storage-entry/gate.json']
    names += ['cpu-04/payload/'+group+extension for group in ('hpc-contract','legacy','native-mapping','chemistry','dbmol-native','dbmol-transport','launcher-adoption') for extension in ('.xml','.log')]
    pins={}
    for name in names:
        value=metadata['run/'+name];raw=(directory/'run'/name).read_bytes()
        assert len(raw)==value['bytes'] and sha(raw)==value['sha256'];pins[name]=value['sha256']
    assert pins['cpu-certificate.json']=='14199ca0444d614385c5bda4863fd449e063a34540085513c3985080e02b5880'
    return pins

def retarget_history(text):
    tree=ast.parse(text);lines=text.splitlines(keepends=True)
    for node in sorted((node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('verify_failed_old_cpu','verify_failed_cpu02','verify_cpu03_history')),key=lambda node:node.lineno,reverse=True):
        lines[node.lineno-1:node.end_lineno]=[re.sub(r'\bROOT\b','HISTORY_ROOT',''.join(lines[node.lineno-1:node.end_lineno]))]
    return ''.join(lines)

def emit():
    files=previous();text=files['scripts/dbmol_storage_ops.py'].decode()
    text=once(text,"ROOT=BASE/'boltz-dbmol-pilot-01'","ROOT=BASE/'boltz-dbmol-pilot-02'\nHISTORY_ROOT=BASE/'boltz-dbmol-pilot-01'")
    text=once(text,"SOURCE=BASE/'boltz-dbmol-pilot-release-01/source'","SOURCE=BASE/'boltz-dbmol-pilot-release-02/source'")
    text=once(text,"BASELINE=BASE/'boltz-lr0025-02'","BASELINE=BASE/'boltz-lr0025-03'")
    text=text.replace("'boltz-lr0025-release-02'","'boltz-lr0025-release-03'")
    text=once(text,"LAUNCHER=BASE/'boltz-dbmol-storage-release-04/source'","LAUNCHER=BASE/'boltz-dbmol-storage-release-05/source'")
    text=once(text,"SCIENCE_MANIFEST='"+OLD_SCIENCE+"'","SCIENCE_MANIFEST='"+SCIENCE+"'")
    text=once(text,"CPU_PARENT='cpu-04'","CPU_PARENT='cpu-05'")
    text=once(text,"AFTER_CPU_PARENT='after-cpu-04'","AFTER_CPU_PARENT='after-cpu-05'")
    text=retarget_history(text)
    text=once(text,'+verify_failed_cpu02()+verify_cpu03_history()',
                      '+verify_failed_cpu02()+verify_cpu03_history()+verify_predecessor_history()')
    text=once(text,"def submit(name,phase,manifest,dependency=None):\n    p=ROOT/name;",
        "def submit(name,phase,manifest,dependency=None):\n    startup=None\n    if phase in ['native','segment']:\n        _,index=decode(name);startup=verify_startup_entry(phase,index,manifest)\n    p=ROOT/name;")
    text=once(text,"exclusive(p/'submission-started.json',r)","if startup is not None:r['startup_smoke']=startup\n    exclusive(p/'submission-started.json',r)")
    text=once(text,"def queue_native(index,manifest):\n    r=submit", "def queue_native(index,manifest):\n    ensure_startup('native',index,manifest)\n    r=submit")
    text=once(text,"('launcher-adoption',0)]:","('launcher-adoption',0),('startup',0)]:")
    text=once(text,"        counts[group]=len(cases);proofs +=",
        "        if group=='startup':require(len(cases)==STARTUP_CASES and {'test_premodel_startup'}<={part for c in cases for part in c.get('classname','').split('.')},'Missing exact allocated startup contracts')\n        counts[group]=len(cases);proofs +=")
    text=once(text,"        r=submit(target.name,'segment',manifest);", "        ensure_startup('segment',index,manifest)\n        r=submit(target.name,'segment',manifest);")
    text=once(text,"phase,index=decode(parent.name);produce(parent,phase,index,a.manifest);return",
        "phase,index=decode(parent.name);pin=verify_startup_entry(phase,index,a.manifest)\n        require(read(parent/'submission.json').get('startup_smoke')==pin,'Actual GPU entry startup proof differs')\n        produce(parent,phase,index,a.manifest);return")
    # A known failed peer cannot satisfy done(EXIT); retain the full projection
    # blocker and its exact metadata instead of creating a doomed watcher.
    text=once(text,"            dependencies.append('done('+r['job_id']+')')",
        "            if pro and peer_failed_projection(root,name,r,peer_manifest):return\n            dependencies.append('done('+r['job_id']+')')")
    text=once(text,"if __name__=='__main__':main()",'')
    text+='\nSTARTUP_CASES='+str(STARTUP_CASES)+'\nPREDECESSOR_PINS='+repr(predecessor_pins())+'\n'+SUPPORT+"\nif __name__=='__main__':main()\n"
    ast.parse(text);(HERE/'dbmol_storage_ops.py').write_text(text,encoding='utf8',newline='\n')
    cpu=files['scripts/hpc/dbmol_storage_cpu.lsf'].decode().replace('$root/cpu-04/payload','$root/cpu-05/payload').replace(OLD_SCIENCE,SCIENCE)
    cpu+='\n"$python" -m pytest "$launcher/tests/test_premodel_startup.py" --rootdir "$launcher" --import-mode=importlib -o addopts= -o cache_dir="$run/cache-startup" --basetemp "$TMPDIR/startup-contracts" --junitxml="$run/startup.xml" > "$run/startup.log" 2>&1\n'
    (HERE/'dbmol_storage_cpu.lsf').write_text(cpu,encoding='utf8',newline='\n')
    (HERE/'dbmol_storage_stage.lsf').write_text(files['scripts/hpc/dbmol_storage_stage.lsf'].decode().replace(OLD_SCIENCE,SCIENCE),encoding='utf8',newline='\n')
    for name in ('gradient_contract','pilot_contract','bridge'):
        (HERE/('fixture_'+name+'.py')).write_bytes(files['tests/test_dbmol_'+name+'.py'])

SUPPORT='''
def verify_predecessor_history():
    for name,digest in PREDECESSOR_PINS.items():require(sha(HISTORY_ROOT/name)==digest,'Historical CPU04/native failure proof changed')
    certificate=read(HISTORY_ROOT/'cpu-certificate.json');verify_proofs(certificate)
    require(certificate['cases']==1135 and certificate['source_manifest_sha256']=='24c207da5510fa579576a04ad32bf1afc6e1f23f51dcb5251d82a4ebdfa8f436'
        and certificate['launcher_manifest_sha256']=='df89def61047469e0db8ee6f4b235c75344dd4ad9f35a0833b0839f23859e9fa','Historical CPU04 pass scope differs')
    for parent,job,status in [('cpu-04','29802621','Done'),('after-cpu-04','29802622','Done'),('pilot-native-0000','29802648','Exited'),('pilot-after-native-0000','29802649','Exited')]:
        submission=read(HISTORY_ROOT/parent/'submission.json');text=(HISTORY_ROOT/parent/('stdout.'+job+'.log')).read_text()
        require(submission['job_id']==job and submission['outcome']=='submitted'
            and re.findall(r'^Subject:\\s*Job\\s+(\\d+):[^\\r\\n]*\\b(Done|Exited)\\s*$',text,re.M)==[(job,status)],'Historical CPU04/native same-job outcome differs')
        require(('Successfully completed.' in text and 'Exited with exit' not in text) if status=='Done' else 'Exited with exit code 1' in text,'Historical actual LSF terminal footer differs')
    for group,count in [('hpc-contract',500),('legacy',239),('native-mapping',34),('chemistry',59),('dbmol-native',83),('dbmol-transport',148),('launcher-adoption',72)]:
        path=HISTORY_ROOT/'cpu-04/payload'/(group+'.xml');tree=ET.parse(path)
        require(len(list(tree.iter('testcase')))==count and not any(list(tree.iter(k)) for k in ['failure','error','skipped']),'Historical CPU04 passing cases relabeled')
    text=(HISTORY_ROOT/'pilot-native-0000/stderr.29802648.log').read_text()
    require("KeyError: 'git_revision_reference'" in text and 'source_identity' in text,'Historical pre-model metadata failure differs')
    study=HISTORY_ROOT/'pilot-native-0000/payload/study'
    require(not list(study.rglob('*')),'Historical pre-model failure unexpectedly has scientific output')
    return [{'path':str(HISTORY_ROOT/name),'sha256':digest} for name,digest in PREDECESSOR_PINS.items()]

def peer_failed_projection(root,name,submission,peer_manifest):
    path=root/name/('stdout.'+submission['job_id']+'.log')
    if not path.is_file():return False
    text=path.read_text();rows=re.findall(r'^Subject:\\s*Job\\s+(\\d+):[^\\r\\n]*\\b(Done|Exited)\\s*$',text,re.M)
    if rows!=[(submission['job_id'],'Exited')]:return False
    require('Exited with exit code 1' in text,'Unclassified failed peer callback')
    record={'status':'awaiting_failed_peer_native_projection','peer_root':str(root),'callback':name,'job_id':submission['job_id'],
        'source_manifest_sha256':peer_manifest,'full_projection_established':False,'gate_waived':False,
        'proofs':[{'path':str(p),'sha256':sha(p)} for p in [path,root/name/'submission.json']]}
    target=ROOT/('awaiting-failed-peer-'+name+'.json')
    if target.exists():require(read(target)==record,'Preserved failed peer decision changed')
    else:exclusive(target,record)
    return True

def startup_path(phase,index):
    require(phase in ['native','segment'] and type(index) is int and index in range(4),'Undeclared startup scope')
    return ROOT/('startup-'+phase+'-'+format(index,'04d')+'.json')

def verify_startup_entry(phase,index,manifest):
    path=startup_path(phase,index);record=read(path);r=record['result']
    require(record['status']=='startup_only_passed' and record['source_manifest_sha256']==manifest==SCIENCE_MANIFEST and record['launcher_manifest_sha256']==LAUNCHER_MANIFEST
        and record['phase']==phase and record['index']==index,'Startup source/phase/index differs')
    require(r['version']=='mosaic-actual-generated-command-premodel-v1' and r['arm']=='boltz2_dbmol'
        and r['status']=='startup_only_passed' and r['model_loaded'] is False and r['scientific_readiness_passed'] is False
        and r['gradient_passed'] is False and r['optimizer_updates']==0 and r['saved_results_changed'] is False
        and r['phase']==phase and r['index']==index and r['source_manifest_sha256']==manifest
        and r['source']==str(SOURCE) and r['git_revision_reference']=='c74c659344123937608077e3e4078f75f09cd743','Actual startup-only result differs')
    require(r['summary']['status']=='failed' and r['summary']['error']=='PremodelBoundaryReached: Intentional startup-only boundary; no native model loaded'
        and r['summary']['source_manifest_sha256']==manifest and not r['summary'].get('native_gradient_checks') and not r['summary'].get('cases'),'Startup cannot certify native readiness or optimization')
    owner=record['owner'];submission=read(owner['path'])
    require(sha(owner['path'])==owner['sha256'] and submission['job_id']==owner['job_id']==r['job_id']
        and submission['outcome']=='submitted' and submission['phase'] in ['after-cpu','after-native','after-segment','dispatch'],'Actual startup callback owner differs')
    parent=Path(owner['path']).parent
    require(parent.parent==ROOT and submission['args']==command(parent.name,submission['phase'],manifest,submission['dependency']),'Actual startup owner resources/source differs')
    require(record['cpu_sha256']==sha(cpu_certificate(manifest)),'Startup CPU proof changed')
    expected_inputs={'config','cpu-certificate','reference','pocket','prepared_context'}
    if phase=='segment':expected_inputs.add('ready')
    if index:expected_inputs.add('resume-snapshot')
    require(set(r['inputs'])==expected_inputs,'Startup authenticated input scope differs')
    for value in r['inputs'].values():require(Path(value['path']).stat().st_size==value['bytes'] and sha(value['path'])==value['sha256'],'Startup original input bytes changed')
    args=r['args'];require(args[:2]==[PYTHON,str(SOURCE/'scripts/dbmol_driver.py')],'Startup driver/interpreter differs')
    def argument(name):
        require(args.count(name)==1,'Missing/duplicate startup argument');return args[args.index(name)+1]
    for flag,value in [('--manifest',manifest),('--mode','validate' if phase=='native' else 'run'),('--config',str(config())),
        ('--cpu-certificate',str(ROOT/'cpu-certificate.json')),('--cpu-certificate-sha256',record['cpu_sha256']),
        ('--offset',str(index*SEGMENT)),('--segment-steps',str(SEGMENT))]:require(argument(flag)==value,'Startup actual driver command differs')
    snapshot=input_snapshot(index)
    require(record['input_snapshot']==snapshot,'Startup exact predecessor differs')
    if phase=='segment':
        ready,_=readiness(index,manifest);require(record['ready']=={'path':str(ready),'sha256':sha(ready)} and argument('--ready')==str(ready) and argument('--ready-sha256')==sha(ready),'Startup actual readiness differs')
    else:require(record['ready'] is None and '--ready' not in args,'Validation cannot invent readiness')
    from dbmol_protocol import CONTEXT_FILE,POCKET
    expected=[PYTHON,str(SOURCE/'scripts/dbmol_driver.py'),'--mode','validate' if phase=='native' else 'run',
        '--config',str(config()),'--output',argument('--output'),'--manifest',manifest,
        '--cpu-certificate',str(ROOT/'cpu-certificate.json'),'--cpu-certificate-sha256',record['cpu_sha256'],
        '--reference',str(ROOT/CPU_PARENT/'payload/composition_reference.json'),
        '--pocket',str(ROOT/CPU_PARENT/'payload/pocket_definition.json'),'--pocket-sha256',POCKET,
        '--prepared',PREPARED,'--context-sha256',CONTEXT_FILE,'--worker-python',WORKER,
        '--offset',str(index*SEGMENT),'--segment-steps',str(SEGMENT)]
    if snapshot:expected+=['--resume-snapshot',snapshot['path'],'--resume-sha256',snapshot['sha256']]
    if phase=='segment':expected+=['--ready',str(ready),'--ready-sha256',sha(ready)]
    require(args==expected,'Complete actual startup driver arguments differ from production')
    expected_paths={'config':str(config()),'cpu-certificate':str(ROOT/'cpu-certificate.json'),
        'reference':str(ROOT/CPU_PARENT/'payload/composition_reference.json'),
        'pocket':str(ROOT/CPU_PARENT/'payload/pocket_definition.json'),'prepared_context':str(Path(PREPARED)/'context.pt')}
    if phase=='segment':expected_paths['ready']=str(ready)
    if snapshot:expected_paths['resume-snapshot']=snapshot['path']
    require({name:value['path'] for name,value in r['inputs'].items()}==expected_paths,'Exact startup input paths differ')
    return {'path':str(path),'sha256':sha(path)}

def ensure_startup(phase,index,manifest):
    path=startup_path(phase,index)
    if path.exists():return verify_startup_entry(phase,index,manifest)
    marker=path.with_name(path.stem+'-started.json')
    require(not marker.exists(),'Unknown startup outcome; no blind replay')
    owner=[]
    for parent in ROOT.iterdir():
        candidate=parent/'submission.json'
        if parent.is_dir() and candidate.is_file() and read(candidate).get('job_id')==os.environ.get('LSB_JOBID'):owner.append(candidate)
    require(len(owner)==1,'Actual allocated startup owner missing/ambiguous')
    exclusive(marker,{'phase':phase,'index':index,'owner':str(owner[0]),'outcome':'unknown'})
    from premodel_startup import run_startup_smoke
    result=run_startup_smoke(sys.modules[__name__],manifest,Path(os.environ['TMPDIR'])/('dbmol-startup-'+phase+'-'+format(index,'04d')),phase=phase,index=index)
    ready=readiness(index,manifest)[0] if phase=='segment' else None
    record={'status':'startup_only_passed','source_manifest_sha256':manifest,'launcher_manifest_sha256':LAUNCHER_MANIFEST,
        'phase':phase,'index':index,'cpu_sha256':sha(ROOT/'cpu-certificate.json'),'input_snapshot':input_snapshot(index),
        'ready':None if ready is None else {'path':str(ready),'sha256':sha(ready)},
        'owner':{'path':str(owner[0]),'sha256':sha(owner[0]),'job_id':os.environ['LSB_JOBID']},'result':result}
    exclusive(path,record);return verify_startup_entry(phase,index,manifest)
'''

def seal():
    names={'dbmol_storage_ops.py':'scripts/dbmol_storage_ops.py','dbmol_storage_cpu.lsf':'scripts/hpc/dbmol_storage_cpu.lsf',
        'dbmol_storage_stage.lsf':'scripts/hpc/dbmol_storage_stage.lsf','fixture_storage_adoption.py':'tests/fixture_storage_adoption.py',
        'fixture_gradient_contract.py':'tests/test_dbmol_gradient_contract.py','fixture_pilot_contract.py':'tests/test_dbmol_pilot_contract.py',
        'fixture_bridge.py':'tests/test_dbmol_bridge.py','README.md':'README.md',
        str(REPO/'scripts/hpc/startup/premodel_startup.py'):'scripts/premodel_startup.py',
        str(REPO/'tests/core/fixture_premodel_startup.py'):'tests/test_premodel_startup.py'}
    files={target:((HERE/name) if not Path(name).is_absolute() else Path(name)).read_bytes().replace(b'\r\n',b'\n') for name,target in names.items()}
    for name,data in files.items():
        if name.endswith('.py'):ast.parse(data,filename=name)
    manifest={'version':'mosaic-dbmol-archived-revision-launcher-v1','scientific_source_manifest_sha256':SCIENCE,
        'previous_scientific_source_manifest_sha256':OLD_SCIENCE,'previous_launcher_manifest_sha256':PREVIOUS,
        'cpu_cases':1135+STARTUP_CASES,'history_root':'/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/boltz-dbmol-pilot-01',
        'files':{name:sha(data) for name,data in sorted(files.items())}}
    raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    output=REPO/'results/hpc'/HERE.name/'source01';output.mkdir(parents=True,exist_ok=False)
    with tarfile.open(output/'source.tar.gz','w:gz') as tar:
        for name,data in {**files,'source_manifest.json':raw}.items():
            member=tarfile.TarInfo('source/'+name);member.mode=0o444;member.size=len(data);tar.addfile(member,io.BytesIO(data))
    (output/'source_manifest.json').write_bytes(raw)
    package={'tag':HERE.name,'files':len(files),'manifest_sha256':sha(raw),'archive_bytes':(output/'source.tar.gz').stat().st_size,
        'archive_sha256':sha((output/'source.tar.gz').read_bytes()),'scientific_source_manifest_sha256':SCIENCE,
        'previous_scientific_source_manifest_sha256':OLD_SCIENCE,'cpu_cases':1135+STARTUP_CASES}
    (HERE/'package.json').write_text(json.dumps(package,indent=2)+'\n',encoding='utf8');print(json.dumps(package,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--emit-only',action='store_true');args=parser.parse_args()
    emit()
    if not args.emit_only:seal()
