"""Build a separate attribution-only launcher from authenticated launcher03."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
SCIENCE='24c207da5510fa579576a04ad32bf1afc6e1f23f51dcb5251d82a4ebdfa8f436'
PREVIOUS='7b67e6221e4d227b6e501478802ab4ebc8536d74272ca3a7d305bb6f15e7ffee'

def sha(data):return hashlib.sha256(data).hexdigest()

def replace_once(text,old,new):
    assert text.count(old)==1,repr(old)
    return text.replace(old,new)

def previous():
    archive=REPO/'results/hpc/20261023-boltz-dbmol-storage-03/source01/source.tar.gz'
    assert archive.stat().st_size==31721 and sha(archive.read_bytes())=='e8d60f4a03f0d6dcd05b9dbe66add8885fa7253d95537b08b3e170ed7ca763ca'
    files={}
    with tarfile.open(archive) as tar:
        for member in tar:
            assert member.isfile() and member.name.startswith('source/') and '..' not in Path(member.name).parts
            name=member.name.removeprefix('source/');assert name not in files
            files[name]=tar.extractfile(member).read()
    raw=files.pop('source_manifest.json');assert sha(raw)==PREVIOUS
    manifest=json.loads(raw);assert len(files)==7 and set(files)==set(manifest['files'])
    for name,data in files.items():assert sha(data)==manifest['files'][name]
    return files

def original_bridge():
    archive=REPO/'results/hpc/20261023-boltz-dbmol-pilot-01/source01/source.tar.gz'
    assert archive.stat().st_size==39676321 and sha(archive.read_bytes())=='ec333fd3e0b33d9e5b82e0cd5e4e12c14ffb4f91782c18cd820d4bfab53de1f8'
    with tarfile.open(archive) as tar:
        raw=tar.extractfile('source/source_manifest.json').read();assert sha(raw)==SCIENCE
        manifest=json.loads(raw);data=tar.extractfile('source/tests/test_dbmol_bridge.py').read()
    assert sha(data)==manifest['files']['tests/test_dbmol_bridge.py']
    return data

def history03():
    folder=REPO/'results/hpc/20261023-boltz-dbmol-storage-03/cpu-evidence-01'
    metadata=json.loads((folder/'metadata.json').read_bytes())['files']
    names=['cpu-03/submission.json','after-cpu-03/submission.json',
           'cpu-03/stdout.29798769.log','after-cpu-03/stdout.29798770.log',
           'after-cpu-03/stderr.29798770.log','cpu-03/storage-cpu-entry/gate.json']
    names += ['cpu-03/payload/'+group+extension for group in ('hpc-contract','legacy','native-mapping','chemistry','dbmol-native','dbmol-transport','launcher-adoption') for extension in ('.xml','.log')]
    pins={}
    for name in names:
        record=metadata['run/'+name];data=(folder/'run'/name).read_bytes()
        assert len(data)==record['bytes'] and sha(data)==record['sha256'];pins[name]=record['sha256']
    for group,count in [('hpc-contract',500),('legacy',239),('native-mapping',34),('chemistry',59),('dbmol-native',83),('dbmol-transport',148),('launcher-adoption',72)]:
        tree=ET.parse(folder/'run/cpu-03/payload'/(group+'.xml'))
        assert len(list(tree.iter('testcase')))==count and not any(list(tree.iter(k)) for k in ('failure','error','skipped'))
    assert sha((folder/'launcher/source_manifest.json').read_bytes())==PREVIOUS
    return pins

def emit():
    files=previous();pins=history03();text=files['scripts/dbmol_storage_ops.py'].decode()
    text=replace_once(text,"LAUNCHER=BASE/'boltz-dbmol-storage-release-03/source'","LAUNCHER=BASE/'boltz-dbmol-storage-release-04/source'")
    text=replace_once(text,"CPU_PARENT='cpu-03'","CPU_PARENT='cpu-04'")
    text=replace_once(text,"AFTER_CPU_PARENT='after-cpu-03'","AFTER_CPU_PARENT='after-cpu-04'")
    text=replace_once(text,"return [{'path':str(ROOT/name),'sha256':digest} for name,digest in pins.items()]+verify_failed_cpu02()",
                      "return [{'path':str(ROOT/name),'sha256':digest} for name,digest in pins.items()]+verify_failed_cpu02()+verify_cpu03_history()")
    text=replace_once(text,"if __name__=='__main__':main()",'')
    text += '\nCPU03_HISTORY_PINS='+repr(pins)+'\nCPU03_SOURCE=BASE/\'boltz-dbmol-storage-release-03/source\'\nCPU03_MANIFEST='+repr(PREVIOUS)+'\n'+SUPPORT+'\nif __name__==\'__main__\':main()\n'
    ast.parse(text);(HERE/'dbmol_storage_ops.py').write_text(text,encoding='utf8',newline='\n')
    cpu=files['scripts/hpc/dbmol_storage_cpu.lsf'].decode().replace('$root/cpu-03/payload','$root/cpu-04/payload')
    cpu=replace_once(cpu,'"$source/tests/test_dbmol_bridge.py"','"$launcher/tests/test_dbmol_bridge.py"')
    (HERE/'dbmol_storage_cpu.lsf').write_text(cpu,encoding='utf8',newline='\n')
    (HERE/'dbmol_storage_stage.lsf').write_bytes(files['scripts/hpc/dbmol_storage_stage.lsf'])
    (HERE/'fixture_gradient_contract.py').write_bytes(files['tests/test_dbmol_gradient_contract.py'])
    (HERE/'fixture_pilot_contract.py').write_bytes(files['tests/test_dbmol_pilot_contract.py'])
    (HERE/'fixture_bridge.py').write_bytes(original_bridge())

SUPPORT='''
def verify_cpu03_history():
    manifest=CPU03_SOURCE/'source_manifest.json'
    require(sha(manifest)==CPU03_MANIFEST,'Historical launcher03 source manifest changed')
    for name,digest in CPU03_HISTORY_PINS.items():
        require(sha(ROOT/name)==digest,'Historical CPU03 or callback proof changed')
    require(read(ROOT/'cpu-03/storage-cpu-entry/gate.json')['status']=='passed','Historical CPU03 capacity pass changed')
    for name,job,status in [('cpu-03','29798769','Done'),('after-cpu-03','29798770','Exited')]:
        submission=read(ROOT/name/'submission.json');text=(ROOT/name/('stdout.'+job+'.log')).read_text()
        require(submission['job_id']==job and submission['outcome']=='submitted'
            and re.findall(r'^Subject:\\s*Job\\s+(\\d+):[^\\r\\n]*\\b(Done|Exited)\\s*$',text,re.M)==[(job,status)],'Historical CPU03/callback same-job status differs')
        if status=='Done':require('Successfully completed.' in text and 'Exited with exit' not in text,'Historical CPU03 success missing')
        else:require('Exited with exit code 1' in text,'Historical callback03 failure missing')
    for group,count in [('hpc-contract',500),('legacy',239),('native-mapping',34),('chemistry',59),('dbmol-native',83),('dbmol-transport',148),('launcher-adoption',72)]:
        path=ROOT/'cpu-03/payload'/(group+'.xml');tree=ET.parse(path);cases=list(tree.iter('testcase'))
        require(len(cases)==count and not any(list(tree.iter(k)) for k in ['failure','error','skipped']),'Historical CPU03 passing cases changed or relabeled')
        require(re.search(r'\\b'+str(count)+r'\\s+passed\\b',path.with_suffix('.log').read_text()) is not None,'Historical CPU03 XML/log count differs')
    error=(ROOT/'after-cpu-03/stderr.29798770.log').read_text()
    require('Missing transport/scope fixture attribution' in error,'Historical callback03 cause changed')
    return [{'path':str(ROOT/name),'sha256':digest} for name,digest in CPU03_HISTORY_PINS.items()]+[{'path':str(manifest),'sha256':CPU03_MANIFEST}]
'''

def seal():
    names={'dbmol_storage_ops.py':'scripts/dbmol_storage_ops.py',
           'dbmol_storage_cpu.lsf':'scripts/hpc/dbmol_storage_cpu.lsf',
           'dbmol_storage_stage.lsf':'scripts/hpc/dbmol_storage_stage.lsf',
           'fixture_storage_adoption.py':'tests/fixture_storage_adoption.py',
           'fixture_gradient_contract.py':'tests/test_dbmol_gradient_contract.py',
           'fixture_pilot_contract.py':'tests/test_dbmol_pilot_contract.py',
           'fixture_bridge.py':'tests/test_dbmol_bridge.py','README.md':'README.md'}
    files={target:(HERE/name).read_bytes().replace(b'\r\n',b'\n') for name,target in names.items()}
    for name,data in files.items():
        if name.endswith('.py'):ast.parse(data,filename=name)
    manifest={'version':'mosaic-dbmol-transport-attribution-adoption-v1','scientific_source_manifest_sha256':SCIENCE,
              'previous_launcher_manifest_sha256':PREVIOUS,'previous_cpu_status':'passed','previous_callback_status':'failed',
              'files':{name:sha(data) for name,data in sorted(files.items())}}
    raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    output=REPO/'results/hpc'/HERE.name/'source01';output.mkdir(parents=True,exist_ok=False)
    with tarfile.open(output/'source.tar.gz','w:gz') as tar:
        for name,data in {**files,'source_manifest.json':raw}.items():
            member=tarfile.TarInfo('source/'+name);member.mode=0o444;member.size=len(data);tar.addfile(member,io.BytesIO(data))
    package={'tag':HERE.name,'files':len(files),'manifest_sha256':sha(raw),
             'archive_bytes':(output/'source.tar.gz').stat().st_size,'archive_sha256':sha((output/'source.tar.gz').read_bytes()),
             'scientific_source_manifest_sha256':SCIENCE,'scientific_source_changed':False,'previous_launcher_manifest_sha256':PREVIOUS,
             'previous_cpu_status':'passed','previous_callback_status':'failed'}
    (output/'source_manifest.json').write_bytes(raw)
    (HERE/'package.json').write_text(json.dumps(package,indent=2)+'\n',encoding='utf8');print(json.dumps(package,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--emit-only',action='store_true');args=parser.parse_args()
    emit()
    if not args.emit_only:seal()
