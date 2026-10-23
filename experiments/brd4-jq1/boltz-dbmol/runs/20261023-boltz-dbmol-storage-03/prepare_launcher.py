"""Build a separate fixture-only launcher from authenticated launcher02 bytes."""
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
PREVIOUS='5f479f2e5342469efd8606c9457c946a51dbcf0dd90d837ee472babfdd820a4b'

def sha(data):return hashlib.sha256(data).hexdigest()

def replace_once(text,old,new):
    assert text.count(old)==1,repr(old)
    return text.replace(old,new)

def previous():
    archive=REPO/'results/hpc/20261023-boltz-dbmol-storage-02/source01/source.tar.gz'
    assert archive.stat().st_size==20406 and sha(archive.read_bytes())=='a26dafcac4fbc342a275dedb1c93ed8880062f9e3dafe48d188937c3137dd750'
    files={}
    with tarfile.open(archive) as tar:
        for m in tar:
            assert m.isfile() and m.name.startswith('source/') and '..' not in Path(m.name).parts
            name=m.name.removeprefix('source/');assert name not in files
            files[name]=tar.extractfile(m).read()
    raw=files.pop('source_manifest.json');assert sha(raw)==PREVIOUS
    manifest=json.loads(raw);assert len(files)==5 and set(files)==set(manifest['files'])
    for name,data in files.items():assert sha(data)==manifest['files'][name]
    return files

def failed02():
    folder=REPO/'results/hpc/20261023-boltz-dbmol-storage-02/cpu-evidence-03'
    metadata=json.loads((folder/'metadata.json').read_bytes())['files']
    names=['cpu-02/submission.json','after-cpu-02/submission.json',
           'cpu-02/stdout.29798699.log','after-cpu-02/stdout.29798700.log',
           'cpu-02/storage-cpu-entry/gate.json']
    names += ['cpu-02/payload/'+group+extension for group in ('hpc-contract','legacy','native-mapping','chemistry','dbmol-native','dbmol-transport') for extension in ('.xml','.log')]
    pins={}
    for name in names:
        record=metadata['run/'+name];data=(folder/'run'/name).read_bytes()
        assert len(data)==record['bytes'] and sha(data)==record['sha256']
        pins[name]=record['sha256']
    tree=ET.parse(folder/'run/cpu-02/payload/dbmol-transport.xml')
    cases=list(tree.iter('testcase'));assert len(cases)==148 and len(list(tree.iter('failure')))==7
    failures=sorted(case.attrib['name'] for case in cases if case.find('failure') is not None)
    return pins,failures

def emit():
    files=previous();pins,failures=failed02()
    text=files['scripts/dbmol_storage_ops.py'].decode()
    text=replace_once(text,"LAUNCHER=BASE/'boltz-dbmol-storage-release-02/source'","LAUNCHER=BASE/'boltz-dbmol-storage-release-03/source'")
    text=replace_once(text,"CPU_PARENT='cpu-02'","CPU_PARENT='cpu-03'")
    text=replace_once(text,"AFTER_CPU_PARENT='after-cpu-02'","AFTER_CPU_PARENT='after-cpu-03'")
    text=replace_once(text,"return [{'path':str(ROOT/name),'sha256':digest} for name,digest in pins.items()]",
                      "return [{'path':str(ROOT/name),'sha256':digest} for name,digest in pins.items()]+verify_failed_cpu02()")
    text=replace_once(text,"if __name__=='__main__':main()",'')
    text += '\nFAILED_CPU02_PINS='+repr(pins)+'\nFAILED_CPU02_NAMES='+repr(failures)+'\n'+SUPPORT+'\nif __name__==\'__main__\':main()\n'
    ast.parse(text)
    (HERE/'dbmol_storage_ops.py').write_text(text,encoding='utf8',newline='\n')
    cpu=files['scripts/hpc/dbmol_storage_cpu.lsf'].decode().replace('$root/cpu-02/payload','$root/cpu-03/payload')
    cpu=replace_once(cpu,'"$python" -m pytest tests/test_dbmol_gradient_contract.py tests/test_dbmol_bridge.py tests/test_dbmol_pilot_contract.py --rootdir "$source"',
                     '"$python" -m pytest "$launcher/tests/test_dbmol_gradient_contract.py" "$source/tests/test_dbmol_bridge.py" "$launcher/tests/test_dbmol_pilot_contract.py" --rootdir "$launcher"')
    (HERE/'dbmol_storage_cpu.lsf').write_text(cpu,encoding='utf8',newline='\n')
    (HERE/'dbmol_storage_stage.lsf').write_bytes(files['scripts/hpc/dbmol_storage_stage.lsf'])

SUPPORT='''
def verify_failed_cpu02():
    for name,digest in FAILED_CPU02_PINS.items():
        require(sha(ROOT/name)==digest,'Historical failed CPU02 evidence changed')
    require(read(ROOT/'cpu-02/storage-cpu-entry/gate.json')['status']=='passed','Historical CPU02 capacity pass changed')
    for name,job in [('cpu-02','29798699'),('after-cpu-02','29798700')]:
        submission=read(ROOT/name/'submission.json');text=(ROOT/name/('stdout.'+job+'.log')).read_text()
        require(submission['job_id']==job and submission['outcome']=='submitted'
            and re.findall(r'^Subject:\\s*Job\\s+(\\d+):[^\\r\\n]*\\b(Done|Exited)\\s*$',text,re.M)==[(job,'Exited')]
            and 'Exited with exit code 1' in text,'Original CPU02/callback failure differs')
    for group,count in [('hpc-contract',500),('legacy',239),('native-mapping',34),('chemistry',59),('dbmol-native',83),('dbmol-transport',148)]:
        path=ROOT/'cpu-02/payload'/(group+'.xml');tree=ET.parse(path);cases=list(tree.iter('testcase'))
        require(len(cases)==count and not any(list(tree.iter(k)) for k in ['error','skipped']),'Historical CPU02 case inventory differs')
        names=sorted(case.attrib['name'] for case in cases if case.find('failure') is not None)
        require(names==(FAILED_CPU02_NAMES if group=='dbmol-transport' else []),'Historical seven fixture failures changed or relabeled')
    text=(ROOT/'cpu-02/payload/dbmol-transport.log').read_text()
    require(re.search(r'\\b7 failed, 141 passed\\b',text) is not None,'Historical CPU02 transport failure log differs')
    require(not (ROOT/'cpu-02/payload/launcher-adoption.xml').exists(),'Historical unexecuted launcher group fabricated')
    return [{'path':str(ROOT/name),'sha256':digest} for name,digest in FAILED_CPU02_PINS.items()]
'''

def seal():
    names={'dbmol_storage_ops.py':'scripts/dbmol_storage_ops.py',
           'dbmol_storage_cpu.lsf':'scripts/hpc/dbmol_storage_cpu.lsf',
           'dbmol_storage_stage.lsf':'scripts/hpc/dbmol_storage_stage.lsf',
           'fixture_storage_adoption.py':'tests/fixture_storage_adoption.py',
           'fixture_gradient_contract.py':'tests/test_dbmol_gradient_contract.py',
           'fixture_pilot_contract.py':'tests/test_dbmol_pilot_contract.py',
           'README.md':'README.md'}
    files={target:(HERE/name).read_bytes().replace(b'\r\n',b'\n') for name,target in names.items()}
    for name,data in files.items():
        if name.endswith('.py'):ast.parse(data,filename=name)
    manifest={'version':'mosaic-dbmol-fixture-adoption-v1','scientific_source_manifest_sha256':SCIENCE,
              'previous_failed_launcher_manifest_sha256':PREVIOUS,
              'files':{name:sha(data) for name,data in sorted(files.items())}}
    raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    output=REPO/'results/hpc'/HERE.name/'source01';output.mkdir(parents=True,exist_ok=False)
    with tarfile.open(output/'source.tar.gz','w:gz') as tar:
        for name,data in {**files,'source_manifest.json':raw}.items():
            member=tarfile.TarInfo('source/'+name);member.mode=0o444;member.size=len(data);tar.addfile(member,io.BytesIO(data))
    package={'tag':HERE.name,'files':len(files),'manifest_sha256':sha(raw),
             'archive_bytes':(output/'source.tar.gz').stat().st_size,'archive_sha256':sha((output/'source.tar.gz').read_bytes()),
             'scientific_source_manifest_sha256':SCIENCE,'scientific_source_changed':False,'previous_failed_launcher_manifest_sha256':PREVIOUS}
    (output/'source_manifest.json').write_bytes(raw)
    (HERE/'package.json').write_text(json.dumps(package,indent=2)+'\n',encoding='utf8')
    print(json.dumps(package,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--emit-only',action='store_true');args=parser.parse_args()
    emit()
    if not args.emit_only:seal()
