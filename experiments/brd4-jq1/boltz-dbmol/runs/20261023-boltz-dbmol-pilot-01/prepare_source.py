"""Local standard-library byte/AST packaging; no numerical imports or tests."""
import ast,hashlib,io,json,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]

def digest(data):return hashlib.sha256(data).hexdigest()

def main():
    archive=REPO/'results/hpc/20261023-protenix-extended-01/source02/source.tar.gz'
    assert digest(archive.read_bytes())=='8ba1c3370fee1f09b96af139dbf1403bb3c187c52d447ec4b0b23ebe4a65f63d'
    files={}
    with tarfile.open(archive) as tar:
        for m in tar:
            assert m.isfile() and m.name.startswith('source/')
            n=m.name.removeprefix('source/');assert n not in files and '..' not in Path(n).parts
            files[n]=tar.extractfile(m).read()
    raw=files.pop('source_manifest.json');parent=json.loads(raw)
    assert digest(raw)=='a30f980fd8fe95001fdfc6780d720bdde6486d2421d7e4942b119ace61777f00' and len(files)==641 and set(files)==set(parent['files'])
    for n,d in files.items():assert digest(d)==parent['files'][n]
    original=dict(files)
    trajectory='src/mosaic/small_molecule/chemistry/trajectory.py'
    text=files[trajectory].decode()
    assert text.count('if name not in ("contact", "affinity"):')==1
    assert text.count('if name == "affinity":')==1
    text=text.replace('if name not in ("contact", "affinity"):','if name not in ("contact", "affinity", "dbmol_total"):')
    text=text.replace('if name == "affinity":','if name in ("affinity", "dbmol_total"):')
    files[trajectory]=text.encode()
    aliases={
      HERE/'dbmol_protocol.py':'scripts/dbmol_protocol.py',HERE/'dbmol_driver.py':'scripts/dbmol_driver.py',
      HERE/'dbmol_ops.py':'scripts/dbmol_ops.py',HERE/'dbmol_cpu.lsf':'scripts/hpc/dbmol_cpu.lsf',
      HERE/'dbmol_stage.lsf':'scripts/hpc/dbmol_stage.lsf',HERE/'README.md':'docs/dbmol_pilot.md',
      HERE.parents[1]/'scripts/bridge.py':'scripts/dbmol_bridge.py',
      HERE.parents[1]/'scripts/worker.py':'scripts/dbmol_worker.py',
      HERE.parents[1]/'tests/native_validation.py':'scripts/dbmol_native_validation.py',
      REPO/'src/mosaic/small_molecule/boltz2/dbmol.py':'src/mosaic/small_molecule/boltz2/dbmol.py',
    }
    for n in ['objective','native_paths','gradient_contract','bridge','pilot_contract']:
        aliases[HERE.parents[1]/'tests'/('fixture_'+n+'.py')]='tests/test_dbmol_'+n+'.py'
    for p,n in aliases.items():files[n]=p.read_bytes().replace(b'\r\n',b'\n')
    c=json.loads(original['examples/configs/brd4_jq1_25_fixed_lrreset_boltz2_10000.json'])
    c.update(version='mosaic-brd4-jq1-dbmol-lr0025-10000-v1',initial_step_size=.0025,
        label='JQ125 / DBMol binding-only / fixedLR0.0025 / fresh10000',
        dbmol={'version':'mosaic-dbmol-pilot-native-components-v1','arithmetic':'contact_head_and_assembly_float64_v1',
            'contact_weight':2.5,'confidence_weight':2.5,'affinity':'mean(value)-mean(raw_binary_logit)',
            'segment_updates':2500,'chemistry_enabled':False})
    data=(json.dumps(c,indent=2)+'\n').encode();(HERE/'pilot.json').write_bytes(data);files['examples/configs/dbmol_pilot.json']=data
    for n,d in files.items():
        if n.endswith('.py'):ast.parse(d,filename=n)
    changed={n:digest(d) for n,d in files.items() if original.get(n)!=d}
    assert [n for n in original if n.startswith('src/') and original[n]!=files[n]]==[trajectory]
    record={'files':{n:digest(d) for n,d in sorted(files.items())},'parent_manifest_sha256':digest(raw),
            'scope':'Fresh independent DBMol binding-only pilot; graph/gradientFP32, contact-head/assemblyFP64, raw component+total FD mandatory; original releases/results unchanged'}
    raw=(json.dumps(record,sort_keys=True,indent=2)+'\n').encode()
    out=REPO/'results/hpc'/HERE.name/'source01';out.mkdir(parents=True,exist_ok=False)
    with tarfile.open(out/'source.tar.gz','w:gz') as tar:
        for n,d in {**files,'source_manifest.json':raw}.items():
            m=tarfile.TarInfo('source/'+n);m.mode=0o444;m.size=len(d);tar.addfile(m,io.BytesIO(d))
    package={'tag':HERE.name,'manifest_sha256':digest(raw),'source_files':len(files),
             'archive_bytes':(out/'source.tar.gz').stat().st_size,'archive_sha256':digest((out/'source.tar.gz').read_bytes()),'changed_members':changed}
    (out/'source_manifest.json').write_bytes(raw);(HERE/'package.json').write_text(json.dumps(package,indent=2)+'\n')
    (HERE/'overlay-src__mosaic__small_molecule__chemistry__trajectory.py').write_bytes(files[trajectory])
    print(json.dumps(package,indent=2))

if __name__=='__main__':main()
