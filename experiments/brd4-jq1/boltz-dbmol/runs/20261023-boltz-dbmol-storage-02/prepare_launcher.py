"""Emit the launcher fork and seal small bytes only; no scientific imports."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
SCIENCE_MANIFEST='24c207da5510fa579576a04ad32bf1afc6e1f23f51dcb5251d82a4ebdfa8f436'

def digest(data):return hashlib.sha256(data).hexdigest()

def source_members():
    archive=REPO/'results/hpc/20261023-boltz-dbmol-pilot-01/source01/source.tar.gz'
    assert archive.stat().st_size==39676321 and digest(archive.read_bytes())=='ec333fd3e0b33d9e5b82e0cd5e4e12c14ffb4f91782c18cd820d4bfab53de1f8'
    files={}
    with tarfile.open(archive) as tar:
        for m in tar:
            assert m.isfile() and m.name.startswith('source/') and '..' not in Path(m.name).parts
            n=m.name.removeprefix('source/');assert n not in files
            files[n]=tar.extractfile(m).read()
    raw=files.pop('source_manifest.json');assert digest(raw)==SCIENCE_MANIFEST
    manifest=json.loads(raw);assert len(files)==657 and set(files)==set(manifest['files'])
    for n,d in files.items():assert digest(d)==manifest['files'][n]
    return files

def replace_once(text,old,new):
    assert text.count(old)==1,repr(old)
    return text.replace(old,new)

def emit():
    files=source_members();text=files['scripts/dbmol_ops.py'].decode()
    text=replace_once(text,"if __name__=='__main__':main()",'')
    text=replace_once(text,"if phase in ['cpu','after-cpu']:require(name==phase,'Wrong parent')",
                      "if phase in ['cpu','after-cpu']:require(name==(CPU_PARENT if phase=='cpu' else AFTER_CPU_PARENT),'Wrong adopted CPU parent')")
    text=replace_once(text,"def command(name,phase,manifest,dependency=None):",
                      "def command(name,phase,manifest,dependency=None):\n    require(manifest==SCIENCE_MANIFEST and LAUNCHER_MANIFEST is not None,'Missing exact dual source identities')")
    text=replace_once(text,"str(SOURCE/'scripts/hpc'/('dbmol_cpu.lsf' if phase=='cpu' else 'dbmol_stage.lsf')),str(SOURCE),str(ROOT),manifest",
                      "str(LAUNCHER/'scripts/hpc'/('dbmol_storage_cpu.lsf' if phase=='cpu' else 'dbmol_storage_stage.lsf')),str(LAUNCHER),LAUNCHER_MANIFEST,str(SOURCE),str(ROOT),manifest")
    text=text.replace("ROOT/'cpu'",'ROOT/CPU_PARENT').replace("ROOT/'after-cpu'",'ROOT/AFTER_CPU_PARENT')
    text=text.replace("ROOT/'cpu/payload", "ROOT/CPU_PARENT/'payload")
    text=replace_once(text,"require(c['suite']==SUITE and c['source_manifest_sha256']==manifest,'CPU scope differs');verify_proofs(c);return p",
                      "require(c['suite']==SUITE and c['source_manifest_sha256']==manifest==SCIENCE_MANIFEST and c['launcher_manifest_sha256']==LAUNCHER_MANIFEST and c['launcher_source']==str(LAUNCHER),'CPU dual-source scope differs');verify_launcher(manifest,LAUNCHER_MANIFEST);verify_proofs(c);return p")
    text=replace_once(text,"require(all(original[a]<=OLD_FULL[a] for a in OLD_FULL),'Original projection exceeded')",
                      "original_future,terminal_evidence=original_remaining(original)")
    text=replace_once(text,"sum(OLD_FULL[a]-original[a] for a in OLD_FULL)","sum(original_future.values())")
    text=replace_once(text,"original_current=original,own_current=own",
                      "original_current=original,original_future=original_future,terminal_original_evidence=terminal_evidence,launcher_manifest_sha256=LAUNCHER_MANIFEST,science_manifest_sha256=SCIENCE_MANIFEST,own_current=own")
    text=replace_once(text,"proofs=final_job(ROOT/CPU_PARENT,'hpc');counts={}",
                      "proofs=final_job(ROOT/CPU_PARENT,'hpc')+verify_launcher(manifest,LAUNCHER_MANIFEST)+verify_failed_old_cpu();counts={}")
    text=replace_once(text,"('dbmol-transport',0)]","('dbmol-transport',0),('launcher-adoption',0)]")
    text=replace_once(text,"        counts[group]=len(cases);proofs +=",
                      "        if group=='launcher-adoption':require({'fixture_storage_adoption'}<={c.get('classname','').split('.')[0] for c in cases},'Missing allocated launcher fixture attribution')\n        counts[group]=len(cases);proofs +=")
    text=text.replace("{c.get('classname','').split('.')[0] for c in cases}",
                      "{part for c in cases for part in c.get('classname','').split('.')}")
    text=replace_once(text,"require([counts[k] for k in ['hpc-contract','legacy','native-mapping','chemistry']]==[500,239,34,59],'Inherited actual suite changed')",
                      "require([counts[k] for k in ['hpc-contract','legacy','native-mapping','chemistry','dbmol-native','dbmol-transport','launcher-adoption']]==[500,239,34,59,83,148,72],'Exact allocated scientific/launcher suite changed')")
    text=replace_once(text,"'suite':SUITE,'source_manifest_sha256':manifest,'cases':sum(counts.values())",
                      "'suite':SUITE,'source_manifest_sha256':manifest,'launcher_manifest_sha256':LAUNCHER_MANIFEST,'launcher_source':str(LAUNCHER),'cases':sum(counts.values())")
    text=replace_once(text,"'original_saved_results_read_only':True}",
                      "'original_saved_results_read_only':True,'launcher_source':str(LAUNCHER),'launcher_manifest_sha256':LAUNCHER_MANIFEST,'science_manifest_sha256':manifest}")
    text=replace_once(text,"p.add_argument('--phase',required=True);p.add_argument('--manifest',required=True);a=p.parse_args()",
                      "p.add_argument('--launcher',type=Path,required=True);p.add_argument('--launcher-manifest',required=True)\n    p.add_argument('--phase',required=True);p.add_argument('--manifest',required=True);a=p.parse_args()\n    require(a.launcher.resolve()==LAUNCHER,'Wrong launcher source')\n    verify_launcher(a.manifest,a.launcher_manifest);verify_failed_old_cpu()")
    text=text.rstrip()+'\n'+(HERE/'terminal_support.txt').read_text()+'\n\nif __name__==\'__main__\':main()\n'
    ast.parse(text)
    (HERE/'dbmol_storage_ops.py').write_text(text,encoding='utf8',newline='\n')
    # The old scientific CPU suite/driver/test bytes are referenced, never copied
    # into a scientific fork. A launcher-local seventh group is additive only.
    cpu=files['scripts/hpc/dbmol_cpu.lsf'].decode()
    cpu=replace_once(cpu,'source=$1\nroot=$2\nmanifest=$3',
                     'launcher=$1\nlauncher_manifest=$2\nsource=$3\nroot=$4\nmanifest=$5')
    cpu=replace_once(cpu,'"$python" "$source/scripts/verify_stage0_release.py" "$source" "$manifest"',AUTHENTICATE)
    cpu=replace_once(cpu,'PYTHONPATH="$source/scripts:$source/src:$source/tests"',
                     'PYTHONPATH="$launcher/scripts:$source/scripts:$source/src:$source/tests"')
    cpu=replace_once(cpu,'export CUDA_VISIBLE_DEVICES', 'unset PYTEST_ADDOPTS\nexport CUDA_VISIBLE_DEVICES')
    cpu=replace_once(cpu,'"$python" "$source/scripts/dbmol_ops.py" --phase cpu-entry --root "$root" --source "$source" --manifest "$manifest"',
                     '"$python" "$launcher/scripts/dbmol_storage_ops.py" --phase cpu-entry --root "$root" --source "$source" --manifest "$manifest" --launcher "$launcher" --launcher-manifest "$launcher_manifest"')
    cpu=cpu.replace('$root/cpu/payload','$root/cpu-02/payload')
    cpu+='\n"$python" -m pytest "$launcher/tests/fixture_storage_adoption.py" --rootdir "$launcher" --import-mode=importlib -o addopts= -o cache_dir="$run/cache-launcher" --basetemp "$TMPDIR/launcher-adoption" --junitxml="$run/launcher-adoption.xml" > "$run/launcher-adoption.log" 2>&1\n'
    (HERE/'dbmol_storage_cpu.lsf').write_text(cpu,encoding='utf8',newline='\n')
    stage=files['scripts/hpc/dbmol_stage.lsf'].decode()
    stage=replace_once(stage,'release=$1\nroot=$2\nmanifest=$3\nphase=$4\nparent=$5',
                       'launcher=$1\nlauncher_manifest=$2\nrelease=$3\nroot=$4\nmanifest=$5\nphase=$6\nparent=$7')
    stage=replace_once(stage,'"$python" "$release/scripts/verify_stage0_release.py" "$release" "$manifest"',AUTHENTICATE.replace('"$source"','"$release"'))
    stage=replace_once(stage,'PYTHONPATH="$release/scripts:$release/tests:$release/src"',
                       'PYTHONPATH="$launcher/scripts:$release/scripts:$release/tests:$release/src"')
    stage=replace_once(stage,'"$python" scripts/dbmol_ops.py --phase "$phase" --root "$root" --source "$release" --manifest "$manifest" --parent "$parent"',
                       '"$python" "$launcher/scripts/dbmol_storage_ops.py" --phase "$phase" --root "$root" --source "$release" --manifest "$manifest" --parent "$parent" --launcher "$launcher" --launcher-manifest "$launcher_manifest"')
    (HERE/'dbmol_storage_stage.lsf').write_text(stage,encoding='utf8',newline='\n')

AUTHENTICATE='''"$python" - "$launcher" "$launcher_manifest" "$source" "$manifest" <<'MOSAIC_AUTH'
import hashlib,json,sys
from pathlib import Path
expected_science='24c207da5510fa579576a04ad32bf1afc6e1f23f51dcb5251d82a4ebdfa8f436'
assert sys.argv[4]==expected_science
for root,expected in [(Path(sys.argv[1]),sys.argv[2]),(Path(sys.argv[3]),sys.argv[4])]:
 raw=(root/'source_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
 files=json.loads(raw)['files'];actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() or p.is_symlink()}
 assert actual==set(files)|{'source_manifest.json'}
 for p in [root,*root.rglob('*')]:assert not p.is_symlink() and (p.is_file() or p.is_dir()) and not(p.stat().st_mode&0o222)
 for name,digest in files.items():
  p=root/name;assert p.resolve().is_relative_to(root.resolve())
  with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
MOSAIC_AUTH'''

def seal():
    names={'dbmol_storage_ops.py':'scripts/dbmol_storage_ops.py',
           'dbmol_storage_cpu.lsf':'scripts/hpc/dbmol_storage_cpu.lsf',
           'dbmol_storage_stage.lsf':'scripts/hpc/dbmol_storage_stage.lsf',
           'fixture_storage_adoption.py':'tests/fixture_storage_adoption.py',
           'README.md':'README.md'}
    files={target:(HERE/name).read_bytes().replace(b'\r\n',b'\n') for name,target in names.items()}
    for name,data in files.items():
        if name.endswith('.py'):ast.parse(data,filename=name)
    manifest={'version':'mosaic-dbmol-completion-aware-storage-adoption-v1',
              'scientific_source_manifest_sha256':SCIENCE_MANIFEST,
              'files':{name:digest(data) for name,data in sorted(files.items())}}
    raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    output=REPO/'results/hpc'/HERE.name/'source01';output.mkdir(parents=True,exist_ok=False)
    with tarfile.open(output/'source.tar.gz','w:gz') as tar:
        for name,data in {**files,'source_manifest.json':raw}.items():
            member=tarfile.TarInfo('source/'+name);member.mode=0o444;member.size=len(data);tar.addfile(member,io.BytesIO(data))
    package={'tag':HERE.name,'files':len(files),'manifest_sha256':digest(raw),
             'archive_bytes':(output/'source.tar.gz').stat().st_size,
             'archive_sha256':digest((output/'source.tar.gz').read_bytes()),
             'scientific_source_manifest_sha256':SCIENCE_MANIFEST,'scientific_source_changed':False}
    (output/'source_manifest.json').write_bytes(raw)
    (HERE/'package.json').write_text(json.dumps(package,indent=2)+'\n',encoding='utf8')
    print(json.dumps(package,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--emit-only',action='store_true');a=p.parse_args()
    emit()
    if not a.emit_only:seal()
