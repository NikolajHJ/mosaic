"""One-shot archive-authenticated metadata bootstrap; never replays old jobs."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / 'results/hpc/20261025-nonchem-stride20-startup-03'
PACKAGE = LOCAL / 'source02'
REMOTE = r'''
import ast,base64,hashlib,io,json,os,stat,subprocess,sys,tarfile,types
from pathlib import Path,PurePosixPath
assert not sys.flags.optimize
archive=base64.b64decode(ARCHIVE)
assert len(archive)==ARCHIVE_BYTES and hashlib.sha256(archive).hexdigest()==ARCHIVE_SHA
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
release=base/'nonchem-stride20-startup-release-03';source=release/'source'
control=base/'nonchem-stride20-02/startup-fix-03';upload=base/'nonchem-stride20-startup-03.tar.gz'
noise=Path('/work3/s234818/mosaic/runs/20261025-protenix-noisy-05')
assert noise.is_dir() and (noise/'cpu-ready.json').is_file(),'Keep the existing independently queued noisy study'
targets=[release,control,upload]
for name in ['boltz-lr0025-03','boltz-dbmol-pilot-02','protenix-extended-03']:
 root=base/name
 targets.extend([root/'nonchem-stride20-v3-owner.json',root/'dispatch-0004'])
assert all(not os.path.lexists(p) for p in targets),'Existing/unknown selected outcome; inspect, never duplicate'
with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
 rows=stream.getmembers();assert len(rows)==13 and len({r.name for r in rows})==len(rows)
 assert all(r.isfile() and not r.issym() and not r.islnk() and not PurePosixPath(r.name).is_absolute() and '..' not in PurePosixPath(r.name).parts for r in rows)
 data={r.name:stream.extractfile(r).read() for r in rows}
raw=data['source_manifest.json'];assert hashlib.sha256(raw).hexdigest()==MANIFEST
m=json.loads(raw);assert set(data)==set(m['files'])|{'source_manifest.json'}
for name,digest in m['files'].items():
 assert hashlib.sha256(data[name]).hexdigest()==digest
 if name.endswith('.py'):ast.parse(data[name],filename=name)
# Execute only authenticated standard-library policy modules for the preflight.
for name in ['legacy_contract','retention','selected_contract','selected_storage']:
 module=types.ModuleType(name);module.__file__=str(source/(name+'.py'));sys.modules[name]=module
 exec(compile(data[name+'.py'],module.__file__,'exec'),module.__dict__)
c=sys.modules['selected_contract'];storage=sys.modules['selected_storage']
for name in c.PROFILES:
 c.native_units(name)
 root=c.profile(name)['root']
 assert not c.production_entries(root)
c.deferred_absence();c.verify_noise_source();c.previous_admission()
assert subprocess.run(['bash','-n'],input=data['stage.lsf'],capture_output=True).returncode==0,'Selected wrapper syntax failed'
study=c.NOISE_SOURCE/c.NOISE_STUDY
for p in [c.NOISE_SOURCE/'tests/test_protenix_initialization.py',study/'tests/test_noise_protocol.py',study/'tests/test_driver_contract.py',study/'scripts/driver.py']:
 assert p.is_file() and not p.is_symlink(),'Original noisy CPU/driver input absent'
control.mkdir()
preflight=control/'preflight';preflight.mkdir()
gate=storage.storage_gate(preflight,'preflight')
with upload.open('xb') as f:f.write(archive);f.flush();os.fsync(f.fileno())
source.mkdir(parents=True)
for name,raw in data.items():
 p=source/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(raw)
 p.chmod(0o444)
for p in sorted([p for p in source.rglob('*') if p.is_dir()],reverse=True):p.chmod(0o555)
source.chmod(0o555);release.chmod(0o555)
c.verify_own(MANIFEST)
subprocess.run(['bash','-n',str(source/'stage.lsf')],check=True,capture_output=True)
(control/'source-staging.json').write_text(json.dumps({'status':'passed','source':str(source),'manifest_sha256':MANIFEST,'archive_sha256':ARCHIVE_SHA,'archive_bytes':ARCHIVE_BYTES,'preflight_gate':c.file_record(preflight/'selected-storage-preflight/gate.json')},indent=2)+'\n')
sys.path.insert(0,str(source))
import selected_ops as op
post=control/'post-stage';post.mkdir();storage.storage_gate(post,'post-stage')
cpu=op.submit('control','cpu','cpu',MANIFEST)
after=op.submit('control','after-cpu','after-cpu',MANIFEST,'ended('+cpu['job_id']+')')
receipt={'status':'metadata_cpu_and_callback_submitted_not_production','cpu':cpu,'after_cpu':after,'source_manifest_sha256':MANIFEST,'full_preflight_storage_gate':gate,'saved_results_and_settings_changed':False}
c.exclusive(control/'initial-submissions-complete.json',receipt)
print(json.dumps(receipt))
'''


def main():
    label=sys.argv[1] if len(sys.argv)>1 else '01'
    assert label.isdigit()
    out = LOCAL / ('bootstrap-'+label)
    out.mkdir(exist_ok=False)
    prep = json.loads((PACKAGE / 'preparation.json').read_text())
    archive = (PACKAGE / 'source.tar.gz').read_bytes()
    assert len(archive) == prep['archive_bytes'] and hashlib.sha256(archive).hexdigest() == prep['archive_sha256']
    bindings = {'ARCHIVE': base64.b64encode(archive).decode(), 'ARCHIVE_BYTES': len(archive),
                'ARCHIVE_SHA': prep['archive_sha256'], 'MANIFEST': prep['manifest_sha256']}
    code = '\n'.join(key + '=' + repr(value) for key, value in bindings.items()) + '\n' + REMOTE
    (out / 'remote-program.py').write_text(code, encoding='utf-8', newline='\n')
    (out / 'started.json').write_text(json.dumps({'started_utc': datetime.now(timezone.utc).isoformat(),
        'remote_program_sha256': hashlib.sha256(code.encode()).hexdigest(), 'preparation': prep}, indent=2)+'\n', encoding='utf-8')
    args = ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
            'bash -l -c ' + shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    with (out / 'stdout.bin').open('xb') as stdout, (out / 'stderr.bin').open('xb') as stderr:
        result = subprocess.run(args, input=code.encode(), stdout=stdout, stderr=stderr, timeout=240)
    receipt = {'returncode': result.returncode, 'completed_utc': datetime.now(timezone.utc).isoformat(),
               'stdout_bytes': (out/'stdout.bin').stat().st_size,
               'stdout_sha256': hashlib.sha256((out/'stdout.bin').read_bytes()).hexdigest()}
    (out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt, indent=2))
    if result.returncode:
        print((out/'stderr.bin').read_text(errors='replace')[-5000:])
    else:
        d=json.loads((out/'stdout.bin').read_bytes())
        print(json.dumps({'status':d['status'],'cpu_job':d['cpu']['job_id'],'callback_job':d['after_cpu']['job_id']},indent=2))


if __name__ == '__main__':
    main()
