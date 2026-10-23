"""Immutable fixture-isolation retry; local stdlib bytes/AST only."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
OLD=HERE.parent/'20261023-boltz-lr0025-01'
ARCHIVE=REPO/'results/hpc/20261023-boltz-lr0025-01/source01/source.tar.gz'
ARCHIVE_SHA='d19a5d8671b12990702ee0502f78e247e36bde97d33373461a6de1de25dcc78f'
MANIFEST='43767a8564c220821fa06fa68c3a3690870cda2e22baeb7bf956b7f849094e76'
ALIASES={
    'boltz_serial_protocol.py':'scripts/boltz_serial_protocol.py',
    'boltz_serial_ops.py':'scripts/boltz_serial_ops.py',
    'boltz_serial_driver.py':'examples/brd4_25_atoms.py',
    'boltz_serial_contract.py':'scripts/campaign25_contract.py',
    'boltz_serial_cpu.lsf':'scripts/hpc/boltz_serial_cpu.lsf',
    'boltz_serial_stage.lsf':'scripts/hpc/boltz_serial_stage.lsf',
    'fixture_boltz_serial.py':'tests/test_boltz_serial.py',
    'fixture_protenix_initialization.py':'tests/test_protenix_initialization.py',
    'boltz2.json':'examples/configs/boltz_serial_boltz2.json',
    'boltz2_chemistry.json':'examples/configs/boltz_serial_boltz2_chemistry.json',
    'README.md':'docs/boltz_serial30k.md',
}


def sha(data):return hashlib.sha256(data).hexdigest()


def replace(text,old,new):
    if text.count(old)!=1:raise ValueError((old,text.count(old)))
    return text.replace(old,new)


def source():
    if sha(ARCHIVE.read_bytes())!=ARCHIVE_SHA:raise ValueError('Original failed633 archive changed')
    files={}
    with tarfile.open(ARCHIVE) as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith('source/') or '..' in Path(member.name).parts:raise ValueError('Unsafe source archive member')
            name=member.name.removeprefix('source/')
            if name in files:raise ValueError('Duplicate source member')
            files[name]=archive.extractfile(member).read()
    raw=files.pop('source_manifest.json');manifest=json.loads(raw)
    if sha(raw)!=MANIFEST or len(files)!=633 or set(files)!=set(manifest['files']):raise ValueError('Original633 complete scope differs')
    for name,data in files.items():
        if sha(data)!=manifest['files'][name]:raise ValueError('Original member changed '+name)
    return files


def write(path,data):
    if path.exists():raise FileExistsError('Never overwrite owner initialization '+str(path))
    path.write_bytes(data.replace(b'\r\n',b'\n'))


def initialize(files):
    for local,target in ALIASES.items():
        if local=='README.md':continue
        data=files[target]
        if local=='fixture_protenix_initialization.py':
            text=data.decode()
            # Collection imports this before any temporary monkeypatch executes.
            marker='import numpy as np\n'
            text=replace(text,marker,marker+'# Bind the real lazy coupled checker before any test monkeypatch.\nimport mosaic.small_molecule.chemistry.checks\n')
            data=text.encode()
        elif local=='fixture_boltz_serial.py':
            data+=b'''

def test_lazy_checker_keeps_real_evaluator_after_temporary_monkeypatch(monkeypatch):
    import mosaic.small_molecule.chemistry.trajectory as trajectory
    import mosaic.small_molecule.chemistry.checks as checks
    original=trajectory.evaluate_combined
    assert checks.evaluate_combined is original
    def temporary(*args,**kwargs):
        raise AssertionError('Temporary unit fixture must not leak to real checker')
    with monkeypatch.context() as patch:
        patch.setattr(trajectory,'evaluate_combined',temporary)
        assert trajectory.evaluate_combined is temporary
        assert checks.evaluate_combined is original
    assert trajectory.evaluate_combined is original
    assert checks.evaluate_combined is original
'''
        elif local=='boltz_serial_ops.py':
            text=data.decode().replace("ROOT = BASE / 'boltz-lr0025-01'","ROOT = BASE / 'boltz-lr0025-02'").replace("SOURCE = BASE / 'boltz-lr0025-release-01/source'","SOURCE = BASE / 'boltz-lr0025-release-02/source'")
            text=replace(text,"GIB = 1024**3", """GIB = 1024**3
NOISY_ROOT = Path('/work3/s234818/mosaic/runs/20261023-protenix-noisy-01')
NOISY_RELEASE = Path('/work3/s234818/mosaic/releases/20261023-protenix-noisy-01')
NOISY_UPLOAD = Path('/work3/s234818/mosaic/releases/20261023-protenix-noisy-01.tar.gz')
NOISY_MANIFEST = '7bb970f48f6f907b7dc05e74a9bc2645a313addf814f08a0eeca11582f3b84ef'
NOISY_STAGE = 'experiments/brd4-jq1/protenix-noisy-1000/scripts/stage.py'
NOISY_STAGE_SHA = '48e3adc68cdb4aed0268b83e9f09508ac7836e0aec14c0113b4b441f947dc6fc'""")
            text=replace(text,"        budgets = {arm: GIB for arm in ARMS} if full is None else full", """        # CPU phases have one TOTAL1GiB allowance, unchanged originalCPU policy.
        # Two native validation arms retain1GiB each; full longs use pilot totals.
        per_arm = GIB // 2 if phase in ('preflight', 'cpu-entry') else GIB
        budgets = {arm: per_arm for arm in ARMS} if full is None else full""")
            text=replace(text,"        dbmol = dbmol_budget()", """        dbmol = dbmol_budget()
        noisy_current = tree(NOISY_ROOT)
        require(0 <= noisy_current <= GIB, 'Independent noisyProtenix output exceeds declared1GiB')
        noisy_bound = 'pending_validation_allowance_only'
        if NOISY_ROOT.exists():
            manifest_path = NOISY_RELEASE / 'source/source_manifest.json'
            bound = manifest_path.is_file() and sha(manifest_path) == NOISY_MANIFEST
            if bound:
                noisy_manifest = read(manifest_path)
                stage_path = NOISY_RELEASE / 'source' / NOISY_STAGE
                bound = (noisy_manifest['files'].get(NOISY_STAGE) == NOISY_STAGE_SHA
                         and stage_path.is_file() and sha(stage_path) == NOISY_STAGE_SHA)
            require(bound or not existing_full, 'Await authenticated noisyProtenix full persistent-output bound')
            if bound:
                noisy_bound = 'authenticated_immutable_controller_one_GiB_output_cap'
        noisy = {'full':GIB,'current':noisy_current,'remaining':GIB-noisy_current,
                 'reporting':noisy_bound,
                 'scope':'Independent bounded1000-update Protenix pilot prospective allowance'}""")
            text=replace(text," + pro['remaining'] + dbmol['remaining'] + 2 * GIB", " + pro['remaining'] + dbmol['remaining'] + noisy['remaining'] + 2 * GIB")
            text=replace(text,"storage.stale_work_bytes(), shutil.disk_usage(ROOT).free", "storage.stale_work_bytes() + tree(NOISY_ROOT) + tree(NOISY_RELEASE) + tree(NOISY_UPLOAD), shutil.disk_usage(ROOT).free")
            text=replace(text,"full_boltz=budgets, existing_protenix=pro, dbmol=dbmol,", "full_boltz=budgets, existing_protenix=pro, dbmol=dbmol, noisy_protenix=noisy,")
            text=text.replace('independent DBMol prospective; observations not reservations','independent DBMol and noisyProtenix prospective; observations not reservations')
            data=text.encode()
        write(HERE/local,data)
    for name in ['submit.py','observe.py']:
        text=(OLD/name).read_text().replace('boltz-lr0025-01','boltz-lr0025-02').replace('boltz-lr0025-release-01','boltz-lr0025-release-02')
        write(HERE/name,text.encode())


def seal(files):
    original=dict(files)
    for local,target in ALIASES.items():files[target]=(HERE/local).read_bytes().replace(b'\r\n',b'\n')
    for name,data in files.items():
        if name.endswith('.py'):ast.parse(data,filename=name)
    scientific=[name for name in files if name.startswith('src/')]
    if len(scientific)!=145 or any(original[name]!=files[name] for name in scientific):raise ValueError('Fixture-only retry changed scientific library bytes')
    unchanged=[name for name in ['examples/brd4_25_atoms.py','scripts/campaign25_contract.py','scripts/boltz_serial_protocol.py','examples/configs/boltz_serial_boltz2.json','examples/configs/boltz_serial_boltz2_chemistry.json'] if original[name]!=files[name]]
    if unchanged:raise ValueError('Declared unchanged driver/contract/config/protocol changed '+repr(unchanged))
    changed={name:sha(data) for name,data in files.items() if original.get(name)!=data}
    raw=(json.dumps({'files':{name:sha(data) for name,data in sorted(files.items())},'parent_manifest_sha256':MANIFEST,'scope':'Engineering retry of failed633: unit-only lazy-checker import isolation+regression; launch roots/capacity/noisy prospectives only; scientific library/driver/config/native checks unchanged'},sort_keys=True,indent=2)+'\n').encode()
    output=REPO/'results/hpc'/HERE.name/'source01';output.mkdir(parents=True,exist_ok=False)
    with tarfile.open(output/'source.tar.gz','w:gz') as archive:
        for name,data in {**files,'source_manifest.json':raw}.items():
            member=tarfile.TarInfo('source/'+name);member.mode=0o444;member.size=len(data);archive.addfile(member,io.BytesIO(data))
    (output/'source_manifest.json').write_bytes(raw)
    package={'tag':HERE.name,'manifest_sha256':sha(raw),'source_files':len(files),'archive_bytes':(output/'source.tar.gz').stat().st_size,'archive_sha256':sha((output/'source.tar.gz').read_bytes()),'parent_manifest_sha256':MANIFEST,'changed_members':changed,'scientific_src_changes':[],'unchanged_scientific_library_members':145,'execution':'Local stdlib AST/byte packaging only; allocated checks pending'}
    (HERE/'package.json').write_bytes((json.dumps(package,indent=2)+'\n').encode());print(json.dumps(package,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--initialize',action='store_true');parser.add_argument('--seal',action='store_true');args=parser.parse_args()
    if args.initialize==args.seal:raise ValueError('Choose one explicit preparation phase')
    files=source()
    if args.initialize:initialize(files)
    else:seal(files)


if __name__=='__main__':main()
