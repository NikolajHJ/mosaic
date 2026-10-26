"""Local stdlib AST/byte packaging of the prospective two-arm production seal.

This helper does not import numerical libraries, execute tests, connect to HPC
or submit jobs. The source binds the actual Pro05 and Bol02 producer IDs; only
allocated validation of their complete proofs can later admit production.
"""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parents[1]
REPO=HERE.parents[2]
LOCAL=REPO/'results/hpc/20261026-chemistry-production-02'
RUN=HERE/'runs/20261026-chemistry-production-02'
PRODUCERS={
    'protenix':{
        'manifest_sha256':'ab875927a328c34650d2e43baf6dfbe323e8f9bcf5600f39bac78f3ccdfb5d8c',
        'cpu_certificate_sha256':'c4a88a77f1ae5b4d20aa8c6f3b7fa59152cb295b4791b70c2bc737e1e0857980',
        'job_id':'29818553','local_run':'20261026-chemistry-full-model-05'},
    'boltz':{
        'manifest_sha256':'05e316a7b1bcceaf286b03231200a2d1f66e3f6a7ce927d4c1e73f08a24f86f9',
        'cpu_certificate_sha256':'c46f39e679f503720ae31b561b8b3273a1d38c160bf3c7f3d6feba28f2b5c687',
        'job_id':'29816184','local_run':'20261025-chemistry-full-model-02'},
}
CASE_COUNTS={
    'tests.test_production_contract.ProductionContractTests':50,
    'tests.test_production_ops.ProductionRoutingTests':33,
    'tests.test_production_integration.OriginalCallSiteTests':14,
    'tests.test_production_context.ProductionContextTests':6,
}
IMPLEMENTATION=(
    'scripts/production_contract.py','scripts/production_ops.py',
    'scripts/production_run.py','scripts/production_stage.lsf',
    'tests/test_production_contract.py','tests/test_production_ops.py',
    'tests/test_production_integration.py','scripts/production_context.py','tests/test_production_context.py',
)


def require(value,message):
    if not value:raise ValueError(message)


def raw_json(value):
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf8')


def main():
    import sys
    sys.path.insert(0,str(HERE/'scripts'))
    import production_contract as c
    require(REPO.name=='mosaic' and HERE.is_relative_to(REPO/'experiments'), 'Wrong authorized repository')
    require((RUN/'README.md').is_file(),'Write the dated run record before packaging/launch')
    members={};observed={};plan={'version':c.VERSION,'pilots':{}};parents=[]
    for arm,proof in PRODUCERS.items():
        path=HERE/'runs'/proof['local_run']/'source-manifest.json'
        data=path.read_bytes()
        require(hashlib.sha256(data).hexdigest()==proof['manifest_sha256'],
                'Original local per-arm immutable manifest differs')
        manifest=json.loads(data)
        require(all(name in manifest['files'] for name in c.ADAPTER_FILES),'Per-arm numerical adapter member missing')
        observed[arm]={name:manifest['files'][name] for name in c.ADAPTER_FILES}
        plan['pilots'][arm]={**c.PILOTS[arm],**{key:proof[key] for key in
            ('manifest_sha256','cpu_certificate_sha256','job_id')}}
        parents.append({'path':c.PILOTS[arm]['source'],'manifest_sha256':proof['manifest_sha256'],
                        'files':len(manifest['files'])})
    c.validate_plan(plan)
    require(observed['protenix']==observed['boltz'],'The two producer seals changed numerical adapter or plan bytes')
    actual_counts={}
    for name in IMPLEMENTATION:
        raw=(HERE/name).read_bytes().replace(b'\r\n',b'\n')
        if name.endswith('.py'):
            tree=ast.parse(raw,filename=name)
            if name.startswith('tests/'):
                module='tests.'+Path(name).stem
                for node in tree.body:
                    if isinstance(node,ast.ClassDef):
                        count=sum(isinstance(item,ast.FunctionDef) and item.name.startswith('test_') for item in node.body)
                        if count:actual_counts[module+'.'+node.name]=count
        members[name]=raw
    require(actual_counts==CASE_COUNTS and sum(actual_counts.values())==103,
            'Declared allocated fixture class attribution/count changed; inspect before sealing')
    members['README.md']=(
        '# Prospective accelerated chemistry production\n\n'
        'Fresh Protenix50k LR0.3 and Boltz+chemistry30k LR0.0025, unchanged fixed-step science.\n'
        'The delayed0.001-through5000 coefficient reaches100 at each arm final update.\n'
        'Both original native/readiness/coupled/runtime checks and genuine new producer passes are mandatory.\n'
        'Losses are saved every update; matrices every20 plus initial/terminal/boundaries/latest/best.\n'
        'Current complete inherited storage/reserves remain unchanged; both full chemistry futures and own64MiB control are additional.\n'
        'CPU waits actual new Pro producer; callback waits own accepted CPU. Both pilot passes remain mandatory. Submission is not a pass/start.\n'
        'Storage failure preserves actual gates/adoptions and records waiting-storage.json; no production submission or callback replay.\n'
        'All numerical/scientific work is allocated DTU HPC only. See protocol.md and plan.json.\n').encode('utf8')
    members['scripts/README.md']=b'# Production execution and metadata\n\nSealed contracts, allocated controller, exact original-driver wrapper and shell entrypoint. No optimizer arithmetic or original source bytes are changed.\n'
    members['tests/README.md']=b'# Allocated production contracts\n\n103 declared metadata/routing/original-call-site cases; actual raw JUnit and same-job scheduler/LSF proofs are required. Fixtures are not native scientific evidence.\n'
    members['protocol.md']=(RUN/'README.md').read_bytes().replace(b'\r\n',b'\n')
    members['plan.json']=raw_json(plan)
    parents += [
        {'path':c.PROFILES[arm]['source'],'manifest_sha256':c.PROFILES[arm]['manifest'],
         'files':643 if arm=='protenix' else 635} for arm in c.PROFILES]
    parents += [
        {'path':c.BASE+'/protenix-endpoint-fix-release-02/source','files':14,
         'manifest_sha256':'09f759a7befb8a7b8f0e1d203e15acf939126fba493b30e6a2bf9cd2eea2216b'},
        {'path':c.BASE+'/nonchem-stride20-startup-release-03/source','files':12,
         'manifest_sha256':'704fab033d97342a14dc12b58dc32df44874232c5c30aad47c4d02041a502aab'}]
    manifest={'version':c.VERSION,'cpu_counts':CASE_COUNTS,
              'cpu_modules':['test_production_contract','test_production_ops','test_production_integration'],
              'parent_sources':parents,'numerical_adapter_sha256_map':observed['protenix'],
              'own_source_CPU_control_cap_bytes':c.CONTROL_CAP,'matrix_stride':20,'loss_stride':1,
              'monitor_remains_paused':True,'original_results_changed':False,
              'scientific_settings_changed':False,
              'files':{name:hashlib.sha256(raw).hexdigest() for name,raw in sorted(members.items())}}
    manifest_raw=raw_json(manifest);members['source_manifest.json']=manifest_raw
    LOCAL.mkdir(exist_ok=True)
    out=LOCAL/'source02';out.mkdir(exist_ok=False)
    (out/'source_manifest.json').write_bytes(manifest_raw)
    archive=out/'source.tar.gz'
    with tarfile.open(archive,'x:gz') as target:
        for name,raw in sorted(members.items()):
            item=tarfile.TarInfo(name);item.size=len(raw);item.mode=0o444;item.mtime=0
            target.addfile(item,io.BytesIO(raw))
    with tarfile.open(archive) as target:
        rows=target.getmembers()
        require(len(rows)==len(members) and all(row.isfile() for row in rows)
                and len({row.name for row in rows})==len(rows),'Archive scope/type differs')
        for row in rows:require(target.extractfile(row).read()==members[row.name],'Archive member changed')
    result={'status':'local_stdlib_AST_byte_packaging_only','files':len(members)-1,
            'implementation_members':list(IMPLEMENTATION),'manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),
            'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_bytes':archive.stat().st_size,
            'plan_sha256':hashlib.sha256(members['plan.json']).hexdigest(),'plan':plan,'cpu_counts':CASE_COUNTS,
            'cpu_cases':103,'allocated_CPU_pass':False,'production_readiness_passed':False,
            'jobs_submitted':False,'monitor_resumed':False}
    (out/'preparation.json').write_bytes(raw_json(result))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
