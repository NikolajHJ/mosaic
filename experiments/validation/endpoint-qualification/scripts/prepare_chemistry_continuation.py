"""Local stdlib-only sealing; no scientific evaluation or automatic submission."""
import ast
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tarfile

OWNER=Path(__file__).resolve().parents[1]
REPO=OWNER.parents[2]
sys.path.insert(0,str(OWNER/'scripts'))
import chemistry_continuation_contract as c


def raw(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def read(path):return json.loads(Path(path).read_bytes())
def digest(value):return hashlib.sha256(value).hexdigest()
def replace(text,before,after,count=1):
    if text.count(before)!=count:raise ValueError('Authenticated original metadata shape differs: '+before)
    return text.replace(before,after)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--draft',choices=('source01','source02'),default='source02')
    args=parser.parse_args()
    output=REPO/'results/hpc/20261029-chemistry-continuation-03'/args.draft
    if output.exists():raise ValueError('Immutable local draft already exists; inspect before a new attempt')
    prior=REPO/'results/hpc/20261029-precision-continuation-25/source01'
    original=prior/'source'
    if not original.exists():original=prior
    old_manifest=read(original/'source_manifest.json')
    if digest((original/'source_manifest.json').read_bytes())!=c.CURRENT_PIN:raise ValueError('Original25 source pin differs')
    qual=read(REPO/'results/hpc/20261029-chemistry-native-qualification-08/source01/plan.json')
    capture=read(REPO/'results/hpc/20261029-chemistry-continuation-preparation/20261029T123449Z/raw.json')
    current=read(original/'plan.json');plan={
        'version':c.VERSION,'root':c.ROOT.as_posix(),'source_root':c.SOURCE.as_posix(),
        'scientific_output_root':c.OUTPUTS.as_posix(),'original_results_changed':False,'optimization_settings_changed':False,
        'original_failures_preserved':True,'raw_policy':'prospective-endpoint-admission-v3',
        'logit_policy':'prospective-fixed-reference-logit-admission-v1','component_only_can_submit':False,
        'both_complete_pilots_required':True,'actual_coefficient_check_required':True,
        'monitor_remains_paused':True,'protected_inputs':qual['protected_inputs'],
        'protenix_reference':current['protenix_reference'],
        'reference_configuration':{'arms':{'protenix':deepcopy(current['reference_configuration']['arms']['protenix'])}},
        'arms':{},'execution_context':{'protenix':qual['qualified_production_context'],
                'boltz':{'arm':'boltz','environment':{},'scientific_settings_changed':False}},
        'retention_source':{'path':(c.BASE/'nonchem-stride20-startup-release-03/source').as_posix(),
            'manifest_sha256':'704fab033d97342a14dc12b58dc32df44874232c5c30aad47c4d02041a502aab'},
        'components':{'protenix':{'parent':(c.SCOPE/'chemistry-native-qualification-08/gpu-protenix').as_posix(),
              'job_id':'29847869','manifest_sha256':'82eb292f66ee8f8dcd86d9480775c480c1b086a1192d6ac724fbdd771a08438e'},
            'boltz':{'parent':(c.BASE/'chemistry-native-qualification-04/gpu-boltz').as_posix(),
              'job_id':'29834601','manifest_sha256':'5d8f4ef0f8bb820e76d2e92442492f604393e79b58725fd01f9bb042b3114863'}}}
    input_metadata=read(REPO/'results/hpc/20261029-chemistry-continuation-preparation/original-input-metadata/raw.json')
    protected={v['path']:v for v in [*plan['protected_inputs'],*input_metadata['argv_inputs']]}
    for name in ('chemistry-production-03/protenix/random-segment-0000/payload/driver-invocation.json',
                 'chemistry-production-04/boltz/boltz2_chemistry-segment-0000/payload/driver-invocation.json'):
        item=capture[name];protected[item['path']]={k:item[k] for k in ('path','bytes','sha256')}
    plan['protected_inputs']=list(protected.values())
    for name,scope in c.ARMS.items():
        arm=deepcopy(qual['arms'][name]);arm.update(scope)
        old_path=('chemistry-production-03/protenix/random-segment-0000/payload/driver-invocation.json' if name=='protenix'
                  else 'chemistry-production-04/boltz/boltz2_chemistry-segment-0000/payload/driver-invocation.json')
        original_argv=capture[old_path]['json']['args'];arm['original_argv']=original_argv
        arm.update(chemistry_enabled=True,element_js_weight=0,confidence_margin_enabled=False,matrix_stride=20,
                loss_stride=1,max_backtracks=1,acceptance_policy='fixed_step',start_global=10000,
                profile='protenix-random50k' if name=='protenix' else 'boltz-chem30k',
                repair_schedule={'version':'delayed-geometric-v1','initial':.001,'final':100.,'hold_updates':5000,'total_updates':scope['target']})
        arm['original_directions']={'X':[],'E_upper':[]}
        # Preserve exact old Boltz raw directions in the new native checker.
        if name=='boltz':
            p=REPO/'results/hpc/20261029-chemistry-continuation-preparation/original-input-metadata/boltz-original-terminal-report.json'
            if p.stat().st_size!=arm['failed_report']['bytes'] or digest(p.read_bytes())!=arm['failed_report']['sha256']:
                raise ValueError('Exact archived Boltz failed report differs')
            failed=read(p)
            for field in ('X','E_upper'):
                arm['original_directions'][field]=[{k:r[k] for k in ('row','source','target')} for r in failed.get('rows',[]) if r.get('field')==field]
            if not any(arm['original_directions'].values()):
                # Source04's plan records this exact archived endpoint direction.
                # A missing local raw report must be fetched/authenticated; do
                # not infer the direction from a different experiment.
                raise ValueError('Exact original Boltz raw report must be available locally before sealing')
        arm['argv_inputs']=[]
        for flag in ('--cpu-certificate','--reference','--pocket','--ready'):
            path=original_argv[original_argv.index(flag)+1]
            if path not in protected:raise ValueError('Missing protected original argv input: '+path)
            arm['argv_inputs'].append(protected[path])
        arm['old_ready']=protected[original_argv[original_argv.index('--ready')+1]]
        plan['arms'][name]=arm
    reference=plan['reference_configuration']['arms']['protenix']
    reference['command_prefix']=[p.replace(c.CURRENT.as_posix(),c.SOURCE.as_posix()).replace('scripts/protenix_reference_worker.py','scripts/chemistry_reference_worker.py') for p in reference['command_prefix']]
    reference['historical_global']=10000;reference['historical_original_directions']={'X':[],'E_upper':[]}
    reference['historical_failure']=plan['arms']['protenix']['failed_report']
    files={}
    for name in ('chemistry_continuation_contract.py','chemistry_continuation_hooks.py','chemistry_continuation_driver.py',
                 'chemistry_continuation_ops.py','chemistry_continuation_stage.lsf'):
        files['scripts/'+name]=(OWNER/'scripts'/name).read_bytes().replace(b'\r\n',b'\n')
    worker=(original/'scripts/protenix_reference_worker.py').read_bytes()
    if digest(worker)!=old_manifest['files']['scripts/protenix_reference_worker.py']:raise ValueError('Reference source bytes changed')
    text=worker.decode();text=replace(text,'ROOT = BASE / "nonchem-precision-continuation-25"','ROOT = BASE / "nonchem-precision-continuation-25/chemistry-continuation-03"')
    text=replace(text,'SOURCE = BASE / "nonchem-precision-release-25/source"','SOURCE = BASE / "nonchem-precision-continuation-25/chemistry-continuation-release-03/source"')
    text=replace(text,'scripts/protenix_reference_worker.py','scripts/chemistry_reference_worker.py',2)
    text=replace(text,'declared == "4"','declared == ("4" if startup else "8")')
    text=replace(text,'path.is_relative_to(ROOT)','(path.is_relative_to(ROOT) or path.is_relative_to(BASE / "chemistry-continuation-03"))')
    files['scripts/chemistry_reference_worker.py']=text.encode()
    cuda=(original/'scripts/cuda_ownership.py').read_bytes()
    if digest(cuda)!=old_manifest['files']['scripts/cuda_ownership.py']:raise ValueError('CUDA source bytes changed')
    text=cuda.decode();text=replace(text,'environment.get("LSB_DJOB_NUMPROC") == "4"','environment.get("LSB_DJOB_NUMPROC") == "8"')
    text=replace(text,'"cpu_slots": 4','"cpu_slots": 8')
    text=replace(text,'("-n", "4")','("-n", "8")')
    text=replace(text,'rusage[mem=8GB]','rusage[mem=4GB]')
    files['scripts/chemistry_cuda_ownership.py']=text.encode()
    files['tests/test_chemistry_continuation.py']=(OWNER/'tests/test_chemistry_continuation.py').read_bytes().replace(b'\r\n',b'\n')
    files['README.md']=(OWNER/'runs/20261029-chemistry-continuation-03/README.md').read_bytes().replace(b'\r\n',b'\n')
    files['plan.json']=raw(plan)
    for name,data in files.items():
        if name.endswith('.py'):ast.parse(data,filename=name)
    parents={r['path']:r for r in qual['parent_sources']}
    parents[c.CURRENT.as_posix()]={'path':c.CURRENT.as_posix(),'files':50,'manifest_sha256':c.CURRENT_PIN}
    qualifier=c.SCOPE/'chemistry-native-qualification-release-08/source'
    parents[qualifier.as_posix()]={'path':qualifier.as_posix(),'files':13,'manifest_sha256':'82eb292f66ee8f8dcd86d9480775c480c1b086a1192d6ac724fbdd771a08438e'}
    previous=c.SCOPE/'chemistry-continuation-release-02/source'
    parents[previous.as_posix()]={'path':previous.as_posix(),'files':10,
            'manifest_sha256':'d87701b3eb422d3e1c1f9255eb61607011b96eb621a67d286ceefe961e36c250'}
    manifest={'version':c.VERSION,'files':{name:digest(data) for name,data in files.items()},
              'parent_sources':list(parents.values()),'capacity_source':parents[c.CURRENT.as_posix()],
              'cpu_counts':{'tests.test_chemistry_continuation':34},'original_results_changed':False}
    output.mkdir(parents=True)
    for name,data in files.items():
        path=output/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    manifest_bytes=raw(manifest);(output/'source/source_manifest.json').write_bytes(manifest_bytes)
    with tarfile.open(output/'source.tar.gz','w:gz') as archive:
        for path in sorted((output/'source').rglob('*')):
            if path.is_file():archive.add(path,arcname=path.relative_to(output/'source').as_posix(),recursive=False)
    archive=(output/'source.tar.gz').read_bytes()
    receipt={'status':'local_stdlib_preparation_only_not_allocated_pass','source_root':c.SOURCE.as_posix(),
        'root':c.ROOT.as_posix(),'files':len(files),'manifest_sha256':digest(manifest_bytes),
        'archive_bytes':len(archive),'archive_sha256':digest(archive),'plan_sha256':digest(files['plan.json'])}
    (output/'preparation.json').write_bytes(raw(receipt));print(json.dumps(receipt))


if __name__=='__main__':main()
