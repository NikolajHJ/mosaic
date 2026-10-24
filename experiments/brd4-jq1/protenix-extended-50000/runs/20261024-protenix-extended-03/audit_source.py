"""Independent post-seal archive/AST audit; no scientific modules are imported."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
CURRENT=('20261024-protenix-extended-03','source01',643,
         'bde714a20a49308365744ab5018c06a6d68dff82c3485dc075848827a5f7a8ef',
         'da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562')
PARENT=('20261023-protenix-extended-01','source02',641,
        '8ba1c3370fee1f09b96af139dbf1403bb3c187c52d447ec4b0b23ebe4a65f63d',
        'a30f980fd8fe95001fdfc6780d720bdde6486d2421d7e4942b119ace61777f00')
ANCESTOR=('20261022-brd4-jq1-25-fixed-lrreset-04','source02',625,
          '4d6d297451d9820543abc3298622e6599014c4b96bc999e104dd732825f56e4e',
          'c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057')
REVISION='c74c659344123937608077e3e4078f75f09cd743'
ALIASES={
    'extended_ops.py':'scripts/extended_ops.py','extended_protocol.py':'scripts/extended_protocol.py',
    'extended_driver.py':'examples/brd4_25_atoms.py','extended_contract.py':'scripts/campaign25_contract.py',
    'extended_cpu.lsf':'scripts/hpc/extended_cpu.lsf','extended_stage.lsf':'scripts/hpc/extended_stage.lsf',
    'fixture_extended.py':'tests/test_extended_protenix.py','random.json':'examples/configs/extended_random.json',
    'continuation.json':'examples/configs/extended_continuation.json','parent.json':'docs/extended_parent.json',
    'predecessor-retirement.json':'docs/extended_retirement.json','README.md':'docs/protenix_extended.md',
    'extended_startup.py':'scripts/extended_startup.py','fixture_extended_startup.py':'tests/test_extended_startup.py'}
CHANGED={'scripts/extended_ops.py','scripts/hpc/extended_cpu.lsf','docs/protenix_extended.md',
         'scripts/extended_startup.py','tests/test_extended_startup.py'}


def sha(data):return hashlib.sha256(data).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)


def unpack(spec):
    tag,folder,count,archive_sha,manifest_sha=spec
    path=REPO/'results/hpc'/tag/folder/'source.tar.gz'
    raw_archive=path.read_bytes();require(sha(raw_archive)==archive_sha,'Archive changed')
    files={}
    with tarfile.open(path) as archive:
        for member in archive:
            name=member.name.removeprefix('source/');safe=PurePosixPath(name)
            require(member.isfile() and member.name.startswith('source/') and name not in files
                    and not safe.is_absolute() and '..' not in safe.parts and '\\' not in name and ':' not in name,
                    'Unsafe/duplicate member')
            require(not member.mode & 0o222,'Writable sealed member')
            data=archive.extractfile(member).read();require(len(data)==member.size,'Truncated member');files[name]=data
    raw=files.pop('source_manifest.json');require(sha(raw)==manifest_sha,'Manifest identity changed')
    manifest=json.loads(raw)
    require(len(files)==count and set(files)==set(manifest['files']),'Full file scope changed')
    require(all(sha(data)==manifest['files'][name] for name,data in files.items()),'Source member changed')
    return files,manifest,raw,len(raw_archive)


def functions(data):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(data).body if isinstance(n,ast.FunctionDef)}


def literal_cases(data):
    tree=ast.parse(data);count=0
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name.startswith('test_'):
            cases=1
            for d in n.decorator_list:
                if isinstance(d,ast.Call) and isinstance(d.func,ast.Attribute) and d.func.attr=='parametrize':
                    cases*=len(ast.literal_eval(d.args[1]))
            count+=cases
    require(count==34,'Startup fixture AST row count differs')
    return count


def main():
    output=HERE/'local-static-verification.json';require(not output.exists(),'Existing audit; no blind overwrite')
    files,manifest,raw,size=unpack(CURRENT);previous,parent,_,_=unpack(PARENT);_,ancestor,_,_=unpack(ANCESTOR)
    package=json.loads((HERE/'package.json').read_bytes())
    require(package['source_files']==643 and package['archive_bytes']==size==39644558
            and package['archive_sha256']==CURRENT[3] and package['manifest_sha256']==CURRENT[4],'Package fields changed')
    require((REPO/'results/hpc'/CURRENT[0]/CURRENT[1]/'source_manifest.json').read_bytes()==raw,'Sibling manifest changed')
    require('git_revision_reference' not in parent and manifest['git_revision_reference']==ancestor['git_revision_reference']==REVISION
            and manifest['parent_manifest_sha256']==PARENT[4],'Authentic provenance inheritance differs')
    require(not set(previous)-set(files),'Historical source members removed')
    changed={name for name in files if previous.get(name)!=files[name]};require(changed==CHANGED,'Undeclared source bytes changed')
    require({name:sha(files[name]) for name in changed}==package['changed_members'],'Changed member map differs')
    protected=[name for name in previous if name not in CHANGED]
    require(len(protected)==638 and all(files[name]==previous[name] for name in protected),'Protected641 members differ')
    scientific=[name for name in previous if name.startswith('src/')]
    require(len(scientific)==145 and all(files[name]==previous[name] for name in scientific),'Scientific source changed')
    for local,member in ALIASES.items():require((HERE/local).read_bytes().replace(b'\r\n',b'\n')==files[member],'Canonical alias changed: '+local)
    python=[name for name in files if name.endswith('.py')]
    for name in python:ast.parse(files[name],filename=name)
    original_functions=functions(previous['scripts/extended_ops.py']);current_functions=functions(files['scripts/extended_ops.py'])
    changed_functions=sorted(name for name in original_functions if original_functions[name]!=current_functions[name])
    require(changed_functions==sorted(['resources','command','storage_gate','all_budgets','queue_native','certify_cpu','after_native','certify_segment','produce','main']),
            'Inherited controller function changes differ')
    require(original_functions['native_assessment']==current_functions['native_assessment'],'Original native checks changed')
    driver=ast.parse(files['examples/brd4_25_atoms.py'])
    calls=[n for n in ast.walk(driver) if isinstance(n,ast.Call)]
    gpu_line=min(n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='devices')
    identity_line=min(n.lineno for n in calls if isinstance(n.func,ast.Name) and n.func.id=='source_identity')
    model_line=min(n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='load'
                   and isinstance(n.func.value,ast.Name) and n.func.value.id=='FrozenProtenixTiny')
    require(gpu_line<identity_line<model_line,'Actual unchanged GPU guard/provenance/model order differs')
    helper=ast.parse(files['scripts/extended_startup.py']);execute=next(n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name=='execute_command')
    patches=[n for n in ast.walk(execute) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='object']
    require(len(patches)==1 and ast.literal_eval(patches[0].args[1])=='load'
            and ast.dump(patches[0].args[0],include_attributes=False)=="Attribute(value=Name(id='module', ctx=Load()), attr='FrozenProtenixTiny', ctx=Load())",
            'GPU startup patches more than model constructor')
    gpu_startup=next(n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name=='gpu_startup')
    require(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='execute_command' for n in ast.walk(gpu_startup)),
            'Long-lived parent must not execute real runpy directly')
    child=next(n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name=='run_startup_child')
    calls=[n for n in ast.walk(child) if isinstance(n,ast.Call)]
    require(sum(isinstance(n.func,ast.Attribute) and n.func.attr=='Popen' for n in calls)==1
            and any(isinstance(n.func,ast.Attribute) and n.func.attr=='communicate' for n in calls)
            and any(isinstance(n.func,ast.Attribute) and n.func.attr=='kill' for n in calls),
            'Owned startup child spawn/wait/timeout cleanup missing')
    old02,old02manifest,_,_=unpack(('20261024-protenix-extended-02','source01',643,
            'e04c57d588ef3bc531902f8ccc8fbb2d634ceea78dc8cbe6640cfbb3607d6b36',
            'f1c15dc99052a8c4764e4ea0d0e9d8ca068e1c496e12c8c76c55251233b1220e'))
    require(manifest['local_engineering_predecessor']['source_manifest_sha256']=='f1c15dc99052a8c4764e4ea0d0e9d8ca068e1c496e12c8c76c55251233b1220e'
            and manifest['local_engineering_predecessor']['status']=='unsubmitted_local_candidate_preserved',
            'Local unsubmitted02 predecessor preservation record differs')
    cases=literal_cases(files['tests/test_extended_startup.py'])
    record={'status':'passed','verified_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'Post-seal independent stdlib archive/byte/JSON/AST audit only',
            'source_manifest_sha256':CURRENT[4],'archive_bytes':size,'archive_sha256':CURRENT[3],
            'source_files':len(files),'safe_members':len(files)+1,'python_members_ast_parsed':len(python),
            'parent_source_manifest_sha256':PARENT[4],'ancestor_source_manifest_sha256':ANCESTOR[4],
            'git_revision_reference':REVISION,'unchanged_original641_members':len(protected),
            'unchanged_scientific_src_members':len(scientific),'canonical_aliases_checked':len(ALIASES),
            'changed_members':package['changed_members'],'changed_controller_functions':changed_functions,
            'original_native_assessment_ast_unchanged':True,
            'unchanged_gpu_guard_line':gpu_line,'unchanged_source_identity_line':identity_line,'unchanged_model_boundary_line':model_line,
            'gpu_smoke_patches_only_native_model_load':True,'real_gpu_runpy_only_in_short_lived_child':True,'local_unsubmitted02_archive_preserved':True,'startup_fixture_cases_ast':cases,'expected_allocated_cpu_cases':832+cases,
            'local_numerical_tests_executed':False,'allocated_cpu_pass_established':False,
            'actual_gpu_startup_pass_established':False,'native_pass_established':False,'remote_submission_established':False}
    with output.open('x',encoding='utf8',newline='\n') as handle:json.dump(record,handle,indent=2,sort_keys=True);handle.write('\n')
    print(json.dumps({'status':'passed','receipt':str(output),'sha256':sha(output.read_bytes()),'source_files':len(files),'expected_cpu':832+cases}))


if __name__=='__main__':main()
