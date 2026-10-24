"""Root-owned one-shot bootstrap for a prepared peer-adoption metadata release.

This local operator is outside the seven runtime members. It does not prepare
or seal a package. Its only submissions are one metadata CPU and its own-ended
callback. Unknown local/remote outcomes are exclusive and cannot be replayed.
Scientific sources, initial native jobs, saved results and settings stay intact.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
import os
import re
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
REVISION = 'c74c659344123937608077e3e4078f75f09cd743'
PROFILES = {
    'boltz30k': {'tag': '20261024-boltz-pro02-adoption-04',
        'consumer': 'experiments/brd4-jq1/boltz-lr0025/runs/20261024-boltz-pro02-adoption-04',
        'source': BASE + '/boltz-lr0025-release-03/source',
        'manifest': 'c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6',
        'root': BASE + '/boltz-lr0025-03', 'control': BASE + '/boltz-pro02-adoption-04',
        'launcher': BASE + '/boltz-pro02-adoption-release-04/source'},
    'dbmol10k': {'tag': '20261024-dbmol-pro02-adoption-06',
        'consumer': 'experiments/brd4-jq1/boltz-dbmol/runs/20261024-dbmol-pro02-adoption-06',
        'source': BASE + '/boltz-dbmol-pilot-release-02/source',
        'manifest': '600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed',
        'root': BASE + '/boltz-dbmol-pilot-02', 'control': BASE + '/dbmol-pro02-adoption-06',
        'launcher': BASE + '/dbmol-pro02-adoption-release-06/source'},
}
MAX_METADATA = 4 * 1024**2
MAX_ARCHIVE = 4 * 1024**2
BOOTSTRAP = r'''
import hashlib,sys
header=sys.stdin.buffer.readline(32)
assert header.endswith(b'\n') and header[:-1].isdigit()
length=int(header);assert 0<length<=128*1024
code=sys.stdin.buffer.read(length)
assert len(code)==length and hashlib.sha256(code).hexdigest()==sys.argv[1]
exec(compile(code,'<reviewed-peer-adoption-bootstrap>','exec'),globals())
'''

REMOTE = r'''
import ast,hashlib,io,json,os,re,stat,subprocess,sys,tarfile
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
assert not sys.flags.optimize
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def strict(raw):
 def pairs(rows):
  out={}
  for k,v in rows:require(k not in out,'Duplicate JSON field');out[k]=v
  return out
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def read(path,cap=64*1024**2):
 path=Path(path);require(path.is_file() and not path.is_symlink() and path.resolve()==path,'Missing/linked proof')
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 with os.fdopen(fd,'rb') as stream:
  before=os.fstat(stream.fileno());require(stat.S_ISREG(before.st_mode) and before.st_size<=cap,'Oversized/nonregular proof')
  data=stream.read(cap+1);after=os.fstat(stream.fileno())
 literal=path.stat();sig=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns,x.st_mode)
 require(len(data)==before.st_size and sig(before)==sig(after)==sig(literal),'Proof changed during read')
 return data
def save(path,value):
 raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444)
 with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
def verify_source(root,expected):
 root=Path(root);require(root.is_dir() and not root.is_symlink() and root.resolve()==root and not root.stat().st_mode&0o222,'Source root differs/writable')
 raw=read(root/'source_manifest.json');require(sha(raw)==expected,'Source manifest differs')
 m=strict(raw);require(m['git_revision_reference']=='c74c659344123937608077e3e4078f75f09cd743','Unauthenticated revision')
 observed=set()
 for directory,dirs,names in os.walk(root,followlinks=False):
  require(not Path(directory).stat().st_mode&0o222,'Writable source directory')
  require(not any((Path(directory)/n).is_symlink() for n in dirs),'Linked source directory')
  for n in names:
   p=Path(directory)/n;require(not p.is_symlink() and stat.S_ISREG(p.stat().st_mode) and not p.stat().st_mode&0o222,'Special/writable source file')
   observed.add(p.relative_to(root).as_posix())
 require(observed==set(m['files'])|{'source_manifest.json'},'Source inventory differs')
 for name,digest in m['files'].items():
  p=PurePosixPath(name);require(not p.is_absolute() and '..' not in p.parts and p.as_posix()==name,'Unsafe source member')
  require(sha(read(root/name))==digest,'Source member differs')
 return m
header=sys.stdin.buffer.readline(32)
require(header.endswith(b'\n') and header[:-1].isdigit(),'Missing metadata frame')
length=int(header);require(0<length<=4*1024**2,'Metadata frame too large')
raw_payload=sys.stdin.buffer.read(length);require(len(raw_payload)==length,'Truncated metadata frame')
p=strict(raw_payload)
profiles={
 'boltz30k':('boltz-pro02-adoption-04','boltz-pro02-adoption-release-04','boltz-lr0025-release-03','boltz-lr0025-03','c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6'),
 'dbmol10k':('dbmol-pro02-adoption-06','dbmol-pro02-adoption-release-06','boltz-dbmol-pilot-release-02','boltz-dbmol-pilot-02','600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed')}
require(p['profile'] in profiles,'Unknown profile')
control_name,release_name,science_name,output_name,science_manifest=profiles[p['profile']]
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
control=base/control_name;release=base/release_name;launcher=release/'source';source=base/science_name/'source';output=base/output_name
upload=base/(control_name+'.tar.gz');started=base/(control_name+'-bootstrap-started.json');completed=base/(control_name+'-bootstrap-complete.json')
require(base.is_dir() and base.resolve()==base and output.is_dir() and output.resolve()==output,'Base/output scope differs')
require(not any(os.path.lexists(x) for x in (control,release,upload,started,completed)),'Fresh target/unknown bootstrap already exists; inspect, never replay')
require(not os.path.lexists(output/'protenix02-adoption-owner.json'),'Existing scientific dispatch ownership')
verify_source(source,science_manifest)
require(p['science_manifest_sha256']==science_manifest and p['source']==str(source) and p['output_root']==str(output)
 and p['control']==str(control) and p['launcher']==str(launcher),'Profile path/source differs')
archive=sys.stdin.buffer.read(4*1024**2+1)
require(len(archive)==p['archive_bytes']<=4*1024**2 and sha(archive)==p['archive_sha256'],'Exact source archive differs')
aliases={'README.md','adoption-plan.json','scripts/adoption_contract.py','scripts/adoption_ops.py','scripts/hpc/peer_adoption_cpu.lsf','scripts/hpc/peer_adoption_stage.lsf','tests/test_peer_adoption.py'}
members={}
with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
 for item in stream:
  name=PurePosixPath(item.name)
  require(item.isfile() and not item.linkname and not name.is_absolute() and '..' not in name.parts and name.as_posix()==item.name
   and '\\' not in item.name and ':' not in item.name and item.name in aliases|{'source_manifest.json'} and item.name not in members,'Unsafe/duplicate source member')
  require(0<=item.size<=4*1024**2,'Source member too large')
  data=stream.extractfile(item).read(4*1024**2+1);require(len(data)==item.size,'Truncated source member');members[item.name]=data
require(set(members)==aliases|{'source_manifest.json'} and sha(members['source_manifest.json'])==p['launcher_manifest_sha256'],'Seven-member manifest scope differs')
manifest=strict(members['source_manifest.json'])
require(manifest['version']=='mosaic-provenance-peer-adoption-launcher-v1' and manifest['profile']==p['profile']
 and manifest['git_revision_reference']=='c74c659344123937608077e3e4078f75f09cd743' and type(manifest['metadata_cpu_cases']) is int and manifest['metadata_cpu_cases']==p['metadata_cases'] and p['metadata_cases']>0
 and manifest['scientific_source_unchanged'] is True and manifest['metadata_cpu_is_scientific_readiness'] is False
 and set(manifest['files'])==aliases,'Metadata manifest scope differs')
for name in aliases:
 require(sha(members[name])==manifest['files'][name]==p['member_sha256'][name],'Authenticated runtime member differs')
 if name.endswith('.py'):ast.parse(members[name],filename=name)
plan=strict(members['adoption-plan.json'])
require(plan['profile']==p['profile'] and plan['scientific_source_manifest_sha256']==science_manifest
 and plan['scientific_source']==str(source) and plan['scientific_output_root']==str(output)
 and plan['adoption_control_root']==str(control) and plan['metadata_cpu_case_count']==p['metadata_cases']
 and plan['scientific_cpu_certificate_unchanged'] is True and plan['new_metadata_cpu_certificate_separate'] is True,'Declared adoption plan differs')
# Authenticate all existing own scientific proofs BEFORE creating any target.
# The exact archived contract is standard-library metadata code, already bound
# above to the reviewed seven-member manifest; this is not a model import.
import types
precontract=types.ModuleType('authenticated_peer_adoption_preflight_contract')
precontract.__file__=str(launcher/'scripts/adoption_contract.py')
exec(compile(members['scripts/adoption_contract.py'],'<authenticated-stdlib-peer-proof-contract>','exec'),precontract.__dict__)
preprofile=precontract.PROFILES[p['profile']]
require(preprofile['source']==source and preprofile['root']==output and preprofile['control']==control
 and preprofile['launcher']==launcher and preprofile['manifest']==science_manifest,'Preflight proof contract scope differs')
pre_native=precontract.verify_native_units(p['profile'])
precontract.absence_of_production(p['profile'])
# The first mutation is an exclusive external unknown-outcome marker.
save(started,{'status':'unknown_until_cpu_and_callback_submitted','profile':p['profile'],'utc':datetime.now(timezone.utc).isoformat(),
 'source_manifest_sha256':science_manifest,'launcher_manifest_sha256':p['launcher_manifest_sha256'],
 'archive_bytes':len(archive),'archive_sha256':sha(archive),'local_preparation_sha256':p['local_preparation_sha256'],
 'scientific_results_or_settings_changed':False,'bootstrap_remote_code_sha256':sys.argv[1]})
with upload.open('xb') as stream:stream.write(archive);stream.flush();os.fsync(stream.fileno())
require(sha(read(upload,4*1024**2))==p['archive_sha256'],'Upload changed')
release.mkdir();launcher.mkdir()
for name,data in members.items():
 target=launcher/name;target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
for directory,dirs,names in os.walk(launcher,topdown=False):
 for name in names:(Path(directory)/name).chmod(0o444)
 Path(directory).chmod(0o555)
release.chmod(0o555)
verify_source(launcher,p['launcher_manifest_sha256'])
control.mkdir()
save(control/'bootstrap-input.json',{'profile':p['profile'],'launcher_manifest_sha256':p['launcher_manifest_sha256'],
 'source_manifest_sha256':science_manifest,'archive_bytes':len(archive),'archive_sha256':sha(archive),'local_preparation_sha256':p['local_preparation_sha256']})
save(control/'jobs.json',{})
sys.dont_write_bytecode=True;os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.path.insert(0,str(launcher/'scripts'))
import adoption_contract as contract
import adoption_ops as ops
runtime=ops.control_config(p['profile'])
require(runtime['launcher']==launcher and runtime['control']==control and runtime['source']==source and runtime['root']==output
 and runtime['manifest']==science_manifest and ops.METADATA_CASES==p['metadata_cases'],'Imported controller profile differs')
ops.verify_sources(p['profile'],p['launcher_manifest_sha256'])
# Existing scientific CPU and genuine own native/readiness chains are required;
# future Pro03 results are not fabricated or required for metadata CPU.
own=contract.verify_native_units(p['profile']);contract.absence_of_production(p['profile'])
require(pre_native['source_manifest_sha256']==own['source_manifest_sha256'] and pre_native['cpu_certificate']==own['cpu_certificate'], 'Original scientific CPU changed during staging')
save(control/'original-scientific-proof-recheck.json',{'before_first_mutation':pre_native,'after_staging':own})
original=ops.load_original(p['profile'],p['launcher_manifest_sha256'])
preflight=control/'bootstrap-preflight';preflight.mkdir()
if p['profile']=='boltz30k':
 # Original own full maxima are already authenticated above. Missing future
 # peers retain only their declared validation allowances, never a long pass.
 gate=original.storage_gate(preflight,'metadata-bootstrap-preflight',original.all_budgets())
else:
 gate=original.storage_gate(preflight,'metadata-bootstrap-preflight',False)
require(gate['status']=='passed' and gate['reserve_bytes']==4*1024**3 and gate['file_reserve']==100000,'Full original metadata admission failed')
save(control/'bootstrap-admission.json',{'status':'passed','profile':p['profile'],
 'source_manifest_sha256':science_manifest,'launcher_manifest_sha256':p['launcher_manifest_sha256'],
 'gate':gate,'future_peer_results_invented':False,'full_long_admission_claimed':False,'scope':'Original whole-campaign gate plus both metadata-control allowances; observations are not reservations.'})
# Exactly two submissions through the unchanged reviewed metadata controller.
cpu=ops.submit(p['profile'],'cpu','cpu',p['launcher_manifest_sha256'])
callback=ops.submit(p['profile'],'after-cpu','after-cpu',p['launcher_manifest_sha256'],'ended('+cpu['job_id']+')')
require(cpu['phase']=='cpu' and callback['phase']=='after-cpu' and callback['dependency']=='ended('+cpu['job_id']+')','Own-ended metadata vector differs')
jobs=contract.read(control/'jobs.json')
require(jobs.get('cpu')==cpu['job_id'] and jobs.get('after-cpu')==callback['job_id'],'Bootstrap actual job mapping differs')
# An already completed fast CPU callback may legitimately append an observer
# before this bootstrap returns. Authenticate it; never replay or overwrite it.
for label,job in jobs.items():
 parent=control/label;actual=contract.read(parent/'submission.json')
 require(actual['phase'] in ('cpu','after-cpu','observe','dispatch') and actual['outcome']=='submitted' and actual['job_id']==job
  and parent==ops.parent_path(p['profile'],label,actual['phase'])
  and actual['args']==ops.command(p['profile'],label,actual['phase'],p['launcher_manifest_sha256'],actual['dependency']), 'Concurrent metadata job mapping is unknown')
record={'status':'metadata_cpu_and_own_ended_callback_submitted','profile':p['profile'],'utc':datetime.now(timezone.utc).isoformat(),
 'source_manifest_sha256':science_manifest,'launcher_manifest_sha256':p['launcher_manifest_sha256'],
 'runtime_source_files':7,'cpu':cpu,'after_cpu':callback,'observed_control_jobs':jobs,'metadata_cpu_cases_expected':p['metadata_cases'],
 'admission':contract.file_record(control/'bootstrap-admission.json'),'original_proof_recheck':contract.file_record(control/'original-scientific-proof-recheck.json'),
 'native_or_segment_submitted_by_bootstrap':False,'allocated_metadata_pass_claimed':False,'scientific_results_or_settings_changed':False}
save(completed,record)
print(json.dumps(record))
'''


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, cap=MAX_METADATA):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/linked local metadata: ' + str(path))
    literal = path.stat()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno()); raw = stream.read(cap + 1); after = os.fstat(stream.fileno())
    last = path.stat()
    signature = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns, x.st_mode, x.st_nlink)
    shared = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_mode, x.st_nlink)
    require(len(raw) == before.st_size <= cap and signature(before) == signature(after)
            and signature(literal) == signature(last) and shared(before) == shared(literal), 'Local metadata changed/oversized')
    return raw


def strict(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate JSON field'); result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def aliases(profile):
    consumer = REPO / PROFILES[profile]['consumer']
    return {'README.md': consumer / 'README.md', 'adoption-plan.json': consumer / 'adoption-plan.json',
        'scripts/adoption_contract.py': HERE / 'adoption_contract.py', 'scripts/adoption_ops.py': HERE / 'adoption_ops.py',
        'scripts/hpc/peer_adoption_cpu.lsf': HERE / 'cpu.lsf', 'scripts/hpc/peer_adoption_stage.lsf': HERE / 'stage.lsf',
        'tests/test_peer_adoption.py': REPO / 'experiments/brd4-jq1/boltz-lr0025/runs/20261024-boltz-pro02-adoption-04/fixture_adoption.py'}



def declared_cases():
    fixture = aliases('boltz30k')['tests/test_peer_adoption.py']
    declarations = []
    for path in (HERE / 'adoption_ops.py', fixture):
        tree = ast.parse(read(path))
        values = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'METADATA_CASES' for target in node.targets)]
        require(len(values) == 1 and type(values[0]) is int and values[0] > 0, 'Missing exact positive metadata fixture count')
        declarations.append(values[0])
    require(declarations[0] == declarations[1], 'Controller and fixture case declarations differ')
    return declarations[0]


def authenticate_package(profile, directory):
    p = PROFILES[profile]; directory = Path(directory)
    cases = declared_cases()
    results = REPO / 'results/hpc' / p['tag']
    require(directory.resolve().is_relative_to(results.resolve()) and not directory.is_symlink(), 'Package outside named result scope')
    preparation_raw = read(directory / 'preparation.json'); preparation = strict(preparation_raw)
    require(preparation['status'] == 'prepared_local_only' and preparation['profile'] == profile
            and preparation['source_file_count'] == 7 and preparation['metadata_cpu_cases_planned'] == cases
            and all(preparation[k] is False for k in ('numerical_execution', 'allocated_cpu_passed', 'staged', 'submitted')),
            'No exact unsubmitted seven-member package')
    source = directory / 'source'; manifest_raw = read(source / 'source_manifest.json'); manifest = strict(manifest_raw)
    require(sha(manifest_raw) == preparation['manifest_sha256'] and manifest['profile'] == profile
            and manifest['metadata_cpu_cases'] == cases and manifest['git_revision_reference'] == REVISION,
            'Prepared metadata manifest differs')
    expected = aliases(profile)
    require(set(manifest['files']) == set(expected) and set(preparation['canonical_members']) == set(expected), 'Prepared member scope differs')
    for name, canonical in expected.items():
        raw = read(source / name)
        require(sha(raw) == manifest['files'][name] and raw == read(canonical), 'Canonical/runtime bytes differ: ' + name)
        pin = preparation['canonical_members'][name]
        require(Path(pin['path']).resolve() == canonical.resolve() and pin['bytes'] == len(raw) and pin['sha256'] == sha(raw),
                'Canonical provenance differs')
    archive = Path(preparation['archive']['path'])
    require(archive.resolve() == (directory / 'source.tar.gz').resolve(), 'Archive escaped package')
    archive_raw = read(archive, MAX_ARCHIVE)
    require(len(archive_raw) == preparation['archive']['bytes'] and sha(archive_raw) == preparation['archive']['sha256'], 'Prepared archive differs')
    observed = {}
    with tarfile.open(fileobj=io.BytesIO(archive_raw)) as stream:
        for member in stream:
            name = PurePosixPath(member.name)
            require(member.isfile() and not member.linkname and name.as_posix() == member.name
                    and member.name in set(expected) | {'source_manifest.json'} and member.name not in observed,
                    'Unsafe/duplicate source archive member')
            raw = stream.extractfile(member).read(MAX_METADATA + 1)
            require(len(raw) == member.size <= MAX_METADATA, 'Oversized/truncated source member')
            require(raw == read(source / member.name), 'Archive differs from prepared source')
            observed[member.name] = sha(raw)
    require(set(observed) == set(expected) | {'source_manifest.json'}, 'Incomplete source archive')
    payload = {'profile': profile, 'source': p['source'], 'science_manifest_sha256': p['manifest'],
        'output_root': p['root'], 'control': p['control'], 'launcher': p['launcher'],
        'launcher_manifest_sha256': sha(manifest_raw), 'member_sha256': manifest['files'],
        'archive_bytes': len(archive_raw), 'archive_sha256': sha(archive_raw), 'local_preparation_sha256': sha(preparation_raw), 'metadata_cases': cases}
    return payload, archive_raw, preparation_raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=tuple(PROFILES), required=True)
    parser.add_argument('--package', type=Path, required=True, help='Existing prepared package directory, not a source archive alone')
    args = parser.parse_args()
    payload, archive, preparation = authenticate_package(args.profile, args.package)
    ast.parse(REMOTE); ast.parse(BOOTSTRAP)
    raw_code = REMOTE.encode(); raw_payload = json.dumps(payload, allow_nan=False).encode()
    require(len(raw_code) <= 128 * 1024 and len(raw_payload) <= MAX_METADATA, 'Bootstrap metadata frame too large')
    framed = str(len(raw_code)).encode() + b'\n' + raw_code + str(len(raw_payload)).encode() + b'\n' + raw_payload + archive
    folder = REPO / 'results/hpc' / PROFILES[args.profile]['tag'] / 'bootstrap-01'
    folder.mkdir(exist_ok=False)
    argv = ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', '-c', BOOTSTRAP, sha(raw_code)]
    ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
        'dtu-hpc', 'bash -l -c ' + shlex.quote(' '.join(shlex.quote(x) for x in argv))]
    started = {'status': 'unknown_until_remote_outcome_authenticated', 'utc': datetime.now(timezone.utc).isoformat(),
        'profile': args.profile, 'args': ssh, 'launcher_manifest_sha256': payload['launcher_manifest_sha256'],
        'scientific_source_manifest_sha256': payload['science_manifest_sha256'], 'archive_bytes': len(archive), 'archive_sha256': sha(archive),
        'preparation_sha256': sha(preparation), 'remote_code_bytes': len(raw_code), 'remote_code_sha256': sha(raw_code),
        'bootstrap_sha256': sha(BOOTSTRAP.encode()), 'caller_sha256': sha(read(__file__)), 'allowed_submissions': ['cpu', 'after-cpu'], 'scientific_results_or_settings_changed': False}
    with (folder / 'started.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(started, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    (folder / 'remote-bootstrap.py').write_bytes(raw_code)
    try:
        with (folder / 'stdout.bin').open('xb') as out, (folder / 'stderr.bin').open('xb') as err:
            result = subprocess.run(ssh, input=framed, stdout=out, stderr=err, timeout=600)
    except BaseException as error:
        with (folder / 'failed-unknown.json').open('x') as stream:
            json.dump({'status': 'unknown_inspect_remote_before_any_recovery', 'error': repr(error)}, stream); stream.write('\n')
        raise
    with (folder / 'result.json').open('x') as stream:
        json.dump({'returncode': result.returncode}, stream); stream.write('\n')
    require(result.returncode == 0, 'Bootstrap failed/unknown; inspect exclusive controls, never replay')
    record = strict(read(folder / 'stdout.bin'))
    require(record['status'] == 'metadata_cpu_and_own_ended_callback_submitted' and record['profile'] == args.profile
            and record['source_manifest_sha256'] == payload['science_manifest_sha256']
            and record['launcher_manifest_sha256'] == payload['launcher_manifest_sha256']
            and record['runtime_source_files'] == 7 and record['metadata_cpu_cases_expected'] == payload['metadata_cases']
            and record['native_or_segment_submitted_by_bootstrap'] is False and record['allocated_metadata_pass_claimed'] is False,
            'Remote bootstrap result scope differs')
    p = PROFILES[args.profile]
    cpu_id = record['cpu']['job_id']
    for label, dependency in (('cpu', None), ('after-cpu', 'ended(' + cpu_id + ')')):
        value = record['cpu' if label == 'cpu' else 'after_cpu']
        parent = p['control'] + '/' + label
        expected = ['bsub', '-q', 'hpc', '-n', '2', '-R', 'span[hosts=1]', '-R', 'rusage[mem=4GB]', '-W', '00:30',
            '-J', 'mosaic-peer-' + args.profile + '-' + label, '-oo', parent + '/stdout.%J.log', '-eo', parent + '/stderr.%J.log']
        if dependency is not None:
            expected += ['-w', dependency]
        wrapper = p['launcher'] + '/scripts/hpc/' + ('peer_adoption_cpu.lsf' if label == 'cpu' else 'peer_adoption_stage.lsf')
        expected += ['bash', '-l', wrapper, p['launcher'], payload['launcher_manifest_sha256'], args.profile,
                     p['source'], p['root'], p['manifest'], p['control'], label, parent]
        require(re.fullmatch(r'[0-9]+', value['job_id']) and value['outcome'] == 'submitted' and value['phase'] == label
                and value['dependency'] == dependency and value['args'] == expected,
                'Actual metadata submission/resource/dependency vector differs')
    require(record['cpu']['job_id'] != record['after_cpu']['job_id'], 'Duplicate actual metadata job IDs')
    print(json.dumps({'status': record['status'], 'profile': args.profile, 'cpu_job_id': record['cpu']['job_id'],
        'after_cpu_job_id': record['after_cpu']['job_id'], 'allocated_metadata_pass_claimed': False}))


if __name__ == '__main__':
    main()
