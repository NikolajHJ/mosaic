"""Seal/stage metadata-only successors to the observed one-based LSF failure.

No model/numerical execution locally or on the login host. Full allocated
fixture/startup gates and real two-client/native acceptance remain mandatory.
Existing or unknown stage/submission outcomes are never replayed.
"""
import argparse
import ast
import base64
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

OWNER=Path(__file__).resolve().parents[1]
REPO=OWNER.parents[2]
BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
HELPER=r'C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe'
PYTHON='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
MAIN_TAG='20261029-precision-continuation-26'
CHEM_TAG='20261029-chemistry-native-qualification-09'


def raw(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())


def source_packet(tag):
    root=REPO/'results/hpc'/tag/'source01';manifest=read(root/'source_manifest.json')
    members={n:(root/n).read_bytes() for n in manifest['files']}
    assert all(sha(b)==manifest['files'][n] for n,b in members.items())
    return root,manifest,members


def seal(tag,root,source,manifest,members,audit):
    out=REPO/'results/hpc'/tag/'source01';record=OWNER/'runs'/tag
    assert not out.exists(),'Existing local package is immutable; inspect before preparation'
    members['namespace-repair.json']=raw(audit)
    # Reuse an existing metadata member instead of adding a scientific module.
    inherited=read_json= json.loads(members['startup-adoption.json']) if 'startup-adoption.json' in members else None
    if inherited is not None:
        inherited['one_based_namespace_repair']=audit;members['startup-adoption.json']=raw(inherited)
        del members['namespace-repair.json']
    manifest=deepcopy(manifest);manifest['files']={n:sha(b) for n,b in sorted(members.items())}
    manifest['one_based_namespace_repair']=audit
    members['source_manifest.json']=raw(manifest)
    for n,b in members.items():
        if n.endswith('.py'):ast.parse(b,filename=n)
    out.mkdir(parents=True);record.mkdir(parents=True,exist_ok=True)
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as tar:
        for n,b in sorted(members.items()):
            p=out/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
            info=tarfile.TarInfo(n);info.size=len(b);info.mode=0o444;info.mtime=0;tar.addfile(info,io.BytesIO(b))
    archive=buffer.getvalue();(out/'source.tar.gz').write_bytes(archive)
    prep={'status':'prepared_not_allocated_pass','tag':tag,'root':root,'source':source,
          'files':len(manifest['files']),'manifest_sha256':sha(members['source_manifest.json']),
          'archive_bytes':len(archive),'archive_sha256':sha(archive),'plan_sha256':sha(members['plan.json']),
          'cpu_cases':sum(manifest['cpu_counts'].values()),'settings_changed':False,'saved_results_changed':False}
    (out/'preparation.json').write_bytes(raw(prep));(record/'preparation.json').write_bytes(raw(prep))
    (record/'source-manifest.json').write_bytes(members['source_manifest.json'])
    (record/'campaign.json').write_bytes(raw(prep));print(json.dumps(prep))


def prepare_main():
    prior,m,members=source_packet('20261029-precision-continuation-25')
    assert sha((prior/'source_manifest.json').read_bytes())=='62a3db1b97b81dd0ef6911ee93f11eb1379da794e01f86f035c0c036e1822f8b'
    root=BASE+'/nonchem-precision-continuation-26';source=BASE+'/nonchem-precision-release-26/source'
    changes={}
    for n in ('scripts/precision_continuation_ops.py','scripts/protenix_reference_worker.py',
              'scripts/dbmol_reference.py','tests/test_dbmol_reference.py'):
        old=members[n];text=old.decode().replace('nonchem-precision-continuation-25','nonchem-precision-continuation-26')
        text=text.replace('nonchem-precision-release-25','nonchem-precision-release-26').replace('mosaic-precision25-','mosaic-precision26-')
        members[n]=text.encode();changes[n]={'old_sha256':sha(old),'new_sha256':sha(members[n]),'scope':'owner namespace literals only'}
    plan=json.loads(members['plan.json'])
    plan.update(root=root,control_root=root,source_root=source)
    for cfg in plan['reference_configuration']['arms'].values():
        cfg['command_prefix']=[v.replace(BASE+'/nonchem-precision-release-25/source',source) for v in cfg['command_prefix']]
    members['plan.json']=raw(plan)
    old_cuda=members['scripts/cuda_ownership.py'];new_cuda=(OWNER/'scripts/cuda_ownership.py').read_bytes().replace(b'\r\n',b'\n')
    old_nodes={n.name:ast.dump(n) for n in ast.parse(old_cuda).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    new_nodes={n.name:ast.dump(n) for n in ast.parse(new_cuda).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert old_nodes.keys()==new_nodes.keys()
    assert [n for n in old_nodes if old_nodes[n]!=new_nodes[n]]==['resolve_visibility']
    members['scripts/cuda_ownership.py']=new_cuda
    portable=ast.parse((OWNER/'tests/test_cuda_visibility.py').read_bytes())
    addition='\n\n'+'\n\n'.join(ast.unparse(n) for n in portable.body if isinstance(n,ast.FunctionDef))+'\n'
    members['tests/test_cuda_ownership.py']+=addition.encode()
    m['cpu_counts']['tests.test_cuda_ownership']+=14
    m['parent_sources'].append({'path':BASE+'/nonchem-precision-release-25/source','files':50,
                               'manifest_sha256':'62a3db1b97b81dd0ef6911ee93f11eb1379da794e01f86f035c0c036e1822f8b'})
    audit={'version':'observed-one-based-LSF-routing-repair-v1','original_parent_source':m['parent_sources'][-1],
           'changed_metadata':changes,'cuda_only_changed_function':'resolve_visibility',
           'native_GPU_two_client_smoke_unchanged':True,'scientific_driver_hooks_identical':True,
           'new_portable_metadata_cases':14,'old_failed_jobs':['29847826','29847828','29847869'],
           'one_based_requires_complete_positive_slot_map':True,'GPU_ordinals_guessed':False,
           'all_scientific_checks_preserved':True,'settings_changed':False,'saved_results_changed':False}
    assert len(members)==50 and sum(m['cpu_counts'].values())==958
    seal(MAIN_TAG,root,source,m,members,audit)


def prepare_native():
    main=read(REPO/'results/hpc'/MAIN_TAG/'source01/preparation.json')
    prior,m,members=source_packet('20261029-chemistry-native-qualification-08')
    root=BASE+'/nonchem-precision-continuation-26/chemistry-native-qualification-09'
    source=BASE+'/nonchem-precision-continuation-26/chemistry-native-qualification-release-09/source'
    changes={}
    for n in ('scripts/chemistry_native_qualification.py','scripts/chemistry_qualification_ops.py',
              'scripts/manage_chemistry_qualification.py','tests/test_chemistry_native_qualification.py',
              'scripts/chemistry_endpoint_stage.lsf'):
        old=members[n];text=old.decode().replace('nonchem-precision-continuation-25','nonchem-precision-continuation-26')
        text=text.replace('nonchem-precision-release-25','nonchem-precision-release-26')
        text=text.replace('chemistry-native-qualification-release-08','chemistry-native-qualification-release-09')
        text=text.replace('chemistry-native-qualification-08','chemistry-native-qualification-09')
        text=text.replace('mosaic-chem-qualification08-','mosaic-chem-qualification09-')
        text=text.replace('62a3db1b97b81dd0ef6911ee93f11eb1379da794e01f86f035c0c036e1822f8b',main['manifest_sha256'])
        members[n]=text.encode();changes[n]={'old_sha256':sha(old),'new_sha256':sha(members[n]),'scope':'owner/current capacity literals only'}
    plan=json.loads(members['plan.json']);plan['output_root']=root
    cap={'path':main['source'],'files':50,'manifest_sha256':main['manifest_sha256']}
    plan['current_capacity_source']=cap;plan['current_control_cap_scope']='Inside26 entire funded2GiB control tree; no optimizer admission'
    plan['parent_sources'] += [cap,{'path':BASE+'/nonchem-precision-continuation-25/chemistry-native-qualification-release-08/source',
        'files':13,'manifest_sha256':'82eb292f66ee8f8dcd86d9480775c480c1b086a1192d6ac724fbdd771a08438e'}]
    members['plan.json']=raw(plan);m['parent_sources']=plan['parent_sources']
    # Exactly13 files: use protocol.md for the additional metadata audit text.
    audit={'version':'chemistry-native-namespace09-v1','changed_metadata':changes,'current_capacity':cap,
           'numeric_algorithms_changed':False,'all_scientific_checks_preserved':True,'old_failed_job':'29847869',
           'settings_changed':False,'saved_results_changed':False}
    members['startup-adoption.json']=raw(audit)
    # seal() folds its audit into this metadata member, preserving13 members.
    del members['protocol.md']
    assert len(members)==13 and sum(m['cpu_counts'].values())==43
    seal(CHEM_TAG,root,source,m,members,audit)


STAGE=r'''
from pathlib import Path,PurePosixPath
import base64,hashlib,io,json,os,subprocess,sys,tarfile
request=json.loads(sys.stdin.buffer.read());p=request['preparation'];root=Path(p['root']);source=Path(p['source'])
B=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
assert p['tag'] in ('20261029-precision-continuation-26','20261029-chemistry-native-qualification-09')
main=p['tag'].endswith('continuation-26')
assert root==(B/'nonchem-precision-continuation-26' if main else B/'nonchem-precision-continuation-26/chemistry-native-qualification-09')
assert source==(B/'nonchem-precision-release-26/source' if main else B/'nonchem-precision-continuation-26/chemistry-native-qualification-release-09/source')
assert not root.exists() and not source.parent.exists(),'Existing/unknown outcome; never blindly repeat'
if main:
 for label,jid in [('protenix-validate-20000','29847826'),('dbmol-validate-07500','29847828')]:
  old=B/'nonchem-precision-continuation-25'/label
  text=' '.join((old/('stdout.'+jid+'.log')).read_text().split())
  assert text.count('Subject: Job '+jid+':')==1 and 'Exited with exit code 1.' in text and 'queue <gpul40s>' in text
  failure=json.loads((old/'cuda-ownership/assessment.json').read_bytes())
  assert failure['status']=='failed' and failure['optimizer_updates']==0 and not failure['records']
  assert not (old/'payload/study').exists(),'Scientific output exists; inspect before replacement'
data=base64.b64decode(request['archive'],validate=True)
assert len(data)==p['archive_bytes'] and hashlib.sha256(data).hexdigest()==p['archive_sha256']
members={}
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
 for item in tar:
  rel=PurePosixPath(item.name);assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts and '\\' not in item.name and item.name not in members
  members[item.name]=tar.extractfile(item).read()
manifest=json.loads(members['source_manifest.json']);assert hashlib.sha256(members['source_manifest.json']).hexdigest()==p['manifest_sha256']
assert set(members)==set(manifest['files'])|{'source_manifest.json'} and len(manifest['files'])==p['files']
assert all(hashlib.sha256(members[n]).hexdigest()==s for n,s in manifest['files'].items())
root.mkdir();source.mkdir(parents=True)
for n,b in members.items():
 target=source/n;target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as f:f.write(b)
 target.chmod(0o444)
for folder in sorted([q for q in source.rglob('*') if q.is_dir()],reverse=True):folder.chmod(0o555)
source.chmod(0o555);sys.path.insert(0,str(source/'scripts'));os.environ['PYTHONDONTWRITEBYTECODE']='1'
if main:
 import precision_continuation_ops as op
 assert op.ROOT==root and op.SOURCE==source
 op.verify(source,p['manifest_sha256']);op.external(manifest,json.loads(members['plan.json']))
 subprocess.run(['bash','-n',str(source/'scripts/precision_continuation_stage.lsf')],check=True)
 gate=root/'preflight';gate.mkdir();storage=op.storage(gate,'preflight')
 cpu=op.submit('cpu',p['manifest_sha256']);callback=op.submit('after-cpu',p['manifest_sha256'],dependency='ended('+cpu['job_id']+')')
else:
 import chemistry_qualification_ops as adapter
 op=adapter.inherited();assert op.ROOT==root and op.SOURCE==source
 op.external(op.verify(source,p['manifest_sha256']))
 subprocess.run(['bash','-n',str(source/'scripts/chemistry_endpoint_stage.lsf')],check=True)
 gate=root/'preflight';gate.mkdir();storage=op.storage(gate,'preflight')
 cpu=op.submit('cpu',p['manifest_sha256']);callback=op.submit('after-cpu',p['manifest_sha256'],'ended('+cpu['job_id']+')')
result={'status':'CPU_and_conditional_controller_submitted_not_passed','preparation':p,'cpu':cpu,'callback':callback,'whole_storage':storage,'settings_changed':False,'saved_results_changed':False}
op.write(root/'initial-submissions.json',result);print(json.dumps(result))
'''


def stage(tag):
    local=REPO/'results/hpc'/tag;packet=local/'source01';p=read(packet/'preparation.json');archive=(packet/'source.tar.gz').read_bytes()
    assert len(archive)==p['archive_bytes'] and sha(archive)==p['archive_sha256']
    out=local/datetime.now(timezone.utc).strftime('stage-%Y%m%dT%H%M%SZ');out.mkdir()
    (out/'program.py').write_bytes(STAGE.encode());(out/'started.json').write_bytes(raw({'status':'unknown_until_receipt','preparation':p}))
    args=[HELPER,'exec','--timeout','360','bash -lc '+shlex.quote(PYTHON+' -B -c '+shlex.quote(STAGE))]
    request=raw({'preparation':p,'archive':base64.b64encode(archive).decode()})
    run=subprocess.run(args,input=request,capture_output=True,timeout=390)
    (out/'stdout.json').write_bytes(run.stdout);(out/'stderr.txt').write_bytes(run.stderr);(out/'transport.json').write_bytes(raw({'returncode':run.returncode}))
    print(out)
    if run.returncode:print(run.stderr.decode());raise SystemExit(run.returncode)
    result=json.loads(run.stdout);(OWNER/'runs'/tag/'submission.json').write_bytes(raw(result))
    campaign=read(OWNER/'runs'/tag/'campaign.json');campaign.update(status=result['status'],cpu_job=result['cpu']['job_id'],callback_job=result['callback']['job_id'])
    (OWNER/'runs'/tag/'campaign.json').write_bytes(raw(campaign));print(json.dumps({'CPU':result['cpu']['job_id'],'callback':result['callback']['job_id'],'status':result['status']}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare-main','stage-main','prepare-native','stage-native'))
    a=p.parse_args()
    if a.action=='prepare-main':prepare_main()
    elif a.action=='prepare-native':prepare_native()
    else:stage(MAIN_TAG if a.action=='stage-main' else CHEM_TAG)


if __name__=='__main__':main()
