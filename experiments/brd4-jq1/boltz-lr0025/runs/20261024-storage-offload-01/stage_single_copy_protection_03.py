"""Unexecuted one-shot source04/CPU44-authenticated protection03 staging only.

No backup directory, archive, job, deletion or scientific change is created.
The old plans/protection/partials stay unchanged. Root alone executes this caller.
"""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
RESULTS=REPO/'results/hpc/20261024-storage-offload-01'
ROOT='/work3/s234818/mosaic/runs/20261024-storage-offload-01'
REMOTE="\nimport hashlib,json,os,re,sys,socket,stat\nfrom pathlib import Path\nimport xml.etree.ElementTree as ET\nroot=Path('/work3/s234818/mosaic/runs/20261024-storage-offload-01')\nsource=root/'operator-source-04'\ninput_raw=sys.stdin.buffer.read(64*1024**2+1);assert len(input_raw)<=64*1024**2\npayload=json.loads(input_raw)\ndef raw(path):\n p=Path(path);assert p.is_file() and not p.is_symlink() and p.resolve()==p\n fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)\n with os.fdopen(fd,'rb') as f:\n  before=os.fstat(f.fileno());assert stat.S_ISREG(before.st_mode) and before.st_size<=64*1024**2\n  data=f.read(64*1024**2+1);after=os.fstat(f.fileno())\n literal=p.stat()\n def sig(x):return(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns,x.st_mode)\n assert len(data)==before.st_size and sig(before)==sig(after)==sig(literal)\n return data\ndef sha(data):return hashlib.sha256(data).hexdigest()\nmanifest=raw(source/'source-manifest.json')\nassert sha(manifest)==payload['source_manifest_sha256']=='8a6420047f7260eedd1219c4313c835dae4bb2a13cd39d4914f9e87abcd5bafe'\nassert not (source/'source-manifest.json').stat().st_mode&0o222\nm=json.loads(manifest);assert len(m['files'])==11\nassert set(p.name for p in source.iterdir())==set(m['files'])|{'source-manifest.json'}\nfor name,digest in m['files'].items():\n p=source/name;assert sha(raw(p))==digest and not p.stat().st_mode&0o222\nassert not source.stat().st_mode&0o222\nfor name,pin in payload['cpu_proofs'].items():\n data=raw(name);assert len(data)==pin['bytes'] and sha(data)==pin['sha256']\nparent=root/'cpu-04'\nexpected_proofs={str(parent/name) for name in ('submission-started.json','submission.json','stdout.29806439.log','stderr.29806439.log','payload/assessment.json','payload/invocation.json','payload/operator-contract.xml','payload/operator-contract.log','payload/storage-gate.json')}\nassert set(payload['cpu_proofs'])==expected_proofs\nsubmission=json.loads(raw(parent/'submission.json'))\nexpected=['bsub','-q','hpc','-n','2','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','00:30','-J','mosaic-storage-offload-cpu-04','-oo',str(parent/'stdout.%J.log'),'-eo',str(parent/'stderr.%J.log'),'bash','-l',str(source/'cleanup_cpu.lsf'),sha(manifest)]\nassert submission['job_id']=='29806439' and submission['outcome']=='submitted' and submission['args']==expected\nassert submission['source_manifest_sha256']==sha(manifest)\nassessment=json.loads(raw(parent/'payload/assessment.json'))\nassert assessment['status']=='passed' and assessment['cases']==44 and assessment['job_id']=='29806439' and assessment['queue']=='hpc'\nassert assessment['source_manifest_sha256']==sha(manifest)\nfor name,pin in m['parent_adoption']['proofs'].items():\n data=raw(name);assert len(data)==pin['bytes'] and sha(data)==pin['sha256']\nrows=list(ET.fromstring(raw(parent/'payload/operator-contract.xml')).iter('testcase'))\nassert len(rows)==44 and all(not list(row) and row.attrib['classname'].split('.')[-1]=='fixture_backup_contract' for row in rows)\nassert json.loads(raw(parent/'payload/assessment.json'))['status']=='passed'\nlog=raw(parent/'stdout.29806439.log').decode()\nassert re.findall(r'^Subject:\\s*Job\\s+(\\d+):[^\\r\\n]*\\b(Done|Exited)\\s*$',log,re.M)==[('29806439','Done')]\nassert 'Successfully completed.' in log\nplan_raw=payload['plan_raw'].encode('utf8');plan=json.loads(plan_raw)\nassert sha(plan_raw)==payload['plan_sha256'] and len(plan_raw)==payload['plan_bytes']\nassert plan['operation_id']=='legacy-offload-03' and plan['cleanup_root']==str(root)\nassert plan['operator_source_manifest_sha256']==sha(manifest) and plan['allocated_operator_job_id']=='29806439' and plan['allocated_operator_cases']==44\nassert plan['selection_policy']=='complete_single_link_inode_groups_only'\nassert plan['protection_manifest']=={'path':str(root/'protection-03.json'),'sha256':payload['protection_sha256']}\nassert plan['audited_active_job_ids']==payload['audited_active_job_ids'] and plan['candidate_owner_job_ids']==[]\nfor name,pin in payload['retirement_files'].items():\n data=raw(name);assert len(data)==pin['bytes'] and sha(data)==pin['sha256']\nretirement_raw=raw(root/'stream-retirement-complete-02.json')\nassert len(retirement_raw)==1132 and sha(retirement_raw)=='3324ff5a10fe9e461ab1946c4072859a8a09d4ffce72eca9d09cebcf2a9a2da2'\nretired=json.loads(retirement_raw)\nassert retired['status']=='intentionally_stopped_own_incomplete_backup' and retired['pid']==177888 and retired['uid']==306121\nassert retired['producer_no_longer_running'] is True and retired['remote_originals_retained'] is True\nassert retired['backup_verified'] is False and retired['original_files_deleted']==0\nassert retired['hostname']=='hpclogin1' and retired['signal']=='SIGTERM'\nassert retired['expected_argv']==['/work3/s234818/mosaic/envs/protenix-tiny/bin/python',str(source/'stream_backup.py'),'--plan-stdin','--plan-sha256','5fc74e3eeb3eedb2df5dfec541bb62b31e2b03dc01b8d0dd20eb3ad277dff6ea','--receipt',str(root/'legacy-offload-02/stream-complete.json')]\nassert {p.name for p in (root/'legacy-offload-02').iterdir()}==set(retired['operation_entries'])=={'active-jobs-before-stream.json','approved-plan.json','stream-started.json'}\nif socket.gethostname().split('.')[0]==retired['hostname']:\n proc=Path('/proc')/str(retired['pid'])/'cmdline'\n if proc.exists():assert proc.read_bytes().split(b'\\0')[:-1]!=[x.encode() for x in retired['expected_argv']],'Exact old producer still running'\nprotection=payload['protection_raw'].encode('utf8')\nassert sha(protection)==payload['protection_sha256']\nassert json.loads(protection)['version']=='mosaic-storage-offload-protection-v1'\nassert root.is_dir() and root.resolve()==root\nassert not os.path.lexists(root/'legacy-offload-03'),'Existing backup operation/unknown outcome; inspect'\ntarget=root/'protection-03.json'\nassert not os.path.lexists(target),'Existing staged protection/unknown outcome; inspect'\nos.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True;sys.path.insert(0,str(source))\nimport offload_protocol as protocol\nobservation=protocol.active_jobs()\nassert {row['job_id'] for row in observation['jobs']}<=set(payload['audited_active_job_ids'])\nfd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444)\nwith os.fdopen(fd,'wb') as stream:\n stream.write(protection);stream.flush();os.fsync(stream.fileno())\nassert raw(target)==protection\nprint(json.dumps({'status':'approved_protection_staged_only','source_manifest_sha256':sha(manifest),\n 'cpu_job_id':'29806439','cpu_cases':len(rows),'protection_path':str(target),\n 'protection_bytes':len(protection),'protection_sha256':sha(protection),'queue':observation,\n 'retired_operation02_sha256':sha(retirement_raw),'operation_directory_created':False,'backup_or_deletion_started':False}))\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-evidence',type=Path,required=True)
    args=parser.parse_args()
    helper=HERE/'approve_single_copy_plan_03_recovery03.py'
    assert hashlib.sha256(helper.read_bytes()).hexdigest()=='aa33b3a442307125c6f09b4c9a3acb39f8e0cd33bf02ae7311c497e07f7c6600'
    spec=importlib.util.spec_from_file_location('plan03_metadata_only',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    retired=module.retired_proof(HERE/'stream-retirement-02.json')
    approval_raw,approval=module.load(HERE/'approved-selection-03.json')
    protection_raw,protection=module.load(HERE/'flat-protection-approved-03.json')
    plan_raw,plan=module.load(RESULTS/'backup-03/approved-plan.json')
    assert approval['status']=='approved_for_backup_then_conditional_identical_copy_removal'
    assert approval['operation_id']==plan['operation_id']=='legacy-offload-03' and approval['plan_sha256']==module.sha(plan_raw) and approval['plan_bytes']==len(plan_raw)
    assert module.sha(protection_raw)==approval['protection_sha256'] and plan['protection_manifest']=={'path':ROOT+'/protection-03.json','sha256':module.sha(protection_raw)}
    assert plan['retired_previous_operation']==retired and plan['selection_policy']==approval['selection_policy']=='complete_single_link_inode_groups_only'
    assert plan['selection_unchanged_from_original_approved_plan'] is False
    binder=HERE/'bind_offload_operation_02.py'
    assert module.sha(binder.read_bytes())=='368142296f3c95e15e5b6e853e51272794fa3851a1bcb8899b2c97a37d845444'
    spec=importlib.util.spec_from_file_location('cpu04_metadata_only',binder)
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    job,audit_raw,bundle_raw=verifier.prove_actual_cpu(args.cpu_evidence)
    assert job=='29806439' and module.sha(audit_raw)==plan['allocated_operator_verification_sha256']
    audit=json.loads(audit_raw);prefix=ROOT+'/cpu-04/'
    proofs={name:pin for name,pin in audit['proofs'].items() if name.startswith(prefix)}
    assert len(proofs)==9
    _,retirement=module.load(HERE/'stream-retirement-02.json')
    payload={'source_manifest_sha256':audit['source_manifest_sha256'],'cpu_proofs':proofs,
        'protection_raw':protection_raw.decode(),'protection_sha256':module.sha(protection_raw),
        'plan_raw':plan_raw.decode(),'plan_sha256':module.sha(plan_raw),'plan_bytes':len(plan_raw),
        'audited_active_job_ids':plan['audited_active_job_ids'],
        'retirement_files':{name:{k:pin[k] for k in ('bytes','sha256')} for name,pin in retirement['files'].items()}}
    ast.parse(REMOTE)
    folder=RESULTS/'protection-staging-03';folder.mkdir(exist_ok=False)
    (folder/'remote-caller.py').write_text(REMOTE,encoding='utf8',newline='\n')
    argv=['/work3/s234818/mosaic/envs/protenix-tiny/bin/python','-c',REMOTE]
    ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -c '+shlex.quote(' '.join(shlex.quote(x) for x in argv))]
    (folder/'started.json').write_text(json.dumps({'outcome':'unknown','args':ssh,'utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':module.sha(plan_raw),'retirement_sha256':retired['sha256']})+'\n')
    try:
        with (folder/'stdout.bin').open('xb') as out,(folder/'stderr.bin').open('xb') as err:
            result=subprocess.run(ssh,input=json.dumps(payload).encode(),stdout=out,stderr=err,timeout=120)
    except BaseException as error:
        (folder/'failed-unknown.json').write_text(json.dumps({'error':repr(error),'outcome':'unknown_inspect_remote'})+'\n')
        raise
    (folder/'result.json').write_text(json.dumps({'returncode':result.returncode})+'\n')
    print((folder/'stdout.bin').read_text());print((folder/'stderr.bin').read_text())
    assert result.returncode==0


if __name__=='__main__':main()
