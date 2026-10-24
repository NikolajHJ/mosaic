"""Root-only, one-shot read-only audit caller; prepared but not executed.

Requires the separately collected stream/deletion receipts and complete local
member-verification record. Stages only approved audit code and pins in a new
exclusive audit directory. It cannot remove, restore, score or submit anything.
Unknown outcomes require inspection; this caller has no retry/resume option.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
BACKUP = REPO / 'results/hpc/20261024-storage-offload-01/backup-01'
ROOT = '/work3/s234818/mosaic/runs/20261024-storage-offload-01'
OPERATION = 'legacy-offload-01'
MANIFEST = '9af242392c14b3fc09804f0868b49f0d650245a1f075d43c3b132d994545230d'
PIN_SPEC_SHA = 'cb8e794d5f67731ff7389c99fd3827a7ad0874f2055879166d932760ad12a9f4'
AUDIT_SHA = '441ec4d6f05a9bf8357edc5fe3b421cbc7b5893ace1f959924cf05b4540ee6c2'
MAX_BYTES = 128 * 1024**2


REMOTE = r'''
import base64,hashlib,json,os,re,stat,sys
from pathlib import Path
import xml.etree.ElementTree as ET
root=Path('/work3/s234818/mosaic/runs/20261024-storage-offload-01')
source=root/'operator-source-03';operation=root/'legacy-offload-01';limit=128*1024**2
def require(value,message):
 if not value:raise ValueError(message)
def sha(data):return hashlib.sha256(data).hexdigest()
def loads(data):
 def pairs(rows):
  result={}
  for k,v in rows:require(k not in result,'Duplicate JSON key');result[k]=v
  return result
 return json.loads(data,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def raw(path):
 path=Path(path);require(path.resolve()==path and not path.is_symlink(),'Linked metadata proof')
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 with os.fdopen(fd,'rb') as stream:
  before=os.fstat(stream.fileno());require(stat.S_ISREG(before.st_mode) and before.st_size<=limit,'Oversized/nonregular metadata proof')
  data=stream.read(limit+1);after=os.fstat(stream.fileno())
 literal=path.lstat()
 def sig(x):return(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns,x.st_mode)
 require(len(data)==before.st_size and sig(before)==sig(after)==sig(literal),'Metadata proof changed')
 return data
def proof(path,data):return {'path':str(path),'bytes':len(data),'sha256':sha(data)}
payload_raw=sys.stdin.buffer.read(limit+1)
require(len(payload_raw)<=limit and not sys.flags.optimize,'Oversized payload/optimized interpreter')
p=loads(payload_raw);require(root.is_dir() and root.resolve()==root,'Cleanup root identity changed')
manifest_raw=raw(source/'source-manifest.json');manifest=loads(manifest_raw)
require(sha(manifest_raw)==p['source_manifest_sha256']=='9af242392c14b3fc09804f0868b49f0d650245a1f075d43c3b132d994545230d','Operator seal differs')
require(len(manifest['files'])==11 and source.resolve()==source and not source.stat().st_mode&0o222,'Operator scope/mode differs')
require({x.name for x in source.iterdir()}==set(manifest['files'])|{'source-manifest.json'},'Operator member scope differs')
source_proofs=[]
for name,digest in [('source-manifest.json',sha(manifest_raw)),*manifest['files'].items()]:
 require(Path(name).name==name,'Unsafe sealed member')
 path=source/name;data=raw(path)
 require(sha(data)==digest and not path.stat().st_mode&0o222,'Operator member/mode differs')
 source_proofs.append(proof(path,data))
plan_path=operation/'approved-plan.json';plan_raw=raw(plan_path);plan=loads(plan_raw)
require(sha(plan_raw)==p['plan_sha256'] and len(plan_raw)==p['plan_bytes'],'Approved plan changed')
require(plan['operation_id']=='legacy-offload-01' and plan['cleanup_root']==str(root) and plan['operator_source_manifest_sha256']==sha(manifest_raw),'Wrong operation/source')
protection_path=root/'protection-01.json';protection_raw=raw(protection_path)
require(plan['protection_manifest']=={'path':str(protection_path),'sha256':sha(protection_raw)} and sha(protection_raw)==p['protection_sha256'],'Protection changed')
audit_raw=p['operator_verification_raw'].encode();audit=loads(audit_raw)
require(sha(audit_raw)==plan['allocated_operator_verification_sha256'] and audit['status']=='passed' and audit['cases']==32 and audit['job_id']=='29806228' and audit['source_manifest_sha256']==sha(manifest_raw),'Allocated CPU proof differs')
cpu=root/'cpu-03';cpu_proofs={k:v for k,v in audit['proofs'].items() if k.startswith(str(cpu)+'/')}
require(set(cpu_proofs)=={str(cpu/x) for x in ('submission-started.json','submission.json','stdout.29806228.log','stderr.29806228.log','payload/assessment.json','payload/invocation.json','payload/operator-contract.xml','payload/operator-contract.log','payload/storage-gate.json')},'CPU proof scope differs')
for path,pin in cpu_proofs.items():
 data=raw(path);require(len(data)==pin['bytes'] and sha(data)==pin['sha256'],'Allocated CPU bytes differ')
xml=ET.fromstring(raw(cpu/'payload/operator-contract.xml'))
require(len(list(xml.iter('testcase')))==32 and not any(list(xml.iter(k)) for k in ('failure','error','skipped')),'Raw CPU cases differ')
require(re.search(r'\b32\s+passed\b',raw(cpu/'payload/operator-contract.log').decode()),'Raw CPU count differs')
assessment=loads(raw(cpu/'payload/assessment.json'))
require(assessment['status']=='passed' and assessment['cases']==32 and assessment['job_id']=='29806228' and assessment['queue']=='hpc' and assessment['source_manifest_sha256']==sha(manifest_raw),'CPU assessment differs')
log=raw(cpu/'stdout.29806228.log').decode()
require(re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$',log,re.M)==[('29806228','Done')] and 'Successfully completed.' in log,'Same-job LSF success absent')
submission=loads(raw(cpu/'submission.json'))
expected=['bsub','-q','hpc','-n','2','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','00:30','-J','mosaic-storage-offload-cpu-03','-oo',str(cpu/'stdout.%J.log'),'-eo',str(cpu/'stderr.%J.log'),'bash','-l',str(source/'cleanup_cpu.lsf'),sha(manifest_raw)]
require(submission['job_id']=='29806228' and submission['outcome']=='submitted' and submission['args']==expected,'Actual CPU resource vector differs')
authority_proofs={}
for name,key in [('stream-complete.json','stream_receipt_sha256'),('local-verification.json','local_verification_sha256'),('deletion-started.json','deletion_started_sha256'),('deletion-complete.json','deletion_receipt_sha256'),('deletion-journal.jsonl','deletion_journal_sha256')]:
 data=raw(operation/name);require(sha(data)==p[key],'Separately authenticated authority changed: '+name)
 authority_proofs[name]=proof(operation/name,data)
complete=loads(raw(operation/'stream-complete.json'));verified=loads(raw(operation/'local-verification.json'));removed=loads(raw(operation/'deletion-complete.json'));started=loads(raw(operation/'deletion-started.json'))
require(complete['status']=='streamed_complete' and complete['all_files_verified'] is True and verified['status']=='verified_all_members' and verified['all_members_verified'] is True,'Backup authority incomplete')
require(removed['status'] in ('exact_deletion_completed','exact_deletion_completed_with_retention') and removed['recursive_delete_used'] is False and removed['scientific_settings_or_saved_retained_results_changed'] is False,'No exact completed deletion receipt')
for row in (complete,verified,removed,started):require(row['operation_id']=='legacy-offload-01' and row['plan_sha256']==sha(plan_raw),'Authority operation differs')
require(started['stream_receipt_sha256']==p['stream_receipt_sha256'] and started['local_verification_sha256']==p['local_verification_sha256'] and removed['journal_sha256']==p['deletion_journal_sha256'],'Authority chain differs')
code=base64.b64decode(p['audit_base64'],validate=True);pins=base64.b64decode(p['pin_spec_base64'],validate=True)
require(sha(code)==p['audit_sha256']=='441ec4d6f05a9bf8357edc5fe3b421cbc7b5893ace1f959924cf05b4540ee6c2' and sha(pins)==p['pin_spec_sha256']=='cb8e794d5f67731ff7389c99fd3827a7ad0874f2055879166d932760ad12a9f4','Unapproved audit code or pins')
approval_raw=p['pin_approval_raw'].encode();approval=loads(approval_raw)
require(sha(approval_raw)==p['pin_approval_sha256'] and approval['status']=='approved_for_read_only_post_deletion_audit_only' and approval['candidate_sha256']==sha(pins) and approval['no_backup_or_deletion_authority_created'] is True,'Read-only pin approval differs')
target=root/'post-audit-01';target.mkdir(exist_ok=False)
def write(path,data):
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444)
 with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
code_path=target/'post_deletion_audit.py';pin_path=target/'pin-spec.json'
write(code_path,code);write(pin_path,pins)
require({x.name for x in target.iterdir()}=={'post_deletion_audit.py','pin-spec.json'} and raw(code_path)==code and raw(pin_path)==pins,'Exclusive audit stage differs')
os.chmod(target,0o555)
args=[sys.executable,str(code_path),'--plan',str(plan_path),'--plan-sha256',sha(plan_raw),'--stream-receipt-sha256',p['stream_receipt_sha256'],'--local-verification-sha256',p['local_verification_sha256'],'--pin-spec',str(pin_path),'--pin-spec-sha256',sha(pins),'--timeout-seconds','1800']
print(json.dumps({'status':'authenticated_read_only_post_audit_bootstrap','source_manifest_sha256':sha(manifest_raw),'source_proofs':source_proofs,'cpu_proofs':cpu_proofs,'plan':proof(plan_path,plan_raw),'protection':proof(protection_path,protection_raw),'authority_proofs':authority_proofs,'pin_approval_sha256':sha(approval_raw),'audit_sha256':sha(code),'pin_spec_sha256':sha(pins),'staged_directory':str(target),'args':args,'operation_files_changed':False,'scientific_results_or_settings_changed':False}),flush=True)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.execv(sys.executable,args)
'''


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate JSON field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/linked local metadata proof: ' + str(path))
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno()); raw = stream.read(MAX_BYTES + 1); after = os.fstat(stream.fileno())
    literal = path.stat()
    signature = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
    require(len(raw) == before.st_size <= MAX_BYTES and signature(before) == signature(after) == signature(literal),
            'Local proof changed/grew beyond metadata cap')
    return raw


def read_json(path):
    raw = read(path)
    return raw, strict_json(raw)


def local_proof(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/linked local backup control')
    path = path.resolve()
    require(path.is_relative_to(BACKUP.resolve()), 'Control outside exact local backup directory')
    return path


def record(path, raw):
    return dict(path=str(path), bytes=len(raw), sha256=sha(raw))


def write_json(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def collected(path, phase, plan_sha):
    path = local_proof(path); raw, value = read_json(path)
    _, index = read_json(path.parent / 'receipt-index.json')
    require(index['status'] == 'authenticated_receipt_metadata_collected' and index['phase'] == phase
            and index['operation_id'] == OPERATION and index['source_manifest_sha256'] == MANIFEST
            and index['plan']['sha256'] == plan_sha
            and index['files'][ROOT + '/' + OPERATION + '/' + path.name]
                == dict(local_path=path.name, bytes=len(raw), sha256=sha(raw))
            and read_json(path.parent / 'result.json')[1]['returncode'] == 0,
            'Separate actual receipt collection is incomplete/different')
    return raw, value


def authorize(args):
    approval_raw, approval = read_json(HERE / 'post-audit-pin-spec-approval-01.json')
    pins_raw, pins = read_json(HERE / 'post-audit-pin-spec-candidate-01.json')
    code = read(HERE / 'post_deletion_audit.py')
    require(sha(pins_raw) == PIN_SPEC_SHA and sha(code) == AUDIT_SHA
            and approval['status'] == 'approved_for_read_only_post_deletion_audit_only'
            and approval['candidate'] == 'post-audit-pin-spec-candidate-01.json'
            and approval['candidate_sha256'] == PIN_SPEC_SHA and approval['execution_status'] == 'unexecuted'
            and approval['no_backup_or_deletion_authority_created'] is True
            and approval['no_scientific_or_storage_admission_pass_claimed'] is True
            and pins['version'] == 'mosaic-storage-offload-post-audit-pins-v1'
            and pins['max_total_hash_bytes'] == approval['maximum_hash_bytes'] == 2 * 1024**3
            and len(pins['sources']) == approval['full_mosaic_source_scopes'] == 7
            and sum(len(s['members']) for s in pins['sources']) == approval['full_mosaic_source_members'] == 3219
            and len(pins['files']) == approval['explicit_file_pins'] == 270,
            'Root-approved bounded read-only code/pin scope differs')
    _, selection = read_json(HERE / 'approved-selection-01.json')
    plan_path = local_proof(BACKUP / 'approved-plan.json'); plan_raw, plan = read_json(plan_path)
    require(selection['status'] == 'approved_for_backup_then_conditional_identical_copy_removal'
            and selection['operation_id'] == plan['operation_id'] == OPERATION
            and selection['plan_sha256'] == sha(plan_raw) and selection['plan_bytes'] == len(plan_raw)
            and plan['cleanup_root'] == ROOT and plan['operator_source_manifest_sha256'] == MANIFEST,
            'Exact approved operation/source differs')
    manifest_raw, manifest = read_json(HERE / 'operator-source-manifest-03.json')
    audit_raw, audit = read_json(HERE / 'operator-verification-03.json')
    require(sha(manifest_raw) == MANIFEST and len(manifest['files']) == 11
            and sha(read(HERE / 'verify_stream_backup_02.py')) == manifest['files']['verify_stream_backup.py']
            and sha(audit_raw) == plan['allocated_operator_verification_sha256']
            and audit['status'] == 'passed' and audit['cases'] == 32 and audit['job_id'] == '29806228'
            and audit['source_manifest_sha256'] == MANIFEST, 'Allocated operator32 proof differs')
    protection_raw, protection = read_json(HERE / 'flat-protection-approved-01.json')
    require(protection['version'] == 'mosaic-storage-offload-protection-v1'
            and sha(protection_raw) == selection['protection_sha256']
            and plan['protection_manifest'] == dict(path=ROOT + '/protection-01.json', sha256=sha(protection_raw)),
            'Approved retained-input protection differs')
    stream_path, verification_path, deletion_path = (local_proof(x) for x in
        (args.stream_receipt, args.verification, args.deletion_receipt))
    require(stream_path.name == 'stream-complete.json' and deletion_path.name == 'deletion-complete.json',
            'Wrong separate receipt filenames')
    stream_raw, stream = collected(stream_path, 'stream', sha(plan_raw))
    deletion_raw, deleted = collected(deletion_path, 'deletion', sha(plan_raw))
    started_raw, started = collected(deletion_path.parent / 'deletion-started.json', 'deletion', sha(plan_raw))
    journal_path = deletion_path.parent / 'deletion-journal.jsonl'; journal = read(journal_path)
    _, deletion_index = read_json(deletion_path.parent / 'receipt-index.json')
    require(deletion_index['files'][ROOT + '/' + OPERATION + '/deletion-journal.jsonl']
            == dict(local_path=journal_path.name, bytes=len(journal), sha256=sha(journal)),
            'Independently collected journal hash differs')
    verified_raw, verified = read_json(verification_path)
    expected = [{k: row[k] for k in ('path', 'member', 'bytes', 'sha256')} for row in plan['files']]
    require(verified['version'] == 'mosaic-storage-offload-local-verification-v1'
            and verified['status'] == 'verified_all_members' and verified['all_members_verified'] is True
            and verified['unexpected_members'] == verified['missing_members'] == [] and verified['files'] == expected
            and verified['plan_sha256'] == sha(plan_raw) and verified['plan_bytes'] == len(plan_raw)
            and verified['operation_id'] == OPERATION and verified['deleted'] is False
            and verified['scientific_passed'] is False
            and verified['remote_stream_receipt'] == record(stream_path, stream_raw)
            and verified['approved_plan'] == record(plan_path, plan_raw), 'Full local member-verification authority differs')
    require(stream['version'] == 'mosaic-storage-offload-stream-v1' and stream['status'] == 'streamed_complete'
            and stream['all_files_verified'] is True and stream['plan_bytes'] == len(plan_raw)
            and deleted['version'] == started['version'] == 'mosaic-storage-offload-delete-v1'
            and deleted['status'] in ('exact_deletion_completed', 'exact_deletion_completed_with_retention')
            and deleted['recursive_delete_used'] is False
            and deleted['scientific_settings_or_saved_retained_results_changed'] is False
            and deleted['journal_sha256'] == sha(journal) and journal.endswith(b'\n')
            and started['stream_receipt_sha256'] == sha(stream_raw)
            and started['local_verification_sha256'] == sha(verified_raw), 'Original deletion authority did not complete exactly')
    for value in (stream, verified, started, deleted):
        require(value['operation_id'] == OPERATION and value['plan_sha256'] == sha(plan_raw), 'Authority operation differs')
    for key in ('archive_sha256', 'archive_bytes', 'files_count', 'member_count', 'logical_bytes'):
        require(stream[key] == verified[key], 'Backup stream/local verification differ')
    require(stream['files_count'] == len(expected) and stream['member_count'] == len(expected) + 1
            and stream['logical_bytes'] == sum(x['bytes'] for x in expected), 'Original backup member scope differs')
    payload = dict(source_manifest_sha256=MANIFEST, plan_sha256=sha(plan_raw), plan_bytes=len(plan_raw),
        protection_sha256=sha(protection_raw), operator_verification_raw=audit_raw.decode(),
        stream_receipt_sha256=sha(stream_raw), local_verification_sha256=sha(verified_raw),
        deletion_started_sha256=sha(started_raw), deletion_receipt_sha256=sha(deletion_raw),
        deletion_journal_sha256=sha(journal), pin_approval_raw=approval_raw.decode(),
        pin_approval_sha256=sha(approval_raw), audit_sha256=AUDIT_SHA, pin_spec_sha256=PIN_SPEC_SHA,
        audit_base64=base64.b64encode(code).decode(), pin_spec_base64=base64.b64encode(pins_raw).decode())
    authority = {'stream-complete.json': stream_raw, 'local-verification.json': verified_raw,
        'deletion-started.json': started_raw, 'deletion-complete.json': deletion_raw, 'deletion-journal.jsonl': journal}
    return payload, plan, plan_raw, pins, audit, manifest, authority, deleted


def validate_output(raw, payload, plan, plan_raw, pins, audit, manifest, authority, deleted):
    lines = raw.splitlines()
    require(len(lines) == 2, 'Expected one bootstrap record and unchanged original audit JSON')
    bootstrap, value = (strict_json(line) for line in lines)
    require(bootstrap['status'] == 'authenticated_read_only_post_audit_bootstrap'
            and bootstrap['source_manifest_sha256'] == MANIFEST
            and bootstrap['audit_sha256'] == AUDIT_SHA and bootstrap['pin_spec_sha256'] == PIN_SPEC_SHA
            and bootstrap['pin_approval_sha256'] == payload['pin_approval_sha256']
            and bootstrap['staged_directory'] == ROOT + '/post-audit-01'
            and bootstrap['operation_files_changed'] is False
            and bootstrap['scientific_results_or_settings_changed'] is False, 'Remote bootstrap scope differs')
    prefix = ROOT + '/operator-source-03/'
    expected_source = {prefix + name: digest for name, digest in manifest['files'].items()}
    expected_source[prefix + 'source-manifest.json'] = MANIFEST
    require(len(bootstrap['source_proofs']) == 12
            and {x['path']: x['sha256'] for x in bootstrap['source_proofs']} == expected_source,
            'Remote original operator-member hashes differ')
    expected_cpu = {p: pin for p, pin in audit['proofs'].items() if p.startswith(ROOT + '/cpu-03/')}
    require(len(expected_cpu) == 9 and bootstrap['cpu_proofs'] == expected_cpu, 'Remote CPU32 proof bytes differ')
    require(bootstrap['plan'] == record(ROOT + '/' + OPERATION + '/approved-plan.json', plan_raw)
            and bootstrap['protection']['sha256'] == payload['protection_sha256'], 'Remote plan/protection proof differs')
    expected_args = ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', ROOT + '/post-audit-01/post_deletion_audit.py',
        '--plan', ROOT + '/' + OPERATION + '/approved-plan.json', '--plan-sha256', sha(plan_raw),
        '--stream-receipt-sha256', payload['stream_receipt_sha256'], '--local-verification-sha256', payload['local_verification_sha256'],
        '--pin-spec', ROOT + '/post-audit-01/pin-spec.json', '--pin-spec-sha256', PIN_SPEC_SHA, '--timeout-seconds', '1800']
    require(bootstrap['args'] == expected_args, 'Generated original read-only audit command differs')
    require(value['version'] == 'mosaic-storage-offload-post-deletion-audit-v1'
            and value['status'] == 'passed_read_only_post_deletion_protection_audit' and value['read_only'] is True
            and value['operation_id'] == OPERATION and value['scientific_results_or_settings_changed'] is False
            and value['files_deleted_by_this_audit'] == 0 and value['plan']['sha256'] == sha(plan_raw)
            and value['plan']['path'] == ROOT + '/' + OPERATION + '/approved-plan.json'
            and value['protection']['sha256'] == payload['protection_sha256']
            and value['pin_spec']['sha256'] == PIN_SPEC_SHA
            and value['pin_spec']['path'] == ROOT + '/post-audit-01/pin-spec.json', 'Original read-only audit failed/incomplete')
    require(set(value['controls']) == set(authority) == set(bootstrap['authority_proofs']), 'Control scope differs')
    for name, expected_raw in authority.items():
        row = value['controls'][name]; decoded = base64.b64decode(row['base64'], validate=True)
        expected = record(ROOT + '/' + OPERATION + '/' + name, expected_raw)
        require(decoded == expected_raw and {k: row[k] for k in expected} == expected
                and bootstrap['authority_proofs'][name] == expected, 'Audit raw authority bytes differ: ' + name)
    # Independently reconcile every journal intent/result and observed exact path.
    planned = {x['path']: x for x in plan['files']}; intents, terminal = set(), {}
    for line in authority['deletion-journal.jsonl'].splitlines():
        row = strict_json(line); event = row['event']
        if event == 'active_jobs_rechecked':
            require({x['job_id'] for x in row['observation']['jobs']} <= set(plan['audited_active_job_ids']), 'Journal unaudited active job')
            continue
        path = row['path']; require(path in planned, 'Journal path outside exact plan')
        if event == 'unlink_intent':
            require(path not in intents and path not in terminal and row['bytes'] == planned[path]['bytes']
                    and row['sha256'] == planned[path]['sha256'], 'Nonunique/mismatched durable unlink intent')
            intents.add(path); continue
        require(event in ('unlinked_verified_copy', 'retained_changed_or_unreadable', 'retained_or_already_absent')
                and path not in terminal, 'Duplicate/unknown journal terminal')
        if event == 'unlinked_verified_copy':
            require(path in intents and row['bytes'] == planned[path]['bytes'] and row['sha256'] == planned[path]['sha256'], 'Unlink without matching intent')
        terminal[path] = event
    observations = value['original_path_observations']
    require(len(observations) == len(planned) and {x['path'] for x in observations} == set(planned)
            and set(terminal) == set(planned), 'Exact original path coverage differs')
    for row in observations:
        require(row['journal_event'] == terminal[row['path']], 'Observed journal outcome differs')
        if row['journal_event'] == 'unlinked_verified_copy':
            require(row['current_lstat'] is None, 'Removed original reappeared')
        if row['journal_event'] == 'retained_changed_or_unreadable':
            require(isinstance(row['current_lstat'], dict), 'Retained changed original absent')
    removed = [p for p, event in terminal.items() if event == 'unlinked_verified_copy']
    account = value['removal_accounting']
    require(account['removed_files'] == deleted['removed_files'] == len(removed)
            and account['removed_logical_bytes'] == deleted['removed_logical_bytes'] == sum(planned[p]['bytes'] for p in removed)
            and account['retained_changed'] == deleted['retained_changed_or_unreadable']
                == sum(event == 'retained_changed_or_unreadable' for event in terminal.values())
            and account['already_absent_not_claimed_removed'] == deleted['already_absent_no_removal_claim']
                == sum(event == 'retained_or_already_absent' for event in terminal.values())
            and account['actual_physical_or_user_quota_bytes_reclaimed_established'] is False, 'Logical/physical accounting differs')
    expected_hashes = Counter((source['root'] + '/' + member, digest)
        for source in pins['sources'] for member, digest in source['members'].items())
    expected_hashes.update((x['path'], x['sha256']) for x in pins['files'])
    hashes = value['protected_pin_rehashes']
    require(Counter((x['path'], x['sha256']) for x in hashes) == expected_hashes
            and all(type(x['bytes']) is int and x['bytes'] >= 0 for x in hashes), 'Protected input/source pin coverage differs')
    for item in pins['files']:
        require(any(x['path'] == item['path'] and x['sha256'] == item['sha256'] and x['bytes'] == item['bytes'] for x in hashes), 'Protected explicit length differs')
    scopes = value['source_scopes']; require(len(scopes) == len(pins['sources']), 'Source scope count differs')
    for actual, expected in zip(scopes, pins['sources']):
        require(actual['root'] == expected['root'] and actual['members_rehashed'] == len(expected['members'])
                and actual['manifest']['path'] == expected['manifest_path']
                and actual['manifest']['sha256'] == expected['manifest_sha256'], 'Full Mosaic source scope differs')
    require(value['protected_bytes_rehashed'] == sum(x['bytes'] for x in hashes) <= pins['max_total_hash_bytes'], 'Protected hash budget differs')
    require(value['queue']['args'] == ['bjobs', '-noheader', '-o', 'jobid stat'] and value['queue']['returncode'] == 0
            and value['work3_quota']['args'] == ['getquota_work3.sh'] and value['work3_quota']['returncode'] == 0,
            'Fresh supported queue/quota query failed')
    for key in ('f_blocks', 'f_bfree', 'f_bavail', 'f_frsize', 'f_files', 'f_ffree', 'f_favail'):
        require(type(value['physical_filesystem_observation'][key]) is int and value['physical_filesystem_observation'][key] >= 0,
                'Malformed physical shared-filesystem observation')
    return bootstrap, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--stream-receipt', type=Path, required=True)
    parser.add_argument('--deletion-receipt', type=Path, required=True)
    args = parser.parse_args()
    payload, plan, plan_raw, pins, audit, manifest, authority, deleted = authorize(args)
    encoded = json.dumps(payload, allow_nan=False).encode(); require(len(encoded) <= MAX_BYTES, 'Oversized metadata payload')
    folder = BACKUP / 'post-deletion-audit-01'; folder.mkdir(exist_ok=False)
    with (folder / 'remote-caller.py').open('x', encoding='utf8', newline='\n') as stream: stream.write(REMOTE)
    remote = ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', '-c', REMOTE]
    command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
        'dtu-hpc', 'bash -l -c ' + shlex.quote(' '.join(shlex.quote(x) for x in remote))]
    write_json(folder / 'started.json', dict(status='unknown_until_read_only_audit_returned', utc=datetime.now(timezone.utc).isoformat(),
        args=command, caller_sha256=sha(read(__file__)), remote_code_sha256=sha(REMOTE.encode()),
        audit_code_sha256=AUDIT_SHA, pin_spec_sha256=PIN_SPEC_SHA, pin_approval_sha256=payload['pin_approval_sha256'],
        plan_sha256=sha(plan_raw), stream_receipt_sha256=payload['stream_receipt_sha256'],
        local_verification_sha256=payload['local_verification_sha256'], deletion_receipt_sha256=payload['deletion_receipt_sha256'],
        original_audit_timeout_seconds=1800, parent_timeout_seconds=1860, no_deletion_restore_or_job_authority=True))
    try:
        with (folder / 'stdout.bin').open('xb') as out, (folder / 'stderr.bin').open('xb') as err:
            result = subprocess.run(command, input=encoded, stdout=out, stderr=err, timeout=1860)
    except BaseException as error:
        write_json(folder / 'failed-unknown.json', dict(status='unknown_preserved_inspect_exclusive_audit_directory', error=repr(error)))
        raise
    stdout, stderr = read(folder / 'stdout.bin'), read(folder / 'stderr.bin')
    write_json(folder / 'result.json', dict(returncode=result.returncode,
        stdout=record(folder / 'stdout.bin', stdout), stderr=record(folder / 'stderr.bin', stderr)))
    require(result.returncode == 0, 'Read-only audit failed/unknown; preserve exact result and inspect, never replay')
    bootstrap, value = validate_output(stdout, payload, plan, plan_raw, pins, audit, manifest, authority, deleted)
    for name, row in value['controls'].items():
        with (folder / ('observed-' + name)).open('xb') as stream:
            stream.write(base64.b64decode(row['base64'], validate=True))
    write_json(folder / 'verification.json', dict(status='independently_verified_read_only_post_deletion_audit',
        utc=datetime.now(timezone.utc).isoformat(), source_manifest_sha256=MANIFEST, operator_cpu_job_id='29806228',
        audit_code_sha256=AUDIT_SHA, pin_spec_sha256=PIN_SPEC_SHA, pin_approval_sha256=payload['pin_approval_sha256'],
        plan_sha256=sha(plan_raw), stdout=record(folder / 'stdout.bin', stdout), stderr=record(folder / 'stderr.bin', stderr),
        original_paths_observed=len(value['original_path_observations']), full_source_scopes=len(value['source_scopes']),
        protected_pin_rehashes=len(value['protected_pin_rehashes']), protected_bytes_rehashed=value['protected_bytes_rehashed'],
        removal_accounting=value['removal_accounting'], queue=value['queue'], work3_quota=value['work3_quota'],
        physical_filesystem_observation=value['physical_filesystem_observation'], files_deleted_by_this_audit=0,
        scientific_or_storage_admission_pass_claimed=False, scientific_results_or_settings_changed=False))
    print(json.dumps(dict(status='independently_verified_read_only_post_deletion_audit', folder=str(folder),
        removed_files=value['removal_accounting']['removed_files'], removed_logical_bytes=value['removal_accounting']['removed_logical_bytes'],
        protected_pin_rehashes=len(value['protected_pin_rehashes']), actual_quota_savings_established=False,
        scientific_or_storage_admission_pass_claimed=False)))


if __name__ == '__main__':
    main()
