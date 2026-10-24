"""Root-only one-shot caller for the original validated exact-file deleter.

Never execute before a completed archive, standalone validated full-member
verification and independently fetched remote stream receipt all agree.
Unknown outcomes are preserved; this caller has no retry or recursive delete.
"""
import argparse
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
BACKUP = REPO / 'results/hpc/20261024-storage-offload-01/backup-03'
ROOT = '/work3/s234818/mosaic/runs/20261024-storage-offload-01'
MANIFEST = '8a6420047f7260eedd1219c4313c835dae4bb2a13cd39d4914f9e87abcd5bafe'
MAX_BYTES = 64 * 1024 ** 2

REMOTE = "\nimport hashlib,json,os,re,subprocess,sys\nfrom pathlib import Path\nimport xml.etree.ElementTree as ET\nroot=Path('/work3/s234818/mosaic/runs/20261024-storage-offload-01')\nsource=root/'operator-source-04';operation=root/'legacy-offload-03';maximum=64*1024**2\npayload_raw=sys.stdin.buffer.read(maximum+1)\nassert len(payload_raw)<=maximum and not sys.flags.optimize\npayload=json.loads(payload_raw)\ndef sha(data):return hashlib.sha256(data).hexdigest()\ndef raw(path):\n path=Path(path);assert path.is_file() and not path.is_symlink() and path.resolve()==path\n with path.open('rb') as stream:data=stream.read(maximum+1)\n assert len(data)<=maximum,'Oversized metadata proof'\n return data\ndef write(path,data):\n fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444)\n with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())\ndef exclusive(path,value):write(path,(json.dumps(value,indent=2,sort_keys=True)+'\\n').encode())\nassert root.is_dir() and root.resolve()==root and source.is_dir() and source.resolve()==source\nmanifest_raw=raw(source/'source-manifest.json')\nassert sha(manifest_raw)==payload['source_manifest_sha256']=='8a6420047f7260eedd1219c4313c835dae4bb2a13cd39d4914f9e87abcd5bafe'\nmanifest=json.loads(manifest_raw);assert len(manifest['files'])==11\nassert {p.name for p in source.iterdir()}==set(manifest['files'])|{'source-manifest.json'}\nassert not source.stat().st_mode&0o222\nfor name in ['source-manifest.json',*manifest['files']]:\n assert Path(name).name==name\n path=source/name;data=raw(path);assert not path.stat().st_mode&0o222\n assert sha(data)==(sha(manifest_raw) if name=='source-manifest.json' else manifest['files'][name])\nassert operation.is_dir() and operation.resolve()==operation\nassert {p.name for p in operation.iterdir()}=={'approved-plan.json','stream-started.json','stream-complete.json','active-jobs-before-stream.json','active-jobs-after-stream.json'},'Deletion/local-verification/caller/unknown operation state exists; inspect, never replay'\nplan_path=operation/'approved-plan.json';plan_raw=raw(plan_path)\nassert sha(plan_raw)==payload['plan_sha256'] and len(plan_raw)==payload['plan_bytes']\nplan=json.loads(plan_raw)\nassert plan['operation_id']=='legacy-offload-03' and plan['cleanup_root']==str(root)\nassert plan['operator_source_manifest_sha256']==sha(manifest_raw)\nassert plan['protection_manifest']=={'path':str(root/'protection-03.json'),'sha256':payload['protection_sha256']}\nassert sha(raw(root/'protection-03.json'))==payload['protection_sha256']\naudit_raw=payload['operator_verification_raw'].encode()\nassert sha(audit_raw)==plan['allocated_operator_verification_sha256']\naudit=json.loads(audit_raw)\nassert audit['status']=='passed' and audit['cases']==44 and audit['job_id']=='29806439' and audit['source_manifest_sha256']==sha(manifest_raw)\ncpu=root/'cpu-04';proofs={p:v for p,v in audit['proofs'].items() if p.startswith(str(cpu)+'/')}\nassert set(proofs)=={str(cpu/p) for p in ('submission-started.json','submission.json','stdout.29806439.log','stderr.29806439.log','payload/assessment.json','payload/invocation.json','payload/operator-contract.xml','payload/operator-contract.log','payload/storage-gate.json')}\nfor path,pin in proofs.items():\n data=raw(path);assert len(data)==pin['bytes'] and sha(data)==pin['sha256']\nxml=ET.fromstring(raw(cpu/'payload/operator-contract.xml'))\nassert len(list(xml.iter('testcase')))==44 and not any(list(xml.iter(k)) for k in ('failure','error','skipped'))\nassert re.search('\\\\b44\\\\s+passed\\\\b',raw(cpu/'payload/operator-contract.log').decode())\nassessment=json.loads(raw(cpu/'payload/assessment.json'))\nassert assessment['status']=='passed' and assessment['cases']==44 and assessment['job_id']=='29806439' and assessment['queue']=='hpc' and assessment['source_manifest_sha256']==sha(manifest_raw)\nlog=raw(cpu/'stdout.29806439.log').decode()\nassert re.findall(r'^Subject:\\s*Job\\s+(\\d+):[^\\r\\n]*\\b(Done|Exited)\\s*$',log,re.M)==[('29806439','Done')] and 'Successfully completed.' in log\nsubmission=json.loads(raw(cpu/'submission.json'))\nexpected=['bsub','-q','hpc','-n','2','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','00:30','-J','mosaic-storage-offload-cpu-04','-oo',str(cpu/'stdout.%J.log'),'-eo',str(cpu/'stderr.%J.log'),'bash','-l',str(source/'cleanup_cpu.lsf'),sha(manifest_raw)]\nassert submission['job_id']=='29806439' and submission['outcome']=='submitted' and submission['args']==expected\ncomplete_raw=raw(operation/'stream-complete.json')\nassert sha(complete_raw)==payload['stream_receipt_sha256']\nverified_raw=payload['local_verification_raw'].encode()\nassert sha(verified_raw)==payload['local_verification_sha256']\ncomplete=json.loads(complete_raw);verified=json.loads(verified_raw)\nassert complete['archive_sha256']==verified['archive_sha256']==payload['archive_sha256']\nassert complete['archive_bytes']==verified['archive_bytes']==payload['archive_bytes']\nsys.dont_write_bytecode=True;sys.path.insert(0,str(source))\nimport offload_protocol as protocol\n# The unchanged original deleter performs FULL plan shape/protection validation\n# once before any unlink. This caller does not duplicate that expensive walk.\nprotocol.verify_delete_authority(plan,plan_raw,complete,verified)\nobservation=protocol.check_active(plan,protocol.active_jobs())\nexclusive(operation/'removal-caller-started.json',{'status':'unknown_until_original_deleter_completed','plan_sha256':payload['plan_sha256'],'source_manifest_sha256':sha(manifest_raw),'local_verification_sha256':sha(verified_raw),'stream_receipt_sha256':sha(complete_raw),'archive_sha256':payload['archive_sha256'],'archive_bytes':payload['archive_bytes'],'active_jobs_before':observation,'scope':'One call to unchanged validated delete_verified.py; no guard bypass or retry'})\nwrite(operation/'local-verification.json',verified_raw)\nargs=[sys.executable,str(source/'delete_verified.py'),'--plan',str(plan_path),'--plan-sha256',payload['plan_sha256'],'--stream-receipt',str(operation/'stream-complete.json'),'--stream-receipt-sha256',sha(complete_raw),'--local-verification',str(operation/'local-verification.json'),'--local-verification-sha256',sha(verified_raw)]\ntry:\n result=subprocess.run(args,capture_output=True,timeout=plan['timeout_seconds']+60)\n write(operation/'removal-caller-stdout.bin',result.stdout);write(operation/'removal-caller-stderr.bin',result.stderr)\n record={'status':'original_deleter_returned','returncode':result.returncode,'args':args,'stdout_sha256':sha(result.stdout),'stderr_sha256':sha(result.stderr),'plan_sha256':payload['plan_sha256']}\n exclusive(operation/'removal-caller-result.json',record)\n print(json.dumps(record))\n assert result.returncode==0,'Original deleter failed/partially completed: inspect durable journal; no replay'\n completion=json.loads(raw(operation/'deletion-complete.json'))\n assert completion['operation_id']=='legacy-offload-03' and completion['plan_sha256']==payload['plan_sha256']\nexcept BaseException as error:\n exclusive(operation/'removal-caller-failed.json',{'status':'failed_or_unknown_inspect_original_journal','error':repr(error),'plan_sha256':payload['plan_sha256']})\n raise\n"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or linked local metadata proof')
    with path.open('rb') as stream:
        raw = stream.read(MAX_BYTES + 1)
    require(len(raw) <= MAX_BYTES, 'Oversized local metadata proof')
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate metadata field')
            result[key] = value
        return result
    return raw, json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def local_file(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(BACKUP.resolve()),
            'Required exact backup proof is outside its dedicated local folder')
    return path.resolve()


def archive_record(path):
    before = path.stat()
    signature = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns, x.st_mode, x.st_nlink)
    shared = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_mode, x.st_nlink)
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        require(not path.is_symlink() and shared(opened) == shared(before), 'Archive changed before rehash')
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        after = os.fstat(stream.fileno())
    literal = path.stat()
    # Windows fstat/stat creation-time precision differs. Both channels must
    # remain independently exact across the whole hash; shared identity fields
    # still cross-match. No content, length, race or remote guard is removed.
    require(not path.is_symlink() and signature(opened) == signature(after)
            and signature(before) == signature(literal)
            and shared(opened) == shared(before) == shared(literal), 'Archive changed during authorization rehash')
    return {'path': str(path), 'bytes': before.st_size, 'sha256': digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--stream-receipt', type=Path, required=True)
    args = parser.parse_args()
    archive, verification, receipt = (local_file(value) for value in
                                     (args.archive, args.verification, args.stream_receipt))
    approval_raw, approval = read_json(HERE / 'approved-selection-03.json')
    plan_path = local_file(BACKUP / 'approved-plan.json'); plan_raw, plan = read_json(plan_path)
    require(approval['status'] == 'approved_for_backup_then_conditional_identical_copy_removal'
            and sha(plan_raw) == approval['plan_sha256'] and len(plan_raw) == approval['plan_bytes']
            and plan['operation_id'] == approval['operation_id'] == 'legacy-offload-03'
            and plan['cleanup_root'] == ROOT and plan['operator_source_manifest_sha256'] == MANIFEST,
            'Exact approved operation/source differs')
    manifest_raw, manifest = read_json(HERE / 'operator-source-manifest-04.json')
    require(sha(manifest_raw) == MANIFEST and sha((HERE / 'verify_stream_backup_02.py').read_bytes())
            == manifest['files']['verify_stream_backup.py'], 'Validated standalone verifier changed')
    audit_raw, audit = read_json(HERE / 'operator-verification-04.json')
    require(sha(audit_raw) == plan['allocated_operator_verification_sha256'] and audit['status'] == 'passed'
            and audit['cases'] == 44 and audit['job_id'] == '29806439' and audit['source_manifest_sha256'] == MANIFEST,
            'Actual allocated operator proof differs')
    receipt_raw, completed = read_json(receipt); verified_raw, verified = read_json(verification)
    index_raw, index = read_json(receipt.parent / 'receipt-index.json')
    collected = index['files'][ROOT + '/legacy-offload-03/stream-complete.json']
    require(index['status'] == 'authenticated_receipt_metadata_collected' and index['phase'] == 'stream'
            and index['source_manifest_sha256'] == MANIFEST and index['plan']['sha256'] == sha(plan_raw)
            and collected == {'local_path': receipt.name, 'bytes': len(receipt_raw), 'sha256': sha(receipt_raw)},
            'Remote completion receipt was not independently fetched/authenticated')
    require(read_json(receipt.parent / 'result.json')[1]['returncode'] == 0, 'Receipt collection failed')
    expected_files = [{key: row[key] for key in ('path', 'member', 'bytes', 'sha256')} for row in plan['files']]
    require(verified['version'] == 'mosaic-storage-offload-local-verification-v1'
            and verified['status'] == 'verified_all_members' and verified['all_members_verified'] is True
            and verified['unexpected_members'] == verified['missing_members'] == []
            and verified['files'] == expected_files and verified['plan_sha256'] == sha(plan_raw)
            and verified['plan_bytes'] == len(plan_raw) and verified['operation_id'] == 'legacy-offload-03'
            and verified['deleted'] is False and verified['scientific_passed'] is False,
            'Standalone full original-member verification is incomplete')
    require(verified['remote_stream_receipt'] == {'path': str(receipt), 'bytes': len(receipt_raw), 'sha256': sha(receipt_raw)}
            and verified['approved_plan'] == {'path': str(plan_path), 'bytes': len(plan_raw), 'sha256': sha(plan_raw)},
            'Standalone verification used different plan or fetched receipt bytes')
    require(completed['version'] == 'mosaic-storage-offload-stream-v1' and completed['status'] == 'streamed_complete'
            and completed['all_files_verified'] is True and completed['plan_sha256'] == sha(plan_raw)
            and completed['operation_id'] == 'legacy-offload-03' and completed['plan_bytes'] == len(plan_raw),
            'Exact remote stream never completed')
    for key in ('archive_sha256', 'archive_bytes', 'member_count', 'files_count', 'logical_bytes'):
        require(verified[key] == completed[key], 'Local verification/remote completed archive differs')
    require(completed['files_count'] == len(plan['files']) and completed['member_count'] == len(plan['files']) + 1
            and completed['logical_bytes'] == sum(row['bytes'] for row in plan['files']), 'Original backup scope differs')
    # This full opaque-byte rehash happens immediately before any authorization.
    current_archive = archive_record(archive)
    require(current_archive == verified['archive']
            and current_archive['sha256'] == completed['archive_sha256']
            and current_archive['bytes'] == completed['archive_bytes'], 'Completed local archive bytes changed')
    protection_raw, protection = read_json(HERE / 'flat-protection-approved-03.json')
    require(plan['protection_manifest'] == {'path': ROOT + '/protection-03.json', 'sha256': sha(protection_raw)}
            and sha(protection_raw) == approval['protection_sha256']
            and protection['version'] == 'mosaic-storage-offload-protection-v1', 'Approved current-input protection differs')
    payload = {'source_manifest_sha256': MANIFEST, 'plan_sha256': sha(plan_raw), 'plan_bytes': len(plan_raw),
        'protection_sha256': sha(protection_raw), 'operator_verification_raw': audit_raw.decode(),
        'stream_receipt_sha256': sha(receipt_raw), 'local_verification_raw': verified_raw.decode(),
        'local_verification_sha256': sha(verified_raw), 'archive_sha256': current_archive['sha256'],
        'archive_bytes': current_archive['bytes']}
    encoded = json.dumps(payload).encode(); require(len(encoded) <= MAX_BYTES, 'Authorization payload exceeds metadata cap')
    folder = BACKUP / 'removal-execution-01'; folder.mkdir(exist_ok=False)
    (folder / 'remote-caller.py').write_text(REMOTE, encoding='utf8', newline='\n')
    remote = ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', '-c', REMOTE]
    command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
               'dtu-hpc', 'bash -l -c ' + shlex.quote(' '.join(shlex.quote(value) for value in remote))]
    with (folder / 'started.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump({'status': 'unknown', 'utc': datetime.now(timezone.utc).isoformat(), 'args': command,
            'caller_sha256': sha(Path(__file__).read_bytes()), 'remote_code_sha256': sha(REMOTE.encode()),
            'plan_sha256': sha(plan_raw), 'archive_rehash': current_archive,
            'local_verification_sha256': sha(verified_raw), 'fetched_stream_receipt_sha256': sha(receipt_raw)}, stream, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    try:
        with (folder / 'stdout.bin').open('xb') as out, (folder / 'stderr.bin').open('xb') as err:
            result = subprocess.run(command, input=encoded, stdout=out, stderr=err, timeout=plan['timeout_seconds'] + 150)
    except BaseException as error:
        (folder / 'failed-unknown.json').write_text(json.dumps({'status': 'unknown_inspect_original_remote_journal', 'error': repr(error)}) + '\n')
        raise
    (folder / 'result.json').write_text(json.dumps({'returncode': result.returncode}) + '\n')
    print((folder / 'stdout.bin').read_text()); print((folder / 'stderr.bin').read_text())
    require(result.returncode == 0, 'Failed/unknown original deletion outcome preserved; never replay this caller')


if __name__ == '__main__':
    main()
