"""One-shot04 allocated metadata validation after authenticated03 retirement.

Prepared only. This caller stages/submits only the44-case CPU check. It cannot
start/replay a backup or delete originals. Unknown outcomes remain exclusive;
source03, legacy-offload-01 and the local partial archive are never modified.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ROOT = '/work3/s234818/mosaic/runs/20261024-storage-offload-01'
MANIFEST04 = '8a6420047f7260eedd1219c4313c835dae4bb2a13cd39d4914f9e87abcd5bafe'
PARENT_SHA = '9af242392c14b3fc09804f0868b49f0d650245a1f075d43c3b132d994545230d'
RETIREMENT_SHA = '41d238499d97bdb0e415c684a5ccf95bb2c73b7264395c310816e144e83988ab'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def retirement_pins(path):
    path = Path(path)
    assert not path.is_symlink() and path.resolve() == HERE / 'stream-retirement-01.json'
    value = json.loads(path.read_bytes())
    assert value['path'] == ROOT + '/stream-retirement-complete-01.json'
    assert value['bytes'] == 1109 and value['sha256'] == RETIREMENT_SHA
    assert value['status'] == 'intentionally_stopped_own_incomplete_backup'
    assert value['old_exact_producer_running'] is False and value['all_remote_originals_retained'] is True
    assert set(value['files']) == {ROOT + '/stream-retirement-intent-01.json', value['path']}
    result = {}
    for name, pin in value['files'].items():
        local = Path(pin['local_path'])
        assert not local.is_symlink() and local.is_file()
        assert local.stat().st_size == pin['bytes'] and sha(local) == pin['sha256']
        result[name] = {key: pin[key] for key in ('bytes', 'sha256')}
    completed = json.loads(Path(value['local_path']).read_bytes())
    assert completed['status'] == value['status'] and completed['pid'] == 4246
    assert completed['hostname'] == 'hpclogin1' and completed['signal'] == 'SIGTERM'
    assert completed['producer_no_longer_running'] is True and completed['remote_originals_retained'] is True
    assert completed['backup_verified'] is False and completed['original_files_deleted'] == 0
    assert completed['operation_entries'] == ['active-jobs-before-stream.json', 'approved-plan.json', 'stream-started.json']
    return result


def remote_source(retirement):
    tree = ast.parse((HERE / 'submit_cleanup_cpu.py').read_bytes())
    body = next(ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'REMOTE' for t in node.targets))
    body = body.replace('operator-source-01', 'operator-source-04').replace('cpu-01', 'cpu-04')
    body = body.replace('operator-staging-started.json', 'operator-staging-started-04.json')
    body = body.replace('operator-staging-complete.json', 'operator-staging-complete-04.json')
    body = body.replace('operator-preflight-01', 'operator-preflight-04')
    audit = json.loads((HERE / 'operator-verification-03.json').read_bytes())
    assert audit['status'] == 'passed' and audit['cases'] == 32 and audit['job_id'] == '29806228'
    assert audit['source_manifest_sha256'] == PARENT_SHA
    before = """assert not os.path.lexists(root),'Existing control root/unknown staging; inspect, do not replay'
assert root.parent.is_dir() and root.parent.resolve()==root.parent
root.mkdir()"""
    guard = r'''assert root.is_dir() and root.resolve()==root
assert expected=='8a6420047f7260eedd1219c4313c835dae4bb2a13cd39d4914f9e87abcd5bafe'
def exact_raw(path,pin):
 path=Path(path);assert path.resolve()==path and not path.is_symlink() and path.is_file()
 before=path.stat();assert before.st_size==pin['bytes'] and before.st_size<=64*1024**2
 data=path.read_bytes();after=path.stat()
 assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
 assert len(data)==pin['bytes'] and hashlib.sha256(data).hexdigest()==pin['sha256']
 return data
old_pins=__OLD_PROOFS__
for name,pin in old_pins.items():exact_raw(name,pin)
old_source=root/'operator-source-03';old_manifest=json.loads((old_source/'source-manifest.json').read_bytes())
assert hashlib.sha256((old_source/'source-manifest.json').read_bytes()).hexdigest()=='9af242392c14b3fc09804f0868b49f0d650245a1f075d43c3b132d994545230d'
assert old_source.resolve()==old_source and not old_source.stat().st_mode&0o222
assert {p.name for p in old_source.iterdir()}==set(old_manifest['files'])|{'source-manifest.json'}
for name,digest in old_manifest['files'].items():
 p=old_source/name;assert not p.is_symlink() and not p.stat().st_mode&0o222
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
import xml.etree.ElementTree as ET
rows=list(ET.parse(root/'cpu-03/payload/operator-contract.xml').iter('testcase'))
assert len(rows)==32 and all(not list(x) for x in rows)
assert all(x.attrib['classname'].split('.')[-1]=='fixture_backup_contract' for x in rows)
a=json.loads((root/'cpu-03/payload/assessment.json').read_bytes())
assert a['status']=='passed' and a['cases']==32 and a['job_id']=='29806228' and a['queue']=='hpc'
assert a['source_manifest_sha256']=='9af242392c14b3fc09804f0868b49f0d650245a1f075d43c3b132d994545230d'
log=(root/'cpu-03/stdout.29806228.log').read_text()
assert re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$',log,re.M)==[('29806228','Done')] and 'Successfully completed.' in log
retirement=__RETIREMENT_PINS__
for name,pin in retirement.items():exact_raw(name,pin)
retired=json.loads((root/'stream-retirement-complete-01.json').read_bytes())
assert retired['status']=='intentionally_stopped_own_incomplete_backup' and retired['pid']==4246
assert retired['hostname']=='hpclogin1' and retired['signal']=='SIGTERM'
assert retired['producer_no_longer_running'] is True and retired['backup_verified'] is False
assert retired['remote_originals_retained'] is True and retired['original_files_deleted']==0
operation=root/'legacy-offload-01'
assert operation.resolve()==operation and {p.name for p in operation.iterdir()}==set(retired['operation_entries'])=={'active-jobs-before-stream.json','approved-plan.json','stream-started.json'}
old_plan=operation/'approved-plan.json'
assert not old_plan.is_symlink() and hashlib.sha256(old_plan.read_bytes()).hexdigest()=='6573d5d471df56b229e5d1f9adb28b74ddf8959b7d6c419332ca36b45f11d07b'
# A newCPU cannot overlap a still-active exact retired producer on this host.
import socket
if socket.gethostname().split('.')[0]==retired['hostname']:
 proc=Path('/proc')/str(retired['pid'])/'cmdline'
 if proc.exists():assert proc.read_bytes().split(b'\0')[:-1]!=[x.encode() for x in retired['expected_argv']],'Oldexact producer still active'
assert not any(os.path.lexists(p) for p in (source,parent,root/'operator-source-04.tar.gz',root/'operator-staging-started-04.json',root/'operator-staging-complete-04.json',root/'operator-preflight-04',root/'legacy-offload-02')),'New04/operation02 target or unknown outcome exists; inspect'
'''
    guard = guard.replace('__OLD_PROOFS__', repr(audit['proofs'])).replace('__RETIREMENT_PINS__', repr(retirement))
    assert before in body
    body = body.replace(before, guard)
    body = body.replace("assert len(m['files'])==11", "assert m['parent_manifest_sha256']=='9af242392c14b3fc09804f0868b49f0d650245a1f075d43c3b132d994545230d' and m['allocated_fixture_rows_expected']==44 and m['gzip_compresslevel']==1\nassert len(m['files'])==11")
    ast.parse(body)
    return body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retirement-receipt', type=Path, required=True)
    args = parser.parse_args()
    retirement = retirement_pins(args.retirement_receipt)
    package = json.loads((HERE / 'operator-package-04.json').read_bytes())
    assert package['manifest_sha256'] == MANIFEST04 and package['allocated_cases_expected'] == 44
    archive = Path(package['archive'])
    assert not archive.is_symlink() and archive.stat().st_size == package['archive_bytes'] and sha(archive) == package['archive_sha256']
    folder = REPO / 'results/hpc/20261024-storage-offload-01/operator-submission-04'
    folder.mkdir(exist_ok=False)
    remote = remote_source(retirement)
    (folder / 'remote-operator.py').write_text(remote, encoding='utf8', newline='\n')
    argv = ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', '-c', remote,
            package['manifest_sha256'], package['archive_sha256'], str(package['archive_bytes'])]
    command = ' '.join(shlex.quote(x) for x in argv)
    ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
           'dtu-hpc', 'bash -l -c ' + shlex.quote(command)]
    (folder / 'started.json').write_text(json.dumps({'outcome': 'unknown', 'args': ssh,
        'retirement_receipt': str(args.retirement_receipt), 'retirement_receipt_sha256': sha(args.retirement_receipt)}) + '\n')
    try:
        with archive.open('rb') as data, (folder / 'stdout.bin').open('xb') as out, (folder / 'stderr.bin').open('xb') as err:
            result = subprocess.run(ssh, stdin=data, stdout=out, stderr=err, timeout=300)
    except BaseException as error:
        (folder / 'failed-unknown.json').write_text(json.dumps({'error': repr(error), 'outcome': 'unknown_inspect_remote'}) + '\n')
        raise
    (folder / 'result.json').write_text(json.dumps({'returncode': result.returncode}) + '\n')
    print('returncode', result.returncode)
    print((folder / 'stdout.bin').read_text())
    print((folder / 'stderr.bin').read_text()[-4000:])
    assert result.returncode == 0


if __name__ == '__main__':
    main()
