"""New exclusive operator02 after authenticated operator01 fixture failure."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def remote_source():
    old = ast.parse((HERE / 'submit_cleanup_cpu.py').read_bytes())
    body = next(ast.literal_eval(node.value) for node in old.body
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'REMOTE' for t in node.targets))
    body = body.replace('operator-source-01', 'operator-source-02').replace('cpu-01', 'cpu-02')
    body = body.replace('operator-staging-started.json', 'operator-staging-started-02.json')
    body = body.replace('operator-staging-complete.json', 'operator-staging-complete-02.json')
    body = body.replace('operator-preflight-01', 'operator-preflight-02')
    failure = json.loads((HERE / 'operator-failure-verification-01.json').read_bytes())
    assert failure['status'] == 'verified_failure_preserved' and failure['passed'] == 18 and failure['failed'] == 1
    before = """assert not os.path.lexists(root),'Existing control root/unknown staging; inspect, do not replay'
assert root.parent.is_dir() and root.parent.resolve()==root.parent
root.mkdir()"""
    replacement = '''assert root.is_dir() and root.resolve()==root
old_pins=__OLD_PROOFS__
for name,pin in old_pins.items():
 p=Path(name);assert p.is_file() and not p.is_symlink()
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 assert p.stat().st_size==pin['bytes'] and digest==pin['sha256'],'Preserved failed01 proof changed'
import xml.etree.ElementTree as ET
old_rows=list(ET.parse(root/'cpu-01/payload/operator-contract.xml').iter('testcase'))
failed=[x for x in old_rows if x.find('failure') is not None]
assert len(old_rows)==19 and len(failed)==1 and 'linkname is too long' in ''.join(failed[0].itertext())
assert not any(x.find('error') is not None or x.find('skipped') is not None for x in old_rows)
old_log=(root/'cpu-01/stdout.29806186.log').read_text()
assert re.findall(r'^Subject:\\s*Job\\s+(\\d+):[^\\r\\n]*\\b(Done|Exited)\\s*$',old_log,re.M)==[('29806186','Exited')] and 'Exited with exit code 1' in old_log
assert not any(os.path.lexists(p) for p in (source,parent,root/'operator-source-02.tar.gz',root/'operator-staging-started-02.json',root/'operator-staging-complete-02.json',root/'operator-preflight-02')),'New02 target/unknown outcome already exists; inspect'
'''.replace('__OLD_PROOFS__', repr(failure['proofs']))
    assert before in body
    body = body.replace(before, replacement)
    body = body.replace("assert len(m['files'])==11", "assert m['parent_manifest_sha256']=='984e39e117ac6de6940d61b9d875b36608f69a60cb9c28eff92b768adf97754b'\nassert len(m['files'])==11")
    ast.parse(body)
    return body


def main():
    package = json.loads((HERE / 'operator-package-02.json').read_bytes())
    archive = Path(package['archive'])
    assert hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest() == package['archive_sha256']
    folder = REPO / 'results/hpc/20261024-storage-offload-01/operator-submission-02'
    folder.mkdir(exist_ok=False)
    remote = remote_source()
    (folder / 'remote-operator.py').write_text(remote, encoding='utf8', newline='\n')
    argv = ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', '-c', remote,
            package['manifest_sha256'], package['archive_sha256'], str(package['archive_bytes'])]
    args = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
            'dtu-hpc', 'bash -l -c ' + shlex.quote(' '.join(shlex.quote(x) for x in argv))]
    (folder / 'started.json').write_text(json.dumps({'outcome': 'unknown', 'args': args}) + '\n')
    try:
        with archive.open('rb') as data, (folder / 'stdout.bin').open('xb') as out, (folder / 'stderr.bin').open('xb') as err:
            result = subprocess.run(args, stdin=data, stdout=out, stderr=err, timeout=300)
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
