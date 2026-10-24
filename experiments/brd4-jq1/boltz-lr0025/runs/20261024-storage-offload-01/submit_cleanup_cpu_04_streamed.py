"""New caller transports identical reviewed metadata code through framed stdin.

The first caller failed Windows CreateProcess WinError206 before an SSH child
could start. Preserve that attempt; this distinct caller still executes every
original remote absence, source, retirement, quota and submission guard.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
BOOTSTRAP=r'''
import hashlib,sys
header=sys.stdin.buffer.readline(32)
assert header.endswith(b'\n') and header[:-1].isdigit()
length=int(header);assert 0<length<=128*1024
code=sys.stdin.buffer.read(length)
assert len(code)==length and hashlib.sha256(code).hexdigest()==sys.argv[4]
sys.argv=sys.argv[:4]
exec(compile(code,'<exact-reviewed-offload-stage04-caller>','exec'),globals())
'''


def main():
    source=HERE/'submit_cleanup_cpu_04.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='8962e5539914d54f1db2e65781d06c57db7609039a2719441a75bfe30170a0b1'
    old=REPO/'results/hpc/20261024-storage-offload-01/operator-submission-04'
    failed=json.loads((old/'failed-unknown.json').read_bytes())
    assert failed['error'] == "FileNotFoundError(2, 'The filename or extension is too long', None, 206, None)"
    assert not (old/'result.json').exists()
    assert (old/'stdout.bin').stat().st_size==(old/'stderr.bin').stat().st_size==0
    spec=importlib.util.spec_from_file_location('exact_original_stage04_caller',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    pins=module.retirement_pins(HERE/'stream-retirement-01.json')
    remote=module.remote_source(pins);raw=remote.encode()
    assert raw==(old/'remote-operator.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='751d3fe6b97a78c5c3bef8bc20fd76057af367bee8efc04f75c432c62267b54a'
    ast.parse(remote);ast.parse(BOOTSTRAP)
    package=json.loads((HERE/'operator-package-04.json').read_bytes())
    archive=Path(package['archive']);data=archive.read_bytes()
    assert len(data)==package['archive_bytes']==31403 and hashlib.sha256(data).hexdigest()==package['archive_sha256']
    folder=REPO/'results/hpc/20261024-storage-offload-01/operator-submission-04-streamed-01'
    folder.mkdir(exist_ok=False)
    argv=['/work3/s234818/mosaic/envs/protenix-tiny/bin/python','-c',BOOTSTRAP,
        package['manifest_sha256'],package['archive_sha256'],str(len(data)),hashlib.sha256(raw).hexdigest()]
    command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1',
        'dtu-hpc','bash -l -c '+shlex.quote(' '.join(shlex.quote(x) for x in argv))]
    (folder/'remote-operator.py').write_bytes(raw)
    (folder/'started.json').write_text(json.dumps({'outcome':'unknown','args':command,
        'remote_code_bytes':len(raw),'remote_code_sha256':hashlib.sha256(raw).hexdigest(),
        'unchanged_original_remote_guards':True,'transport_only':'ASCII code length, exact code, exact archive on one stdin stream'})+'\n')
    framed=str(len(raw)).encode()+b'\n'+raw+data
    try:
        with (folder/'stdout.bin').open('xb') as out,(folder/'stderr.bin').open('xb') as err:
            result=subprocess.run(command,input=framed,stdout=out,stderr=err,timeout=300)
    except BaseException as error:
        (folder/'failed-unknown.json').write_text(json.dumps({'error':repr(error),'inspect_remote_before_any_recovery':True})+'\n')
        raise
    (folder/'result.json').write_text(json.dumps({'returncode':result.returncode})+'\n')
    print((folder/'stdout.bin').read_text());print((folder/'stderr.bin').read_text()[-1500:])
    assert result.returncode==0


if __name__=='__main__':main()
