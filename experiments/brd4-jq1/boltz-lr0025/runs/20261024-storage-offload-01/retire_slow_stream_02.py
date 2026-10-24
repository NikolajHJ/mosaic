"""Intentionally stop only the exact owned incomplete read-only backup producer."""
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
REMOTE = r'''
import hashlib,json,os,signal,socket,sys,time
from pathlib import Path
root=Path('/work3/s234818/mosaic/runs/20261024-storage-offload-01')
operation=root/'legacy-offload-02';pid=177888
expected=['/work3/s234818/mosaic/envs/protenix-tiny/bin/python',str(root/'operator-source-04/stream_backup.py'),'--plan-stdin','--plan-sha256','5fc74e3eeb3eedb2df5dfec541bb62b31e2b03dc01b8d0dd20eb3ad277dff6ea','--receipt',str(operation/'stream-complete.json')]
assert root.is_dir() and root.resolve()==root and operation.is_dir() and operation.resolve()==operation
assert not os.path.lexists(operation/'stream-complete.json'),'Backup actually completed; inspect instead of stopping'
raw=(operation/'approved-plan.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==expected[4]
process=Path('/proc')/str(pid)
assert process.stat().st_uid==os.getuid(),'Process belongs to another account'
command=process.joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0')
assert [x.decode() for x in command]==expected,'Different process/unknown identity; do not signal'
def save(path,value):
 with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
intent={'status':'intentional_owned_incomplete_backup_retirement','pid':pid,'hostname':socket.gethostname(),
 'expected_argv':expected,'uid':os.getuid(),'signal':'SIGTERM','unix_ns':time.time_ns(),
 'reason':'Network pipe waits and redundant alias copies risk the one-hour bound; retain this partial and all originals, prioritize the single-copy subset with unchanged validated tooling.',
 'original_files_deleted':0,'scientific_jobs_or_settings_changed':False}
save(root/'stream-retirement-intent-02.json',intent)
os.kill(pid,signal.SIGTERM)
stopped=False
for _ in range(100):
 try:
  current=process.joinpath('status').read_text()
  if any(line.startswith('State:') and '\tZ' in line for line in current.splitlines()):stopped=True;break
 except FileNotFoundError:stopped=True;break
 time.sleep(.1)
assert stopped,'Owned producer outcome unknown; inspect before a new operation'
assert not os.path.lexists(operation/'stream-complete.json')
result={**intent,'status':'intentionally_stopped_own_incomplete_backup','producer_no_longer_running':True,
 'completed_unix_ns':time.time_ns(),'operation_entries':[p.name for p in sorted(operation.iterdir())],
 'backup_verified':False,'remote_originals_retained':True}
save(root/'stream-retirement-complete-02.json',result)
print(json.dumps(result))
'''


def main():
    folder = REPO / 'results/hpc/20261024-storage-offload-01/backup-02/intentional-retirement-01'
    folder.mkdir(exist_ok=False)
    argv = ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python','-c',REMOTE]
    command = ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1',
        'dtu-hpc','bash -l -c '+shlex.quote(' '.join(shlex.quote(x) for x in argv))]
    (folder/'started.json').write_text(json.dumps({'outcome':'unknown','args':command})+'\n')
    try:
        with (folder/'stdout.bin').open('xb') as out, (folder/'stderr.bin').open('xb') as err:
            result=subprocess.run(command,stdout=out,stderr=err,timeout=45)
    except BaseException as error:
        (folder/'unknown.json').write_text(json.dumps({'error':repr(error),'inspect_before_recovery':True})+'\n')
        raise
    (folder/'result.json').write_text(json.dumps({'returncode':result.returncode})+'\n')
    print((folder/'stdout.bin').read_text());print((folder/'stderr.bin').read_text()[-1500:])
    assert result.returncode==0


if __name__=='__main__':
    main()
