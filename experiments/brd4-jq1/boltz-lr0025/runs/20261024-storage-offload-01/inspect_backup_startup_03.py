"""One bounded read-only inspection of the exact existing backup process."""
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
REMOTE = r'''
import json,os,subprocess,time
from pathlib import Path
root=Path('/work3/s234818/mosaic/runs/20261024-storage-offload-01')
operation=root/'legacy-offload-03'
needle=str(root/'operator-source-04/stream_backup.py')
result=subprocess.run(['ps','-u',str(os.getuid()),'-o','pid=,ppid=,etimes=,time=,pcpu=,stat=,wchan=,args='],capture_output=True,text=True,timeout=15)
assert result.returncode==0
rows=[line for line in result.stdout.splitlines() if needle in line and 'ps -u' not in line]
entries=None
if operation.exists():
 assert operation.is_dir() and operation.resolve()==operation
 entries=[{'name':p.name,'bytes':p.lstat().st_size} for p in sorted(operation.iterdir())]
print(json.dumps({'read_only':True,'utc_unix_ns':time.time_ns(),'existing_backup_process_rows':rows,
 'operation_exists':operation.exists(),'operation_entries':entries,'files_deleted':0,'processes_stopped':0}))
'''


def main():
    folder = REPO / 'results/hpc/20261024-storage-offload-01/backup-03/startup-inspection-01'
    folder.mkdir(exist_ok=False)
    remote = ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', '-c', REMOTE]
    command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
        'dtu-hpc', 'bash -l -c ' + shlex.quote(' '.join(shlex.quote(value) for value in remote))]
    (folder / 'started.json').write_text(json.dumps({'outcome':'unknown','args':command})+'\n')
    try:
        with (folder / 'stdout.bin').open('xb') as out, (folder / 'stderr.bin').open('xb') as err:
            result = subprocess.run(command, stdout=out, stderr=err, timeout=45)
    except BaseException as error:
        (folder / 'failed.json').write_text(json.dumps({'error':repr(error),'read_only':True})+'\n')
        raise
    (folder / 'result.json').write_text(json.dumps({'returncode':result.returncode})+'\n')
    print((folder / 'stdout.bin').read_text())
    assert result.returncode==0


if __name__ == '__main__':
    main()
