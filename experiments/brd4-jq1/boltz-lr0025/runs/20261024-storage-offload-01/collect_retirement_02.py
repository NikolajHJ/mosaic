"""Fetch exact intentional-stop evidence separately; no process mutation."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
REMOTE=r'''
import base64,hashlib,json
from pathlib import Path
root=Path('/work3/s234818/mosaic/runs/20261024-storage-offload-01')
files={}
for name in ('stream-retirement-intent-02.json','stream-retirement-complete-02.json'):
 p=root/name;assert p.is_file() and not p.is_symlink() and p.resolve()==p
 raw=p.read_bytes();assert len(raw)<65536
 files[str(p)]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
complete=json.loads((root/'stream-retirement-complete-02.json').read_bytes())
assert complete['status']=='intentionally_stopped_own_incomplete_backup' and complete['producer_no_longer_running'] is True and complete['original_files_deleted']==0
assert not (root/'legacy-offload-02/stream-complete.json').exists()
proc=Path('/proc')/str(complete['pid'])/'cmdline'
current=proc.read_bytes().rstrip(b'\0').split(b'\0') if proc.exists() else []
assert [x.decode() for x in current]!=complete['expected_argv']
print(json.dumps({'status':'intentional_stop_receipts_fetched','read_only':True,'files':files,'old_exact_producer_running':False}))
'''


def main():
    folder=REPO/'results/hpc/20261024-storage-offload-01/backup-02/retirement-evidence-01'
    folder.mkdir(exist_ok=False)
    argv=['/work3/s234818/mosaic/envs/protenix-tiny/bin/python','-c',REMOTE]
    command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -c '+shlex.quote(' '.join(shlex.quote(x) for x in argv))]
    (folder/'started.json').write_text(json.dumps({'outcome':'unknown','args':command})+'\n')
    with (folder/'stdout.bin').open('xb') as out,(folder/'stderr.bin').open('xb') as err:
        result=subprocess.run(command,stdout=out,stderr=err,timeout=45)
    (folder/'result.json').write_text(json.dumps({'returncode':result.returncode})+'\n')
    assert result.returncode==0
    value=json.loads((folder/'stdout.bin').read_bytes());assert value['old_exact_producer_running'] is False
    records={}
    for name,record in value['files'].items():
        raw=base64.b64decode(record['base64'],validate=True)
        assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256']
        target=folder/Path(name).name
        with target.open('xb') as stream:stream.write(raw)
        records[name]={'path':name,'bytes':len(raw),'sha256':record['sha256'],'local_path':str(target.resolve()),'status':json.loads(raw)['status']}
    complete=records['/work3/s234818/mosaic/runs/20261024-storage-offload-01/stream-retirement-complete-02.json']
    output={**complete,'version':'mosaic-intentional-backup-retirement-proof-v1','old_exact_producer_running':False,'all_remote_originals_retained':True,'files':records}
    with (HERE/'stream-retirement-02.json').open('x') as out:json.dump(output,out,indent=2);out.write('\n')
    print(json.dumps({k:v for k,v in output.items() if k!='files'}))


if __name__=='__main__':main()
