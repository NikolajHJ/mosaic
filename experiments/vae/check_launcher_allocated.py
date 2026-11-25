"""Allocated CPU smoke of the real launcher, including a stalled startup.

No tensor library/model is imported. Use this small test before the recovered
GPU probe; it tests node-local staging, preserved exit status, bounded startup,
home-storage receipts and actual scratch checkpoint readability.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--helper",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--checkpoint",required=True);p.add_argument("--checkpoint-sha256",required=True)
    a=p.parse_args()
    if not os.environ.get("LSB_JOBID","").isdigit():
        p.error("Run this smoke in an allocated job")
    a.output.mkdir(exist_ok=False)
    spec=importlib.util.spec_from_file_location("launch_smoke",a.helper)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    archive=a.output/"tiny.tar.gz"
    with tarfile.open(archive,"w:gz") as stream:
        member=tarfile.TarInfo("sentinel.txt");body=b"verified-local-source\n";member.size=len(body)
        stream.addfile(member,io.BytesIO(body))
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    code="import os,json,pathlib,time; assert pathlib.Path(os.environ['DBMOL_LOCAL_SOURCE'],'sentinel.txt').read_text()=='verified-local-source\\n'; pathlib.Path(os.environ['DBMOL_LAUNCH_READY']).write_text(json.dumps(dict(status='models_and_inputs_ready',nonce=os.environ['DBMOL_LAUNCH_NONCE'],job_id=os.environ['LSB_JOBID']))); time.sleep(.6)"
    def args(label,command,startup):
        return SimpleNamespace(archive=str(archive),archive_sha256=sha,release="/unused/source",
            receipt=str(a.output/(label+".json")),startup_seconds=startup,run_seconds=10,
            command=["/usr/bin/python3","-c",command])
    code0=helper.launch(args("success",code,5))
    success=json.loads((a.output/"success.json").read_text())
    if code0!=0 or success.get("ready") is not True or not success["local_source"].startswith("/tmp/"):
        raise RuntimeError("Actual node-local launch/readiness failed")
    try:
        helper.launch(args("stall","import time; time.sleep(10)",1))
    except TimeoutError:
        pass
    else:
        raise RuntimeError("Stalled startup was not bounded")
    stall=json.loads((a.output/"stall.json").read_text())
    digest=helper.bounded_file(a.checkpoint,"hash",timeout=30,spool=a.output)
    if digest!=a.checkpoint_sha256 or stall.get("status")!="launch_failed":
        raise RuntimeError("Checkpoint read or stall receipt differs")
    helper.atomic_json(a.output/"result.json",dict(status="passed_allocated_launcher_smoke",
        job_id=os.environ["LSB_JOBID"],node=os.uname().nodename,
        native_model_work=False,checkpoint_sha256=digest,node_local_staging=True,
        startup_timeout=True,home_receipts=True))
    print((a.output/"result.json").read_text(),flush=True)


if __name__=="__main__":
    main()
