"""Small standard-library bootstrap for an unreliable shared scratch mount.

Keep this file and control receipts on home storage. Potentially blocking reads
run in separate processes; models execute from a checksum-verified node-local
source snapshot. No model, dataset, loss, optimizer or checkpoint is changed.
The launcher preserves the worker exit code and never retries scientific work.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def terminate(process):
    """Kill only our process group; never wait indefinitely for blocked I/O."""
    if process.poll() is None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


def bounded_file(path, operation="json", *, timeout=30, spool=None):
    """Read/hash scratch in a child, with the result staged on reliable storage."""
    spool = spool or os.environ.get("DBMOL_CONTROL_IO") or tempfile.gettempdir()
    with tempfile.TemporaryDirectory(prefix="dbmol-io-", dir=spool) as directory:
        output = Path(directory)/"result.json"
        with (Path(directory)/"stderr.log").open("w") as errors:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                "--io", operation, "--path", str(path), "--output", str(output)],
                stdout=subprocess.DEVNULL, stderr=errors, start_new_session=os.name=="posix")
            try:
                child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                terminate(child)
                raise TimeoutError("Scratch I/O timed out: " + str(path))
        if child.returncode or not output.exists():
            raise OSError("Scratch reader failed: " + str(path))
        result = json.loads(output.read_text())
        if result.get("missing"):
            raise FileNotFoundError(str(path))
        if result.get("error"):
            raise OSError(result["error"])
        return result["value"]


def io_child(args):
    try:
        path = Path(args.path)
        if args.io == "hash":
            h=hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    h.update(block)
            value=h.hexdigest()
        else:
            with path.open("rb") as stream:
                data=stream.read(8*1024*1024+1)
            if len(data)>8*1024*1024:
                raise ValueError("Control JSON exceeds8MiB bound")
            value=json.loads(data)
        result=dict(value=value)
    except FileNotFoundError:
        result=dict(missing=True)
    except Exception as exc:
        result=dict(error=type(exc).__name__+": "+str(exc))
    atomic_json(args.output,result)


def stage_archive(archive, expected, destination):
    """Authenticate before extraction; reject links, path escapes and collisions."""
    raw=Path(archive).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError("Source archive checksum differs")
    destination=Path(destination)
    destination.mkdir(exist_ok=False)
    local_archive=destination.parent/"source.tar.gz"
    with local_archive.open("xb") as stream:
        stream.write(raw)
    with tarfile.open(local_archive) as source:
        for member in source:
            relative=Path(member.name)
            if PurePosixPath(member.name).is_absolute() or relative.is_absolute() or ".." in relative.parts or member.issym() or member.islnk():
                raise ValueError("Unsafe archive member")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Unsupported archive member")
            target=destination/relative
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Archive escaped local source")
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open("xb") as stream:
                shutil.copyfileobj(source.extractfile(member),stream)
    return local_archive


def launch_child(args):
    source=Path(args.local_source)
    archive=stage_archive(args.archive,args.archive_sha256,source)
    command=args.command[1:] if args.command[:1]==["--"] else args.command
    command=[("PYTHONPATH="+str(source/"src") if item=="PYTHONPATH="+args.release+"/src" else
              "LATENTMOL_SOURCE_ARCHIVE="+str(archive) if item=="LATENTMOL_SOURCE_ARCHIVE="+args.release+".tar.gz" else item)
             for item in command]
    os.environ.update(DBMOL_LAUNCH_READY=args.ready, DBMOL_LAUNCH_NONCE=args.nonce,
                      DBMOL_LOCAL_SOURCE=str(source), PYTHONDONTWRITEBYTECODE="1")
    os.chdir(source)
    os.execvpe(command[0],command,os.environ)


def launch(args):
    receipt=Path(args.receipt)
    receipt.parent.mkdir(parents=True,exist_ok=True)
    if receipt.exists():
        raise FileExistsError("Preserve previous launch receipt")
    ready=receipt.with_suffix(".ready.json")
    if ready.exists():
        raise FileExistsError("Preserve previous launch readiness")
    nonce=uuid.uuid4().hex
    local_root=Path(os.environ.get("DBMOL_NODE_TMP", "/tmp" if os.name=="posix" else tempfile.gettempdir()))
    if shutil.disk_usage(local_root).free < 100*1024*1024:
        raise OSError("Insufficient node-local space for source snapshot")
    directory=Path(tempfile.mkdtemp(prefix="dbmol-source-",dir=local_root))
    started=time.monotonic()
    record=dict(status="starting",job_id=os.environ.get("LSB_JOBID"),
                nonce=nonce,source_archive_sha256=args.archive_sha256,local_source=str(directory/"source"),
                startup_seconds=args.startup_seconds,run_seconds=args.run_seconds)
    atomic_json(receipt,record)
    command=args.command[1:] if args.command[:1]==["--"] else args.command
    child_command=[sys.executable,str(Path(__file__).resolve()),"--child",
        "--archive",args.archive,"--archive-sha256",args.archive_sha256,"--release",args.release,
        "--local-source",str(directory/"source"),"--ready",str(ready),"--nonce",nonce,"--",*command]
    child=subprocess.Popen(child_command,start_new_session=os.name=="posix")
    was_ready=False
    try:
        while child.poll() is None:
            elapsed=time.monotonic()-started
            if not was_ready and ready.exists():
                proof=json.loads(ready.read_text())
                if (proof.get("nonce")!=nonce or proof.get("job_id")!=os.environ.get("LSB_JOBID")
                        or proof.get("status")!="models_and_inputs_ready"):
                    raise ValueError("Foreign worker startup marker")
                was_ready=True
                record.update(status="worker_ready",startup_elapsed_seconds=elapsed)
                atomic_json(receipt,record)
            if elapsed>args.run_seconds or (not was_ready and elapsed>args.startup_seconds):
                raise TimeoutError("Worker startup/run deadline exceeded; possible shared-storage stall")
            time.sleep(.2)
        code=child.returncode
        record.update(status="worker_exited",returncode=code,ready=was_ready,
                      elapsed_seconds=time.monotonic()-started)
        atomic_json(receipt,record)
        return code if code>=0 else 128-code
    except BaseException as exc:
        terminate(child)
        record.update(status="launch_failed",error=type(exc).__name__+": "+str(exc),
                      ready=was_ready,elapsed_seconds=time.monotonic()-started)
        atomic_json(receipt,record)
        raise
    finally:
        # This is our own node-local temporary directory, never work3/home.
        if child.poll() is not None and directory.resolve().is_relative_to(local_root.resolve()):
            shutil.rmtree(directory)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--io",choices=("json","hash"));parser.add_argument("--path")
    parser.add_argument("--output");parser.add_argument("--child",action="store_true")
    parser.add_argument("--archive");parser.add_argument("--archive-sha256");parser.add_argument("--release")
    parser.add_argument("--local-source");parser.add_argument("--ready");parser.add_argument("--nonce")
    parser.add_argument("--receipt");parser.add_argument("--startup-seconds",type=int,default=600)
    parser.add_argument("--run-seconds",type=int,default=13200)
    parser.add_argument("command",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if args.io:
        io_child(args)
    elif args.child:
        launch_child(args)
    else:
        if not os.environ.get("LSB_JOBID","").isdigit():
            parser.error("Model launch requires an allocated job")
        sys.exit(launch(args))


if __name__=="__main__":
    main()
