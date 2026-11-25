"""Shared launch/I/O behavior, independent of molecular numerical code."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest


@pytest.fixture
def launcher():
    path=Path(__file__).resolve().parents[2]/"scripts/hpc/hpc_launch.py"
    spec=importlib.util.spec_from_file_location("launch_fixture",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def archive(tmp_path,name="src/example.py"):
    path=tmp_path/"input.tar.gz"
    with tarfile.open(path,"w:gz") as t:
        member=tarfile.TarInfo(name);body=b"answer=42\n";member.size=len(body)
        t.addfile(member,io.BytesIO(body))
    return path,hashlib.sha256(path.read_bytes()).hexdigest()


def test_archive_is_authenticated_and_staged_without_changing_input(launcher,tmp_path):
    path,sha=archive(tmp_path)
    parent=tmp_path/"local";parent.mkdir()
    result=launcher.stage_archive(path,sha,parent/"source")
    assert (parent/"source/src/example.py").read_bytes()==b"answer=42\n"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
    assert hashlib.sha256(result.read_bytes()).hexdigest()==sha
    with pytest.raises(FileExistsError):
        launcher.stage_archive(path,sha,parent/"source")


def test_archive_checksum_failure_does_not_create_source(launcher,tmp_path):
    path,_=archive(tmp_path)
    with pytest.raises(ValueError,match="checksum"):
        launcher.stage_archive(path,"0"*64,tmp_path/"source")
    assert not (tmp_path/"source").exists()


@pytest.mark.parametrize("member",["../escape","/absolute/file"])
def test_archive_escape_rejected(launcher,tmp_path,member):
    path,sha=archive(tmp_path,member)
    with pytest.raises(ValueError,match="Unsafe"):
        launcher.stage_archive(path,sha,tmp_path/"source")


def test_bounded_file_read_hash_and_missing(launcher,tmp_path):
    path=tmp_path/"asset.json";path.write_text('{"epoch":43}')
    assert launcher.bounded_file(path,spool=tmp_path)==dict(epoch=43)
    assert launcher.bounded_file(path,"hash",spool=tmp_path)==hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileNotFoundError):
        launcher.bounded_file(tmp_path/"missing",spool=tmp_path)


def test_blocked_reader_is_stopped_and_reported(launcher,tmp_path,monkeypatch):
    calls=[]
    class Child:
        def wait(self,timeout):raise subprocess.TimeoutExpired("reader",timeout)
    monkeypatch.setattr(launcher.subprocess,"Popen",lambda *a,**k:Child())
    monkeypatch.setattr(launcher,"terminate",lambda p:calls.append("terminated"))
    with pytest.raises(TimeoutError,match="Scratch I/O"):
        launcher.bounded_file(tmp_path/"blocked",timeout=.01,spool=tmp_path)
    assert calls==["terminated"]
