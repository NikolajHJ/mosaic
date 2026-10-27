"""Fetch/hash public bytes and archive a pinned Git source; never load models.

Run locally before sealing, or transfer the resulting bytes after storage
admission. Native numerical preprocessing remains allocated HPC work.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

COMMIT = "5d0d38d16c8932a0339fd2ce3f67ade98bbdff27"
RECORD = "https://zenodo.org/api/records/8183747"
NAME = "crossdocked_fullatom_cond.ckpt"
SIZE = 17861341
MD5 = "166b0c056b31ffdf31d489a63e91e05b"


def digest(path, name="sha256"):
    h = hashlib.new(name)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(source_checkout, output, receptor):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "-C", str(source_checkout), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT:
        raise ValueError("Native checkout is not the reviewed commit")
    # Archive Git blob bytes, avoiding checkout EOL conversion and unrelated dirt.
    archive = subprocess.check_output(["git", "-C", str(source_checkout), "archive", "--format=tar", COMMIT])
    root = output / "DiffSBDD"
    root.mkdir()
    members = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for member in tar:
            if member.isdir():
                continue
            if not member.isfile() or member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ValueError("Unexpected native Git archive member")
            raw = tar.extractfile(member).read()
            path = root / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            members[member.name] = hashlib.sha256(raw).hexdigest()
    write_json(root / ".diffsbdd-source.json", {"commit":COMMIT,"members":members,"basis":"raw_git_blobs"})
    with urllib.request.urlopen(RECORD, timeout=90) as response:
        record_raw = response.read()
    (output / "zenodo-record.json").write_bytes(record_raw)
    record = json.loads(record_raw)
    files = [x for x in record["files"] if x["key"] == NAME]
    if len(files) != 1 or files[0]["size"] != SIZE or files[0]["checksum"] != "md5:" + MD5:
        raise ValueError("Public record differs from reviewed weight size/checksum")
    url = files[0]["links"].get("content") or files[0]["links"]["self"]
    checkpoint = output / NAME
    with urllib.request.urlopen(url, timeout=120) as response, checkpoint.open("xb") as stream:
        shutil.copyfileobj(response, stream, 4 * 1024**2)
    if checkpoint.stat().st_size != SIZE or digest(checkpoint, "md5") != MD5:
        raise ValueError("Public checkpoint transport checksum failed; preserve failed download")
    target = output / "3MXF.pdb"
    shutil.copyfile(receptor, target)
    if digest(target) != "534dcb8ea9fa29956249b5133b22187c3867363b16a6b98dd8219c18c5d3250f":
        raise ValueError("Crystal source identity differs from the reviewed fixture")
    receipt = {"version":"diffsbdd-public-assets-v1","models_loaded":False,
        "numerical_work":False,"remote_staging":False,"upstream_commit":COMMIT,
        "source_manifest_sha256":digest(root / ".diffsbdd-source.json"),
        "record_url":RECORD,"checkpoint_url":url,"checkpoint_name":NAME,
        "checkpoint_sha256":digest(checkpoint),"checkpoint_md5":MD5,"checkpoint_bytes":SIZE,
        "receptor_sha256":digest(target),"record_sha256":digest(output / "zenodo-record.json")}
    write_json(output / "assets.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receptor", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_checkout, args.output, args.receptor), indent=2))


if __name__ == "__main__":
    main()
