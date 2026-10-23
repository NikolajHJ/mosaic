"""Byte-only native asset staging. No project imports or numerical execution."""

import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

COMMIT = "510faf9f5cc8fbbbb8d9055934d3fa7e6877c76d"
ROOT = Path("/work3/s234818/latentmol/protenix_occupancy/assets")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, name):
    path = ROOT/name
    receipt_path = ROOT/(name+".receipt.json")
    if path.exists():
        if not receipt_path.exists():
            raise FileExistsError("Existing asset has no authenticated receipt")
        old = json.loads(receipt_path.read_text())
        if old["url"] != url or old["sha256"] != sha(path):
            raise ValueError("Existing asset identity changed")
        return old
    partial = ROOT/(name+".part")
    request = urllib.request.Request(url, headers={"User-Agent":"DBMol-scientific-study/1"})
    with urllib.request.urlopen(request, timeout=120) as incoming, partial.open("xb") as outgoing:
        while block := incoming.read(1024*1024):
            outgoing.write(block)
    digest = sha(partial)
    partial.rename(path)
    result = dict(url=url,path=str(path),sha256=digest,bytes=path.stat().st_size)
    with receipt_path.open("x") as handle:
        json.dump(result,handle,indent=2)
    return result


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    source = download("https://codeload.github.com/bytedance/Protenix/tar.gz/"+COMMIT,
                      "Protenix-"+COMMIT+".tar.gz")
    native_root = ROOT/("Protenix-"+COMMIT)
    if not native_root.exists():
        with tarfile.open(source["path"]) as archive:
            for member in archive.getmembers():
                target = (ROOT/member.name).resolve()
                if not target.is_relative_to(ROOT.resolve()) or member.issym() or member.islnk():
                    raise ValueError("Unsafe native source archive member")
            archive.extractall(ROOT,filter="data")
    records = [[p.relative_to(native_root).as_posix(),sha(p)]
               for p in sorted((native_root/"protenix").rglob("*.py"))]
    source_digest = hashlib.sha256(json.dumps(records,separators=(",",":")).encode()).hexdigest()
    weights = download("https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_tiny_default_v0.5.0.pt",
                       "protenix_tiny_default_v0.5.0.pt")
    result = dict(source=source, source_root=str(native_root),source_commit=COMMIT,
                  source_sha256=source_digest, weights=weights,
                  scope="TLS primary-source byte downloads only; numerical loading later in allocation")
    output = ROOT/"assets-ready.json"
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise ValueError("Existing asset readiness record differs")
    else:
        with output.open("x") as handle:json.dump(result,handle,indent=2)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
