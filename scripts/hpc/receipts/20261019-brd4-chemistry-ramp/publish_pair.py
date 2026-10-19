"""Publish only the verified second edition to the new localhost report family."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    base = args.directory.resolve(strict=True)
    repo = Path(__file__).resolve().parents[4]
    family = (repo / "results/hpc/20261019-brd4-delayed-chemistry-report").resolve(
        strict=True
    )
    if not base.is_relative_to(family) or base == family:
        raise ValueError(
            "Only an edition within the new report family may be published"
        )
    verified = json.loads((base / "pair-verification.json").read_text())
    if (
        verified["status"] != "passed"
        or verified["input_jobs"] != ["29753667", "29753668"]
        or verified["states"] != [10001, 10001]
    ):
        raise ValueError("Missing completed verified comparison")
    source = Path(verified["report"]).resolve(strict=True).parent
    if (
        not source.is_relative_to(base)
        or sha(source / "report.html") != verified["report_sha256"]
    ):
        raise ValueError("Verified source report differs")
    receipt = json.loads((source / "report_receipt.json").read_text())
    if sha(source / "report_receipt.json") != verified["report_receipt_sha256"]:
        raise ValueError("Verified report receipt differs")
    viewer = (family / "viewer").resolve(strict=True)
    if not viewer.is_relative_to(family):
        raise ValueError("Viewer path escaped the new family")
    publication = json.loads((family / "publication.json").read_text())
    backup = family / "publication-edition-01.json"
    if not backup.exists():
        backup.write_bytes((family / "publication.json").read_bytes())
    # Validate all bytes before replacing anything. HTML is self-contained and
    # is replaced last, preserving one coherent rendered page for open readers.
    for name, entry in receipt["files"].items():
        if (
            Path(name).name != name
            or (source / name).stat().st_size != entry["bytes"]
            or sha(source / name) != entry["sha256"]
        ):
            raise ValueError("Unsafe or changed report artifact: " + name)
    names = [name for name in receipt["files"] if name != "report.html"]
    names += ["report_receipt.json", "report.html"]
    for name in names:
        temporary = viewer / (name + ".incoming")
        shutil.copyfile(source / name, temporary)
        temporary.replace(viewer / name)
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/serve_protenix_report.py"),
            str(viewer),
            "--port",
            "8773",
        ],
        check=True,
    )
    url = "http://127.0.0.1:8773/report.html"
    with urllib.request.urlopen(
        urllib.request.Request(url, method="HEAD"), timeout=10
    ) as response:
        if (
            response.status != 200
            or int(response.headers["Content-Length"]) != verified["report_bytes"]
        ):
            raise ValueError("Published HTTP response differs")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=30) as response:
        while chunk := response.read(1024**2):
            digest.update(chunk)
    if digest.hexdigest() != verified["report_sha256"]:
        raise ValueError("HTTP-served report bytes differ")
    publication.pop("included_job", None)
    publication.update(
        url=url,
        viewer=str(viewer),
        second_edition=str(source),
        report_sha256=verified["report_sha256"],
        bytes=verified["report_bytes"],
        included_jobs=verified["input_jobs"],
        pending_job=None,
        status="both complete and byte-verified; browser QA pending",
    )
    (family / "publication.json").write_text(json.dumps(publication, indent=2) + "\n")
    print(json.dumps(publication))


if __name__ == "__main__":
    main()
