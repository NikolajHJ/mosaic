"""Serve an existing report on loopback, independently of the launching chat.

This only serves saved files. It performs no rendering or numerical work.
Windows children explicitly leave the parent's job so closing a tool session
does not terminate the report viewer. No startup task or service is installed.
"""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request


def available(url, size):
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD"), timeout=1
        ) as response:
            return (
                response.status == 200
                and int(response.headers["Content-Length"]) == size
            )
    except (OSError, ValueError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    root = args.directory.resolve(strict=True)
    report = root / "report.html"
    if not report.is_file() or not 1 <= args.port <= 65535:
        parser.error("An existing report.html and valid loopback port are required")
    url = f"http://127.0.0.1:{args.port}/report.html"
    receipt_path = root.parent / "server.json"
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if previous.get("root") == str(root) and previous.get("url") == url:
            if available(url, report.stat().st_size):
                print(url)
                return
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError("Port is occupied; choose another --port")
    options = {"start_new_session": True}
    if sys.platform == "win32":
        options = {
            "creationflags": subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_BREAKAWAY_FROM_JOB
        }
    log = root.parent / f"http-{args.port}.log"
    with log.open("ab") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "http.server",
                str(args.port),
                "--bind",
                "127.0.0.1",
                "--directory",
                str(root),
            ],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            close_fds=True,
            **options,
        )
    for _ in range(20):
        if process.poll() is not None:
            raise RuntimeError(f"Report server exited; inspect {log}")
        if available(url, report.stat().st_size):
            receipt_path.write_text(
                json.dumps(
                    {
                        "pid": process.pid,
                        "url": url,
                        "root": str(root),
                        "log": str(log),
                        "lifecycle": "detached from launcher and parent job",
                    },
                    indent=2,
                )
                + "\n"
            )
            print(url)
            return
        time.sleep(0.2)
    raise RuntimeError(f"Report server did not become ready; inspect {log}")


if __name__ == "__main__":
    main()
