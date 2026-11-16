#!/usr/bin/env python3
"""Serve a local visualization library with durable curation and recoverable deletion."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
import threading
from urllib.parse import urlsplit
import webbrowser

ID = re.compile(r"^[a-f0-9]{20}$")


class Library:
    def __init__(self, root):
        self.root = root.resolve()
        self.lock = threading.RLock()

    def read(self, name, fallback):
        p = self.root / name
        return json.loads(p.read_text()) if p.exists() else fallback

    def save(self, curation):
        tmp = self.root / "curation.json.tmp"
        tmp.write_text(json.dumps(curation, indent=2, allow_nan=False) + "\n")
        tmp.replace(self.root / "curation.json")

    def runs(self):
        with self.lock:
            curation = self.read("curation.json", {})
            index = self.read("index.json", {"runs": []})
            result = []
            for item in index["runs"]:
                pref = curation.get(item["id"], {})
                result.append({**item, **pref, "starred": pref.get("starred", False),
                               "deleted": pref.get("deleted", False)})
            return {"runs": result, "errors": index.get("errors", [])}

    def mutate(self, action, body):
        with self.lock:
            known = {r["id"]: r for r in self.runs()["runs"]}
            ids = body.get("ids", [body.get("id")])
            if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
                raise ValueError("Choose one or more distinct runs.")
            if any(not isinstance(rid, str) or not ID.fullmatch(rid) or rid not in known for rid in ids):
                raise ValueError("Unknown run.")
            curation = self.read("curation.json", {})
            moved = []
            if action == "purge":
                if body.get("confirm_permanent") is not True:
                    raise ValueError("Confirm permanent deletion of the selected exported copies.")
                for rid in ids:
                    source = self.root / ".trash" / rid
                    if (not known[rid]["deleted"] or known[rid]["starred"] or
                            not source.is_dir() or source.is_symlink() or
                            source.resolve().parent != (self.root / ".trash").resolve() or
                            not source.resolve().is_relative_to(self.root)):
                        raise ValueError("Only unstarred exported runs in library trash can be permanently deleted.")
                for rid in ids:
                    shutil.rmtree(self.root / ".trash" / rid)
                    curation.setdefault(rid, {}).update(deleted=True, purged=True)
                    self.save(curation)
                return self.runs()
            try:
                for rid in ids:
                    pref = curation.setdefault(rid, {})
                    if action == "star":
                        if not isinstance(body.get("starred"), bool):
                            raise ValueError("starred must be true or false")
                        pref["starred"] = body["starred"]
                    elif action == "description":
                        text = body.get("description")
                        if not isinstance(text, str) or len(text) > 20000:
                            raise ValueError("Description must be text with at most 20,000 characters.")
                        pref["description"] = text
                    elif action in ("delete", "restore"):
                        if action == "delete" and known[rid]["starred"]:
                            raise ValueError("Unstar saved runs before deleting them.")
                        deleting = action == "delete"
                        if known[rid]["deleted"] == deleting:
                            raise ValueError("Run is already in the requested location.")
                        source = self.root / ("runs" if deleting else ".trash") / rid
                        destination = self.root / (".trash" if deleting else "runs") / rid
                        if source.is_symlink() or not source.resolve().is_relative_to(self.root) or destination.exists():
                            raise ValueError("Unsafe or conflicting run directory.")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        source.rename(destination)
                        moved.append((source, destination))
                        pref["deleted"] = deleting
                    else:
                        raise ValueError("Unknown action.")
                self.save(curation)
            except Exception:
                for source, destination in reversed(moved):
                    destination.rename(source)
                raise
            return self.runs()


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, library, **kwargs):
        self.library = library
        super().__init__(*args, **kwargs)

    def reply(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def valid_host(self):
        return self.headers.get("Host") in {
            f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

    def do_GET(self):
        if not self.valid_host():
            return self.reply(403, {"error": "Local requests only."})
        path = urlsplit(self.path).path
        if path == "/api/runs":
            return self.reply(200, self.library.runs())
        if path in ("/", "/index.html", "/viewer.js", "/style.css", "/evaluation.json"):
            filename = "index.html" if path == "/" else path[1:]
            content_type = {"index.html": "text/html; charset=utf-8", "viewer.js": "text/javascript", "style.css": "text/css", "evaluation.json": "application/json"}[filename]
            compressed = False
            file = self.library.root / filename
        elif re.fullmatch(r"/runs/[a-f0-9]{20}/data.json", path):
            file = self.library.root / (path[1:] + ".gz")
            content_type, compressed = "application/json", True
        else:
            return self.reply(404, {"error": "Not found."})
        if not file.is_file() or file.is_symlink() or not file.resolve().is_relative_to(self.library.root):
            return self.reply(404, {"error": "Not found."})
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        if compressed:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(file.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        with file.open("rb") as handle:
            import shutil
            shutil.copyfileobj(handle, self.wfile)

    def do_POST(self):
        origin = self.headers.get("Origin")
        allowed_origin = f"http://{self.headers.get('Host')}"
        if (not self.valid_host() or self.headers.get("X-Mosaic-UI") != "1" or
                (origin is not None and origin != allowed_origin) or
                self.headers.get("Sec-Fetch-Site") == "cross-site"):
            return self.reply(403, {"error": "Use the local visualization page."})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 100000:
                raise ValueError("Invalid request size.")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object.")
            action = urlsplit(self.path).path.removeprefix("/api/")
            if action not in ("star", "description", "delete", "restore", "purge"):
                raise ValueError("Unknown action.")
            self.reply(200, self.library.mutate(action, body))
        except (ValueError, TypeError, OSError) as error:
            self.reply(400, {"error": str(error)})

    def log_message(self, fmt, *args):
        if args and str(args[1] if len(args) > 1 else "") not in ("200", "304"):
            super().log_message(fmt, *args)


def server(root, port=8765):
    return ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, library=Library(root)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if not (args.library / "index.html").is_file():
        raise SystemExit("Build the library first with scripts/build.py.")
    app = server(args.library, args.port)
    url = f"http://127.0.0.1:{app.server_port}"
    print(url, flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        app.server_close()
