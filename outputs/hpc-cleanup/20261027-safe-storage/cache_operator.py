"""Exact-file stdlib cache maintenance; never deletes scientific artifacts.

Run remotely with inventory/execute/verify, --plan and --protected-inputs.
inventory additionally accepts --control (fresh submissions directory).
execute/verify require --plan-sha256 from the reviewed inventory output.
Only generated regular cache files in inactive latentmol releases are eligible.
No model imports, job mutations, directory removal or automatic retry occurs.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import time


ROOT = Path("/work3/s234818/latentmol")
RELEASES = ROOT / "releases"
VERSION = "inactive-release-generated-cache-cleanup-v1"
PYTEST_FILES = {".gitignore", "CACHEDIR.TAG", "README.md", "v/cache/nodeids",
                "v/cache/lastfailed", "v/cache/stepwise", "v/cache/durations"}
MANIFESTS = {"release_manifest.json": "files", "source_manifest.json": "files",
             "embedding-release-manifest.json": "members"}
START = time.monotonic()
DEADLINE_SECONDS = 600


def now():
    return datetime.now(timezone.utc).isoformat()


def deadline():
    if time.monotonic() - START >= DEADLINE_SECONDS:
        raise TimeoutError("Bounded cache maintenance deadline reached; inspect receipts before retry")


def canonical(path):
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("Noncanonical maintenance path: " + str(path))
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError("Symlink path boundary: " + str(ancestor))
    return path


def fingerprint(info):
    return {key: getattr(info, "st_" + key) for key in
            ("dev", "ino", "mode", "uid", "nlink", "size", "mtime_ns", "ctime_ns", "blocks")}


def sha(path):
    canonical(path)
    handle = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(handle)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid():
            raise ValueError("Hash input is not an owned regular file: " + str(path))
        value = hashlib.sha256()
        while True:
            deadline()
            block = os.read(handle, 1024 * 1024)
            if not block:
                break
            value.update(block)
        if fingerprint(os.fstat(handle)) != fingerprint(before):
            raise ValueError("File changed during hashing: " + str(path))
        return value.hexdigest()
    finally:
        os.close(handle)


def sync_directory(path):
    canonical(path)
    handle = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def durable(path, value):
    canonical(path.parent)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def protected_inputs(path):
    canonical(path)
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or not document:
        raise ValueError("Require a nonempty exact protected path/SHA mapping")
    for name, expected in document.items():
        if (not isinstance(name, str) or not Path(name).is_relative_to(ROOT)
                or not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None):
            raise ValueError("Invalid protected-input mapping entry")
    return document


def pin_check(pins):
    observed = {name: sha(Path(name)) for name in pins}
    if observed != pins:
        raise ValueError("Protected checkpoint/source/input hashes changed")
    return observed


def scheduler():
    """Fresh JSON LSF snapshot, resolving actual release and run input roots."""
    deadline()
    command = ["bjobs", "-a", "-json", "-o", "jobid stat job_name command:4096"]
    run = subprocess.run(command, capture_output=True, text=True, timeout=30)
    empty = (not run.stdout.strip() and run.returncode in (0, 255)
             and run.stderr.strip() in ("No unfinished job found", "No unfinished jobs found"))
    if not empty and (run.returncode or run.stderr.strip()):
        raise ValueError("Fresh scheduler state is uncertain: " + repr((run.returncode, run.stderr)))
    document = {"RECORDS": []} if empty else json.loads(run.stdout)
    records = document.get("RECORDS")
    if not isinstance(records, list):
        raise ValueError("Unknown bjobs JSON record schema")
    excluded, jobs = set(), []
    root_pattern = re.compile(re.escape(str(ROOT)) + r"/(releases|runs)/([A-Za-z0-9_.-]+)")
    plan_pattern = re.compile(re.escape(str(ROOT / "submissions")) + r"/[A-Za-z0-9_./-]+\.json")

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)

    for source in records:
        row = {str(key).upper(): value for key, value in source.items()}
        job, state, text = str(row.get("JOBID", "")), row.get("STAT"), row.get("COMMAND")
        if not job.isdecimal() or not isinstance(text, str) or len(text) >= 4096:
            raise ValueError("Opaque or truncated scheduler command")
        if state in ("DONE", "EXIT"):
            continue
        if state not in ("RUN", "PEND", "WAIT", "PSUSP", "USUSP", "SSUSP"):
            raise ValueError("Unknown active scheduler state: " + str(state))
        texts, plans = [text], []
        for name in sorted(set(plan_pattern.findall(text))):
            path = canonical(Path(name))
            if not path.is_relative_to(ROOT / "submissions") or path.stat().st_size > 4 * 1024**2:
                raise ValueError("Unbounded active submission plan")
            before = fingerprint(path.stat())
            values = json.loads(path.read_text())
            if fingerprint(path.stat()) != before:
                raise ValueError("Active plan changed during inspection")
            texts.extend(strings(values))
            plans.append(dict(path=name, sha256=sha(path)))
        roots = set()
        for value in texts:
            for category, name in root_pattern.findall(value):
                if category == "releases" and name.endswith(".tar.gz"):
                    name = name[:-7]
                candidate = ROOT / category / name
                if not candidate.exists() or not candidate.is_dir():
                    continue
                canonical(candidate)
                roots.add(str(candidate))
                if category == "releases":
                    excluded.add(str(candidate))
        dbmol = any(str(ROOT) in value for value in texts)
        if dbmol and not any(Path(value).parent == RELEASES for value in roots):
            raise ValueError("Active DBMol job has unresolved release: " + job)
        if not dbmol and "/work3/s234818/mosaic" not in text:
            raise ValueError("Cannot safely classify an opaque active job: " + job)
        jobs.append(dict(job_id=job, state=state, command=text, roots=sorted(roots), plans=plans))
    return dict(utc=now(), command=command, stdout=run.stdout, stderr=run.stderr,
                returncode=run.returncode, active_jobs=jobs, excluded_releases=sorted(excluded))


def sealed_members(release):
    sealed, manifests = set(), {}
    for name, field in MANIFESTS.items():
        path = release / name
        if not path.exists() and not path.is_symlink():
            continue
        canonical(path)
        if path.stat().st_size > 32 * 1024**2:
            raise ValueError("Oversized source manifest")
        digest = sha(path)
        document = json.loads(path.read_text())
        values = document.get(field)
        if isinstance(values, dict):
            paths = list(values)
        elif isinstance(values, list):
            paths = [row if isinstance(row, str) else row["path"] for row in values]
        else:
            raise ValueError("Unknown sealed manifest schema: " + str(path))
        for value in paths:
            if not isinstance(value, str):
                raise ValueError("Nonstring sealed member path")
            relative = PurePosixPath(value)
            if relative.is_absolute() or ".." in relative.parts or "\\" in value:
                raise ValueError("Unsafe sealed member path")
            sealed.add(relative.as_posix())
        if sha(path) != digest:
            raise ValueError("Manifest changed during inspection")
        manifests[name] = digest
    if not manifests:
        raise ValueError("Release has no recognized source manifest")
    return sealed, manifests


def allowed(path, release):
    if release.parent != RELEASES or not path.is_relative_to(release):
        raise ValueError("Cache path escaped exact release scope")
    parts = path.relative_to(release).parts
    selected = "__pycache__" in parts[:-1] and path.suffix == ".pyc"
    if ".pytest_cache" in parts[:-1]:
        index = parts.index(".pytest_cache")
        selected = selected or "/".join(parts[index + 1:]) in PYTEST_FILES
    if not selected:
        return False
    canonical(release)
    canonical(path)
    if release.stat().st_uid != os.getuid():
        raise ValueError("Release owner changed")
    return True


def inventory(args, pins):
    if args.plan.exists() or args.plan.is_symlink():
        raise FileExistsError("Preserve an existing cache plan")
    control = args.control or args.plan.with_name(args.plan.stem + "-receipts")
    canonical(control.parent)
    if not control.is_relative_to(ROOT / "submissions") or control.exists():
        raise ValueError("Require a fresh control directory under latentmol/submissions")
    protected = pin_check(pins)
    active = scheduler()
    excluded = set(active["excluded_releases"])
    candidates, skipped, visited = [], [], 0
    for release in sorted(RELEASES.iterdir()):
        deadline()
        if release.is_symlink() or not release.is_dir() or release.stat().st_uid != os.getuid():
            continue
        canonical(release)
        if str(release) in excluded:
            skipped.append(dict(path=str(release), reason="active_release"))
            continue
        try:
            sealed, manifests = sealed_members(release)
        except Exception as error:
            skipped.append(dict(path=str(release), reason="manifest_uncertain: " + str(error)))
            continue
        for folder, dirs, names in os.walk(release, followlinks=False):
            deadline()
            visited += 1
            if visited > 250000:
                raise ValueError("Bounded release traversal exhausted")
            dirs[:] = sorted(name for name in dirs if not Path(folder, name).is_symlink())
            for name in sorted(names):
                path = Path(folder, name)
                if path.is_symlink() or not allowed(path, release):
                    continue
                info = path.lstat()
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1
                        or path.relative_to(release).as_posix() in sealed or str(path) in pins):
                    continue
                before = fingerprint(info)
                digest = sha(path)
                if fingerprint(path.lstat()) != before:
                    skipped.append(dict(path=str(path), reason="changed_during_inventory"))
                    continue
                candidates.append(dict(path=str(path), release=str(release), sha256=digest,
                    stat=before, manifests=manifests, allocated_bytes=info.st_blocks * 512))
    if pin_check(pins) != protected:
        raise ValueError("Protected identity changed during inventory")
    canonical(args.plan.parent)
    if not args.plan.is_relative_to(ROOT / "submissions"):
        raise ValueError("Plan must be under exact project submissions root")
    control.mkdir(exist_ok=False)
    sync_directory(control.parent)
    plan = dict(version=VERSION, status="inventory_only_no_deletions", utc=now(), root=str(ROOT),
        uid=os.getuid(), control=str(control), protected_inputs_sha256=sha(args.protected_inputs),
        protected=pins, scheduler=active, candidates=candidates, skipped=skipped,
        candidate_files=len(candidates), candidate_allocated_bytes=sum(row["allocated_bytes"] for row in candidates))
    durable(args.plan, plan)
    durable(control / "inventory.json", plan)
    return dict(status=plan["status"], plan=str(args.plan), plan_sha256=sha(args.plan),
                control=str(control), candidate_files=len(candidates), candidate_allocated_bytes=plan["candidate_allocated_bytes"])


def reviewed(args, pins):
    if not args.plan_sha256 or sha(args.plan) != args.plan_sha256:
        raise ValueError("Execution/verification needs the exact reviewed plan SHA256")
    plan = json.loads(args.plan.read_text())
    if (plan["version"] != VERSION or plan["root"] != str(ROOT) or plan["uid"] != os.getuid()
            or plan["protected"] != pins or plan["protected_inputs_sha256"] != sha(args.protected_inputs)):
        raise ValueError("Reviewed cache scope/protected mapping changed")
    control = canonical(Path(plan["control"]))
    if not control.is_relative_to(ROOT / "submissions"):
        raise ValueError("Control escaped exact submissions root")
    if json.loads((control / "inventory.json").read_text()) != plan:
        raise ValueError("Durable inventory differs from reviewed plan")
    return plan, control


def execute(args, pins):
    plan, control = reviewed(args, pins)
    if any((control / name).exists() for name in ("intent.json", "journal.jsonl", "completion.json")):
        raise FileExistsError("Prior execution exists; verify receipts instead of automatic retry")
    before = pin_check(pins)
    active = scheduler()
    durable(control / "intent.json", dict(version=VERSION, utc=now(), plan_sha256=args.plan_sha256,
        protected_before=before, scheduler=active, candidate_files=len(plan["candidates"])))
    removed, skipped = [], []
    with (control / "journal.jsonl").open("x", encoding="utf-8") as journal:
        sync_directory(control)
        def event(value):
            journal.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
        current_release, sealed, manifests = None, set(), {}
        for row in plan["candidates"]:
            deadline()
            path, release = Path(row["path"]), Path(row["release"])
            unlinked = False
            try:
                if current_release != release:
                    active = scheduler()
                    sealed, manifests = sealed_members(release)
                    current_release = release
                if str(release) in active["excluded_releases"]:
                    raise ValueError("release_is_now_active")
                if manifests != row["manifests"]:
                    raise ValueError("sealed_manifest_changed")
                if not allowed(path, release) or path.relative_to(release).as_posix() in sealed or str(path) in pins:
                    raise ValueError("not_generated_unsealed_cache")
                if fingerprint(path.lstat()) != row["stat"] or sha(path) != row["sha256"]:
                    raise ValueError("inventoried_file_changed")
                parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                    if (fingerprint(info) != row["stat"] or not stat.S_ISREG(info.st_mode)
                            or info.st_uid != os.getuid() or info.st_nlink != 1):
                        raise ValueError("fresh_file_identity_changed")
                    event(dict(event="unlink_intent", path=str(path), sha256=row["sha256"], stat=row["stat"], utc=now()))
                    if fingerprint(os.stat(path.name, dir_fd=parent, follow_symlinks=False)) != row["stat"]:
                        raise ValueError("file_changed_after_intent")
                    os.unlink(path.name, dir_fd=parent)
                    unlinked = True
                    os.fsync(parent)
                finally:
                    os.close(parent)
                removed.append(row)
                event(dict(event="unlinked", path=str(path), sha256=row["sha256"], allocated_bytes=row["allocated_bytes"], utc=now()))
            except Exception as error:
                if unlinked:
                    raise RuntimeError("Unlink occurred but durable completion is uncertain; preserve journal and never retry") from error
                skipped.append(dict(path=str(path), reason=type(error).__name__ + ": " + str(error)))
                event(dict(event="skipped", **skipped[-1], utc=now()))
    after = pin_check(pins)
    completion = dict(version=VERSION, status="completed_exact_cache_cleanup", utc=now(),
        plan_sha256=args.plan_sha256, removed_files=len(removed), skipped_files=len(skipped),
        removed_allocated_bytes=sum(row["allocated_bytes"] for row in removed),
        removed_paths=[row["path"] for row in removed], skipped=skipped,
        protected_before=before, protected_after=after,
        intent_sha256=sha(control / "intent.json"), journal_sha256=sha(control / "journal.jsonl"))
    durable(control / "completion.json", completion)
    return completion


def verify(args, pins):
    plan, control = reviewed(args, pins)
    done = json.loads((control / "completion.json").read_text())
    intent = json.loads((control / "intent.json").read_text())
    events = [json.loads(line) for line in (control / "journal.jsonl").read_text().splitlines()]
    if (done["plan_sha256"] != args.plan_sha256 or intent["plan_sha256"] != args.plan_sha256
            or done["intent_sha256"] != sha(control / "intent.json")
            or done["journal_sha256"] != sha(control / "journal.jsonl")):
        raise ValueError("Completion/intent/journal authentication failed")
    candidates = {row["path"]: row for row in plan["candidates"]}
    if len(candidates) != len(plan["candidates"]):
        raise ValueError("Duplicate inventory candidate")
    pending, terminal, removed, skipped = set(), set(), [], []
    for row in events:
        path = row["path"]
        if path not in candidates or path in terminal:
            raise ValueError("Journal path is outside reviewed allowlist")
        if row["event"] == "unlink_intent":
            if (path in pending or row["sha256"] != candidates[path]["sha256"]
                    or row["stat"] != candidates[path]["stat"]):
                raise ValueError("Unexpected unlink intent")
            pending.add(path)
        elif row["event"] == "unlinked":
            if (path not in pending or row["sha256"] != candidates[path]["sha256"]
                    or row["allocated_bytes"] != candidates[path]["allocated_bytes"]):
                raise ValueError("Unlink has no authenticated preceding intent")
            pending.remove(path)
            removed.append(path)
            terminal.add(path)
        elif row["event"] == "skipped":
            pending.discard(path)
            skipped.append(dict(path=path, reason=row["reason"]))
            terminal.add(path)
        else:
            raise ValueError("Unknown deletion journal event")
    if (pending or removed != done["removed_paths"] or len(removed) != done["removed_files"]
            or skipped != done["skipped"] or len(skipped) != done["skipped_files"]
            or len(removed) + len(skipped) != len(candidates)
            or done["removed_allocated_bytes"] != sum(candidates[path]["allocated_bytes"] for path in removed)
            or intent["candidate_files"] != len(candidates)
            or intent["protected_before"] != pins):
        raise ValueError("Incomplete deletion journal")
    for name in removed:
        path = Path(name)
        if path.exists() or path.is_symlink():
            raise ValueError("Removed cache path exists again: " + name)
    protected = pin_check(pins)
    if protected != done["protected_before"] or protected != done["protected_after"]:
        raise ValueError("Protected scientific identity changed")
    result = dict(status="passed_independent_cache_verification", utc=now(), removed_files=len(removed),
        removed_allocated_bytes=done["removed_allocated_bytes"], protected_hashes=len(protected),
        all_removed_paths_absent=True, plan_sha256=args.plan_sha256,
        completion_sha256=sha(control / "completion.json"), scheduler=scheduler())
    durable(control / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inventory", "execute", "verify"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--protected-inputs", required=True, type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--control", type=Path)
    args = parser.parse_args()
    canonical(ROOT)
    canonical(RELEASES)
    if ROOT.stat().st_uid != os.getuid() or RELEASES.stat().st_uid != os.getuid():
        raise ValueError("Exact project/release owner guard failed")
    pins = protected_inputs(args.protected_inputs)
    result = {"inventory": inventory, "execute": execute, "verify": verify}[args.mode](args, pins)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
