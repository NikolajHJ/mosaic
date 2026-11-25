"""Pin the last published segment-3 save and retire its identified LSF reservations."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


WORKER, CALLBACK, OLD_CANDIDATE = 30264092, 30264093, 30302772
LABEL = "e48-r7-rtearly-s03"
OLD_DIR = "vae-e48-encoder3-nextleg-20261123a"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save(path, value):
    """Publish a complete receipt in one rename. The operator lock serializes calls."""
    with tempfile.NamedTemporaryFile("w", encoding="utf8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    Path(handle.name).replace(path)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(src, copied, first):
    """Open before the rolling save unlinks its predecessor; verify the copied bytes."""
    latest = read(src / "latest.json")
    u = latest["updates"]
    name = f"checkpoint-{u:09d}.pt"
    start = src / name
    require(Path(latest["checkpoint"]) == start and u >= first
            and latest["completed_epoch"] == latest["epoch"] == 47 and
            latest["row"] == latest["real_points"] == (u-575498)*240 and
            0 < latest["row"] < 60000000 and re.fullmatch("[a-f0-9]{64}", latest["sha256"]),
            "Invalid source checkpoint cursor, path or hash")
    copied.mkdir(exist_ok=True)
    dest = copied / name
    if not dest.exists():
        with start.open("rb") as stream, tempfile.NamedTemporaryFile(dir=copied, delete=False) as target:
            tmp = Path(target.name)
            try:
                shutil.copyfileobj(stream, target)
                target.flush()
                require(sha(tmp) == latest["sha256"], "Checkpoint copy checksum differs")
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        tmp.replace(dest)
    require(sha(dest) == latest["sha256"], "Pinned checkpoint has changed")
    return latest


def logged_progress(src, minimum):
    """Read the last fully flushed per-update record; progress.json can lag ten steps."""
    progress = max(minimum, read(src / "progress.json")["updates"])
    with (src / "training.jsonl").open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 256*1024))
        for line in reversed(stream.read().splitlines()):
            try:
                return max(progress, json.loads(line)["updates"])
            except (ValueError, KeyError, UnicodeDecodeError):
                continue
    return progress


def run(directory, action):
    release = read(directory / "release.json")
    prod = Path(release["production_control_dir"])
    sys.path.insert(0, str(prod))
    import epoch48_cutover as cut

    plan = read(prod / "plan.json")
    signature = sha(directory / "release.json")
    require(sha(prod / "plan.json") == release["production_plan_sha256"] and
            plan["job_prefix"] == "e48-r7-rtearly", "Production plan differs")
    src = Path(plan["output_root"]) / "segment-03"
    copied = directory / "source"
    stop = directory / "stop.json"
    intent = directory / "stop-worker.intent.json"
    old_dir = prod.parent / OLD_DIR
    if stop.exists():
        result = read(stop)
        require(result["release_sha256"] == signature and read(copied / "latest.json") == result["latest"]
                and sha(Path(result["latest"]["checkpoint"])) == result["latest"]["sha256"],
                "Completed handoff receipt or checkpoint changed")
        return result
    launch = read(prod / (LABEL + ".launch.json"))
    fork = read(src / "fork.json")
    require(read(prod / (LABEL + ".claim.json")) == {"job": WORKER} and int(launch["job_id"]) == WORKER
            and launch["source_archive_sha256"] == plan["archive_sha256"]
            and fork["config_sha256"] == hashlib.sha256(
                f"{plan['config_sha256']}:{fork['parent']['sha256']}:3".encode()).hexdigest()
            and fork["resume"]["updates"] == 600116, "Unexpected source writer or fork")

    def checked_rows():
        rows = cut.jobs()
        for prefix, folder in (("e48-r7-rtearly-s04", prod), ("e48-r7-rtearly-after-s04", prod),
                               ("e48-encoder3-eval", old_dir)):
            require(not any(folder.glob(prefix + ".*.json")), "Superseded successor was already claimed/submitted")
            require(not any(r["JOB_NAME"] == prefix and r["STAT"] not in ("DONE", "EXIT") for r in rows.values()),
                    "Superseded successor is active")
        require(not (src.parent / "segment-04").exists() and not (old_dir / "candidate").exists(),
                "Superseded successor has already written output")
        return rows

    def held(rows):
        for arm in ("control", "candidate"):
            name = f"e48-fork3-{arm}"
            receipt = read(directory / f"{name}.receipt.json")
            row = cut.reservation(directory, name, rows, receipt["id"])
            require(row["STAT"] == "PSUSP" and cut.unstarted(row) and row["QUEUE"] == "gpuv100"
                    and "-H" in receipt["argv"], f"Expected held {arm} replacement")

    def take_snapshot():
        for _ in range(3):
            try:
                latest = pin(src, copied, fork["resume"]["updates"])
                save(directory / "snapshot.json", latest)
                return latest
            except FileNotFoundError:
                # An atomic rolling save may have replaced the receipt between read and open.
                time.sleep(.1)
        raise RuntimeError("Could not copy a published checkpoint")

    rows = checked_rows()
    if action == "snapshot":
        row = cut.reservation(prod, LABEL, rows, WORKER)
        require(row["STAT"] == "RUN", "Expected the original running worker")
        return take_snapshot()
    if action == "stop":
        held(rows)
        if not intent.exists():
            require(cut.reservation(prod, LABEL, rows, WORKER)["STAT"] == "RUN", "Expected running source")
            # Snapshot before retiring any reservation, then refresh just before stopping.
            take_snapshot()
        for ident, name, folder, queue, dep in (
            (OLD_CANDIDATE, "e48-encoder3-s04", old_dir, "gpuv100", f"done({CALLBACK})"),
            (CALLBACK, "e48-r7-rtearly-after-s03", prod, "hpc", f"ended({WORKER})")):
            for attempt in range(15):
                observed = checked_rows()
                if ident not in observed:
                    observed[ident] = cut.retired_row(directory, signature, ident)
                row = cut.reservation(folder, name, observed, ident)
                require(row["QUEUE"] == queue and cut.unstarted(row), "Old reservation already started")
                if row["STAT"] == "EXIT":
                    break
                require(row["STAT"] == "PEND" and row["DEPENDENCY"] == dep, "Old dependency or state changed")
                if attempt == 0:
                    cut.retire_pending(directory, signature, ident, row)
                time.sleep(.5)
            else:
                raise RuntimeError(f"Awaiting scheduler retirement of {ident}")
        rows = checked_rows()
        held(rows)
        if not intent.exists():
            latest = take_snapshot()
            worker = cut.reservation(prod, LABEL, rows, WORKER)
            require(worker["STAT"] == "RUN" and worker["QUEUE"] == "gpuv100", "Source has stopped independently")
            save(intent, dict(worker=WORKER, row=worker, release_sha256=signature, latest=latest))
        require(read(intent)["release_sha256"] == signature, "Foreign stop intent")
        ack = directory / "stop-worker.ack.json"
        if not ack.exists() and WORKER in rows:
            worker = cut.reservation(prod, LABEL, rows, WORKER)
            if worker["STAT"] == "RUN":
                # The positive ID, command, owner and nonce-bearing source launch identify the writer.
                save(ack, dict(worker=WORKER, reply=cut.command("bkill", str(WORKER))))
        return dict(worker=WORKER, acknowledged=ack.is_file(), snapshot=read(intent)["latest"])

    proof = read(intent)
    require(proof["worker"] == WORKER and proof["release_sha256"] == signature, "Foreign source stop")
    if WORKER in rows:
        row = cut.reservation(prod, LABEL, rows, WORKER)
        require(row["STAT"] == "EXIT", "Wait for terminal source worker before finalizing")
    else:
        history = subprocess.check_output(["bhist", "-n", "1", "-l", str(WORKER)], text=True, timeout=30)
        require(f"Job <{WORKER}>" in history and re.search(r":\s*Exited", history)
                and "s234818" in history and (directory / "stop-worker.ack.json").exists(),
                "Could not verify source worker's terminal state in scheduler history")
        row = dict(proof["row"], STAT="EXIT", terminal_evidence="bhist")
    previous = proof["latest"]
    fallback = None
    try:
        last = pin(src, copied, previous["updates"])
    except (FileNotFoundError, json.JSONDecodeError) as error:
        # SIGTERM can interrupt the writer around publication. Keep the verified pre-stop copy.
        last, fallback = previous, str(error)
    ck = copied / Path(last["checkpoint"]).name
    require(sha(ck) == last["sha256"] and (last["updates"] != previous["updates"] or last == previous),
            "Conflicting final checkpoint")
    latest = dict(last, checkpoint=str(ck))
    progress = logged_progress(src, latest["updates"])
    result = dict(worker=WORKER, release_sha256=signature, production_plan_sha256=release["production_plan_sha256"],
                  terminal=row, original_latest=last, latest=latest, progress_update=progress,
                  discarded_updates_lower_bound=progress-latest["updates"], fallback=fallback)
    save(copied / "latest.json", latest)
    save(stop, result)
    return result


if __name__ == "__main__":
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "stop", "finalize"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    with (root / "stop.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(run(root, args.action)), flush=True)
