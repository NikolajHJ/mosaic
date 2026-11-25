"""Safely replace a pending epoch-48 reservation, or follow its running worker.

On the DTU login node: python3 epoch48_cutover.py /path/to/new/plan.json.
The replacement GPU starts held. Scheduler evidence and authenticated receipts
must agree before any old job is retired. A running GPU is never killed.
"""

import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

try:
    from . import epoch48_segment_chain as chain
except ImportError:
    import epoch48_segment_chain as chain


def jobs():
    fields = "jobid user stat queue start_time run_time job_name:100 dependency:180 command:768"
    raw = subprocess.check_output(["bjobs", "-a", "-u", "s234818", "-json", "-o", fields],
                                  text=True, timeout=40)
    return {int(r["JOBID"]): r for r in json.loads(raw)["RECORDS"] if r.get("JOBID", "").isdigit()}


def command(*argv):
    result = subprocess.run(list(argv), capture_output=True, text=True, timeout=40)
    if result.returncode:
        raise RuntimeError(f"{argv[0]} returned {result.returncode}: {result.stderr}{result.stdout}")
    return result.stdout.strip() + result.stderr.strip()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def unstarted(row):
    return row["RUN_TIME"] == "00:00:00" and row["START_TIME"].strip() in ("", "-")


def reservation(control, label, rows, ident):
    receipt = chain.read(control / (label + ".receipt.json"))
    argv = receipt["argv"]
    row = rows[ident]
    require(int(receipt["id"]) == ident and row["USER"] == "s234818" and row["JOB_NAME"] == label
            and shlex.split(row["COMMAND"]) == argv[argv.index(chain.PYTHON):], f"Reservation mismatch: {ident}")
    return row


def await_state(ident, predicate):
    for _ in range(18):
        row = jobs()[ident]
        if predicate(row):
            return row
        time.sleep(1)
    raise RuntimeError(f"Scheduler state did not settle for {ident}. Read receipts before retrying.")


def retirement_history(ident):
    """Resolve a lost cancellation reply only with an explicit undispatched end in LSF history."""
    raw = subprocess.run(["bhist", "-n", "1", "-l", str(ident)], text=True, capture_output=True, timeout=45)
    body = raw.stdout
    if raw.returncode or f"Job <{ident}>" not in body:
        return False
    started = re.search(r":\s*(Dispatched|Starting|Started|Running|Requeued|Resumed)\b", body, re.I)
    ended = re.search(r":\s*(Exited|Completed <exit>|Job (?:was )?killed before dispatch)", body, re.I)
    return bool(ended and not started and "Done successfully." not in body)


def retire_pending(control, sha, ident, row):
    """Limit LSF's signal to one verified positive job ID *while it is pending*.

    LSF may purge a killed pending job immediately. Persist the authenticated
    observation before the command and its reply before checking for removal.
    """
    require(type(ident) is int and ident > 0 and row["STAT"] == "PEND" and unstarted(row),
            "Refusing to retire anything other than one unstarted job")
    intent = control / f"retire-{ident}.intent.json"
    receipt = control / f"retire-{ident}.ack.json"
    argv = ["bkill", "-stat", "pend", "-s", "TERM", str(ident)]
    value = dict(plan_sha256=sha, id=ident, argv=argv, row=row.copy())
    if not intent.exists():
        chain.write_once(intent, value)
    else:
        require(chain.read(intent) == value, "Original reservation changed since retirement intent")
    if not receipt.exists():
        reply = command(*argv)
        chain.write_once(receipt, dict(plan_sha256=sha, id=ident, reply=reply))


def retired_row(control, sha, ident):
    """Recover LSF's immediately purged row using the exact pending-only operation."""
    intent = control / f"retire-{ident}.intent.json"
    ack = control / f"retire-{ident}.ack.json"
    require(intent.is_file(), f"Original job {ident} missing without a pending-only retirement intent")
    prior = chain.read(intent)
    require(prior["id"] == ident and prior["plan_sha256"] == sha
            and prior["argv"] == ["bkill", "-stat", "pend", "-s", "TERM", str(ident)]
            and prior["row"]["STAT"] == "PEND" and unstarted(prior["row"]), "Retirement intent differs")
    acknowledged = False
    if ack.is_file():
        reply = chain.read(ack)
        require(reply["id"] == ident and reply["plan_sha256"] == sha, "Retirement reply differs")
        message = reply.get("reply") or ""
        acknowledged = (f"<{ident}>" in message and re.search(r"terminat|signal|kill", message, re.I)
                        and not re.search(r"\b(no|not|cannot|failed|unable)\b", message, re.I))
    if not acknowledged:
        require(retirement_history(ident), f"No confirmation of pending-only retirement for {ident}")
    return dict(prior["row"], STAT="EXIT")


def cutover(path, *, stage_only=False):
    path = path.resolve(strict=True)
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    plan = json.loads(raw)
    control = path.parent
    require(not (control / "rollback.intent.json").exists(), "This reservation is being rolled back")
    old_control, old = chain.previous_plan(plan)
    h = plan["handoff"]
    segment, old_gpu, old_cb = h["segment"], h["worker_id"], h["callback_id"]
    original_parent = int(old["first_job_id"])
    labels = [chain.name(segment, plan=plan), chain.name(segment, after=True, plan=plan)]
    authorization = control / "cutover-authorized.json"
    gpu_receipt = control / (labels[0] + ".receipt.json")
    parked = f"ended({chain.read(gpu_receipt)['id']})" if gpu_receipt.is_file() else None

    def owned(rows):
        for jid, row in rows.items():
            if row["STAT"] in ("DONE", "EXIT"):
                continue
            label = row["JOB_NAME"]
            if label.startswith(chain.prefix(plan) + "-"):
                require(label in labels or authorization.exists(), "Another ring worker exists")
                # A dropped submission reply can leave a scheduler row and an exclusive
                # intent but no receipt. Reconcile only an exact, uniquely named job.
                receipt = control / (label + ".receipt.json")
                if not receipt.exists():
                    intent = chain.read(control / (label + ".intent.json"))
                    argv = intent["argv"]
                    require(row["USER"] == "s234818" and argv[argv.index("-J")+1] == label
                            and len([r for r in rows.values() if r["JOB_NAME"] == label]) == 1
                            and shlex.split(row["COMMAND"]) == argv[argv.index(chain.PYTHON):]
                            and row["QUEUE"] == argv[argv.index("-q")+1]
                            and row["STAT"] == ("PSUSP" if "-H" in argv else "PEND")
                            and unstarted(row), "Cannot reconcile ambiguous submission")
                    chain.write_once(receipt, dict(id=jid, argv=argv, reconciled=True))
                reservation(control, label, rows, jid)
            elif re.fullmatch(r"e48-r7-[a-z0-9]+-(?:after-)?s\d+", label):
                require(jid in (old_gpu, old_cb), "An original successor already exists")

    def old_pair(rows):
        observed = dict(rows)
        if old_gpu not in observed and segment == 3:
            require(chain.inspect(old_control, old, 2)["status"] == "paused", "Expected an original segment-2 checkpoint")
            original_worker = dict(STAT="DONE", RUN_TIME="finished", START_TIME="authenticated launch")
        else:
            if old_gpu not in observed:
                observed[old_gpu] = retired_row(control, sha, old_gpu)
            original_worker = reservation(old_control, chain.name(2, plan=old), observed, old_gpu)
        if old_cb not in observed:
            observed[old_cb] = retired_row(control, sha, old_cb)
        return original_worker, reservation(old_control, chain.name(2, after=True, plan=old), observed, old_cb)

    def predecessor_state(rows):
        gpu, cb = old_pair(rows)
        if original_parent in rows:
            require(rows[original_parent]["STAT"] == "DONE", "Expected successful original first leg")
        require(cb["STAT"] in ("PEND", "EXIT") and unstarted(cb), "Original callback already started")
        if cb["STAT"] == "EXIT":
            require((control / f"retire-{old_cb}.intent.json").is_file(), "Original callback exited independently")
        require(cb["DEPENDENCY"] in (f"ended({old_gpu})", parked) or cb["STAT"] == "EXIT",
                "Original callback dependency changed")
        successor = chain.name(3, plan=old)
        require(not any((old_control / (successor + suffix)).exists()
                        for suffix in (".intent.json", ".receipt.json", ".claim.json"))
                and not chain.output(old, 3).exists(),
                "Original callback has already reserved a successor")
        if segment == 2:
            require(gpu["STAT"] in ("PEND", "EXIT") and unstarted(gpu)
                    and not (old_control / (chain.name(2, plan=old) + ".claim.json")).exists()
                    and not chain.output(old, 2).exists(), "Original GPU has started; preserve it for a segment-3 handoff")
            require(gpu["DEPENDENCY"] in (f"done({original_parent})", parked) or gpu["STAT"] == "EXIT",
                    "Original worker dependency changed")
        else:
            require(gpu["STAT"] in ("RUN", "DONE") and not unstarted(gpu), "Segment 3 requires a started segment 2")
        return gpu, cb

    rows = jobs()
    owned(rows)
    if not authorization.exists():
        predecessor_state(rows)
        if segment == 2:
            require(chain.inspect(old_control, old, 1)["status"] == "paused", "Original epoch already complete")
    result = chain.queue(path, plan, segment, sha)
    gpu_id, cb_id = result["worker"], result["callback"]
    parked = f"ended({gpu_id})"
    claim = dict(plan_sha256=sha, worker=gpu_id)
    rows = jobs()
    owned(rows)
    gpu = reservation(control, labels[0], rows, gpu_id)
    cb = reservation(control, labels[1], rows, cb_id)
    if not authorization.exists():
        expected_dep = ("", "-") if segment == 2 else (f"done({result['predecessor']})",)
        require(gpu["STAT"] == "PSUSP" and unstarted(gpu) and gpu["QUEUE"] == "gpuv100"
                and gpu["DEPENDENCY"].strip() in expected_dep, "Replacement GPU is not held as expected")
        require(cb["STAT"] == "PEND" and unstarted(cb) and cb["QUEUE"] == "hpc"
                and cb["DEPENDENCY"] == f"ended({gpu_id})", "Replacement callback dependency differs")
        if stage_only:
            predecessor_state(rows)
            return dict(result, stage="held_and_verified")
        # Retire the callback first: otherwise the outgoing GPU's termination
        # would release its dependent callback. Never send an unconditional kill.
        for ident in ((old_cb, old_gpu) if segment == 2 else (old_cb,)):
            snapshot = jobs()
            owned(snapshot)
            require(snapshot[gpu_id]["STAT"] == "PSUSP", "Replacement ceased being held during cutover")
            prior_gpu, prior_cb = predecessor_state(snapshot)
            r = prior_gpu if ident == old_gpu else prior_cb
            if r["STAT"] == "PEND":
                retire_pending(control, sha, ident, r)
                for _ in range(15):
                    snapshot = jobs()
                    observed = predecessor_state(snapshot)
                    if observed[0 if ident == old_gpu else 1]["STAT"] == "EXIT":
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError(f"Original job {ident} remains pending; replacement remains held")
        prior_gpu, prior_cb = predecessor_state(jobs())
        require(prior_cb["STAT"] == "EXIT" and (segment == 3 or prior_gpu["STAT"] == "EXIT"),
                "Original writer or callback still active")
        chain.write_once(authorization, claim)
    require(chain.read(authorization) == claim, "Authorization belongs to a different release or worker")
    if jobs()[gpu_id]["STAT"] == "PSUSP":
        command("bresume", str(gpu_id))
    released = await_state(gpu_id, lambda r: r["STAT"] in ("PEND", "RUN", "DONE"))
    rows = jobs()
    result["scheduler"] = {str(i): {k: rows[i][k] for k in ("STAT", "JOB_NAME", "QUEUE", "RUN_TIME", "DEPENDENCY")}
                           for i in (old_gpu, old_cb, gpu_id, cb_id) if i in rows}
    require(released["QUEUE"] == "gpuv100", "Replacement queue changed")
    if not (control / "cutover-result.json").exists():
        chain.write_once(control / "cutover-result.json", result)
    return result


def rollback_staged(path):
    """Withdraw only this plan's held reservations before touching the prior callback."""
    plan = chain.read(path)
    control = path.parent
    old_control, old = chain.previous_plan(plan)
    handoff = plan["handoff"]
    prior_cb = handoff["callback_id"]
    require(not (control / "cutover-authorized.json").exists()
            and not (control / f"retire-{prior_cb}.intent.json").exists(), "Old callback retirement has begun")
    rows = jobs()
    old_cb = reservation(old_control, chain.name(2, after=True, plan=old), rows, prior_cb)
    require(old_cb["STAT"] == "PEND" and unstarted(old_cb)
            and old_cb["DEPENDENCY"] == f"ended({handoff['worker_id']})", "Old callback has changed")
    segment = handoff["segment"]
    intent = control / "rollback.intent.json"
    if not intent.exists():
        chain.write_once(intent, {"plan_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for after, expected, selector in ((True, "PEND", "pend"), (False, "PSUSP", "susp")):
        label = chain.name(segment, after=after, plan=plan)
        receipt = control / (label + ".receipt.json")
        if not receipt.exists():
            require(not (control / (label + ".intent.json")).exists(), "Unconfirmed staged submission")
            continue
        ident = int(chain.read(receipt)["id"])
        require(ident > 0 and ident not in (handoff["worker_id"], prior_cb), "Invalid staged job ID")
        rows = jobs()
        if ident not in rows or rows[ident]["STAT"] == "EXIT":
            # Only jobs in this unpublished plan can be missing here. An
            # unfinished submission retains its intent and cannot be resubmitted.
            continue
        row = reservation(control, label, rows, ident)
        require(row["STAT"] == expected and unstarted(row), "Staged job changed before rollback")
        command("bkill", "-stat", selector, "-s", "TERM", str(ident))
        for _ in range(15):
            observed = jobs().get(ident)
            if observed is None or observed["STAT"] == "EXIT":
                break
            time.sleep(1)
        else:
            raise RuntimeError("Wait for the staged job retirement and reconcile before retry")
    return {"stage": "rolled_back", "old_callback": prior_cb}


if __name__ == "__main__":
    import fcntl
    path = Path(sys.argv[1]).resolve(strict=True)
    if sys.argv[2:] not in ([], ["--stage"], ["--rollback-staged"]):
        sys.exit("Usage: epoch48_cutover.py plan.json [--stage|--rollback-staged]")
    with (path.parent / "cutover.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = (rollback_staged(path) if sys.argv[2:] == ["--rollback-staged"]
                  else cutover(path, stage_only=sys.argv[2:] == ["--stage"]))
        print(json.dumps(result), flush=True)
