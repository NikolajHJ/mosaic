"""Stop the identified rings allocation and release a fresh round-trip continuation."""

import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
import time

try:
    from . import epoch48_cutover as cut
    from . import epoch48_segment_chain as chain
except ImportError:
    import epoch48_cutover as cut
    import epoch48_segment_chain as chain


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def persist(path, value):
    if not path.exists():
        chain.write_once(path, value)
    cut.require(chain.read(path) == value, f"Conflicting saved receipt: {path}")


def act(path, action):
    control, plan = path.parent, chain.read(path)
    sha = digest(path)
    early = plan["early_stop"]
    original = Path(early["superseded_control"])
    existing = chain.read(original / "plan.json")
    outgoing = plan["handoff"]["worker_id"]
    old_gpu, old_cb = early["superseded_worker_id"], early["superseded_callback_id"]
    cut.require(plan["handoff"]["segment"] == 3 and digest(original / "plan.json") == early["superseded_plan_sha256"]
                and original != control and plan["output_root"] != existing["output_root"]
                and chain.prefix(plan) != chain.prefix(existing)
                and {k: v for k, v in plan.items() if k not in ("early_stop", "output_root", "job_prefix")}
                == {k: v for k, v in existing.items() if k not in ("output_root", "job_prefix")},
                "Replacement differs from the queued training recipe or its parent")
    ring_control, ring = chain.previous_plan(plan)
    cut.require(chain.read(original / "cutover-authorized.json") == {
        "plan_sha256": digest(original / "plan.json"), "worker": old_gpu}, "Unauthenticated original reservation")
    cut.require(digest(Path(plan["archive"])) == plan["archive_sha256"], "Science archive changed")

    def older(ident, rows):
        is_cb = ident == old_cb
        label = chain.name(3, after=is_cb, plan=existing)
        observed = dict(rows)
        if ident not in observed:
            observed[ident] = cut.retired_row(control, sha, ident)
        row = cut.reservation(original, label, observed, ident)
        expected = f"ended({old_gpu})" if is_cb else f"done({outgoing})"
        pending = row
        if row["STAT"] == "EXIT":
            proof = chain.read(control / f"retire-{ident}.intent.json")
            cut.require(proof["plan_sha256"] == sha and proof["id"] == ident, "Retirement intent differs")
            pending = proof["row"]
        cut.require(cut.unstarted(row) and row["STAT"] in ("PEND", "EXIT")
                    and row["QUEUE"] == ("hpc" if is_cb else "gpuv100")
                    and pending["DEPENDENCY"] == expected,
                    f"Prior reservation changed: {ident}")
        return row

    def lineage(rows):
        for key in ("claim", "launch"):
            cut.require(not (original / (chain.name(3, plan=existing) + f".{key}.json")).exists(),
                        "Prior segment-3 worker has already started")
        cut.require(not chain.output(existing, 3).exists(), "Prior segment-3 output exists")
        old_pair = [older(ident, rows) for ident in (old_cb, old_gpu)]
        identifiers = {old_gpu, old_cb, outgoing}
        for after in (True, False):
            label = chain.name(3, after=after, plan=plan)
            receipt = control / (label + ".receipt.json")
            intent = control / (label + ".intent.json")
            if not receipt.exists() and intent.exists():
                # A lost bsub reply may have created exactly one matching reservation.
                matches = [r for r in rows.values() if r["JOB_NAME"] == label]
                argv = chain.read(intent)["argv"]
                if (len(matches) == 1 and cut.unstarted(matches[0]) and matches[0]["USER"] == "s234818"
                        and matches[0]["STAT"] == ("PEND" if after else "PSUSP")
                        and matches[0]["QUEUE"] == argv[argv.index("-q")+1]
                        and shlex.split(matches[0]["COMMAND"]) == argv[argv.index(chain.PYTHON):]):
                    chain.write_once(receipt, dict(id=int(matches[0]["JOBID"]), argv=argv, reconciled=True))
            if receipt.exists():
                identifiers.add(int(chain.read(receipt)["id"]))
        for ident, row in rows.items():
            if row["STAT"] not in ("DONE", "EXIT") and re.fullmatch(r"e48-r7-[a-z0-9]+-(?:after-)?s\d+", row["JOB_NAME"]):
                cut.require(ident in identifiers, f"Unexpected active epoch-48 reservation: {ident}")
        return old_pair

    def active_parent(rows):
        row = cut.reservation(ring_control, chain.name(2, plan=ring), rows, outgoing)
        cut.require(row["STAT"] in ("RUN", "USUSP", "SSUSP") and not cut.unstarted(row)
                    and row["QUEUE"] == "gpuv100", "Expected a started rings allocation")
        return row

    rows = cut.jobs()
    lineage(rows)
    if action == "stage":
        active_parent(rows)
        result = chain.queue(path, plan, 3, sha)
    else:
        result = dict(segment=3, predecessor=outgoing,
                      worker=chain.job_id(control, plan, 3),
                      callback=int(chain.read(control / (chain.name(3, after=True, plan=plan) + ".receipt.json"))["id"]))
    gpu, cb = result["worker"], result["callback"]

    def replacement(rows, *, released=False):
        worker = cut.reservation(control, chain.name(3, plan=plan), rows, gpu)
        callback = cut.reservation(control, chain.name(3, after=True, plan=plan), rows, cb)
        cut.require(len({gpu, cb, outgoing, old_gpu, old_cb}) == 5
                    and worker["QUEUE"] == "gpuv100" and worker["DEPENDENCY"] == f"ended({outgoing})"
                    and (released or (worker["STAT"] == "PSUSP" and cut.unstarted(worker)))
                    and callback["QUEUE"] == "hpc" and callback["STAT"] == "PEND" and cut.unstarted(callback)
                    and callback["DEPENDENCY"] == f"ended({gpu})", "Replacement reservation changed")
        return worker

    rows = cut.jobs()
    old_pair = lineage(rows)
    authorization = control / "cutover-authorized.json"
    worker = replacement(rows, released=(action == "release" and authorization.is_file()))
    if action == "stage":
        return dict(result, stage="held_and_verified")
    if action == "retire":
        active_parent(rows)
        for ident in (old_cb, old_gpu):
            rows = cut.jobs()
            lineage(rows)
            replacement(rows)
            row = older(ident, rows)
            if row["STAT"] == "PEND":
                cut.retire_pending(control, sha, ident, row)
                for _ in range(20):
                    if older(ident, cut.jobs())["STAT"] == "EXIT":
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError(f"Pending retirement of {ident}, inspect LSF before retry")
        return dict(result, retired=[old_cb, old_gpu])
    cut.require(all(r["STAT"] == "EXIT" for r in old_pair), "Original segment-3 reservations still active")
    stop_intent = control / "stop-parent.intent.json"
    if action == "stop":
        if stop_intent.exists():
            receipt = control / "stop-parent.ack.json"
            return dict(result, previous_stop=chain.read(stop_intent),
                        acknowledgement=chain.read(receipt) if receipt.exists() else None,
                        observed_state=rows.get(outgoing, {}).get("STAT"))
        active_parent(rows)
        latest = chain.saved_checkpoint(plan)
        rows = cut.jobs()
        replacement(rows)
        lineage(rows)
        parent = active_parent(rows)
        argv = ["bkill", str(outgoing)]
        chain.write_once(stop_intent, dict(plan_sha256=sha, argv=argv, row=parent, latest=latest))
        reply = cut.command(*argv)
        chain.write_once(control / "stop-parent.ack.json", dict(id=outgoing, reply=reply))
        return dict(result, stop_reply=reply, verified_before_stop=latest)
    if action != "release":
        raise ValueError(action)
    stop = chain.read(stop_intent)
    cut.require(stop["plan_sha256"] == sha and stop["argv"] == ["bkill", str(outgoing)],
                "Missing prior stop intent for this allocation")
    stopped = control / "stopped.json"
    if not stopped.is_file():
        parent = rows[outgoing]
        cut.require(parent["STAT"] == "EXIT", "Await terminal LSF exit before pinning the final checkpoint")
        value = chain.stopped_checkpoint(control, plan, parent)
        latest = value["latest"]
        cut.require(latest["updates"] >= stop["latest"]["updates"] and
                    (latest["updates"] != stop["latest"]["updates"] or latest == stop["latest"]),
                    "Final cursor regressed or a previously verified checkpoint changed")
        chain.write_once(stopped, value)
    value = chain.read(stopped)
    cut.require(value["worker_id"] == outgoing and value["plan_sha256"] == sha
                and value["latest"] == chain.saved_checkpoint(plan), "Stopped predecessor cursor changed")
    persist(authorization, {"plan_sha256": sha, "worker": gpu})
    if worker["STAT"] == "PSUSP":
        cut.command("bresume", str(gpu))
    scheduler = cut.await_state(gpu, lambda r: r["STAT"] in ("PEND", "RUN", "DONE"))
    return dict(result, latest=value["latest"], scheduler=scheduler)


if __name__ == "__main__":
    import fcntl
    path = Path(sys.argv[1]).resolve(strict=True)
    if len(sys.argv) != 3 or sys.argv[2] not in ("stage", "retire", "stop", "release"):
        sys.exit("Usage: epoch48_early_cutover.py plan.json stage|retire|stop|release")
    with (path.parent / "early-cutover.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(act(path, sys.argv[2])), flush=True)
